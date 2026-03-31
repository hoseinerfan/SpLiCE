#!/usr/bin/env python3
import argparse
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more TREC runs on a sampled subset of queries."
    )
    parser.add_argument("--qrels", type=str, required=True, help="Path to TREC qrels file.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run in name=path format. Repeat for multiple runs.",
    )
    parser.add_argument("--sample-size", type=int, default=20, help="Number of query IDs to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--metrics",
        type=str,
        default="map,recip_rank,ndcg_cut.10",
        help="Comma-separated trec_eval metrics.",
    )
    parser.add_argument(
        "--qids-file",
        type=str,
        default=None,
        help="Optional file with one query_id per line. If set, sampling is skipped.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where sampled qids and filtered files are written.",
    )
    parser.add_argument("--trec-eval-bin", type=str, default="trec_eval")
    return parser.parse_args()


def parse_named_run(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"Invalid --run value '{text}'. Expected name=path.")
    name, path = text.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Invalid --run value '{text}'. Expected name=path.")
    return name, path


def read_qids_from_trec_file(path: Path) -> Set[str]:
    qids: Set[str] = set()
    with open(path, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            qids.add(parts[0])
    return qids


def read_qids_from_list(path: Path) -> List[str]:
    out: List[str] = []
    with open(path, "r") as handle:
        for line in handle:
            qid = line.strip()
            if qid:
                out.append(qid)
    return out


def filter_trec_file(in_path: Path, out_path: Path, keep_qids: Set[str]) -> int:
    kept = 0
    with open(in_path, "r") as in_handle, open(out_path, "w") as out_handle:
        for line in in_handle:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] in keep_qids:
                out_handle.write(line)
                kept += 1
    return kept


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def run_trec_eval(
    trec_eval_bin: str,
    metrics: List[str],
    qrels_path: Path,
    run_path: Path,
) -> Dict[str, float]:
    cmd: List[str] = [trec_eval_bin]
    for metric in metrics:
        cmd.extend(["-m", metric])
    cmd.extend([str(qrels_path), str(run_path)])

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"trec_eval failed for run {run_path}\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    values: Dict[str, float] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        metric, scope, value = parts[0], parts[1], parts[2]
        if scope != "all":
            continue
        try:
            values[metric] = float(value)
        except Exception:
            continue
    return values


def main() -> None:
    args = parse_args()

    trec_eval_path = shutil.which(args.trec_eval_bin)
    if trec_eval_path is None:
        raise FileNotFoundError(
            f"Could not find trec_eval binary '{args.trec_eval_bin}' in PATH."
        )

    qrels_path = Path(args.qrels)
    if not qrels_path.exists():
        raise FileNotFoundError(f"qrels file not found: {qrels_path}")

    run_specs = [parse_named_run(text) for text in args.run]
    run_paths: Dict[str, Path] = {}
    for name, path_text in run_specs:
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"run file not found for '{name}': {path}")
        run_paths[name] = path

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qrels_qids = read_qids_from_trec_file(qrels_path)
    candidate_qids = set(qrels_qids)
    run_qids: Dict[str, Set[str]] = {}
    for name, path in run_paths.items():
        qids = read_qids_from_trec_file(path)
        run_qids[name] = qids
        candidate_qids &= qids

    if not candidate_qids:
        raise ValueError("No overlapping query IDs between qrels and all runs.")

    if args.qids_file:
        requested = read_qids_from_list(Path(args.qids_file))
        sampled = [qid for qid in requested if qid in candidate_qids]
        if not sampled:
            raise ValueError("No query IDs from --qids-file are present in all inputs.")
    else:
        size = min(args.sample_size, len(candidate_qids))
        if size < args.sample_size:
            print(
                f"Requested sample-size={args.sample_size}, but only {len(candidate_qids)} overlapping qids are available."
            )
        rng = random.Random(args.seed)
        sampled = sorted(rng.sample(sorted(candidate_qids), size))

    sampled_set = set(sampled)
    sampled_qids_path = out_dir / "sampled_qids.txt"
    with open(sampled_qids_path, "w") as handle:
        for qid in sampled:
            handle.write(qid + "\n")

    sampled_qrels_path = out_dir / "sampled_qrels.trec"
    qrels_lines = filter_trec_file(qrels_path, sampled_qrels_path, sampled_set)

    sampled_run_paths: Dict[str, Path] = {}
    for name, path in run_paths.items():
        out_run = out_dir / f"sampled_run_{sanitize(name)}.trec"
        kept = filter_trec_file(path, out_run, sampled_set)
        sampled_run_paths[name] = out_run
        print(f"Run '{name}': kept {kept} lines -> {out_run}")

    print(f"Sampled queries: {len(sampled)}")
    print(f"Sampled qids file: {sampled_qids_path}")
    print(f"Filtered qrels lines: {qrels_lines} -> {sampled_qrels_path}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise ValueError("No valid metrics requested.")

    rows: Dict[str, Dict[str, float]] = {}
    for name, path in sampled_run_paths.items():
        rows[name] = run_trec_eval(trec_eval_path, metrics, sampled_qrels_path, path)

    header = ["run"] + metrics
    print("\nRESULTS")
    print("\t".join(header))
    for name in run_paths.keys():
        values = rows.get(name, {})
        cols = [name]
        for m in metrics:
            v = values.get(m)
            cols.append("NA" if v is None else f"{v:.6f}")
        print("\t".join(cols))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
