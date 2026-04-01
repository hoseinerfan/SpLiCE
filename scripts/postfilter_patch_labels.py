#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-filter patch-level concept labels for better precision."
    )
    parser.add_argument("--labels-path", type=str, required=True, help="Input JSONL file or directory.")
    parser.add_argument("--output-path", type=str, required=True, help="Output JSONL file or directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories for JSONL files.")
    parser.add_argument("--topk", type=int, default=5, help="Max concepts kept per patch.")
    parser.add_argument("--min-weight", type=float, default=0.10, help="Absolute min concept weight.")
    parser.add_argument(
        "--min-relative-top1",
        type=float,
        default=0.60,
        help="Keep concept only if weight/top1 >= this threshold (when top1 > 0).",
    )
    parser.add_argument(
        "--drop-concepts",
        type=str,
        default="",
        help="Comma-separated concepts to drop (case-insensitive).",
    )
    parser.add_argument(
        "--drop-concepts-file",
        type=str,
        default=None,
        help="Optional newline-delimited concept list to drop (case-insensitive).",
    )
    parser.add_argument(
        "--keep-concepts",
        type=str,
        default="",
        help="Comma-separated concepts that bypass drop-concepts filtering (case-insensitive).",
    )
    parser.add_argument(
        "--renormalize",
        action="store_true",
        help="Renormalize retained weights to sum to 1.0 for each patch.",
    )
    parser.add_argument(
        "--drop-empty-rows",
        action="store_true",
        help="Drop rows where no concept remains after filtering.",
    )
    parser.add_argument("--print-every-files", type=int, default=100, help="Progress print interval.")
    parser.add_argument("--summary-json", type=str, default=None, help="Optional summary JSON output.")
    return parser.parse_args()


def parse_csv_set(text: str) -> Set[str]:
    if not text:
        return set()
    return {item.strip().lower() for item in text.split(",") if item.strip()}


def parse_file_set(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    out: Set[str] = set()
    with open(path, "r") as handle:
        for line in handle:
            item = line.strip().lower()
            if item:
                out.add(item)
    return out


def parse_top_concepts(row: Dict) -> List[Tuple[str, float]]:
    raw = row.get("top_concepts", [])
    if not isinstance(raw, list):
        return []
    concepts: List[Tuple[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        concept = str(item.get("concept", "")).strip()
        if not concept:
            continue
        try:
            weight = float(item.get("weight", 0.0))
        except Exception:
            weight = 0.0
        concepts.append((concept, weight))
    concepts.sort(key=lambda x: x[1], reverse=True)
    return concepts


def to_top_concept_dicts(concepts: List[Tuple[str, float]]) -> List[Dict]:
    return [{"concept": concept, "weight": round(float(weight), 6)} for concept, weight in concepts]


def list_jsonl_files(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"labels path not found: {path}")
    if recursive:
        files = sorted([p for p in path.rglob("*.jsonl") if p.is_file()])
    else:
        files = sorted([p for p in path.glob("*.jsonl") if p.is_file()])
    if not files:
        raise ValueError(f"No JSONL files found under: {path}")
    return files


def compute_output_file(input_path: Path, root_in: Path, root_out: Path) -> Path:
    if root_in.is_file():
        return root_out
    rel = input_path.relative_to(root_in)
    return root_out / rel


def keep_concept(
    concept: str,
    weight: float,
    top1: float,
    min_weight: float,
    min_relative_top1: float,
    drop_set: Set[str],
    keep_set: Set[str],
) -> bool:
    if weight < min_weight:
        return False
    if top1 > 0 and (weight / top1) < min_relative_top1:
        return False

    c = concept.lower()
    if c in drop_set and c not in keep_set:
        return False
    return True


def main() -> None:
    args = parse_args()

    in_path = Path(args.labels_path)
    out_path = Path(args.output_path)
    files = list_jsonl_files(in_path, args.recursive)

    if in_path.is_file():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.mkdir(parents=True, exist_ok=True)

    drop_set = parse_csv_set(args.drop_concepts) | parse_file_set(args.drop_concepts_file)
    keep_set = parse_csv_set(args.keep_concepts)

    rows_in = 0
    rows_out = 0
    rows_dropped = 0
    rows_empty_after = 0
    malformed_rows = 0
    concepts_in = 0
    concepts_out = 0
    files_done = 0

    for idx, in_file in enumerate(files, start=1):
        out_file = compute_output_file(in_file, in_path, out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(in_file, "r") as src, open(out_file, "w") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                rows_in += 1

                try:
                    row = json.loads(line)
                except Exception:
                    malformed_rows += 1
                    continue

                concepts = parse_top_concepts(row)
                concepts_in += len(concepts)

                if concepts:
                    top1 = concepts[0][1]
                else:
                    top1 = 0.0

                kept: List[Tuple[str, float]] = []
                for concept, weight in concepts:
                    if keep_concept(
                        concept=concept,
                        weight=weight,
                        top1=top1,
                        min_weight=args.min_weight,
                        min_relative_top1=args.min_relative_top1,
                        drop_set=drop_set,
                        keep_set=keep_set,
                    ):
                        kept.append((concept, weight))

                if args.topk > 0:
                    kept = kept[: args.topk]

                if args.renormalize and kept:
                    total = sum(w for _, w in kept)
                    if total > 0:
                        kept = [(c, w / total) for c, w in kept]

                if not kept:
                    rows_empty_after += 1
                    if args.drop_empty_rows:
                        rows_dropped += 1
                        continue

                row["top_concepts"] = to_top_concept_dicts(kept)
                concepts_out += len(kept)
                dst.write(json.dumps(row) + "\n")
                rows_out += 1

        files_done += 1
        if args.print_every_files > 0 and (idx % args.print_every_files == 0 or idx == len(files)):
            print(
                f"Progress: {idx}/{len(files)} files | rows_in={rows_in} rows_out={rows_out} "
                f"avg_kept={concepts_out / max(rows_out, 1):.3f}"
            )

    summary = {
        "input_path": str(in_path),
        "output_path": str(out_path),
        "files_in": len(files),
        "files_done": files_done,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_dropped": rows_dropped,
        "rows_empty_after_filter": rows_empty_after,
        "malformed_rows": malformed_rows,
        "concepts_in": concepts_in,
        "concepts_out": concepts_out,
        "avg_concepts_before": round(concepts_in / max(rows_in, 1), 6),
        "avg_concepts_after": round(concepts_out / max(rows_out, 1), 6),
        "topk": args.topk,
        "min_weight": args.min_weight,
        "min_relative_top1": args.min_relative_top1,
        "drop_concepts_count": len(drop_set),
        "keep_concepts_count": len(keep_set),
        "renormalize": bool(args.renormalize),
        "drop_empty_rows": bool(args.drop_empty_rows),
    }

    print("\n=== Postfilter Summary ===")
    for key in [
        "files_done",
        "rows_in",
        "rows_out",
        "rows_dropped",
        "rows_empty_after_filter",
        "malformed_rows",
        "avg_concepts_before",
        "avg_concepts_after",
        "topk",
        "min_weight",
        "min_relative_top1",
        "drop_concepts_count",
        "keep_concepts_count",
        "renormalize",
        "drop_empty_rows",
    ]:
        print(f"{key}: {summary[key]}")

    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote summary: {path}")


if __name__ == "__main__":
    main()
