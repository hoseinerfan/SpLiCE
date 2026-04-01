#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare patch top-hits TSV files across multiple runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as name=/abs/path/to/top_hits.tsv (repeatable).",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        required=True,
        help="Comma-separated concept list to compare.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Top-k patches per concept to use for overlap/spread.",
    )
    return parser.parse_args()


def parse_run_specs(specs: List[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --run spec '{spec}'. Expected name=path.")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"Invalid --run spec '{spec}': empty name.")
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"TSV not found for run '{name}': {p}")
        out.append((name, p))
    return out


def parse_concepts(text: str) -> List[str]:
    concepts = [x.strip() for x in text.split(",") if x.strip()]
    if not concepts:
        raise ValueError("No concepts provided.")
    return concepts


def load_tsv(path: Path) -> Dict[str, List[Dict]]:
    data: Dict[str, List[Dict]] = defaultdict(list)
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"concept", "patch_index", "row", "col", "weight"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} missing required columns. Found: {reader.fieldnames}")
        for row in reader:
            concept = str(row["concept"]).strip()
            if not concept:
                continue
            data[concept].append(
                {
                    "patch_index": int(row["patch_index"]),
                    "row": int(row["row"]),
                    "col": int(row["col"]),
                    "weight": float(row["weight"]),
                }
            )
    for concept in data:
        data[concept].sort(key=lambda x: x["weight"], reverse=True)
    return data


def spread(items: List[Dict]) -> float:
    if not items:
        return float("nan")
    rows = [x["row"] for x in items]
    cols = [x["col"] for x in items]
    r0 = mean(rows)
    c0 = mean(cols)
    return mean([math.hypot(r - r0, c - c0) for r, c in zip(rows, cols)])


def patch_set(items: List[Dict], topk: int) -> set:
    return {x["patch_index"] for x in items[:topk]}


def main() -> None:
    args = parse_args()
    runs = parse_run_specs(args.run)
    concepts = parse_concepts(args.concepts)

    run_data = {name: load_tsv(path) for name, path in runs}

    print("=== Run files ===")
    for name, path in runs:
        print(f"{name}: {path}")

    for concept in concepts:
        print(f"\n=== Concept: {concept} ===")
        print("run\tcount\ttop1_patch\ttop1_weight\tavg_top5\tavg_top10\tspread_topk")
        concept_sets = {}
        for name, _ in runs:
            rows = run_data[name].get(concept, [])
            concept_sets[name] = patch_set(rows, args.topk)
            if rows:
                top1_patch = rows[0]["patch_index"]
                top1_weight = rows[0]["weight"]
                avg5 = mean([x["weight"] for x in rows[: min(5, len(rows))]])
                avg10 = mean([x["weight"] for x in rows[: min(10, len(rows))]])
            else:
                top1_patch = -1
                top1_weight = 0.0
                avg5 = 0.0
                avg10 = 0.0
            sp = spread(rows[: min(args.topk, len(rows))])
            sp_text = "nan" if math.isnan(sp) else f"{sp:.3f}"
            print(
                f"{name}\t{len(rows)}\t{top1_patch}\t{top1_weight:.6f}\t"
                f"{avg5:.6f}\t{avg10:.6f}\t{sp_text}"
            )

        if len(runs) >= 2:
            print("\nTop-k patch overlap matrix:")
            names = [name for name, _ in runs]
            print("run\t" + "\t".join(names))
            for a in names:
                row_vals = []
                for b in names:
                    a_set = concept_sets[a]
                    b_set = concept_sets[b]
                    if not a_set and not b_set:
                        row_vals.append("0/0")
                    else:
                        inter = len(a_set & b_set)
                        denom = max(1, min(len(a_set), len(b_set), args.topk))
                        row_vals.append(f"{inter}/{denom}")
                print(a + "\t" + "\t".join(row_vals))


if __name__ == "__main__":
    main()
