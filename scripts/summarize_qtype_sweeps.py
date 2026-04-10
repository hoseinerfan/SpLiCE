#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate qtype sweep runs across seeds and report per-config mean/std "
            "for dev metrics from train_summary_qtype.json files."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        required=True,
        help="Root directory to recursively scan for train_summary_qtype.json files.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional output path for grouped summary JSON.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help="Optional output path for grouped summary CSV.",
    )
    parser.add_argument(
        "--print-runs",
        action="store_true",
        help="Print every discovered run before grouped summary.",
    )
    return parser.parse_args()


def extract_config_name(summary_path: Path) -> str:
    # Expected layout is .../<config>/seed_<n>/train_summary_qtype.json
    parent = summary_path.parent
    if parent.name.startswith("seed_"):
        return parent.parent.name
    return parent.name


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    root = Path(args.runs_root)
    if not root.exists():
        raise FileNotFoundError(f"--runs-root not found: {root}")

    summary_files = sorted(root.rglob("train_summary_qtype.json"))
    if not summary_files:
        raise ValueError(f"No train_summary_qtype.json files found under: {root}")

    runs: List[Dict[str, Any]] = []
    by_config: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for path in summary_files:
        payload = json.load(open(path, "r"))
        best = payload.get("best_dev_metrics") or {}
        run = {
            "summary_path": str(path),
            "config_name": extract_config_name(path),
            "seed": payload.get("args", {}).get("seed"),
            "label_space": payload.get("args", {}).get("label_space"),
            "loss_type": payload.get("args", {}).get("loss_type"),
            "train_sampling": payload.get("args", {}).get("train_sampling"),
            "use_class_weights": payload.get("args", {}).get("use_class_weights"),
            "model_name": payload.get("args", {}).get("model_name"),
            "best_epoch": payload.get("best_epoch"),
            "dev_acc": safe_float(best.get("acc")),
            "dev_macro_recall": safe_float(best.get("macro_recall")),
            "dev_macro_f1": safe_float(best.get("macro_f1")),
            "dev_weighted_f1": safe_float(best.get("weighted_f1")),
        }
        runs.append(run)
        by_config[run["config_name"]].append(run)

    if args.print_runs:
        print("=== Individual Runs ===")
        for run in runs:
            print(json.dumps(run, indent=2))

    grouped: List[Dict[str, Any]] = []
    for config_name, rows in by_config.items():
        def metric_values(key: str) -> List[float]:
            return [r[key] for r in rows if r[key] is not None]

        accs = metric_values("dev_acc")
        recalls = metric_values("dev_macro_recall")
        f1s = metric_values("dev_macro_f1")
        wf1s = metric_values("dev_weighted_f1")

        entry = {
            "config_name": config_name,
            "n_runs": len(rows),
            "seeds": sorted([r["seed"] for r in rows if r["seed"] is not None]),
            "label_space": rows[0]["label_space"],
            "loss_type": rows[0]["loss_type"],
            "train_sampling": rows[0]["train_sampling"],
            "use_class_weights": rows[0]["use_class_weights"],
            "model_name": rows[0]["model_name"],
            "dev_macro_recall_mean": mean(recalls) if recalls else None,
            "dev_macro_recall_std": pstdev(recalls) if len(recalls) > 1 else 0.0,
            "dev_macro_f1_mean": mean(f1s) if f1s else None,
            "dev_macro_f1_std": pstdev(f1s) if len(f1s) > 1 else 0.0,
            "dev_acc_mean": mean(accs) if accs else None,
            "dev_acc_std": pstdev(accs) if len(accs) > 1 else 0.0,
            "dev_weighted_f1_mean": mean(wf1s) if wf1s else None,
            "dev_weighted_f1_std": pstdev(wf1s) if len(wf1s) > 1 else 0.0,
            "runs": rows,
        }
        grouped.append(entry)

    grouped.sort(
        key=lambda x: (
            x["dev_macro_recall_mean"] is not None,
            x["dev_macro_recall_mean"],
            x["dev_macro_f1_mean"] if x["dev_macro_f1_mean"] is not None else -1.0,
        ),
        reverse=True,
    )

    print("=== Grouped Summary ===")
    print(
        f"{'config':36s} {'n':>2s} {'recall_mean':>12s} {'recall_std':>11s} "
        f"{'f1_mean':>10s} {'f1_std':>9s} {'acc_mean':>10s}"
    )
    print("-" * 100)
    for row in grouped:
        print(
            f"{row['config_name'][:36]:36s} "
            f"{row['n_runs']:2d} "
            f"{(row['dev_macro_recall_mean'] or 0.0):12.4f} "
            f"{(row['dev_macro_recall_std'] or 0.0):11.4f} "
            f"{(row['dev_macro_f1_mean'] or 0.0):10.4f} "
            f"{(row['dev_macro_f1_std'] or 0.0):9.4f} "
            f"{(row['dev_acc_mean'] or 0.0):10.4f}"
        )

    if args.summary_json:
        out_json = Path(args.summary_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w") as handle:
            json.dump({"runs_root": str(root), "n_runs": len(runs), "grouped": grouped}, handle, indent=2)
        print(f"Wrote: {out_json}")

    if args.summary_csv:
        out_csv = Path(args.summary_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "config_name",
                    "n_runs",
                    "seeds",
                    "label_space",
                    "loss_type",
                    "train_sampling",
                    "use_class_weights",
                    "model_name",
                    "dev_macro_recall_mean",
                    "dev_macro_recall_std",
                    "dev_macro_f1_mean",
                    "dev_macro_f1_std",
                    "dev_acc_mean",
                    "dev_acc_std",
                    "dev_weighted_f1_mean",
                    "dev_weighted_f1_std",
                ],
            )
            writer.writeheader()
            for row in grouped:
                writer.writerow(
                    {
                        **{k: row[k] for k in writer.fieldnames if k in row},
                        "seeds": ",".join(str(x) for x in row["seeds"]),
                    }
                )
        print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
