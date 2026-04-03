#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply concept-level score thresholds plus optional page-level gate "
            "to patch label JSONL files."
        )
    )
    parser.add_argument("--labels-path", type=str, required=True, help="Input JSONL file or directory.")
    parser.add_argument("--output-path", type=str, required=True, help="Output JSONL file or directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories for JSONL files.")
    parser.add_argument(
        "--concepts",
        type=str,
        required=True,
        help="Comma-separated concepts to gate (case-insensitive).",
    )
    parser.add_argument(
        "--concept-min-score",
        action="append",
        default=[],
        help="Per-concept threshold as concept=value (repeatable).",
    )
    parser.add_argument(
        "--concept-min-score-file",
        type=str,
        default=None,
        help="Optional TSV/CSV file with concept<tab|,>value.",
    )
    parser.add_argument(
        "--default-min-score",
        type=float,
        default=0.0,
        help="Fallback threshold for target concepts not listed in --concept-min-score.",
    )
    parser.add_argument(
        "--page-gate-concept",
        type=str,
        default="",
        help=(
            "Optional gate concept. A page must have enough hits of this concept "
            "to keep target concepts."
        ),
    )
    parser.add_argument(
        "--page-gate-min-score",
        type=float,
        default=0.0,
        help="A patch counts as a gate hit if gate-concept weight >= this value.",
    )
    parser.add_argument(
        "--page-gate-min-hits",
        type=int,
        default=1,
        help="Page passes only if gate-concept hit count >= this value.",
    )
    parser.add_argument(
        "--page-gate-min-max",
        type=float,
        default=0.0,
        help="Page also requires max gate-concept score >= this value.",
    )
    parser.add_argument(
        "--keep-other-concepts",
        action="store_true",
        help="Keep non-target concepts unchanged. If unset, only target concepts remain.",
    )
    parser.add_argument("--topk", type=int, default=0, help="Cap concepts per patch after gating (0 disables cap).")
    parser.add_argument("--renormalize", action="store_true", help="Renormalize kept concept weights per patch.")
    parser.add_argument("--summary-tsv", type=str, default=None, help="Optional page/concept activity TSV.")
    parser.add_argument("--summary-json", type=str, default=None, help="Optional summary JSON.")
    parser.add_argument("--print-every-files", type=int, default=100, help="Progress print interval.")
    return parser.parse_args()


def parse_csv_set(text: str) -> Set[str]:
    return {x.strip().lower() for x in text.split(",") if x.strip()}


def parse_concept_threshold_specs(specs: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --concept-min-score spec '{spec}'. Use concept=value.")
        concept, value = spec.split("=", 1)
        concept = concept.strip().lower()
        if not concept:
            raise ValueError(f"Invalid --concept-min-score spec '{spec}': empty concept.")
        try:
            out[concept] = float(value.strip())
        except Exception as exc:
            raise ValueError(f"Invalid threshold in spec '{spec}'.") from exc
    return out


def parse_concept_threshold_file(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Threshold file not found: {p}")
    out: Dict[str, float] = {}
    with open(p, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                left, right = line.split("\t", 1)
            elif "," in line:
                left, right = line.split(",", 1)
            else:
                raise ValueError(f"Invalid threshold line: {line}")
            concept = left.strip().lower()
            if not concept:
                continue
            try:
                out[concept] = float(right.strip())
            except Exception as exc:
                raise ValueError(f"Invalid threshold value in line: {line}") from exc
    return out


def list_jsonl_files(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")
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


def infer_page_id(row: Dict) -> str:
    page_id = str(row.get("page_id", "")).strip()
    if page_id:
        return page_id
    rid = str(row.get("id", "")).strip()
    if "#patch" in rid:
        return rid.split("#patch", 1)[0]
    if rid:
        return rid
    doc_id = str(row.get("doc_id", "")).strip()
    if doc_id:
        patch_index = row.get("patch_index", "")
        return f"{doc_id}:{patch_index}"
    return "unknown_page"


def parse_top_concepts(row: Dict) -> List[Dict]:
    raw = row.get("top_concepts", [])
    if not isinstance(raw, list):
        return []
    out: List[Dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        concept = str(item.get("concept", "")).strip()
        if not concept:
            continue
        rec = dict(item)
        try:
            rec["_weight"] = float(item.get("weight", 0.0))
        except Exception:
            rec["_weight"] = 0.0
        rec["_concept_lc"] = concept.lower()
        out.append(rec)
    out.sort(key=lambda x: x["_weight"], reverse=True)
    return out


def strip_internal_fields(items: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for item in items:
        rec = dict(item)
        rec.pop("_weight", None)
        rec.pop("_concept_lc", None)
        out.append(rec)
    return out


def renormalize_weights(items: List[Dict]) -> None:
    if not items:
        return
    total = sum(max(float(x.get("_weight", 0.0)), 0.0) for x in items)
    if total <= 0:
        return
    for item in items:
        weight = max(float(item.get("_weight", 0.0)), 0.0) / total
        item["_weight"] = weight
        item["weight"] = round(weight, 6)


def main() -> None:
    args = parse_args()

    target_concepts = parse_csv_set(args.concepts)
    if not target_concepts:
        raise ValueError("No target concepts parsed from --concepts.")

    concept_thresholds = parse_concept_threshold_specs(args.concept_min_score)
    concept_thresholds.update(parse_concept_threshold_file(args.concept_min_score_file))

    gate_concept = args.page_gate_concept.strip().lower()
    if not gate_concept:
        # Empty gate means "always pass".
        gate_concept = ""

    in_path = Path(args.labels_path)
    out_path = Path(args.output_path)
    files = list_jsonl_files(in_path, args.recursive)

    if in_path.is_file():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict] = []
    files_done = 0
    rows_in = 0
    rows_out = 0
    malformed_rows = 0
    target_items_before = 0
    target_items_after = 0
    page_pass_true = 0
    page_pass_false = 0

    for idx, in_file in enumerate(files, start=1):
        page_gate_hits: Dict[str, int] = defaultdict(int)
        page_gate_max: Dict[str, float] = defaultdict(float)
        page_set: Set[str] = set()

        # Pass 1: compute page-level gate stats.
        with open(in_file, "r") as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    malformed_rows += 1
                    continue
                page_id = infer_page_id(row)
                page_set.add(page_id)
                if not gate_concept:
                    continue
                for item in parse_top_concepts(row):
                    if item["_concept_lc"] != gate_concept:
                        continue
                    weight = float(item["_weight"])
                    if weight >= args.page_gate_min_score:
                        page_gate_hits[page_id] += 1
                    if weight > page_gate_max[page_id]:
                        page_gate_max[page_id] = weight
                    break

        page_pass: Dict[str, bool] = {}
        for page_id in page_set:
            if not gate_concept:
                ok = True
            else:
                ok = (
                    page_gate_hits.get(page_id, 0) >= args.page_gate_min_hits
                    and page_gate_max.get(page_id, 0.0) >= args.page_gate_min_max
                )
            page_pass[page_id] = ok
            if ok:
                page_pass_true += 1
            else:
                page_pass_false += 1

        # Pass 2: apply gating and write.
        out_file = compute_output_file(in_file, in_path, out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        page_concept_nonzero: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        page_concept_max_after: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        with open(in_file, "r") as src, open(out_file, "w") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    malformed_rows += 1
                    continue

                page_id = infer_page_id(row)
                rows_in += 1
                concepts = parse_top_concepts(row)
                kept: List[Dict] = []

                for item in concepts:
                    concept_lc = item["_concept_lc"]
                    weight = float(item["_weight"])

                    if concept_lc in target_concepts:
                        target_items_before += 1
                        min_score = float(concept_thresholds.get(concept_lc, args.default_min_score))
                        if (not page_pass.get(page_id, True)) or (weight < min_score):
                            continue
                        target_items_after += 1
                        page_concept_nonzero[page_id][concept_lc] += 1
                        if weight > page_concept_max_after[page_id][concept_lc]:
                            page_concept_max_after[page_id][concept_lc] = weight
                        kept.append(item)
                    else:
                        if args.keep_other_concepts:
                            kept.append(item)

                kept.sort(key=lambda x: x["_weight"], reverse=True)
                if args.topk > 0:
                    kept = kept[: args.topk]
                if args.renormalize and kept:
                    renormalize_weights(kept)
                else:
                    for item in kept:
                        item["weight"] = round(float(item["_weight"]), 6)

                row["top_concepts"] = strip_internal_fields(kept)
                dst.write(json.dumps(row) + "\n")
                rows_out += 1

        files_done += 1
        for page_id in sorted(page_set):
            for concept in sorted(target_concepts):
                summary_rows.append(
                    {
                        "input_file": str(in_file),
                        "page_id": page_id,
                        "concept": concept,
                        "nonzero": int(page_concept_nonzero[page_id].get(concept, 0)),
                        "max": round(float(page_concept_max_after[page_id].get(concept, 0.0)), 6),
                        "min_score": round(float(concept_thresholds.get(concept, args.default_min_score)), 6),
                        "page_gate_pass": bool(page_pass.get(page_id, True)),
                        "page_gate_hits": int(page_gate_hits.get(page_id, 0)),
                        "page_gate_max": round(float(page_gate_max.get(page_id, 0.0)), 6),
                    }
                )

        if args.print_every_files > 0 and (idx % args.print_every_files == 0 or idx == len(files)):
            print(
                f"Progress: {idx}/{len(files)} files | rows_in={rows_in} rows_out={rows_out} "
                f"target_kept={target_items_after}/{target_items_before}"
            )

    if args.summary_tsv:
        tsv_path = Path(args.summary_tsv)
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tsv_path, "w") as handle:
            handle.write(
                "page_id\tconcept\tnonzero\tmax\tmin_score\tpage_gate_pass\tpage_gate_hits\tpage_gate_max\tinput_file\n"
            )
            for row in summary_rows:
                handle.write(
                    f"{row['page_id']}\t{row['concept']}\t{row['nonzero']}\t{row['max']:.6f}\t"
                    f"{row['min_score']:.6f}\t{str(row['page_gate_pass'])}\t"
                    f"{row['page_gate_hits']}\t{row['page_gate_max']:.6f}\t{row['input_file']}\n"
                )
        print(f"Wrote: {tsv_path}")

    summary = {
        "input_path": str(in_path),
        "output_path": str(out_path),
        "files_in": len(files),
        "files_done": files_done,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "malformed_rows": malformed_rows,
        "target_concepts_count": len(target_concepts),
        "target_items_before": target_items_before,
        "target_items_after": target_items_after,
        "keep_ratio": round(
            float(target_items_after) / float(target_items_before) if target_items_before else 0.0,
            6,
        ),
        "page_gate_concept": gate_concept,
        "page_gate_min_score": float(args.page_gate_min_score),
        "page_gate_min_hits": int(args.page_gate_min_hits),
        "page_gate_min_max": float(args.page_gate_min_max),
        "page_pass_true": page_pass_true,
        "page_pass_false": page_pass_false,
        "default_min_score": float(args.default_min_score),
        "keep_other_concepts": bool(args.keep_other_concepts),
        "topk": int(args.topk),
        "renormalize": bool(args.renormalize),
        "concept_min_scores": {k: float(v) for k, v in sorted(concept_thresholds.items())},
    }

    print("\n=== Adaptive Gate Summary ===")
    for key in [
        "files_done",
        "rows_in",
        "rows_out",
        "target_items_before",
        "target_items_after",
        "keep_ratio",
        "page_gate_concept",
        "page_pass_true",
        "page_pass_false",
    ]:
        print(f"{key}: {summary[key]}")

    if args.summary_json:
        json_path = Path(args.summary_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
