#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Constrain concept labels by OCR word boxes. "
            "Target concepts are kept only where matched OCR text exists."
        )
    )
    parser.add_argument("--labels-path", type=str, required=True, help="Input labels JSONL file or directory.")
    parser.add_argument("--ocr-jsonl", type=str, required=True, help="OCR JSONL from extract_ocr_word_boxes.py.")
    parser.add_argument("--output-path", type=str, required=True, help="Output labels JSONL file or directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan labels directories for JSONL.")
    parser.add_argument(
        "--concepts",
        type=str,
        required=True,
        help="Comma-separated target concepts to constrain by OCR.",
    )
    parser.add_argument(
        "--concept-min-score",
        action="append",
        default=[],
        help="Per-concept minimum weight as concept=value (repeatable).",
    )
    parser.add_argument(
        "--default-min-score",
        type=float,
        default=0.0,
        help="Fallback minimum weight for target concepts.",
    )
    parser.add_argument(
        "--concept-terms",
        action="append",
        default=[],
        help="Optional explicit mapping: concept=term1|term2|... (repeatable).",
    )
    parser.add_argument(
        "--concept-terms-file",
        type=str,
        default=None,
        help="Optional TSV/CSV file: concept<tab|,>term1|term2|...",
    )
    parser.add_argument(
        "--term-match-mode",
        type=str,
        default="auto",
        choices=["auto", "any", "all"],
        help="How OCR terms must match for a concept to be active on a page.",
    )
    parser.add_argument("--min-word-conf", type=float, default=35.0, help="Ignore OCR words below this confidence.")
    parser.add_argument("--grid-size", type=int, default=32, help="Patch grid side length.")
    parser.add_argument("--image-token-start", type=int, default=0, help="Image patch start index.")
    parser.add_argument("--image-token-count", type=int, default=1024, help="Image patch count.")
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.01,
        help="Min overlap fraction (of patch area) between patch and OCR box.",
    )
    parser.add_argument(
        "--expand-cells",
        type=int,
        default=0,
        help="Expand allowed patch cells by this Manhattan radius around OCR hits.",
    )
    parser.add_argument(
        "--keep-other-concepts",
        action="store_true",
        help="Keep non-target concepts unchanged.",
    )
    parser.add_argument("--topk", type=int, default=0, help="Cap concepts per patch after filtering (0 disables).")
    parser.add_argument("--renormalize", action="store_true", help="Renormalize weights per patch after filtering.")
    parser.add_argument("--summary-tsv", type=str, default=None, help="Optional summary TSV path.")
    parser.add_argument("--summary-json", type=str, default=None, help="Optional summary JSON path.")
    parser.add_argument("--print-every-files", type=int, default=100, help="Progress interval.")
    return parser.parse_args()


def parse_csv_set(text: str) -> Set[str]:
    return {x.strip().lower() for x in text.split(",") if x.strip()}


def norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def concept_terms_from_name(concept: str) -> Set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", concept.lower()) if tok}


def parse_concept_score_specs(specs: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --concept-min-score spec '{spec}'.")
        concept, value = spec.split("=", 1)
        concept = concept.strip().lower()
        if not concept:
            raise ValueError(f"Invalid --concept-min-score spec '{spec}'.")
        out[concept] = float(value.strip())
    return out


def parse_concept_terms_specs(specs: List[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --concept-terms spec '{spec}'.")
        concept, terms = spec.split("=", 1)
        concept = concept.strip().lower()
        if not concept:
            raise ValueError(f"Invalid --concept-terms spec '{spec}'.")
        items = {norm_token(t) for t in terms.split("|") if norm_token(t)}
        if items:
            out[concept] = items
    return out


def parse_concept_terms_file(path: Optional[str]) -> Dict[str, Set[str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"concept-terms-file not found: {p}")
    out: Dict[str, Set[str]] = {}
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
                raise ValueError(f"Invalid concept-terms-file line: {line}")
            concept = left.strip().lower()
            if not concept:
                continue
            items = {norm_token(t) for t in right.split("|") if norm_token(t)}
            if items:
                out[concept] = items
    return out


def infer_page_id(row: Dict) -> str:
    page_id = str(row.get("page_id", "")).strip()
    if page_id:
        return page_id
    rid = str(row.get("id", "")).strip()
    if "#patch" in rid:
        return rid.split("#patch", 1)[0]
    return rid if rid else "unknown_page"


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
        rec["_concept_lc"] = concept.lower()
        try:
            rec["_weight"] = float(item.get("weight", 0.0))
        except Exception:
            rec["_weight"] = 0.0
        out.append(rec)
    out.sort(key=lambda x: x["_weight"], reverse=True)
    return out


def strip_internal(items: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for item in items:
        rec = dict(item)
        rec.pop("_concept_lc", None)
        rec.pop("_weight", None)
        out.append(rec)
    return out


def renormalize(items: List[Dict]) -> None:
    if not items:
        return
    s = sum(max(float(x.get("_weight", 0.0)), 0.0) for x in items)
    if s <= 0:
        return
    for item in items:
        w = max(float(item.get("_weight", 0.0)), 0.0) / s
        item["_weight"] = w
        item["weight"] = round(w, 6)


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
    return root_out / input_path.relative_to(root_in)


def load_ocr_map(path: Path, min_word_conf: float) -> Dict[str, Dict]:
    ocr_map: Dict[str, Dict] = {}
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            page_id = str(row.get("page_id", "")).strip()
            if not page_id:
                continue
            width = int(row.get("width", 0))
            height = int(row.get("height", 0))
            words = []
            for w in row.get("words", []):
                if not isinstance(w, dict):
                    continue
                conf = float(w.get("conf", -1.0))
                if conf < min_word_conf:
                    continue
                box = w.get("bbox_xyxy", [])
                if not isinstance(box, list) or len(box) != 4:
                    continue
                x0, y0, x1, y1 = [float(v) for v in box]
                if width <= 0 or height <= 0 or x1 <= x0 or y1 <= y0:
                    continue
                norm = str(w.get("norm", "")).strip().lower()
                if not norm:
                    norm = norm_token(str(w.get("text", "")))
                if not norm:
                    continue
                words.append(
                    {
                        "norm": norm,
                        "bbox": [x0 / width, y0 / height, x1 / width, y1 / height],
                        "conf": conf,
                    }
                )
            ocr_map[page_id] = {
                "width": width,
                "height": height,
                "words": words,
            }
    return ocr_map


def patch_bbox_from_index(patch_index: int, grid_size: int, image_token_start: int) -> Tuple[int, int, List[float]]:
    rel = patch_index - image_token_start
    row = rel // grid_size
    col = rel % grid_size
    x0 = col / grid_size
    y0 = row / grid_size
    x1 = (col + 1) / grid_size
    y1 = (row + 1) / grid_size
    return row, col, [x0, y0, x1, y1]


def overlap_fraction_of_patch(a: List[float], b: List[float]) -> float:
    # a is patch bbox, b is OCR bbox in normalized coordinates.
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    patch_area = (a[2] - a[0]) * (a[3] - a[1])
    return inter / patch_area if patch_area > 0 else 0.0


def term_match_pass(matched_terms: Set[str], required_terms: Set[str], mode: str) -> bool:
    if not required_terms:
        return False
    if mode == "any":
        return len(matched_terms) > 0
    if mode == "all":
        return required_terms.issubset(matched_terms)
    # auto
    if len(required_terms) <= 1:
        return len(matched_terms) > 0
    return required_terms.issubset(matched_terms)


def expand_allowed(allowed: Set[int], grid_size: int, image_token_start: int, radius: int) -> Set[int]:
    if radius <= 0 or not allowed:
        return allowed
    out: Set[int] = set(allowed)
    for patch_index in list(allowed):
        rel = patch_index - image_token_start
        row = rel // grid_size
        col = rel % grid_size
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = row + dr
                cc = col + dc
                if rr < 0 or rr >= grid_size or cc < 0 or cc >= grid_size:
                    continue
                out.add(image_token_start + rr * grid_size + cc)
    return out


def main() -> None:
    args = parse_args()

    if args.image_token_count != args.grid_size * args.grid_size:
        raise ValueError("--image-token-count must equal grid-size^2 in this script.")

    target_concepts = parse_csv_set(args.concepts)
    if not target_concepts:
        raise ValueError("No target concepts parsed from --concepts.")

    concept_scores = parse_concept_score_specs(args.concept_min_score)
    concept_terms = parse_concept_terms_specs(args.concept_terms)
    concept_terms.update(parse_concept_terms_file(args.concept_terms_file))

    for concept in target_concepts:
        concept_terms.setdefault(concept, concept_terms_from_name(concept))

    ocr_map = load_ocr_map(Path(args.ocr_jsonl), args.min_word_conf)

    in_path = Path(args.labels_path)
    out_path = Path(args.output_path)
    files = list_jsonl_files(in_path, args.recursive)
    if in_path.is_file():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.mkdir(parents=True, exist_ok=True)

    rows_in = 0
    rows_out = 0
    files_done = 0
    malformed_rows = 0
    target_before = 0
    target_after = 0
    summary_rows: List[Dict] = []

    for file_idx, in_file in enumerate(files, start=1):
        rows_by_page: Dict[str, List[Dict]] = defaultdict(list)

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
                rows_by_page[infer_page_id(row)].append(row)

        out_file = compute_output_file(in_file, in_path, out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w") as dst:
            for page_id in sorted(rows_by_page.keys()):
                page_rows = rows_by_page[page_id]
                ocr = ocr_map.get(page_id, {"words": []})
                words = ocr.get("words", [])

                concept_allowed: Dict[str, Set[int]] = {}
                concept_term_pass: Dict[str, bool] = {}
                concept_matched_terms: Dict[str, List[str]] = {}

                # Build allowed patch set per target concept for this page.
                for concept in sorted(target_concepts):
                    required_terms = concept_terms.get(concept, set())
                    matched_words = [w for w in words if w.get("norm", "") in required_terms]
                    matched_terms = {w.get("norm", "") for w in matched_words}
                    pass_terms = term_match_pass(matched_terms, required_terms, args.term_match_mode)
                    concept_term_pass[concept] = pass_terms
                    concept_matched_terms[concept] = sorted(matched_terms)

                    allowed: Set[int] = set()
                    if pass_terms:
                        for row in page_rows:
                            patch_index = int(row.get("patch_index", -1))
                            if patch_index < args.image_token_start:
                                continue
                            if patch_index >= args.image_token_start + args.image_token_count:
                                continue
                            _, _, pbox = patch_bbox_from_index(
                                patch_index,
                                grid_size=args.grid_size,
                                image_token_start=args.image_token_start,
                            )
                            ok = False
                            for w in matched_words:
                                overlap = overlap_fraction_of_patch(pbox, w["bbox"])
                                if overlap >= args.min_overlap:
                                    ok = True
                                    break
                            if ok:
                                allowed.add(patch_index)
                    allowed = expand_allowed(
                        allowed,
                        grid_size=args.grid_size,
                        image_token_start=args.image_token_start,
                        radius=args.expand_cells,
                    )
                    concept_allowed[concept] = allowed

                # Apply filter to page rows.
                page_before = defaultdict(int)
                page_after = defaultdict(int)
                page_max_after = defaultdict(float)

                for row in page_rows:
                    rows_in += 1
                    patch_index = int(row.get("patch_index", -1))
                    concepts = parse_top_concepts(row)
                    kept: List[Dict] = []

                    for item in concepts:
                        concept_lc = item["_concept_lc"]
                        weight = float(item["_weight"])
                        if concept_lc in target_concepts:
                            target_before += 1
                            page_before[concept_lc] += 1
                            min_score = float(concept_scores.get(concept_lc, args.default_min_score))
                            if weight < min_score:
                                continue
                            if not concept_term_pass.get(concept_lc, False):
                                continue
                            if patch_index not in concept_allowed.get(concept_lc, set()):
                                continue
                            target_after += 1
                            page_after[concept_lc] += 1
                            if weight > page_max_after[concept_lc]:
                                page_max_after[concept_lc] = weight
                            kept.append(item)
                        else:
                            if args.keep_other_concepts:
                                kept.append(item)

                    kept.sort(key=lambda x: x["_weight"], reverse=True)
                    if args.topk > 0:
                        kept = kept[: args.topk]
                    if args.renormalize and kept:
                        renormalize(kept)
                    else:
                        for item in kept:
                            item["weight"] = round(float(item["_weight"]), 6)

                    row["top_concepts"] = strip_internal(kept)
                    dst.write(json.dumps(row) + "\n")
                    rows_out += 1

                for concept in sorted(target_concepts):
                    summary_rows.append(
                        {
                            "input_file": str(in_file),
                            "page_id": page_id,
                            "concept": concept,
                            "before": int(page_before.get(concept, 0)),
                            "after": int(page_after.get(concept, 0)),
                            "max_after": round(float(page_max_after.get(concept, 0.0)), 6),
                            "term_pass": bool(concept_term_pass.get(concept, False)),
                            "matched_terms": "|".join(concept_matched_terms.get(concept, [])),
                            "allowed_patches": int(len(concept_allowed.get(concept, set()))),
                        }
                    )

        files_done += 1
        if args.print_every_files > 0 and (file_idx % args.print_every_files == 0 or file_idx == len(files)):
            print(
                f"Progress: {file_idx}/{len(files)} files | rows_in={rows_in} rows_out={rows_out} "
                f"target_kept={target_after}/{target_before}"
            )

    if args.summary_tsv:
        tsv_path = Path(args.summary_tsv)
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tsv_path, "w") as handle:
            handle.write(
                "page_id\tconcept\tbefore\tafter\tmax_after\tterm_pass\tmatched_terms\tallowed_patches\tinput_file\n"
            )
            for r in summary_rows:
                handle.write(
                    f"{r['page_id']}\t{r['concept']}\t{r['before']}\t{r['after']}\t{r['max_after']:.6f}\t"
                    f"{str(r['term_pass'])}\t{r['matched_terms']}\t{r['allowed_patches']}\t{r['input_file']}\n"
                )
        print(f"Wrote: {tsv_path}")

    summary = {
        "input_path": str(in_path),
        "ocr_jsonl": str(Path(args.ocr_jsonl)),
        "output_path": str(out_path),
        "files_in": len(files),
        "files_done": files_done,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "malformed_rows": malformed_rows,
        "target_concepts_count": len(target_concepts),
        "target_items_before": target_before,
        "target_items_after": target_after,
        "keep_ratio": round(float(target_after) / max(float(target_before), 1.0), 6),
        "grid_size": int(args.grid_size),
        "image_token_start": int(args.image_token_start),
        "image_token_count": int(args.image_token_count),
        "min_overlap": float(args.min_overlap),
        "expand_cells": int(args.expand_cells),
        "term_match_mode": args.term_match_mode,
        "min_word_conf": float(args.min_word_conf),
        "keep_other_concepts": bool(args.keep_other_concepts),
        "topk": int(args.topk),
        "renormalize": bool(args.renormalize),
    }

    print("\n=== OCR Constrained Summary ===")
    for k in [
        "files_done",
        "rows_in",
        "rows_out",
        "target_items_before",
        "target_items_after",
        "keep_ratio",
        "term_match_mode",
        "min_overlap",
        "expand_cells",
    ]:
        print(f"{k}: {summary[k]}")

    if args.summary_json:
        p = Path(args.summary_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
