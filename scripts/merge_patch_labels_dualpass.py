#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge patch-level concept labels from two passes "
            "(typically textual dictionary + visual-attribute dictionary)."
        )
    )
    parser.add_argument("--text-labels-path", type=str, required=True, help="Text-side patch labels JSONL file or directory.")
    parser.add_argument("--visual-labels-path", type=str, required=True, help="Visual-side patch labels JSONL file or directory.")
    parser.add_argument("--output-path", type=str, required=True, help="Output JSONL file or directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories for JSONL files.")
    parser.add_argument("--text-weight", type=float, default=1.0)
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=10, help="Top-k merged concepts per patch.")
    parser.add_argument("--min-weight", type=float, default=0.0, help="Drop merged concepts below this score before renorm.")
    parser.add_argument("--id-field", type=str, default="", help="Row id field (auto if empty).")
    parser.add_argument("--include-visual-only", action="store_true", help="Include rows present only in visual labels.")
    parser.add_argument("--preserve-component-scores", action="store_true", help="Add text_score/visual_score to each merged concept.")
    parser.add_argument("--print-every-files", type=int, default=100)
    parser.add_argument("--summary-json", type=str, default=None)
    return parser.parse_args()


def list_jsonl(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")
    if recursive:
        files = sorted([p for p in path.rglob("*.jsonl") if p.is_file()])
    else:
        files = sorted([p for p in path.glob("*.jsonl") if p.is_file()])
    if not files:
        raise ValueError(f"No JSONL files found under: {path}")
    return files


def first_row(path: Path) -> Dict:
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"Empty JSONL: {path}")


def pick_id_field(row: Dict, preferred: str = "") -> str:
    if preferred and preferred in row:
        return preferred
    for key in ("id", "patch_id", "page_id", "doc_id"):
        if key in row:
            return key
    raise KeyError(f"Could not infer id field from keys: {list(row.keys())}")


def parse_concepts(row: Dict) -> List[Dict]:
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
        try:
            weight = float(item.get("weight", 0.0))
        except Exception:
            weight = 0.0
        out.append({"concept": concept, "weight": max(weight, 0.0)})
    return out


def normalize(items: List[Dict]) -> List[Dict]:
    if not items:
        return []
    total = sum(max(float(x.get("weight", 0.0)), 0.0) for x in items)
    if total <= 0:
        uniform = 1.0 / len(items)
        return [{"concept": x["concept"], "weight": uniform} for x in items]
    return [{"concept": x["concept"], "weight": max(float(x["weight"]), 0.0) / total} for x in items]


def load_rows(path: Path, id_field: str) -> Tuple[List[str], Dict[str, Dict]]:
    order: List[str] = []
    rows: Dict[str, Dict] = {}
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = str(row.get(id_field, "")).strip()
            if not rid:
                continue
            if rid not in rows:
                order.append(rid)
            rows[rid] = row
    return order, rows


def merge_concepts(
    text_concepts: List[Dict],
    visual_concepts: List[Dict],
    text_weight: float,
    visual_weight: float,
    topk: int,
    min_weight: float,
    preserve_component_scores: bool,
) -> Tuple[List[Dict], Dict[str, int]]:
    t_norm = normalize(text_concepts)
    v_norm = normalize(visual_concepts)

    # key -> [display, score, t_score, v_score, t_hit, v_hit]
    merged: Dict[str, List] = {}

    def add(items: List[Dict], mult: float, src: str) -> None:
        for item in items:
            concept = str(item["concept"]).strip()
            if not concept:
                continue
            key = concept.lower()
            score = float(item["weight"]) * mult
            if key not in merged:
                merged[key] = [concept, 0.0, 0.0, 0.0, 0, 0]
            merged[key][1] += score
            if src == "text":
                merged[key][2] += score
                merged[key][4] = 1
            else:
                merged[key][3] += score
                merged[key][5] = 1

    add(t_norm, text_weight, "text")
    add(v_norm, visual_weight, "visual")

    out: List[Dict] = []
    shared = 0
    for _, (concept, score, t_score, v_score, t_hit, v_hit) in merged.items():
        if score < min_weight:
            continue
        if t_hit and v_hit:
            source = "both"
            shared += 1
        elif t_hit:
            source = "text"
        else:
            source = "visual"
        rec = {
            "concept": concept,
            "weight": score,
            "source": source,
        }
        if preserve_component_scores:
            rec["text_score"] = t_score
            rec["visual_score"] = v_score
        out.append(rec)

    out.sort(key=lambda x: x["weight"], reverse=True)
    if topk > 0:
        out = out[:topk]

    total = sum(float(x["weight"]) for x in out)
    if total > 0:
        for x in out:
            x["weight"] = round(float(x["weight"]) / total, 6)
            if "text_score" in x:
                x["text_score"] = round(float(x["text_score"]), 6)
            if "visual_score" in x:
                x["visual_score"] = round(float(x["visual_score"]), 6)
    else:
        out = []

    info = {
        "text_concepts": len(t_norm),
        "visual_concepts": len(v_norm),
        "shared_concepts": shared,
        "merged_concepts": len(out),
    }
    return out, info


def ordered_union(primary: List[str], secondary: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in primary:
        if x not in seen:
            seen.add(x)
            out.append(x)
    for x in secondary:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def pair_files(
    text_path: Path,
    visual_path: Path,
    recursive: bool,
    include_visual_only: bool,
) -> List[Tuple[Optional[Path], Optional[Path], Path]]:
    """
    Returns tuples: (text_file_or_stub, visual_file_or_none, relative_key).
    If text_path is file, visual_path must be file.
    If both dirs, pairs by relative path from roots.
    """
    if text_path.is_file() and visual_path.is_file():
        return [(text_path, visual_path, Path(text_path.name))]
    if text_path.is_file() != visual_path.is_file():
        raise ValueError("Both inputs must be files or both must be directories.")

    text_files = list_jsonl(text_path, recursive)
    visual_files = list_jsonl(visual_path, recursive)
    text_map = {p.relative_to(text_path): p for p in text_files}
    visual_map = {p.relative_to(visual_path): p for p in visual_files}

    rels: Set[Path] = set(text_map.keys())
    if include_visual_only:
        rels.update(visual_map.keys())

    pairs: List[Tuple[Optional[Path], Optional[Path], Path]] = []
    for rel in sorted(rels):
        pairs.append((text_map.get(rel), visual_map.get(rel), rel))
    return pairs


def main() -> None:
    args = parse_args()

    text_path = Path(args.text_labels_path)
    visual_path = Path(args.visual_labels_path)
    output_path = Path(args.output_path)

    pairs = pair_files(text_path, visual_path, args.recursive, args.include_visual_only)

    if text_path.is_file():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    files_done = 0
    files_missing_visual = 0
    rows_written = 0
    rows_both = 0
    rows_text_only = 0
    rows_visual_only = 0

    for idx, (text_file, visual_file, rel) in enumerate(pairs, start=1):
        if visual_file is None:
            files_missing_visual += 1

        if text_file is not None:
            text_first = first_row(text_file)
            id_field = pick_id_field(text_first, args.id_field)
        elif visual_file is not None:
            visual_first = first_row(visual_file)
            id_field = pick_id_field(visual_first, args.id_field)
        else:
            continue

        if text_file is None:
            text_order = []
            text_rows = {}
        else:
            text_order, text_rows = load_rows(text_file, id_field)

        if visual_file is None:
            visual_rows = {}
        else:
            visual_first = first_row(visual_file)
            _ = pick_id_field(visual_first, id_field)  # Validate compatibility.
            _, visual_rows = load_rows(visual_file, id_field)

        if args.include_visual_only:
            write_order = ordered_union(text_order, visual_rows.keys())
        else:
            write_order = list(text_order)

        if text_path.is_file():
            out_file = output_path
        else:
            out_file = output_path / rel
            out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w") as out_handle:
            for rid in write_order:
                trow = text_rows.get(rid)
                vrow = visual_rows.get(rid)

                if trow is None and vrow is None:
                    continue

                if trow is None:
                    if not args.include_visual_only:
                        continue
                    rows_visual_only += 1
                    trow = {id_field: rid, "top_concepts": []}
                    base_row = dict(vrow)
                elif vrow is None:
                    rows_text_only += 1
                    vrow = {id_field: rid, "top_concepts": []}
                    base_row = dict(trow)
                else:
                    rows_both += 1
                    base_row = dict(trow)

                merged, info = merge_concepts(
                    text_concepts=parse_concepts(trow),
                    visual_concepts=parse_concepts(vrow),
                    text_weight=args.text_weight,
                    visual_weight=args.visual_weight,
                    topk=args.topk,
                    min_weight=args.min_weight,
                    preserve_component_scores=args.preserve_component_scores,
                )

                base_row[id_field] = rid
                base_row["top_concepts"] = merged
                base_row["merge_info"] = info
                base_row["l0_norm"] = float(len(merged))
                out_handle.write(json.dumps(base_row) + "\n")
                rows_written += 1

        files_done += 1
        if args.print_every_files > 0 and (idx % args.print_every_files == 0 or idx == len(pairs)):
            print(
                f"Progress: {idx}/{len(pairs)} files | rows_written={rows_written} "
                f"missing_visual_files={files_missing_visual}"
            )

    summary = {
        "text_labels_path": str(text_path),
        "visual_labels_path": str(visual_path),
        "output_path": str(output_path),
        "files_total": len(pairs),
        "files_done": files_done,
        "files_missing_visual": files_missing_visual,
        "rows_written": rows_written,
        "rows_with_both_sources": rows_both,
        "rows_text_only": rows_text_only,
        "rows_visual_only": rows_visual_only,
        "text_weight": args.text_weight,
        "visual_weight": args.visual_weight,
        "topk": args.topk,
        "min_weight": args.min_weight,
        "preserve_component_scores": bool(args.preserve_component_scores),
        "include_visual_only": bool(args.include_visual_only),
    }

    print("\n=== Merge Patch Labels Summary ===")
    for key in [
        "files_done",
        "files_missing_visual",
        "rows_written",
        "rows_with_both_sources",
        "rows_text_only",
        "rows_visual_only",
        "text_weight",
        "visual_weight",
        "topk",
        "min_weight",
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
