#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge page concept labels from two passes (e.g., full dictionary + visual-attribute dictionary)."
        )
    )
    parser.add_argument("--base-labels-jsonl", type=str, required=True, help="Primary page labels JSONL.")
    parser.add_argument("--attribute-labels-jsonl", type=str, required=True, help="Secondary/attribute-focused page labels JSONL.")
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--attribute-weight", type=float, default=0.8)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--include-attribute-only", action="store_true")
    parser.add_argument("--id-field", type=str, default="", help="Optional id field (auto-detected if empty).")
    return parser.parse_args()


def first_json_row(path: str) -> Dict:
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"Empty JSONL: {path}")


def pick_id_field(row: Dict, preferred: str) -> str:
    if preferred and preferred in row:
        return preferred
    for key in ("id", "page_id", "doc_id"):
        if key in row:
            return key
    raise KeyError(f"Could not detect id field. Keys: {list(row.keys())}")


def concept_field(row: Dict) -> Optional[str]:
    for key in ("top_concepts", "query_top_concepts"):
        if key in row and isinstance(row[key], list):
            return key
    return None


def parse_concepts(row: Dict) -> List[Dict]:
    field = concept_field(row)
    if field is None:
        return []
    out: List[Dict] = []
    for item in row.get(field, []):
        if isinstance(item, dict):
            concept = str(item.get("concept", "")).strip()
            if not concept:
                continue
            try:
                weight = float(item.get("weight", 0.0))
            except Exception:
                weight = 0.0
            out.append({"concept": concept, "weight": max(weight, 0.0)})
        elif isinstance(item, str):
            concept = item.strip()
            if concept:
                out.append({"concept": concept, "weight": 1.0})
    return out


def normalize_weights(items: List[Dict]) -> List[Dict]:
    if not items:
        return []
    total = sum(max(float(x.get("weight", 0.0)), 0.0) for x in items)
    if total <= 0:
        w = 1.0 / len(items)
        return [{"concept": x["concept"], "weight": w} for x in items]
    return [{"concept": x["concept"], "weight": max(float(x["weight"]), 0.0) / total} for x in items]


def load_rows(path: str, id_field: str) -> Tuple[List[str], Dict[str, Dict]]:
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
    base_concepts: List[Dict],
    attr_concepts: List[Dict],
    base_weight: float,
    attribute_weight: float,
    topk: int,
    min_weight: float,
) -> Tuple[List[Dict], Dict[str, int]]:
    base_norm = normalize_weights(base_concepts)
    attr_norm = normalize_weights(attr_concepts)

    # key -> [display_concept, score, base_hit, attr_hit]
    merged: Dict[str, List] = {}

    def add(items: List[Dict], mult: float, source: str) -> None:
        for c in items:
            concept = str(c["concept"]).strip()
            if not concept:
                continue
            key = concept.lower()
            score = float(c["weight"]) * mult
            if key not in merged:
                merged[key] = [concept, 0.0, 0, 0]
            merged[key][1] += score
            if source == "base":
                merged[key][2] = 1
            else:
                merged[key][3] = 1

    add(base_norm, base_weight, "base")
    add(attr_norm, attribute_weight, "attr")

    items = []
    shared = 0
    for _, (concept, score, base_hit, attr_hit) in merged.items():
        if score < min_weight:
            continue
        if base_hit and attr_hit:
            shared += 1
        items.append(
            {
                "concept": concept,
                "weight": score,
                "source": "both" if (base_hit and attr_hit) else ("base" if base_hit else "attribute"),
            }
        )
    items.sort(key=lambda x: x["weight"], reverse=True)
    if topk > 0:
        items = items[:topk]

    total = sum(x["weight"] for x in items)
    if total > 0:
        for x in items:
            x["weight"] = round(float(x["weight"] / total), 6)
    else:
        items = []

    info = {
        "base_concepts": len(base_norm),
        "attribute_concepts": len(attr_norm),
        "shared_concepts": shared,
        "merged_concepts": len(items),
    }
    return items, info


def ordered_union(primary: List[str], secondary: Iterable[str]) -> List[str]:
    seen = set()
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


def main() -> None:
    args = parse_args()

    base_first = first_json_row(args.base_labels_jsonl)
    attr_first = first_json_row(args.attribute_labels_jsonl)
    id_field = pick_id_field(base_first, args.id_field)
    # Ensure compatibility with secondary file if user specified custom id field.
    _ = pick_id_field(attr_first, id_field)

    base_order, base_rows = load_rows(args.base_labels_jsonl, id_field)
    _, attr_rows = load_rows(args.attribute_labels_jsonl, id_field)

    if args.include_attribute_only:
        write_order = ordered_union(base_order, attr_rows.keys())
    else:
        write_order = list(base_order)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged_count = 0
    base_only = 0
    attr_only = 0
    with_attr = 0

    with open(args.output_jsonl, "w") as out_handle:
        for rid in write_order:
            base_row = base_rows.get(rid)
            attr_row = attr_rows.get(rid)
            if base_row is None and attr_row is None:
                continue
            if base_row is None:
                base_row = {id_field: rid, "top_concepts": []}
                attr_only += 1
            elif attr_row is None:
                base_only += 1
                attr_row = {id_field: rid, "top_concepts": []}
            else:
                with_attr += 1

            merged_concepts, info = merge_concepts(
                base_concepts=parse_concepts(base_row),
                attr_concepts=parse_concepts(attr_row),
                base_weight=args.base_weight,
                attribute_weight=args.attribute_weight,
                topk=args.topk,
                min_weight=args.min_weight,
            )

            out_row = dict(base_row)
            out_row[id_field] = rid
            out_row["top_concepts"] = merged_concepts
            out_row["merge_info"] = info
            if "l0_norm" in out_row:
                out_row["base_l0_norm"] = out_row["l0_norm"]
            out_row["l0_norm"] = float(len(merged_concepts))
            out_handle.write(json.dumps(out_row) + "\n")
            merged_count += 1

    print(f"Wrote merged page labels: {args.output_jsonl}")
    print(f"Rows written: {merged_count}")
    print(f"Rows with both sources: {with_attr}")
    print(f"Base-only rows: {base_only}")
    if args.include_attribute_only:
        print(f"Attribute-only rows included: {attr_only}")
    print(f"Base weight: {args.base_weight}")
    print(f"Attribute weight: {args.attribute_weight}")
    print(f"Top-k: {args.topk}")
    print(f"ID field: {id_field}")


if __name__ == "__main__":
    main()
