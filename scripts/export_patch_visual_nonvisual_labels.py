#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive patch-level visual/non-visual classes from patch label JSONL "
            "(expects page_id, patch_index, top_concepts)."
        )
    )
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument(
        "--visual-concepts",
        type=str,
        default="visual_region,image_region,image,photo,figure,illustration",
    )
    parser.add_argument(
        "--non-visual-concepts",
        type=str,
        default="ocr_text,table_text,table_structure,table_all,text_region,text",
    )
    return parser.parse_args()


def parse_csv(raw: str) -> List[str]:
    return [x.strip().lower() for x in str(raw).split(",") if x.strip()]


def concept_weight(item: Dict[str, Any]) -> Tuple[str, float]:
    c = str(item.get("concept", "")).strip().lower()
    w = float(item.get("weight", 0.0))
    return c, w


def infer_patch_class(
    top_concepts: List[Dict[str, Any]],
    visual_names: List[str],
    non_visual_names: List[str],
) -> Tuple[str, float, float]:
    visual_score = 0.0
    non_visual_score = 0.0
    for item in top_concepts:
        c, w = concept_weight(item)
        if not c:
            continue
        if any(name in c for name in visual_names):
            visual_score += w
        if any(name in c for name in non_visual_names):
            non_visual_score += w

    if visual_score > 0 and non_visual_score > 0:
        return "mixed", visual_score, non_visual_score
    if visual_score > 0:
        return "visual", visual_score, non_visual_score
    if non_visual_score > 0:
        return "non_visual", visual_score, non_visual_score
    return "unknown", visual_score, non_visual_score


def main() -> None:
    args = parse_args()
    visual_names = parse_csv(args.visual_concepts)
    non_visual_names = parse_csv(args.non_visual_concepts)

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with in_path.open("r") as in_f, out_path.open("w") as out_f:
        for line in in_f:
            row = json.loads(line)
            top_concepts = row.get("top_concepts", []) or []
            patch_class, visual_score, non_visual_score = infer_patch_class(
                top_concepts=top_concepts,
                visual_names=visual_names,
                non_visual_names=non_visual_names,
            )

            out = {
                "id": row.get("id", ""),
                "page_id": row.get("page_id", ""),
                "patch_index": row.get("patch_index", -1),
                "top_concepts": top_concepts,
                "patch_class": patch_class,
                "visual_score": visual_score,
                "non_visual_score": non_visual_score,
            }
            out_f.write(json.dumps(out) + "\n")
            count += 1

    print(f"Wrote: {out_path}")
    print(f"rows: {count}")


if __name__ == "__main__":
    main()

