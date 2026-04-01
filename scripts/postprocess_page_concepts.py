#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process page concept labels: filter noisy/generic concepts, "
            "renormalize, and optionally reduce top1 dominance."
        )
    )
    parser.add_argument("--labels-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--max-concepts", type=int, default=10)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--drop-generic-concepts", action="store_true")
    parser.add_argument("--generic-min-token-len", type=int, default=3)
    parser.add_argument(
        "--cap-top1-weight",
        type=float,
        default=0.0,
        help="If >0 and <1, smoothly cap top1 concept weight to this value when possible.",
    )
    parser.add_argument(
        "--keep-at-least-one",
        action="store_true",
        help="If filtering removes all concepts, keep original top concept.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def is_generic_concept(concept: str, min_token_len: int) -> bool:
    toks = tokenize(concept)
    if not toks:
        return True
    meaningful = [t for t in toks if t not in STOPWORDS]
    if not meaningful:
        return True
    if all(len(t) < min_token_len for t in meaningful):
        return True
    return False


def parse_concepts(row: Dict) -> List[Dict]:
    field = None
    for key in ("top_concepts", "query_top_concepts"):
        if key in row:
            field = key
            break
    if field is None:
        return []

    value = row.get(field, [])
    if not isinstance(value, list):
        return []

    out: List[Dict] = []
    for item in value:
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


def cap_top1(concepts: List[Dict], cap: float) -> List[Dict]:
    if not concepts:
        return concepts
    n = len(concepts)
    if n <= 1 or not (0.0 < cap < 1.0):
        return concepts

    concepts = normalize_weights(concepts)
    sorted_items = sorted(concepts, key=lambda x: x["weight"], reverse=True)
    w1 = sorted_items[0]["weight"]
    if w1 <= cap:
        return sorted_items

    uniform = 1.0 / n
    denom = w1 - uniform
    if denom <= 0:
        return sorted_items

    alpha = (w1 - cap) / denom
    alpha = min(max(alpha, 0.0), 1.0)
    out = []
    for x in sorted_items:
        w = (1.0 - alpha) * x["weight"] + alpha * uniform
        out.append({"concept": x["concept"], "weight": max(w, 0.0)})
    out = normalize_weights(out)
    out.sort(key=lambda x: x["weight"], reverse=True)
    return out


def shannon_entropy(weights: List[float]) -> float:
    vals = [w for w in weights if w > 0]
    if not vals:
        return 0.0
    return -sum(w * math.log(w + 1e-12) for w in vals)


def main() -> None:
    args = parse_args()

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    kept_total = 0
    filtered_generic = 0
    filtered_weight = 0
    top1_dom_before = 0
    top1_dom_after = 0
    entropy_before = 0.0
    entropy_after = 0.0

    with open(args.output_jsonl, "w") as out_handle:
        for line in open(args.labels_jsonl, "r"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1

            original = parse_concepts(row)
            original = normalize_weights(original)
            if original and original[0]["weight"] >= 0.85:
                top1_dom_before += 1
            entropy_before += shannon_entropy([x["weight"] for x in original])

            concepts = list(original)
            if args.min_weight > 0:
                before = len(concepts)
                concepts = [c for c in concepts if c["weight"] >= args.min_weight]
                filtered_weight += max(before - len(concepts), 0)

            if args.drop_generic_concepts:
                before = len(concepts)
                concepts = [
                    c
                    for c in concepts
                    if not is_generic_concept(c["concept"], args.generic_min_token_len)
                ]
                filtered_generic += max(before - len(concepts), 0)

            if not concepts and args.keep_at_least_one and original:
                concepts = [original[0]]

            concepts = normalize_weights(concepts)
            if args.cap_top1_weight > 0:
                concepts = cap_top1(concepts, args.cap_top1_weight)

            if args.max_concepts > 0:
                concepts = concepts[: args.max_concepts]
                concepts = normalize_weights(concepts)
            concepts.sort(key=lambda x: x["weight"], reverse=True)

            if concepts and concepts[0]["weight"] >= 0.85:
                top1_dom_after += 1
            entropy_after += shannon_entropy([x["weight"] for x in concepts])
            kept_total += len(concepts)

            row["top_concepts"] = concepts
            out_handle.write(json.dumps(row) + "\n")

    avg_kept = kept_total / max(n, 1)
    print(f"Wrote page labels: {args.output_jsonl}")
    print(f"Pages processed: {n}")
    print(f"Average concepts/page: {avg_kept:.3f}")
    print(f"Top1 dominance >=0.85 before: {top1_dom_before / max(n, 1):.3f}")
    print(f"Top1 dominance >=0.85 after : {top1_dom_after / max(n, 1):.3f}")
    print(f"Average entropy before: {entropy_before / max(n, 1):.4f}")
    print(f"Average entropy after : {entropy_after / max(n, 1):.4f}")
    if args.min_weight > 0:
        print(f"Filtered by min-weight: {filtered_weight}")
    if args.drop_generic_concepts:
        print(f"Filtered generic concepts: {filtered_generic}")


if __name__ == "__main__":
    main()
