#!/usr/bin/env python3
import argparse
import json
import random
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
    parser = argparse.ArgumentParser(description="Audit page concept labels on a sampled subset.")
    parser.add_argument("--labels-jsonl", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dominance-threshold", type=float, default=0.85)
    parser.add_argument("--min-token-len", type=int, default=3)
    parser.add_argument("--show-examples", type=int, default=10)
    parser.add_argument("--output-sample-jsonl", type=str, default=None)
    parser.add_argument("--output-summary-json", type=str, default=None)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def is_generic(concept: str, min_token_len: int) -> bool:
    toks = tokenize(concept)
    if not toks:
        return True
    meaningful = [t for t in toks if t not in STOPWORDS]
    if not meaningful:
        return True
    if all(len(t) < min_token_len for t in meaningful):
        return True
    return False


def is_numeric(concept: str) -> bool:
    c = normalize_text(concept).replace(" ", "")
    return bool(c) and c.isdigit()


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
    total = sum(x["weight"] for x in out)
    if total > 0:
        out = [{"concept": x["concept"], "weight": x["weight"] / total} for x in out]
    out.sort(key=lambda x: x["weight"], reverse=True)
    return out


def main() -> None:
    args = parse_args()

    rows = []
    for line in open(args.labels_jsonl, "r"):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {args.labels_jsonl}")

    sample_size = min(args.sample_size, len(rows))
    rng = random.Random(args.seed)
    sample_rows = rng.sample(rows, sample_size)

    audited = []
    for row in sample_rows:
        pid = str(row.get("id", row.get("page_id", row.get("doc_id", "unknown"))))
        concepts = parse_concepts(row)
        num_concepts = len(concepts)
        top1 = concepts[0]["weight"] if concepts else 0.0
        has_generic = any(is_generic(c["concept"], args.min_token_len) for c in concepts)
        has_numeric = any(is_numeric(c["concept"]) for c in concepts)

        flags = []
        if num_concepts == 0:
            flags.append("no_concepts")
        if top1 >= args.dominance_threshold:
            flags.append("top1_dominant")
        if has_generic:
            flags.append("generic_concepts")
        if has_numeric:
            flags.append("numeric_concepts")

        score = 100.0
        if num_concepts == 0:
            score -= 40.0
        if top1 >= args.dominance_threshold:
            score -= 15.0
        if has_generic:
            score -= 10.0
        if has_numeric:
            score -= 5.0
        score = max(score, 0.0)

        audited.append(
            {
                "page_id": pid,
                "top_concepts": concepts,
                "num_concepts": num_concepts,
                "top1_weight": top1,
                "flags": flags,
                "health_score": score,
            }
        )

    n = len(audited)
    avg_concepts = sum(r["num_concepts"] for r in audited) / max(n, 1)
    avg_health = sum(r["health_score"] for r in audited) / max(n, 1)
    zero_rate = sum(1 for r in audited if r["num_concepts"] == 0) / max(n, 1)
    dominant_rate = sum(1 for r in audited if r["top1_weight"] >= args.dominance_threshold) / max(n, 1)
    generic_rate = sum(1 for r in audited if "generic_concepts" in r["flags"]) / max(n, 1)
    numeric_rate = sum(1 for r in audited if "numeric_concepts" in r["flags"]) / max(n, 1)

    print("=== Page Concept Health Audit ===")
    print(f"Sample size: {n}")
    print(f"Avg concepts/page: {avg_concepts:.3f}")
    print(f"Avg health score: {avg_health:.2f} / 100")
    print(f"Zero-concepts rate: {zero_rate:.3f}")
    print(f"Top1-dominance rate (>= {args.dominance_threshold:.2f}): {dominant_rate:.3f}")
    print(f"Generic concept rate: {generic_rate:.3f}")
    print(f"Numeric concept rate: {numeric_rate:.3f}")

    flagged = [r for r in audited if r["flags"]]
    flagged.sort(key=lambda r: (r["health_score"], -len(r["flags"])))
    healthy = [r for r in audited if not r["flags"]]
    healthy.sort(key=lambda r: (-r["health_score"], -r["num_concepts"]))

    k = max(args.show_examples, 0)
    if k > 0:
        print("\n--- Flagged examples ---")
        for r in flagged[:k]:
            concepts = [f'{c["concept"]}:{c["weight"]:.3f}' for c in r["top_concepts"][:8]]
            print(f'PAGE={r["page_id"]} score={r["health_score"]:.1f} flags={r["flags"]}')
            print(f"Concepts: {concepts}")
            print("")

        print("--- Healthy examples ---")
        for r in healthy[:k]:
            concepts = [f'{c["concept"]}:{c["weight"]:.3f}' for c in r["top_concepts"][:8]]
            print(f'PAGE={r["page_id"]} score={r["health_score"]:.1f}')
            print(f"Concepts: {concepts}")
            print("")

    if args.output_sample_jsonl:
        out_path = Path(args.output_sample_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as handle:
            for r in audited:
                handle.write(json.dumps(r) + "\n")
        print(f"Wrote sample rows: {args.output_sample_jsonl}")

    if args.output_summary_json:
        summary = {
            "sample_size": n,
            "seed": args.seed,
            "dominance_threshold": args.dominance_threshold,
            "avg_concepts": avg_concepts,
            "avg_health_score": avg_health,
            "zero_concepts_rate": zero_rate,
            "top1_dominance_rate": dominant_rate,
            "generic_rate": generic_rate,
            "numeric_rate": numeric_rate,
        }
        out_path = Path(args.output_summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote summary: {args.output_summary_json}")


if __name__ == "__main__":
    main()
