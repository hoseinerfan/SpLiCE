#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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
        description="Audit query concept labels on a sampled subset for lexical health."
    )
    parser.add_argument("--queries-jsonl", type=str, required=True)
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


def first_json_row(path: str) -> Dict:
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"Empty JSONL: {path}")


def pick_field(row: Dict, candidates: Iterable[str], field_label: str) -> str:
    for name in candidates:
        if name in row:
            return name
    raise KeyError(f"Could not detect {field_label}. Keys: {list(row.keys())}")


def parse_concepts(row: Dict) -> List[Dict]:
    concept_field = None
    for key in ("top_concepts", "query_top_concepts"):
        if key in row:
            concept_field = key
            break
    if concept_field is None:
        return []
    value = row.get(concept_field, [])
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
            out.append({"concept": concept, "weight": weight})
        elif isinstance(item, str):
            concept = item.strip()
            if concept:
                out.append({"concept": concept, "weight": 1.0})
    return out


def is_generic(concept: str, min_token_len: int) -> bool:
    toks = tokenize(concept)
    if not toks:
        return True
    if all(t in STOPWORDS for t in toks):
        return True
    if all(len(t) < min_token_len for t in toks):
        return True
    return False


def is_numeric(concept: str) -> bool:
    c = normalize_text(concept).replace(" ", "")
    return bool(c) and c.isdigit()


def lexical_match(query_norm: str, concept: str) -> bool:
    c = normalize_text(concept)
    if not c:
        return False
    q = f" {query_norm} "
    return f" {c} " in q


def normalize_weights(concepts: List[Dict]) -> List[Dict]:
    total = sum(max(float(c.get("weight", 0.0)), 0.0) for c in concepts)
    if total <= 0:
        if not concepts:
            return []
        w = 1.0 / len(concepts)
        return [{"concept": c["concept"], "weight": w} for c in concepts]
    return [
        {"concept": c["concept"], "weight": max(float(c.get("weight", 0.0)), 0.0) / total}
        for c in concepts
    ]


def main() -> None:
    args = parse_args()

    q0 = first_json_row(args.queries_jsonl)
    l0 = first_json_row(args.labels_jsonl)
    query_id_field = pick_field(q0, ("query_id", "id", "qid"), "query id field in queries")
    query_text_field = pick_field(q0, ("query_text", "query", "question", "text"), "query text field in queries")
    label_id_field = pick_field(l0, ("query_id", "id", "qid"), "query id field in labels")

    queries: Dict[str, str] = {}
    for line in open(args.queries_jsonl, "r"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        qid = str(row.get(query_id_field, "")).strip()
        if not qid:
            continue
        queries[qid] = str(row.get(query_text_field, ""))

    labels: Dict[str, List[Dict]] = {}
    for line in open(args.labels_jsonl, "r"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        qid = str(row.get(label_id_field, "")).strip()
        if not qid:
            continue
        labels[qid] = parse_concepts(row)

    common_qids = sorted(set(queries.keys()) & set(labels.keys()))
    if not common_qids:
        raise ValueError("No overlapping query IDs between queries and labels files.")

    sample_size = min(args.sample_size, len(common_qids))
    rng = random.Random(args.seed)
    sampled_qids = sorted(rng.sample(common_qids, sample_size))

    rows: List[Dict] = []
    for qid in sampled_qids:
        qtext = queries[qid]
        qnorm = normalize_text(qtext)
        concepts = normalize_weights(labels.get(qid, []))
        num_concepts = len(concepts)

        lex_matches = [lexical_match(qnorm, c["concept"]) for c in concepts]
        lexical_count = sum(1 for x in lex_matches if x)
        lexical_ratio = (lexical_count / num_concepts) if num_concepts > 0 else 0.0

        top1_weight = max((c["weight"] for c in concepts), default=0.0)
        dominant = top1_weight >= args.dominance_threshold
        has_non_lex = lexical_count < num_concepts
        has_generic = any(is_generic(c["concept"], args.min_token_len) for c in concepts)
        has_numeric = any(is_numeric(c["concept"]) for c in concepts)

        flags = []
        if num_concepts == 0:
            flags.append("no_concepts")
        if has_non_lex:
            flags.append("non_lexical_concepts")
        if dominant:
            flags.append("top1_dominant")
        if has_generic:
            flags.append("generic_concepts")
        if has_numeric:
            flags.append("numeric_concepts")

        score = 100.0
        if num_concepts == 0:
            score -= 40.0
        if has_non_lex:
            score -= 30.0
        if dominant:
            score -= 15.0
        if has_generic:
            score -= 10.0
        if has_numeric:
            score -= 5.0
        score = max(score, 0.0)

        rows.append(
            {
                "query_id": qid,
                "query_text": qtext,
                "concepts": concepts,
                "num_concepts": num_concepts,
                "lexical_ratio": lexical_ratio,
                "top1_weight": top1_weight,
                "flags": flags,
                "health_score": score,
            }
        )

    n = len(rows)
    avg_concepts = sum(r["num_concepts"] for r in rows) / max(n, 1)
    avg_health = sum(r["health_score"] for r in rows) / max(n, 1)
    zero_rate = sum(1 for r in rows if r["num_concepts"] == 0) / max(n, 1)
    all_lex_rate = sum(1 for r in rows if r["lexical_ratio"] == 1.0 and r["num_concepts"] > 0) / max(n, 1)
    dominant_rate = sum(1 for r in rows if r["top1_weight"] >= args.dominance_threshold) / max(n, 1)
    non_lex_rate = sum(1 for r in rows if "non_lexical_concepts" in r["flags"]) / max(n, 1)
    generic_rate = sum(1 for r in rows if "generic_concepts" in r["flags"]) / max(n, 1)

    summary = {
        "sample_size": n,
        "seed": args.seed,
        "dominance_threshold": args.dominance_threshold,
        "avg_concepts": avg_concepts,
        "avg_health_score": avg_health,
        "zero_concepts_rate": zero_rate,
        "all_lexical_rate": all_lex_rate,
        "top1_dominance_rate": dominant_rate,
        "non_lexical_rate": non_lex_rate,
        "generic_rate": generic_rate,
        "detected_fields": {
            "queries_id": query_id_field,
            "queries_text": query_text_field,
            "labels_id": label_id_field,
        },
    }

    print("=== Query Concept Health Audit ===")
    print(f"Sample size: {n}")
    print(f"Avg concepts/query: {avg_concepts:.3f}")
    print(f"Avg health score: {avg_health:.2f} / 100")
    print(f"Zero-concepts rate: {zero_rate:.3f}")
    print(f"All-lexical rate: {all_lex_rate:.3f}")
    print(f"Top1-dominance rate (>= {args.dominance_threshold:.2f}): {dominant_rate:.3f}")
    print(f"Non-lexical concept rate: {non_lex_rate:.3f}")
    print(f"Generic concept rate: {generic_rate:.3f}")

    flagged = [r for r in rows if r["flags"]]
    flagged.sort(key=lambda r: (r["health_score"], -len(r["flags"])))
    healthy = [r for r in rows if not r["flags"]]
    healthy.sort(key=lambda r: (-r["health_score"], -r["num_concepts"]))

    k = max(args.show_examples, 0)
    if k > 0:
        print("\n--- Flagged examples ---")
        for r in flagged[:k]:
            concepts = [f'{c["concept"]}:{c["weight"]:.3f}' for c in r["concepts"][:8]]
            print(f'QID={r["query_id"]} score={r["health_score"]:.1f} flags={r["flags"]}')
            print(f'Q: {r["query_text"]}')
            print(f'Concepts: {concepts}')
            print("")

        print("--- Healthy examples ---")
        for r in healthy[:k]:
            concepts = [f'{c["concept"]}:{c["weight"]:.3f}' for c in r["concepts"][:8]]
            print(f'QID={r["query_id"]} score={r["health_score"]:.1f}')
            print(f'Q: {r["query_text"]}')
            print(f'Concepts: {concepts}')
            print("")

    if args.output_sample_jsonl:
        out_path = Path(args.output_sample_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as handle:
            for r in rows:
                handle.write(json.dumps(r) + "\n")
        print(f"Wrote sample rows: {args.output_sample_jsonl}")

    if args.output_summary_json:
        out_path = Path(args.output_summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote summary: {args.output_summary_json}")


if __name__ == "__main__":
    main()
