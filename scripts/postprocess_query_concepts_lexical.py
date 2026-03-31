#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


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
            "Post-process query concept labels to keep lexical concepts only, "
            "optionally backfilling empty queries with query tokens."
        )
    )
    parser.add_argument("--queries-jsonl", type=str, required=True)
    parser.add_argument("--labels-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument(
        "--zero-output-jsonl",
        type=str,
        default=None,
        help="Optional file with queries that had zero lexical concepts before backfill.",
    )
    parser.add_argument("--max-concepts", type=int, default=20)
    parser.add_argument(
        "--backfill-missing",
        action="store_true",
        help="If lexical filtering yields zero concepts, backfill from query tokens.",
    )
    parser.add_argument("--backfill-max-concepts", type=int, default=5)
    parser.add_argument("--backfill-use-bigrams", action="store_true")
    parser.add_argument(
        "--min-concepts",
        type=int,
        default=0,
        help="Ensure at least this many concepts by lexical token backfill (0 disables).",
    )
    parser.add_argument(
        "--min-concepts-long-query-tokens",
        type=int,
        default=0,
        help="Apply --min-concepts only when query token count >= this threshold (0 applies to all queries).",
    )
    parser.add_argument("--min-token-len", type=int, default=3)
    parser.add_argument("--keep-stopwords", action="store_true")
    return parser.parse_args()


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


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def normalize_weights(items: List[Dict]) -> List[Dict]:
    total = sum(max(float(x.get("weight", 0.0)), 0.0) for x in items)
    if not items:
        return []
    if total <= 0:
        uniform = 1.0 / len(items)
        return [{"concept": x["concept"], "weight": uniform} for x in items]
    return [{"concept": x["concept"], "weight": max(float(x["weight"]), 0.0) / total} for x in items]


def parse_concept_items(row: Dict, concept_field: str) -> List[Dict]:
    concepts = row.get(concept_field, [])
    if not isinstance(concepts, list):
        return []

    out: List[Dict] = []
    for item in concepts:
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


def lexical_filter(query_norm: str, concept_items: List[Dict]) -> List[Dict]:
    qtext = f" {query_norm} "
    kept: List[Dict] = []
    for c in concept_items:
        c_norm = normalize_text(c["concept"])
        if not c_norm:
            continue
        if f" {c_norm} " in qtext:
            kept.append({"concept": c["concept"], "weight": c["weight"]})
    return kept


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def build_backfill_concepts(
    query_text: str,
    max_concepts: int,
    use_bigrams: bool,
    min_token_len: int,
    keep_stopwords: bool,
) -> List[Dict]:
    toks = tokenize(query_text)
    if not keep_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    toks = [t for t in toks if len(t) >= min_token_len]

    candidates: List[str] = []
    if use_bigrams:
        candidates.extend([f"{a} {b}" for a, b in zip(toks, toks[1:])])
    candidates.extend(toks)
    candidates = unique_preserve_order(candidates)

    if max_concepts > 0:
        candidates = candidates[:max_concepts]
    if not candidates:
        return []

    w = 1.0 / len(candidates)
    return [{"concept": c, "weight": w} for c in candidates]


def augment_to_min_concepts(
    current: List[Dict],
    query_text: str,
    target: int,
    use_bigrams: bool,
    min_token_len: int,
    keep_stopwords: bool,
    candidate_max: int,
) -> List[Dict]:
    if target <= 0 or len(current) >= target:
        return current

    pool = build_backfill_concepts(
        query_text=query_text,
        max_concepts=candidate_max,
        use_bigrams=use_bigrams,
        min_token_len=min_token_len,
        keep_stopwords=keep_stopwords,
    )

    existing = {normalize_text(c["concept"]) for c in current if normalize_text(c["concept"])}
    out = list(current)
    for cand in pool:
        c_norm = normalize_text(cand["concept"])
        if not c_norm or c_norm in existing:
            continue
        out.append({"concept": cand["concept"], "weight": cand["weight"]})
        existing.add(c_norm)
        if len(out) >= target:
            break
    return out


def main() -> None:
    args = parse_args()

    q0 = first_json_row(args.queries_jsonl)
    l0 = first_json_row(args.labels_jsonl)

    query_id_field = pick_field(q0, ["query_id", "id", "qid"], "query id field in queries")
    query_text_field = pick_field(q0, ["query_text", "query", "question", "text"], "query text field in queries")
    label_id_field = pick_field(l0, ["query_id", "id", "qid"], "query id field in labels")
    concept_field = pick_field(l0, ["top_concepts", "query_top_concepts"], "concept field in labels")

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

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zero_handle = None
    if args.zero_output_jsonl:
        zero_path = Path(args.zero_output_jsonl)
        zero_path.parent.mkdir(parents=True, exist_ok=True)
        zero_handle = open(zero_path, "w")

    n = 0
    with_lexical = 0
    with_backfill = 0
    with_min_concepts_augment = 0
    zero_before_backfill = 0
    zero_after_backfill = 0
    avg_kept = 0.0

    with open(args.output_jsonl, "w") as out_handle:
        for line in open(args.labels_jsonl, "r"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1

            qid = str(row.get(label_id_field, "")).strip()
            query_text = queries.get(qid, "")
            query_norm = normalize_text(query_text)

            original_concepts = parse_concept_items(row, concept_field)
            lexical_kept = lexical_filter(query_norm, original_concepts)
            had_lexical = bool(lexical_kept)

            if had_lexical:
                with_lexical += 1
            else:
                zero_before_backfill += 1
                if zero_handle is not None:
                    zero_handle.write(
                        json.dumps(
                            {
                                "query_id": qid,
                                "query_text": query_text,
                                "original_top_concepts": original_concepts[:10],
                            }
                        )
                        + "\n"
                    )

            backfilled = False
            min_augmented = False
            final_concepts = lexical_kept
            if not final_concepts and args.backfill_missing:
                final_concepts = build_backfill_concepts(
                    query_text=query_text,
                    max_concepts=args.backfill_max_concepts,
                    use_bigrams=args.backfill_use_bigrams,
                    min_token_len=args.min_token_len,
                    keep_stopwords=args.keep_stopwords,
                )
                backfilled = bool(final_concepts)
                if backfilled:
                    with_backfill += 1

            if args.min_concepts > 0:
                token_count = len(tokenize(query_text))
                apply_min = (
                    args.min_concepts_long_query_tokens <= 0
                    or token_count >= args.min_concepts_long_query_tokens
                )
                if apply_min and len(final_concepts) < args.min_concepts:
                    before = len(final_concepts)
                    candidate_max = max(args.min_concepts * 3, args.backfill_max_concepts, 10)
                    final_concepts = augment_to_min_concepts(
                        current=final_concepts,
                        query_text=query_text,
                        target=args.min_concepts,
                        use_bigrams=args.backfill_use_bigrams,
                        min_token_len=args.min_token_len,
                        keep_stopwords=args.keep_stopwords,
                        candidate_max=candidate_max,
                    )
                    min_augmented = len(final_concepts) > before
                    if min_augmented:
                        with_min_concepts_augment += 1

            if not final_concepts:
                zero_after_backfill += 1

            final_concepts = normalize_weights(final_concepts)
            if args.max_concepts > 0:
                final_concepts = final_concepts[: args.max_concepts]
            avg_kept += len(final_concepts)

            out_row = {
                "query_id": qid,
                "query_text": query_text,
                "top_concepts": final_concepts,
                "original_concepts": len(original_concepts),
                "lexical_kept": len(lexical_kept),
                "backfilled": backfilled,
                "min_concepts_augmented": min_augmented,
            }
            out_handle.write(json.dumps(out_row) + "\n")

    if zero_handle is not None:
        zero_handle.close()

    print(f"Wrote lexical labels: {args.output_jsonl}")
    print(f"Queries processed: {n}")
    print(f"Queries with lexical concepts: {with_lexical} ratio={with_lexical / max(n, 1):.3f}")
    print(f"Queries zero before backfill: {zero_before_backfill} ratio={zero_before_backfill / max(n, 1):.3f}")
    if args.backfill_missing:
        print(f"Queries backfilled: {with_backfill} ratio={with_backfill / max(n, 1):.3f}")
    if args.min_concepts > 0:
        print(
            f"Queries min-concepts augmented: {with_min_concepts_augment} "
            f"ratio={with_min_concepts_augment / max(n, 1):.3f}"
        )
    print(f"Queries zero after backfill: {zero_after_backfill} ratio={zero_after_backfill / max(n, 1):.3f}")
    print(f"Average kept concepts: {avg_kept / max(n, 1):.3f}")

    print(
        "Detected fields:",
        f"queries.id={query_id_field}",
        f"queries.text={query_text_field}",
        f"labels.id={label_id_field}",
        f"labels.concepts={concept_field}",
    )
    if args.zero_output_jsonl:
        print(f"Wrote zero-concept report: {args.zero_output_jsonl}")


if __name__ == "__main__":
    main()
