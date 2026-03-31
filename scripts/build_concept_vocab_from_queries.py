#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Set


DEFAULT_QUERY_FIELDS = ["query_text", "query", "question", "question_text", "text"]

DEFAULT_STOPWORDS = {
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
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "please",
    "there",
    "their",
    "them",
    "than",
    "then",
    "these",
    "those",
    "has",
    "have",
    "had",
    "into",
    "about",
    "over",
    "under",
    "up",
    "down",
    "left",
    "right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a concept vocabulary (unigrams+bigrams) from query JSONL."
    )
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-concepts-txt", type=str, required=True)
    parser.add_argument("--output-counts-tsv", type=str, default=None)
    parser.add_argument("--query-field", type=str, default="")
    parser.add_argument("--top-unigrams", type=int, default=10000)
    parser.add_argument("--top-bigrams", type=int, default=5000)
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument("--min-unigram-count", type=int, default=2)
    parser.add_argument("--min-bigram-count", type=int, default=2)
    parser.add_argument("--stopwords-file", type=str, default=None)
    parser.add_argument("--keep-numbers", action="store_true")
    return parser.parse_args()


def load_stopwords(path: Optional[str]) -> Set[str]:
    stopwords = set(DEFAULT_STOPWORDS)
    if path is None:
        return stopwords
    with open(path, "r") as handle:
        for line in handle:
            token = line.strip().lower()
            if token:
                stopwords.add(token)
    return stopwords


def first_query_text(record: dict, preferred_field: str) -> Optional[str]:
    if preferred_field:
        value = record.get(preferred_field)
        if isinstance(value, str):
            return value
        return None
    for key in DEFAULT_QUERY_FIELDS:
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def tokenize(text: str, min_len: int, keep_numbers: bool, stopwords: Set[str]) -> List[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9'_-]*", text.lower())
    clean: List[str] = []
    for token in tokens:
        if len(token) < min_len:
            continue
        if token in stopwords:
            continue
        if not keep_numbers and token.isdigit():
            continue
        clean.append(token)
    return clean


def iter_queries(input_jsonl: str, preferred_field: str) -> Iterable[str]:
    with open(input_jsonl, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = first_query_text(record, preferred_field=preferred_field)
            if text is None:
                continue
            text = text.strip()
            if text:
                yield text


def main() -> None:
    args = parse_args()
    stopwords = load_stopwords(args.stopwords_file)

    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    total_queries = 0

    for query_text in iter_queries(args.input_jsonl, args.query_field):
        total_queries += 1
        tokens = tokenize(
            query_text,
            min_len=args.min_token_len,
            keep_numbers=args.keep_numbers,
            stopwords=stopwords,
        )
        if not tokens:
            continue
        unigram_counts.update(tokens)
        if len(tokens) > 1:
            bigram_counts.update(f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1))

    unigrams = [
        token
        for token, count in unigram_counts.most_common(args.top_unigrams)
        if count >= args.min_unigram_count
    ]
    bigrams = [
        token
        for token, count in bigram_counts.most_common(args.top_bigrams)
        if count >= args.min_bigram_count
    ]

    concepts = unigrams + [b for b in bigrams if b not in set(unigrams)]

    output_concepts = Path(args.output_concepts_txt)
    output_concepts.parent.mkdir(parents=True, exist_ok=True)
    with open(output_concepts, "w") as handle:
        for concept in concepts:
            handle.write(concept + "\n")

    if args.output_counts_tsv is not None:
        output_counts = Path(args.output_counts_tsv)
        output_counts.parent.mkdir(parents=True, exist_ok=True)
        with open(output_counts, "w") as handle:
            handle.write("type\tconcept\tcount\n")
            for concept in unigrams:
                handle.write(f"unigram\t{concept}\t{unigram_counts[concept]}\n")
            for concept in bigrams:
                handle.write(f"bigram\t{concept}\t{bigram_counts[concept]}\n")

    print(f"Queries read: {total_queries}")
    print(f"Unigrams kept: {len(unigrams)}")
    print(f"Bigrams kept: {len(bigrams)}")
    print(f"Total concepts written: {len(concepts)}")
    print(f"Wrote concepts to: {output_concepts}")
    if args.output_counts_tsv is not None:
        print(f"Wrote counts to: {args.output_counts_tsv}")


if __name__ == "__main__":
    main()
