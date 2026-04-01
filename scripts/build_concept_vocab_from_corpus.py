#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


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


EMBED_EXTS = {".safetensors", ".pt", ".pth", ".bin", ".npy", ".npz"}
DEFAULT_ID_FIELDS = ["id", "doc_id", "page_id", "document_id", "qid", "query_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a concept vocabulary (unigrams+bigrams) from one or more corpus JSONL files "
            "using recursive string extraction."
        )
    )
    parser.add_argument("--input-jsonl", action="append", required=True, help="Input JSONL path (repeatable).")
    parser.add_argument("--output-concepts-txt", type=str, required=True)
    parser.add_argument("--output-counts-tsv", type=str, default=None)
    parser.add_argument("--top-unigrams", type=int, default=30000)
    parser.add_argument("--top-bigrams", type=int, default=20000)
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument("--min-unigram-count", type=int, default=2)
    parser.add_argument("--min-bigram-count", type=int, default=2)
    parser.add_argument("--stopwords-file", type=str, default=None)
    parser.add_argument("--keep-numbers", action="store_true")
    parser.add_argument("--max-records", type=int, default=0, help="0 = no limit")
    parser.add_argument("--max-texts-per-record", type=int, default=0, help="0 = no limit")
    parser.add_argument("--dedupe-texts-per-record", action="store_true")
    parser.add_argument("--id-field", type=str, default="", help="Preferred record id field.")
    parser.add_argument(
        "--restrict-ids-file",
        type=str,
        default=None,
        help="Optional file with one allowed record id per line.",
    )
    parser.add_argument(
        "--restrict-embeddings-path",
        type=str,
        default=None,
        help="Optional embedding directory; file stems are used as allowed record ids.",
    )
    parser.add_argument("--restrict-embeddings-recursive", action="store_true")
    parser.add_argument(
        "--include-key-regex",
        type=str,
        default="",
        help="Only include string fields whose key-path matches this regex.",
    )
    parser.add_argument(
        "--exclude-key-regex",
        type=str,
        default=r"(^|\.)(id|doc_id|page_id|document_id|url|image_url|filename|file_path|md5|sha1|sha256)$",
        help="Exclude string fields whose key-path matches this regex.",
    )
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


def likely_noise_text(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    if len(s) > 256 and " " not in s:
        return True
    if re.fullmatch(r"[0-9a-fA-F]{24,}", s):
        return True
    return False


def iter_strings(value: object, key_path: str = "") -> Iterator[Tuple[str, str]]:
    if isinstance(value, dict):
        for k, v in value.items():
            child = f"{key_path}.{k}" if key_path else str(k)
            yield from iter_strings(v, child)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            child = f"{key_path}[{idx}]"
            yield from iter_strings(item, child)
    elif isinstance(value, str):
        yield key_path, value


def preferred_record_id(record: Dict, preferred: str) -> Optional[str]:
    if preferred:
        v = record.get(preferred)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for key in DEFAULT_ID_FIELDS:
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def load_ids_file(path: str) -> Set[str]:
    ids: Set[str] = set()
    with open(path, "r") as handle:
        for line in handle:
            s = line.strip()
            if s:
                ids.add(s)
    return ids


def collect_embedding_ids(path: str, recursive: bool) -> Set[str]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Embeddings path not found: {path}")
    ids: Set[str] = set()
    it = root.rglob("*") if recursive else root.glob("*")
    for p in it:
        if not p.is_file():
            continue
        if p.suffix.lower() in EMBED_EXTS:
            ids.add(p.stem)
    return ids


def should_keep_key(
    key_path: str,
    include_re: Optional[re.Pattern],
    exclude_re: Optional[re.Pattern],
) -> bool:
    if include_re is not None and include_re.search(key_path) is None:
        return False
    if exclude_re is not None and exclude_re.search(key_path) is not None:
        return False
    return True


def iter_texts_from_record(
    record: Dict,
    include_re: Optional[re.Pattern],
    exclude_re: Optional[re.Pattern],
    max_texts_per_record: int,
    dedupe: bool,
) -> Iterable[str]:
    seen: Set[str] = set()
    emitted = 0
    for key_path, text in iter_strings(record):
        if not should_keep_key(key_path, include_re, exclude_re):
            continue
        if likely_noise_text(text):
            continue
        t = text.strip()
        if not t:
            continue
        if dedupe:
            if t in seen:
                continue
            seen.add(t)
        yield t
        emitted += 1
        if max_texts_per_record > 0 and emitted >= max_texts_per_record:
            break


def main() -> None:
    args = parse_args()
    stopwords = load_stopwords(args.stopwords_file)

    include_re = re.compile(args.include_key_regex) if args.include_key_regex else None
    exclude_re = re.compile(args.exclude_key_regex) if args.exclude_key_regex else None

    allowed_ids: Optional[Set[str]] = None
    if args.restrict_ids_file:
        allowed_ids = load_ids_file(args.restrict_ids_file)
    if args.restrict_embeddings_path:
        emb_ids = collect_embedding_ids(args.restrict_embeddings_path, args.restrict_embeddings_recursive)
        allowed_ids = emb_ids if allowed_ids is None else (allowed_ids & emb_ids)

    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    records_read = 0
    records_used = 0
    texts_used = 0

    for input_path in args.input_jsonl:
        with open(input_path, "r") as handle:
            for line in handle:
                if args.max_records > 0 and records_read >= args.max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                records_read += 1
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue

                rec_id = preferred_record_id(record, args.id_field)
                if allowed_ids is not None:
                    if rec_id is None or rec_id not in allowed_ids:
                        continue

                used_this_record = False
                for text in iter_texts_from_record(
                    record=record,
                    include_re=include_re,
                    exclude_re=exclude_re,
                    max_texts_per_record=args.max_texts_per_record,
                    dedupe=args.dedupe_texts_per_record,
                ):
                    tokens = tokenize(
                        text=text,
                        min_len=args.min_token_len,
                        keep_numbers=args.keep_numbers,
                        stopwords=stopwords,
                    )
                    if not tokens:
                        continue
                    unigram_counts.update(tokens)
                    if len(tokens) > 1:
                        bigram_counts.update(f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1))
                    texts_used += 1
                    used_this_record = True

                if used_this_record:
                    records_used += 1

            if args.max_records > 0 and records_read >= args.max_records:
                break

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
    unigram_set = set(unigrams)
    concepts = unigrams + [b for b in bigrams if b not in unigram_set]

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

    print(f"Records read: {records_read}")
    print(f"Records used: {records_used}")
    print(f"Texts used: {texts_used}")
    if allowed_ids is not None:
        print(f"Allowed ids size: {len(allowed_ids)}")
    print(f"Unigrams kept: {len(unigrams)}")
    print(f"Bigrams kept: {len(bigrams)}")
    print(f"Total concepts written: {len(concepts)}")
    print(f"Wrote concepts to: {output_concepts}")
    if args.output_counts_tsv is not None:
        print(f"Wrote counts to: {args.output_counts_tsv}")


if __name__ == "__main__":
    main()
