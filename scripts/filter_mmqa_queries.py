#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set


DEFAULT_ID_FIELDS = ["query_id", "qid", "question_id", "id"]
DEFAULT_QUERY_FIELDS = ["query", "question", "question_text", "text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and filter MMQA queries into a clean JSONL for runtime embedding."
    )
    parser.add_argument("--input-jsonl", type=str, required=True, help="Path to MMQA JSONL file.")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--id-field", type=str, default="", help="Optional explicit ID field (supports dot.path).")
    parser.add_argument("--query-field", type=str, default="", help="Optional explicit query field (supports dot.path).")
    parser.add_argument("--allowlist-file", type=str, default=None, help="Optional file with allowed query IDs, one per line.")
    parser.add_argument("--contains", type=str, default=None, help="Optional case-insensitive substring filter on query text.")
    parser.add_argument("--max-queries", type=int, default=None, help="Stop after writing this many queries.")
    parser.add_argument("--dedupe-text", action="store_true", help="Drop duplicate query strings.")
    parser.add_argument("--keep-full-record", action="store_true", help="Include original record in output under `record`.")
    return parser.parse_args()


def read_allowlist(path: Optional[str]) -> Optional[Set[str]]:
    if path is None:
        return None
    allowed: Set[str] = set()
    with open(path, "r") as handle:
        for line in handle:
            value = line.strip()
            if value:
                allowed.add(value)
    return allowed


def get_by_path(record: Dict[str, Any], dot_path: str) -> Any:
    value: Any = record
    for token in dot_path.split("."):
        if not isinstance(value, dict) or token not in value:
            return None
        value = value[token]
    return value


def first_existing(record: Dict[str, Any], candidates: Iterable[str]) -> Any:
    for key in candidates:
        value = get_by_path(record, key)
        if value is not None:
            return value
    return None


def main() -> None:
    args = parse_args()

    allowlist = read_allowlist(args.allowlist_file)
    contains = args.contains.lower() if args.contains is not None else None
    seen_queries: Set[str] = set()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    id_fields = [args.id_field] if args.id_field else DEFAULT_ID_FIELDS
    query_fields = [args.query_field] if args.query_field else DEFAULT_QUERY_FIELDS

    total = 0
    written = 0
    dropped_no_query = 0
    dropped_allowlist = 0
    dropped_contains = 0
    dropped_dedupe = 0

    with open(input_path, "r") as in_handle, open(output_path, "w") as out_handle:
        for line_idx, line in enumerate(in_handle):
            line = line.strip()
            if not line:
                continue

            total += 1
            record = json.loads(line)

            query_text = first_existing(record, query_fields)
            if query_text is None:
                dropped_no_query += 1
                continue
            query_text = str(query_text).strip()
            if not query_text:
                dropped_no_query += 1
                continue

            query_id = first_existing(record, id_fields)
            if query_id is None:
                query_id = f"line-{line_idx}"
            query_id = str(query_id)

            if allowlist is not None and query_id not in allowlist:
                dropped_allowlist += 1
                continue

            if contains is not None and contains not in query_text.lower():
                dropped_contains += 1
                continue

            if args.dedupe_text:
                if query_text in seen_queries:
                    dropped_dedupe += 1
                    continue
                seen_queries.add(query_text)

            output_record: Dict[str, Any] = {
                "query_id": query_id,
                "query_text": query_text,
            }
            if args.keep_full_record:
                output_record["record"] = record

            out_handle.write(json.dumps(output_record) + "\n")
            written += 1

            if args.max_queries is not None and written >= args.max_queries:
                break

    print(f"Input records read: {total}")
    print(f"Output records written: {written}")
    print(f"Dropped (no query text): {dropped_no_query}")
    print(f"Dropped (allowlist): {dropped_allowlist}")
    print(f"Dropped (contains): {dropped_contains}")
    print(f"Dropped (dedupe): {dropped_dedupe}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
