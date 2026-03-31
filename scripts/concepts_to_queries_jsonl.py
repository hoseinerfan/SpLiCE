#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a concept list into query JSONL for runtime ColPali embedding."
    )
    parser.add_argument("--concepts-txt", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--id-prefix", type=str, default="concept")
    return parser.parse_args()


def make_query_id(prefix: str, text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def main() -> None:
    args = parse_args()

    with open(args.concepts_txt, "r") as handle:
        concepts = [line.strip() for line in handle if line.strip()]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as handle:
        for concept in concepts:
            row = {
                "query_id": make_query_id(args.id_prefix, concept),
                "query_text": concept,
            }
            handle.write(json.dumps(row) + "\n")

    print(f"Concepts read: {len(concepts)}")
    print(f"Wrote query JSONL: {output_path}")


if __name__ == "__main__":
    main()
