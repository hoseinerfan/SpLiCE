#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple


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
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
}

GENERIC_BLACKLIST = {
    "image",
    "picture",
    "photo",
    "poster",
    "logo",
    "cover",
    "movie",
    "film",
    "song",
    "person",
    "people",
    "man",
    "woman",
    "whose",
    "text",
    "his",
    "her",
    "its",
    "their",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build doc-seed text concepts (and optional dictionary) from MMQA linked context."
        )
    )
    parser.add_argument("--doc-id", action="append", default=[], help="Target doc id (repeatable).")
    parser.add_argument("--doc-ids-file", type=str, default=None, help="Optional doc id list file.")
    parser.add_argument("--mmqa-dev-jsonl", type=str, required=True)
    parser.add_argument("--mmqa-texts-jsonl", type=str, required=True)
    parser.add_argument("--mmqa-tables-jsonl", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True, help="Output root dir.")
    parser.add_argument("--top-unigrams", type=int, default=300)
    parser.add_argument("--top-bigrams", type=int, default=200)
    parser.add_argument("--min-token-len", type=int, default=3)
    parser.add_argument("--min-unigram-count", type=int, default=2)
    parser.add_argument("--min-bigram-count", type=int, default=2)
    parser.add_argument(
        "--strict-max-concepts",
        type=int,
        default=80,
        help="Max concepts in strict text concept list.",
    )
    parser.add_argument(
        "--strict-min-count-nonquery",
        type=int,
        default=3,
        help="For concepts not appearing in linked query lexicon, minimum corpus count to keep.",
    )
    parser.add_argument(
        "--strict-require-query-overlap",
        action="store_true",
        help=(
            "Keep only concepts that overlap linked query tokens/phrases "
            "(recommended to reduce noisy concepts)."
        ),
    )
    parser.add_argument(
        "--fallback-text-ids-max",
        type=int,
        default=8,
        help=(
            "If no evidence-linked text ids are found for a doc, fallback to at most this many "
            "metadata.text_doc_ids (in first-seen order)."
        ),
    )
    parser.add_argument(
        "--build-dictionary",
        action="store_true",
        help="Also run concepts->queries->embed->dictionary for strict text concepts.",
    )
    parser.add_argument("--model-name", type=str, default="vidore/colpali-v1.2-hf")
    parser.add_argument("--backend", type=str, default="transformers")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--expected-dim", type=int, default=128)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict]]:
    with open(path, "r") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield line_no, row


def read_doc_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    ids.extend([x.strip() for x in args.doc_id if x.strip()])
    if args.doc_ids_file:
        with open(args.doc_ids_file, "r") as handle:
            for line in handle:
                s = line.strip()
                if s:
                    ids.append(s)
    # preserve order, dedupe
    seen = set()
    out = []
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    if not out:
        raise ValueError("No doc ids provided. Use --doc-id or --doc-ids-file.")
    return out


def list_contains(values: object, needle: str) -> bool:
    if not isinstance(values, list):
        return False
    return any(str(x) == needle for x in values)


def get_nested(record: Dict, *path: str):
    cur = record
    for token in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(token)
    return cur


def row_mentions_doc(dev_row: Dict, doc_id: str) -> bool:
    meta = dev_row.get("metadata", {}) if isinstance(dev_row.get("metadata"), dict) else {}
    if list_contains(meta.get("image_doc_ids"), doc_id):
        return True
    if list_contains(meta.get("text_doc_ids"), doc_id):
        return True
    table_id = meta.get("table_id")
    if isinstance(table_id, str) and table_id.strip() == doc_id:
        return True
    supp = dev_row.get("supporting_context", [])
    if isinstance(supp, list):
        for item in supp:
            if isinstance(item, dict) and str(item.get("doc_id", "")) == doc_id:
                return True
    return False


def tokenize(text: str, min_len: int) -> List[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9'_-]*", text.lower())
    out = []
    for t in toks:
        if len(t) < min_len:
            continue
        if t in STOPWORDS:
            continue
        if t.isdigit():
            continue
        out.append(t)
    return out


def build_query_lexicon(linked_dev_rows: List[Dict], min_len: int) -> Set[str]:
    lex: Set[str] = set()
    for row in linked_dev_rows:
        texts = [
            str(row.get("question", "")),
            str(get_nested(row, "metadata", "pseudo_language_question") or ""),
        ]
        for text in texts:
            toks = tokenize(text, min_len=min_len)
            for t in toks:
                lex.add(t)
            for a, b in zip(toks, toks[1:]):
                lex.add(f"{a} {b}")
    return lex


def query_token_set(query_lex: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for item in query_lex:
        for tok in item.split():
            tok = tok.strip()
            if tok:
                out.add(tok)
    return out


def extract_doc_ids_from_instances(instances: object) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(instances, list):
        return out
    for item in instances:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id", "")).strip()
        if doc_id:
            out.add(doc_id)
    return out


def iter_answer_like_records(dev_row: Dict) -> Iterator[Dict]:
    answers = dev_row.get("answers", [])
    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, dict):
                yield item

    meta = dev_row.get("metadata", {}) if isinstance(dev_row.get("metadata"), dict) else {}
    inter = meta.get("intermediate_answers", [])
    if isinstance(inter, list):
        for group in inter:
            if not isinstance(group, list):
                continue
            for item in group:
                if isinstance(item, dict):
                    yield item


def flatten_table_record(table_row: Dict) -> str:
    table_obj = table_row.get("table", {})
    if not isinstance(table_obj, dict):
        return ""

    chunks: List[str] = []
    header = table_obj.get("header", [])
    if isinstance(header, list):
        cols = []
        for h in header:
            if isinstance(h, dict):
                c = str(h.get("column_name", "")).strip()
                if c:
                    cols.append(c)
        if cols:
            chunks.append(" | ".join(cols))

    rows = table_obj.get("table_rows", [])
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, list):
                continue
            vals = []
            for cell in r:
                if isinstance(cell, dict):
                    text = str(cell.get("text", "")).strip()
                    if text:
                        vals.append(text)
            if vals:
                chunks.append(" | ".join(vals))

    return "\n".join(chunks)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            n += 1
    return n


def run_cmd(cmd: List[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_counts_tsv(path: Path) -> List[Tuple[str, str, int]]:
    out: List[Tuple[str, str, int]] = []
    with open(path, "r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            ctype = str(row.get("type", "")).strip()
            concept = str(row.get("concept", "")).strip().lower()
            if not concept:
                continue
            try:
                cnt = int(float(row.get("count", "0")))
            except Exception:
                cnt = 0
            out.append((ctype, concept, cnt))
    return out


def strict_filter(
    counts: List[Tuple[str, str, int]],
    query_lex: Set[str],
    query_tokens: Set[str],
    min_len: int,
    min_nonquery_count: int,
    max_concepts: int,
    require_query_overlap: bool,
) -> List[str]:
    kept: List[str] = []
    seen: Set[str] = set()
    # preserve upstream ordering by counts file
    for _, concept, cnt in counts:
        if concept in seen:
            continue
        toks = concept.split()
        if not toks:
            continue
        if any(len(t) < min_len for t in toks):
            continue
        if all(t in STOPWORDS for t in toks):
            continue
        if concept in GENERIC_BLACKLIST and concept not in query_lex:
            continue
        has_query_overlap = (concept in query_lex) or any(t in query_tokens for t in toks)
        if require_query_overlap and not has_query_overlap:
            continue
        if concept not in query_lex and cnt < min_nonquery_count:
            continue
        seen.add(concept)
        kept.append(concept)
        if max_concepts > 0 and len(kept) >= max_concepts:
            break
    return kept


def main() -> None:
    args = parse_args()
    doc_ids = read_doc_ids(args)

    dev_path = Path(args.mmqa_dev_jsonl)
    texts_path = Path(args.mmqa_texts_jsonl)
    tables_path = Path(args.mmqa_tables_jsonl)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Per-doc state.
    state = {
        d: {
            "linked_dev_full": [],
            "text_ids": set(),
            "table_ids": set(),
            "fallback_text_ids": [],
            "fallback_text_seen": set(),
        }
        for d in doc_ids
    }

    # Pass 1: linked dev rows + ids.
    dev_total = 0
    for line_no, row in iter_jsonl(dev_path):
        dev_total += 1
        for doc_id in doc_ids:
            if not row_mentions_doc(row, doc_id):
                continue
            state[doc_id]["linked_dev_full"].append({"source_file": str(dev_path), "line": line_no, "record": row})
            meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}

            # Prefer evidence-linked context over broad candidate pools.
            for ctx in row.get("supporting_context", []) if isinstance(row.get("supporting_context"), list) else []:
                if not isinstance(ctx, dict):
                    continue
                ctx_id = str(ctx.get("doc_id", "")).strip()
                if not ctx_id:
                    continue
                part = str(ctx.get("doc_part", ctx.get("part", ""))).strip().lower()
                if part == "text":
                    state[doc_id]["text_ids"].add(ctx_id)
                elif part == "table":
                    state[doc_id]["table_ids"].add(ctx_id)

            for ans in iter_answer_like_records(row):
                state[doc_id]["text_ids"].update(extract_doc_ids_from_instances(ans.get("text_instances", [])))

            table_id = meta.get("table_id")
            if isinstance(table_id, str) and table_id.strip():
                state[doc_id]["table_ids"].add(table_id.strip())

            text_doc_ids = meta.get("text_doc_ids", [])
            if isinstance(text_doc_ids, list):
                for tid in text_doc_ids:
                    if isinstance(tid, str) and tid.strip():
                        tid = tid.strip()
                        if tid not in state[doc_id]["fallback_text_seen"]:
                            state[doc_id]["fallback_text_seen"].add(tid)
                            state[doc_id]["fallback_text_ids"].append(tid)

    for doc_id in doc_ids:
        if state[doc_id]["text_ids"]:
            continue
        fallback = state[doc_id]["fallback_text_ids"][: max(0, int(args.fallback_text_ids_max))]
        for tid in fallback:
            state[doc_id]["text_ids"].add(tid)

    all_text_ids: Set[str] = set()
    all_table_ids: Set[str] = set()
    for doc_id in doc_ids:
        all_text_ids.update(state[doc_id]["text_ids"])
        all_table_ids.update(state[doc_id]["table_ids"])

    # Pass 2: collect text/table rows for all needed ids once.
    text_rows_by_id: Dict[str, Dict] = {}
    for line_no, row in iter_jsonl(texts_path):
        rid = str(row.get("id", "")).strip()
        if rid and rid in all_text_ids and rid not in text_rows_by_id:
            text_rows_by_id[rid] = {"source_file": str(texts_path), "line": line_no, "record": row}
    table_rows_by_id: Dict[str, Dict] = {}
    for line_no, row in iter_jsonl(tables_path):
        rid = str(row.get("id", "")).strip()
        if rid and rid in all_table_ids and rid not in table_rows_by_id:
            table_rows_by_id[rid] = {"source_file": str(tables_path), "line": line_no, "record": row}

    for doc_id in doc_ids:
        doc_dir = out_root / f"debug_{doc_id}_linked_context_auto"
        doc_dir.mkdir(parents=True, exist_ok=True)

        linked_dev_full = state[doc_id]["linked_dev_full"]
        linked_dev_rows = [x["record"] for x in linked_dev_full]

        # Compact dev rows
        linked_dev_compact = []
        for x in linked_dev_full:
            r = x["record"]
            meta = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
            linked_dev_compact.append(
                {
                    "source_file": x["source_file"],
                    "line": x["line"],
                    "qid": r.get("qid", ""),
                    "question": r.get("question", ""),
                    "table_id": meta.get("table_id", ""),
                    "text_doc_ids_count": len(meta.get("text_doc_ids", []))
                    if isinstance(meta.get("text_doc_ids", []), list)
                    else 0,
                    "image_doc_ids_count": len(meta.get("image_doc_ids", []))
                    if isinstance(meta.get("image_doc_ids", []), list)
                    else 0,
                    "supporting_context": r.get("supporting_context", []),
                }
            )

        linked_text_rows = [text_rows_by_id[x] for x in sorted(state[doc_id]["text_ids"]) if x in text_rows_by_id]
        linked_table_rows = [table_rows_by_id[x] for x in sorted(state[doc_id]["table_ids"]) if x in table_rows_by_id]

        write_jsonl(doc_dir / "linked_dev_rows_full.jsonl", linked_dev_full)
        write_jsonl(doc_dir / "linked_dev_rows_compact.jsonl", linked_dev_compact)
        write_jsonl(doc_dir / "linked_texts_rows.jsonl", linked_text_rows)
        write_jsonl(doc_dir / "linked_tables_rows.jsonl", linked_table_rows)
        with open(doc_dir / "linked_qids.txt", "w") as handle:
            for row in linked_dev_compact:
                qid = str(row.get("qid", "")).strip()
                if qid:
                    handle.write(qid + "\n")

        # Build corpus jsonl for vocab extraction.
        corpus_rows = []
        for item in linked_text_rows:
            r = item["record"]
            corpus_rows.append(
                {
                    "doc_id": doc_id,
                    "source": "text",
                    "id": r.get("id", ""),
                    "title": r.get("title", ""),
                    "text": r.get("text", ""),
                }
            )
        for item in linked_table_rows:
            r = item["record"]
            corpus_rows.append(
                {
                    "doc_id": doc_id,
                    "source": "table",
                    "id": r.get("id", ""),
                    "title": r.get("title", ""),
                    "table_text": flatten_table_record(r),
                }
            )
        corpus_path = doc_dir / "doc_seed_corpus.jsonl"
        write_jsonl(corpus_path, corpus_rows)

        raw_concepts = doc_dir / "doc_seed_concepts_text_raw.txt"
        raw_counts = doc_dir / "doc_seed_concepts_text_counts.tsv"
        run_cmd(
            [
                sys.executable,
                "scripts/build_concept_vocab_from_corpus.py",
                "--input-jsonl",
                str(corpus_path),
                "--output-concepts-txt",
                str(raw_concepts),
                "--output-counts-tsv",
                str(raw_counts),
                "--top-unigrams",
                str(args.top_unigrams),
                "--top-bigrams",
                str(args.top_bigrams),
                "--min-token-len",
                str(args.min_token_len),
                "--min-unigram-count",
                str(args.min_unigram_count),
                "--min-bigram-count",
                str(args.min_bigram_count),
                "--dedupe-texts-per-record",
            ]
        )

        query_lex = build_query_lexicon(linked_dev_rows, min_len=args.min_token_len)
        query_tokens = query_token_set(query_lex)
        counts = load_counts_tsv(raw_counts)
        strict_concepts = strict_filter(
            counts=counts,
            query_lex=query_lex,
            query_tokens=query_tokens,
            min_len=args.min_token_len,
            min_nonquery_count=args.strict_min_count_nonquery,
            max_concepts=args.strict_max_concepts,
            require_query_overlap=args.strict_require_query_overlap,
        )

        strict_path = doc_dir / "doc_seed_concepts_text_strict.txt"
        with open(strict_path, "w") as handle:
            for c in strict_concepts:
                handle.write(c + "\n")

        summary = {
            "doc_id": doc_id,
            "dev_total_rows": dev_total,
            "linked_dev_rows": len(linked_dev_full),
            "linked_text_ids": len(state[doc_id]["text_ids"]),
            "linked_table_ids": len(state[doc_id]["table_ids"]),
            "found_text_rows": len(linked_text_rows),
            "found_table_rows": len(linked_table_rows),
            "corpus_rows": len(corpus_rows),
            "raw_concepts_path": str(raw_concepts),
            "raw_counts_path": str(raw_counts),
            "strict_concepts_path": str(strict_path),
            "strict_concepts_count": len(strict_concepts),
        }

        if args.build_dictionary:
            q_jsonl = doc_dir / "doc_seed_text_queries.jsonl"
            emb_dir = doc_dir / "doc_seed_text_query_emb"
            dict_pt = doc_dir / "doc_seed_text_dict.pt"

            run_cmd(
                [
                    sys.executable,
                    "scripts/concepts_to_queries_jsonl.py",
                    "--concepts-txt",
                    str(strict_path),
                    "--output-jsonl",
                    str(q_jsonl),
                ]
            )
            run_cmd(
                [
                    sys.executable,
                    "scripts/embed_colpali_queries.py",
                    "--input-jsonl",
                    str(q_jsonl),
                    "--output-dir",
                    str(emb_dir),
                    "--model-name",
                    args.model_name,
                    "--backend",
                    args.backend,
                    "--query-id-field",
                    "query_id",
                    "--query-text-field",
                    "query_text",
                    "--batch-size",
                    str(args.batch_size),
                    "--dtype",
                    args.dtype,
                    "--device",
                    args.device,
                    "--skip-existing",
                ]
            )
            run_cmd(
                [
                    sys.executable,
                    "scripts/build_dictionary_from_concept_embeddings.py",
                    "--embeddings-path",
                    str(emb_dir),
                    "--concept-jsonl",
                    str(q_jsonl),
                    "--output-dictionary-pt",
                    str(dict_pt),
                    "--output-vocab-txt",
                    str(strict_path),
                    "--recursive",
                    "--normalize",
                    "--expected-dim",
                    str(args.expected_dim),
                ]
            )
            summary["dictionary_path"] = str(dict_pt)

        with open(doc_dir / "summary.json", "w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")

        print(f"\n=== {doc_id} ===")
        for k in [
            "linked_dev_rows",
            "found_text_rows",
            "found_table_rows",
            "corpus_rows",
            "strict_concepts_count",
        ]:
            print(f"{k}: {summary[k]}")
        print(f"Output dir: {doc_dir}")


if __name__ == "__main__":
    main()
