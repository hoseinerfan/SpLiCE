#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge concept-overlap rankings with a base retriever using fallback flags."
    )
    parser.add_argument("--concept-ranking-jsonl", type=str, required=True)
    parser.add_argument("--base-ranking", type=str, required=True, help="Base ranking file (JSONL or TREC run).")
    parser.add_argument("--base-format", type=str, default="auto", choices=["auto", "jsonl", "trec"])
    parser.add_argument("--output-merged-jsonl", type=str, required=True)
    parser.add_argument("--output-trec-run", type=str, default=None, help="Optional output path for TREC run format.")
    parser.add_argument("--topk-pages", type=int, default=100)
    parser.add_argument("--fallback-on-low-confidence", action="store_true", help="Fallback to base ranking when concept confidence=low.")
    parser.add_argument("--fallback-on-flag", action="store_true", help="Fallback to base ranking when concept fallback_recommended=true.")
    parser.add_argument("--default-source", type=str, default="concept", choices=["concept", "base"], help="Source used when fallback conditions are not met.")
    parser.add_argument("--include-base-only-queries", action="store_true", help="Include queries that exist only in base ranking.")
    parser.add_argument("--trec-run-tag", type=str, default="splice-merged")
    return parser.parse_args()


def detect_base_format(path: str) -> str:
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                return "jsonl"
            return "trec"
    raise ValueError(f"Empty base ranking file: {path}")


def parse_pages_from_json_row(row: Dict) -> List[Dict]:
    if "top_pages" in row and isinstance(row["top_pages"], list):
        pages = row["top_pages"]
    elif "results" in row and isinstance(row["results"], list):
        pages = row["results"]
    elif "hits" in row and isinstance(row["hits"], list):
        pages = row["hits"]
    elif "pages" in row and isinstance(row["pages"], list):
        pages = row["pages"]
    else:
        pages = []

    out: List[Dict] = []
    for idx, p in enumerate(pages):
        if not isinstance(p, dict):
            continue
        page_id = p.get("page_id", p.get("doc_id", p.get("id")))
        if page_id is None:
            continue
        score = p.get("score", p.get("similarity", p.get("value", 0.0)))
        try:
            score = float(score)
        except Exception:
            score = 0.0
        item = {"page_id": str(page_id), "score": score}
        if "shared_concepts" in p:
            item["shared_concepts"] = p["shared_concepts"]
        out.append(item)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def load_base_jsonl(path: str) -> Dict[str, List[Dict]]:
    by_query: Dict[str, List[Dict]] = {}
    with open(path, "r") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = row.get("query_id", row.get("id", f"line-{line_idx}"))
            query_id = str(query_id)
            by_query[query_id] = parse_pages_from_json_row(row)
    return by_query


def load_base_trec(path: str) -> Dict[str, List[Dict]]:
    by_query: Dict[str, List[Tuple[int, float, str]]] = defaultdict(list)
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score, _ = parts[:6]
            try:
                rank_i = int(rank)
            except Exception:
                rank_i = 10**9
            try:
                score_f = float(score)
            except Exception:
                score_f = 0.0
            by_query[qid].append((rank_i, score_f, docid))

    out: Dict[str, List[Dict]] = {}
    for qid, rows in by_query.items():
        rows.sort(key=lambda x: (x[0], -x[1]))
        out[qid] = [{"page_id": docid, "score": score} for _, score, docid in rows]
    return out


def load_concept_rows(path: str) -> Tuple[List[Dict], Dict[str, Dict]]:
    rows: List[Dict] = []
    by_query: Dict[str, Dict] = {}
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row["query_id"])
            rows.append(row)
            by_query[qid] = row
    return rows, by_query


def choose_source(
    concept_row: Dict,
    has_base: bool,
    args: argparse.Namespace,
) -> str:
    fallback_triggered = False
    if args.fallback_on_flag and bool(concept_row.get("fallback_recommended", False)):
        fallback_triggered = True
    if args.fallback_on_low_confidence and str(concept_row.get("confidence", "")).lower() == "low":
        fallback_triggered = True

    if fallback_triggered and has_base:
        return "base"
    return args.default_source


def to_trec_lines(query_id: str, pages: List[Dict], run_tag: str) -> Iterable[str]:
    for rank, page in enumerate(pages, start=1):
        page_id = str(page.get("page_id"))
        score = float(page.get("score", 0.0))
        yield f"{query_id} Q0 {page_id} {rank} {score:.8f} {run_tag}"


def main() -> None:
    args = parse_args()

    base_path = Path(args.base_ranking)
    if not base_path.exists():
        hint = ""
        if args.base_ranking.startswith("/path/to/"):
            hint = (
                " It looks like you used the placeholder path from the docs. "
                "Replace --base-ranking with your real JSONL/TREC file."
            )
        raise FileNotFoundError(
            f"Base ranking file not found: {args.base_ranking}.{hint} "
            "You can locate candidates with: find /mmfs1/scratch -type f "
            "\\( -name '*.trec' -o -name '*.run' -o -name '*.jsonl' \\) | grep -i -E 'colpali|base|ranking|run'"
        )

    base_format = args.base_format
    if base_format == "auto":
        base_format = detect_base_format(args.base_ranking)

    if base_format == "jsonl":
        base_by_query = load_base_jsonl(args.base_ranking)
    elif base_format == "trec":
        base_by_query = load_base_trec(args.base_ranking)
    else:
        raise ValueError(f"Unsupported base format: {base_format}")

    concept_rows, concept_by_query = load_concept_rows(args.concept_ranking_jsonl)

    out_path = Path(args.output_merged_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trec_handle = None
    if args.output_trec_run is not None:
        trec_path = Path(args.output_trec_run)
        trec_path.parent.mkdir(parents=True, exist_ok=True)
        trec_handle = open(trec_path, "w")

    total = 0
    used_concept = 0
    used_base = 0
    missing_base_for_fallback = 0
    base_only = 0

    with open(out_path, "w") as out_handle:
        for row in concept_rows:
            qid = str(row["query_id"])
            concept_pages = row.get("top_pages", [])
            if not isinstance(concept_pages, list):
                concept_pages = []
            base_pages = base_by_query.get(qid, [])

            source = choose_source(row, has_base=bool(base_pages), args=args)
            if source == "base" and not base_pages:
                source = "concept"
                missing_base_for_fallback += 1

            if source == "base":
                selected_pages = base_pages[: args.topk_pages]
                used_base += 1
            else:
                selected_pages = concept_pages[: args.topk_pages]
                used_concept += 1

            out_row = {
                "query_id": qid,
                "source": source,
                "confidence": row.get("confidence"),
                "fallback_recommended": row.get("fallback_recommended"),
                "fallback_reasons": row.get("fallback_reasons", []),
                "query_top_concepts": row.get("query_top_concepts", []),
                "top_pages": selected_pages,
            }
            out_handle.write(json.dumps(out_row) + "\n")

            if trec_handle is not None:
                for line in to_trec_lines(qid, selected_pages, args.trec_run_tag):
                    trec_handle.write(line + "\n")

            total += 1

        if args.include_base_only_queries:
            for qid, pages in base_by_query.items():
                if qid in concept_by_query:
                    continue
                out_row = {
                    "query_id": qid,
                    "source": "base_only",
                    "confidence": None,
                    "fallback_recommended": None,
                    "fallback_reasons": ["missing_concept_row"],
                    "query_top_concepts": [],
                    "top_pages": pages[: args.topk_pages],
                }
                out_handle.write(json.dumps(out_row) + "\n")
                if trec_handle is not None:
                    for line in to_trec_lines(qid, out_row["top_pages"], args.trec_run_tag):
                        trec_handle.write(line + "\n")
                base_only += 1
                total += 1

    if trec_handle is not None:
        trec_handle.close()

    print(f"Base format: {base_format}")
    print(f"Queries written: {total}")
    print(f"Used concept source: {used_concept}")
    print(f"Used base source: {used_base}")
    print(f"Fallback requested but base missing: {missing_base_for_fallback}")
    if args.include_base_only_queries:
        print(f"Base-only queries appended: {base_only}")
    print(f"Wrote merged JSONL: {out_path}")
    if args.output_trec_run is not None:
        print(f"Wrote TREC run: {args.output_trec_run}")


if __name__ == "__main__":
    main()
