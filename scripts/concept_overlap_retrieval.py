#!/usr/bin/env python3
import argparse
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retrieval rankings from query/page concept label JSONL files."
    )
    parser.add_argument("--query-labels-jsonl", type=str, required=True)
    parser.add_argument("--page-labels-jsonl", type=str, required=True)
    parser.add_argument("--output-ranking-jsonl", type=str, required=True)
    parser.add_argument("--topk-pages", type=int, default=50, help="Number of top pages to keep per query.")
    parser.add_argument("--max-shared-concepts", type=int, default=5, help="Max shared concepts to emit per query-page pair.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Drop query-page matches below this overlap score.")
    parser.add_argument("--min-shared-concepts", type=int, default=1, help="Require at least this many shared concepts for a query-page match.")
    parser.add_argument("--must-match-top-query-concepts", type=int, default=0, help="If >0, require at least one shared concept from the top-N query concepts.")
    parser.add_argument("--idf-weighting", action="store_true", help="Apply IDF weighting per concept during overlap scoring.")
    parser.add_argument("--idf-power", type=float, default=1.0, help="Power applied to IDF multiplier (1.0 = linear).")
    parser.add_argument("--max-concept-df-ratio", type=float, default=1.0, help="Drop concepts that appear in more than this fraction of pages (e.g., 0.05).")
    parser.add_argument("--max-pages-per-concept", type=int, default=0, help="If >0, keep only the top-N pages per concept by concept weight.")
    parser.add_argument("--include-page-top-concepts", action="store_true", help="Include page top concepts in output rows.")
    parser.add_argument("--progress-every", type=int, default=200, help="Print progress every N queries.")
    return parser.parse_args()


def concepts_to_dict(top_concepts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in top_concepts:
        concept = str(item.get("concept", "")).strip()
        if not concept:
            continue
        weight = float(item.get("weight", 0.0))
        if weight <= 0:
            continue
        out[concept] = weight
    return out


def load_label_jsonl(path: str) -> Tuple[List[str], List[Dict[str, float]], List[List[Dict[str, float]]]]:
    ids: List[str] = []
    concept_dicts: List[Dict[str, float]] = []
    raw_top_concepts: List[List[Dict[str, float]]] = []

    with open(path, "r") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            item_id = row.get("id")
            if item_id is None:
                item_id = row.get("query_id", f"line-{line_idx}")
            item_id = str(item_id)

            top_concepts = row.get("top_concepts", [])
            if not isinstance(top_concepts, list):
                top_concepts = []

            concept_weights = concepts_to_dict(top_concepts)

            ids.append(item_id)
            concept_dicts.append(concept_weights)
            raw_top_concepts.append(top_concepts)

    return ids, concept_dicts, raw_top_concepts


def build_page_inverted_index(
    page_ids: List[str],
    page_concept_dicts: List[Dict[str, float]],
) -> Dict[str, List[Tuple[int, float]]]:
    postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for page_idx, concept_weights in enumerate(page_concept_dicts):
        for concept, weight in concept_weights.items():
            postings[concept].append((page_idx, weight))
    return postings


def apply_postings_filters(
    postings: Dict[str, List[Tuple[int, float]]],
    num_pages: int,
    max_concept_df_ratio: float,
    max_pages_per_concept: int,
) -> Tuple[Dict[str, List[Tuple[int, float]]], int]:
    filtered: Dict[str, List[Tuple[int, float]]] = {}
    dropped_df = 0

    for concept, posting_list in postings.items():
        df_ratio = len(posting_list) / max(num_pages, 1)
        if df_ratio > max_concept_df_ratio:
            dropped_df += 1
            continue

        if max_pages_per_concept > 0:
            posting_list = sorted(posting_list, key=lambda x: x[1], reverse=True)[:max_pages_per_concept]

        filtered[concept] = posting_list

    return filtered, dropped_df


def compute_idf_weights(
    postings: Dict[str, List[Tuple[int, float]]],
    num_pages: int,
) -> Dict[str, float]:
    # Smooth IDF to avoid divide-by-zero and keep strictly positive weights.
    return {
        concept: math.log((num_pages + 1.0) / (len(posting_list) + 1.0)) + 1.0
        for concept, posting_list in postings.items()
    }


def get_shared_concepts(
    query_concepts: Dict[str, float],
    page_concepts: Dict[str, float],
    max_shared_concepts: int,
    idf_weights: Dict[str, float],
    idf_power: float,
) -> List[Dict[str, float]]:
    shared: List[Tuple[str, float]] = []
    for concept, q_weight in query_concepts.items():
        p_weight = page_concepts.get(concept)
        if p_weight is None:
            continue
        idf_scale = idf_weights.get(concept, 1.0) ** idf_power
        shared_score = q_weight * p_weight * idf_scale
        shared.append((concept, shared_score))
    shared.sort(key=lambda x: x[1], reverse=True)
    return [
        {"concept": concept, "overlap": round(float(score), 6)}
        for concept, score in shared[:max_shared_concepts]
    ]


def main() -> None:
    args = parse_args()

    query_ids, query_concepts, query_raw_top = load_label_jsonl(args.query_labels_jsonl)
    page_ids, page_concepts, page_raw_top = load_label_jsonl(args.page_labels_jsonl)

    if not query_ids:
        raise ValueError(f"No query labels found: {args.query_labels_jsonl}")
    if not page_ids:
        raise ValueError(f"No page labels found: {args.page_labels_jsonl}")
    if args.max_concept_df_ratio <= 0 or args.max_concept_df_ratio > 1.0:
        raise ValueError("--max-concept-df-ratio must be in (0, 1].")

    raw_postings = build_page_inverted_index(
        page_ids,
        page_concepts,
    )
    postings, dropped_df = apply_postings_filters(
        raw_postings,
        num_pages=len(page_ids),
        max_concept_df_ratio=args.max_concept_df_ratio,
        max_pages_per_concept=args.max_pages_per_concept,
    )
    idf_weights = compute_idf_weights(postings, num_pages=len(page_ids))
    print(f"Loaded {len(query_ids)} queries and {len(page_ids)} pages.")
    print(f"Built inverted index over {len(raw_postings)} unique concepts.")
    if dropped_df > 0:
        print(f"Dropped {dropped_df} concepts by df ratio > {args.max_concept_df_ratio}.")
    print(f"Scoring with {len(postings)} concepts after filters.")
    if args.idf_weighting:
        print(f"IDF weighting enabled (power={args.idf_power}).")
    if args.max_pages_per_concept > 0:
        print(f"Capped postings at {args.max_pages_per_concept} pages/concept.")

    out_path = Path(args.output_ranking_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as out_handle:
        for q_idx, query_id in enumerate(query_ids):
            q_concepts = query_concepts[q_idx]
            score_by_page: Dict[int, float] = defaultdict(float)
            shared_by_page: Dict[int, set] = defaultdict(set)

            top_query_concepts: set = set()
            if args.must_match_top_query_concepts > 0:
                top_query_concepts = {
                    str(item.get("concept", "")).strip()
                    for item in query_raw_top[q_idx][: args.must_match_top_query_concepts]
                    if str(item.get("concept", "")).strip()
                }

            for concept, q_weight in q_concepts.items():
                idf_scale = idf_weights.get(concept, 1.0) ** args.idf_power if args.idf_weighting else 1.0
                for page_idx, p_weight in postings.get(concept, []):
                    contribution = q_weight * p_weight * idf_scale
                    if contribution <= 0:
                        continue
                    score_by_page[page_idx] += contribution
                    shared_by_page[page_idx].add(concept)

            candidates = []
            for pidx, score in score_by_page.items():
                if score < args.min_score:
                    continue
                shared = shared_by_page.get(pidx, set())
                if len(shared) < args.min_shared_concepts:
                    continue
                if top_query_concepts and not (shared & top_query_concepts):
                    continue
                candidates.append((pidx, score))

            top_pairs = heapq.nlargest(args.topk_pages, candidates, key=lambda x: x[1])
            top_pages = []

            for page_idx, score in top_pairs:
                item = {
                    "page_id": page_ids[page_idx],
                    "score": round(float(score), 6),
                    "shared_concepts": get_shared_concepts(
                        q_concepts,
                        page_concepts[page_idx],
                        max_shared_concepts=args.max_shared_concepts,
                        idf_weights=idf_weights if args.idf_weighting else {},
                        idf_power=args.idf_power if args.idf_weighting else 1.0,
                    ),
                }
                if args.include_page_top_concepts:
                    item["page_top_concepts"] = page_raw_top[page_idx]
                top_pages.append(item)

            row = {
                "query_id": query_id,
                "query_top_concepts": query_raw_top[q_idx],
                "top_pages": top_pages,
            }
            out_handle.write(json.dumps(row) + "\n")

            if (q_idx + 1) % args.progress_every == 0:
                print(f"Scored {q_idx + 1}/{len(query_ids)} queries...", flush=True)

    print(f"Done. Wrote rankings to: {out_path}")


if __name__ == "__main__":
    main()
