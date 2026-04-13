#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "many",
    "me",
    "most",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "would",
    "you",
    "your",
}

DEFAULT_ARTIFACT_TOKENS = {"s", "t", "p"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one query's attribution tokens with noise filtering and optional "
            "IG/Occlusion overlap constraint."
        )
    )
    parser.add_argument("--attribution-jsonl", type=str, required=True)
    parser.add_argument("--qid", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--drop-stopwords", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-token-len", type=int, default=3)
    parser.add_argument("--drop-numeric-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-artifact-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--artifact-tokens", type=str, default="s,t,p")
    parser.add_argument("--require-method-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--overlap-methods",
        type=str,
        default="ig,occlusion",
        help="Comma-separated methods among: ig, occlusion, eg",
    )
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def normalize_token(
    token: str,
    stopwords: Optional[set],
    min_token_len: int,
    drop_numeric_only: bool,
    artifact_tokens: Optional[set],
) -> Optional[str]:
    t = str(token)
    if t.startswith("##"):
        t = t[2:]
    while t.startswith("▁") or t.startswith("Ġ"):
        t = t[1:]
    t = t.strip().lower()
    t = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", t)
    if not t:
        return None
    if len(t) < max(min_token_len, 1):
        return None
    if drop_numeric_only and re.fullmatch(r"\d+", t):
        return None
    if stopwords is not None and t in stopwords:
        return None
    if artifact_tokens is not None and t in artifact_tokens:
        return None
    return t


def field_names(method: str) -> Tuple[str, str]:
    if method == "ig":
        return "ig_top_positive", "ig_top_negative"
    if method == "occlusion":
        return "occlusion_top_positive", "occlusion_top_negative"
    if method == "eg":
        return "eg_top_positive", "eg_top_negative"
    raise ValueError(method)


def normalize_items(
    items: List[Dict[str, Any]],
    stopwords: Optional[set],
    min_token_len: int,
    drop_numeric_only: bool,
    artifact_tokens: Optional[set],
) -> List[Dict[str, Any]]:
    out = []
    for it in items or []:
        raw = it.get("token", "")
        norm = normalize_token(
            raw,
            stopwords=stopwords,
            min_token_len=min_token_len,
            drop_numeric_only=drop_numeric_only,
            artifact_tokens=artifact_tokens,
        )
        if not norm:
            continue
        out.append(
            {
                "raw_token": str(raw),
                "token": norm,
                "score": float(it.get("score", 0.0)),
                "index": int(it.get("index", -1)),
            }
        )
    return out


def topk_by_score(items: List[Dict[str, Any]], top_k: int, descending: bool) -> List[Dict[str, Any]]:
    dedup: Dict[str, Dict[str, Any]] = {}
    for it in items:
        key = it["token"]
        if key not in dedup:
            dedup[key] = it
            continue
        # Keep the more extreme score for stable token-level audit.
        if descending and it["score"] > dedup[key]["score"]:
            dedup[key] = it
        if (not descending) and it["score"] < dedup[key]["score"]:
            dedup[key] = it
    rows = list(dedup.values())
    rows.sort(key=lambda x: x["score"], reverse=descending)
    return rows[:top_k]


def main() -> None:
    args = parse_args()
    stopwords = DEFAULT_STOPWORDS if args.drop_stopwords else None
    artifact_tokens = None
    if args.drop_artifact_tokens:
        toks = [t.strip().lower() for t in str(args.artifact_tokens).split(",") if t.strip()]
        artifact_tokens = set(toks) if toks else set(DEFAULT_ARTIFACT_TOKENS)

    row = None
    with open(args.attribution_jsonl, "r") as handle:
        for line in handle:
            x = json.loads(line)
            if str(x.get("query_id")) == str(args.qid):
                row = x
                break

    if row is None:
        raise ValueError(f"qid not found: {args.qid}")

    methods = []
    for m in ["ig", "occlusion", "eg"]:
        pos_field, _ = field_names(m)
        if pos_field in row:
            methods.append(m)

    overlap_methods = [m.strip() for m in str(args.overlap_methods).split(",") if m.strip() in methods]
    overlap_tokens_pos = None
    overlap_tokens_neg = None
    if args.require_method_overlap and len(overlap_methods) >= 2:
        pos_sets = []
        neg_sets = []
        for m in overlap_methods:
            pos_field, neg_field = field_names(m)
            pos = normalize_items(
                row.get(pos_field, []) or [],
                stopwords=stopwords,
                min_token_len=args.min_token_len,
                drop_numeric_only=args.drop_numeric_only,
                artifact_tokens=artifact_tokens,
            )
            neg = normalize_items(
                row.get(neg_field, []) or [],
                stopwords=stopwords,
                min_token_len=args.min_token_len,
                drop_numeric_only=args.drop_numeric_only,
                artifact_tokens=artifact_tokens,
            )
            pos_sets.append({x["token"] for x in pos})
            neg_sets.append({x["token"] for x in neg})
        overlap_tokens_pos = set.intersection(*pos_sets) if len(pos_sets) >= 2 else None
        overlap_tokens_neg = set.intersection(*neg_sets) if len(neg_sets) >= 2 else None

    out: Dict[str, Any] = {
        "query_id": row.get("query_id"),
        "query_text": row.get("query_text"),
        "gold_qtype": row.get("gold_qtype"),
        "gold_label_name": row.get("gold_label_name"),
        "pred_label_name": row.get("pred_label_name"),
        "pred_prob": row.get("pred_prob"),
        "options": {
            "top_k": args.top_k,
            "drop_stopwords": args.drop_stopwords,
            "min_token_len": args.min_token_len,
            "drop_numeric_only": args.drop_numeric_only,
            "drop_artifact_tokens": args.drop_artifact_tokens,
            "artifact_tokens": sorted(list(artifact_tokens)) if artifact_tokens else [],
            "require_method_overlap": args.require_method_overlap,
            "overlap_methods": overlap_methods,
        },
        "methods": {},
    }

    for m in methods:
        pos_field, neg_field = field_names(m)
        pos = normalize_items(
            row.get(pos_field, []) or [],
            stopwords=stopwords,
            min_token_len=args.min_token_len,
            drop_numeric_only=args.drop_numeric_only,
            artifact_tokens=artifact_tokens,
        )
        neg = normalize_items(
            row.get(neg_field, []) or [],
            stopwords=stopwords,
            min_token_len=args.min_token_len,
            drop_numeric_only=args.drop_numeric_only,
            artifact_tokens=artifact_tokens,
        )
        if overlap_tokens_pos is not None:
            pos = [x for x in pos if x["token"] in overlap_tokens_pos]
        if overlap_tokens_neg is not None:
            neg = [x for x in neg if x["token"] in overlap_tokens_neg]

        out["methods"][m] = {
            "top_positive": topk_by_score(pos, args.top_k, descending=True),
            "top_negative": topk_by_score(neg, args.top_k, descending=False),
        }

    print("query_id:", out["query_id"])
    print("query_text:", out["query_text"])
    print("gold_qtype:", out["gold_qtype"])
    print("pred_label:", out["pred_label_name"], "pred_prob:", out["pred_prob"])
    if overlap_tokens_pos is not None:
        print("overlap_pos_tokens:", sorted(list(overlap_tokens_pos)))
    if overlap_tokens_neg is not None:
        print("overlap_neg_tokens:", sorted(list(overlap_tokens_neg)))

    for m in ["ig", "occlusion", "eg"]:
        if m not in out["methods"]:
            continue
        print(f"\n=== {m} ===")
        print("top_positive:")
        for x in out["methods"][m]["top_positive"]:
            print(f"  {x['token']:20s} {x['score']:.4f}")
        print("top_negative:")
        for x in out["methods"][m]["top_negative"]:
            print(f"  {x['token']:20s} {x['score']:.4f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            json.dump(out, handle, indent=2)
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
