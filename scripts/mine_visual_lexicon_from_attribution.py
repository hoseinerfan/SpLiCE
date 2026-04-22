#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set

from export_query_visual_binary_labels import (
    DEFAULT_ARTIFACT_TOKENS,
    DEFAULT_STOPWORDS,
    DEFAULT_VISUAL_LEXICON,
    VISUAL_GOLD_QTYPES,
    field_name,
    normalize_qtype_name,
    normalize_token,
    parse_csv_set,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine an attribution-supported visual lexicon by contrasting tokens in "
            "visual vs non-visual query sets."
        )
    )
    parser.add_argument("--attribution-jsonl", type=str, required=True)
    parser.add_argument("--output-lexicon-txt", type=str, required=True)
    parser.add_argument(
        "--output-stats-tsv",
        type=str,
        default="",
        help="Optional ranked token stats TSV. Defaults next to --output-lexicon-txt.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="ig,occlusion",
        help="Comma-separated attribution methods to mine from.",
    )
    parser.add_argument(
        "--label-source",
        type=str,
        default="gold",
        choices=["gold", "pred"],
        help="Use gold qtypes or predicted binary labels to define visual queries.",
    )
    parser.add_argument(
        "--positive-label-name",
        type=str,
        default="visual_needed",
        help="Positive predicted label name when --label-source pred.",
    )
    parser.add_argument("--min-token-len", type=int, default=3)
    parser.add_argument("--drop-stopwords", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-artifact-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--artifact-tokens", type=str, default="")
    parser.add_argument(
        "--include-default-lexicon-tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If false, output only mined extension tokens not already in the default seed lexicon.",
    )
    parser.add_argument(
        "--min-positive-query-freq",
        type=int,
        default=3,
        help="Minimum number of visual queries a token must appear in.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.75,
        help="Minimum positive-query precision: pos_df / (pos_df + neg_df).",
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=2.0,
        help="Minimum smoothed visual/non-visual rate ratio.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum support-adjusted lift score to keep.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Optional cap on number of selected tokens. 0 keeps all passing tokens.",
    )
    parser.add_argument(
        "--sample-query-count",
        type=int,
        default=3,
        help="How many example positive/non-visual queries to store per token in the stats TSV.",
    )
    return parser.parse_args()


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def row_is_positive(row: Dict[str, Any], label_source: str, positive_label_name: str) -> bool:
    if label_source == "pred":
        return str(row.get("pred_label_name", "")).strip() == positive_label_name
    return normalize_qtype_name(row.get("gold_qtype", "")) in VISUAL_GOLD_QTYPES


def collect_row_tokens(
    row: Dict[str, Any],
    methods: Sequence[str],
    min_token_len: int,
    stopwords: Optional[Set[str]],
    artifact_tokens: Optional[Set[str]],
) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for method in methods:
        vals: Set[str] = set()
        for item in row.get(field_name(method), []) or []:
            token = normalize_token(item.get("token", ""))
            if not token:
                continue
            if len(token) < max(1, min_token_len):
                continue
            if stopwords is not None and token in stopwords:
                continue
            if artifact_tokens is not None and token in artifact_tokens:
                continue
            vals.add(token)
        out[method] = vals
    return out


def safe_query_text(row: Dict[str, Any]) -> str:
    return str(row.get("query_text", "")).strip().replace("\t", " ")


def main() -> None:
    args = parse_args()
    methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    for method in methods:
        field_name(method)

    stopwords = DEFAULT_STOPWORDS if args.drop_stopwords else None
    artifact_tokens: Optional[Set[str]] = None
    if args.drop_artifact_tokens:
        artifact_tokens = set(DEFAULT_ARTIFACT_TOKENS)
        artifact_tokens |= parse_csv_set(args.artifact_tokens)

    output_lexicon = Path(args.output_lexicon_txt)
    output_lexicon.parent.mkdir(parents=True, exist_ok=True)
    output_stats = Path(args.output_stats_tsv) if args.output_stats_tsv else output_lexicon.with_suffix(".tsv")
    output_stats.parent.mkdir(parents=True, exist_ok=True)

    positive_query_ids: Set[str] = set()
    negative_query_ids: Set[str] = set()
    pos_df: Counter[str] = Counter()
    neg_df: Counter[str] = Counter()
    pos_method_df: DefaultDict[str, Counter[str]] = defaultdict(Counter)
    neg_method_df: DefaultDict[str, Counter[str]] = defaultdict(Counter)
    pos_examples: DefaultDict[str, List[str]] = defaultdict(list)
    neg_examples: DefaultDict[str, List[str]] = defaultdict(list)

    sample_query_count = max(0, int(args.sample_query_count))

    for row in read_jsonl(args.attribution_jsonl):
        qid = str(row.get("query_id", "")).strip()
        if not qid:
            continue
        is_pos = row_is_positive(
            row=row,
            label_source=str(args.label_source),
            positive_label_name=str(args.positive_label_name),
        )
        method_tokens = collect_row_tokens(
            row=row,
            methods=methods,
            min_token_len=int(args.min_token_len),
            stopwords=stopwords,
            artifact_tokens=artifact_tokens,
        )
        union_tokens: Set[str] = set()
        for vals in method_tokens.values():
            union_tokens |= vals

        if is_pos:
            positive_query_ids.add(qid)
        else:
            negative_query_ids.add(qid)

        if not union_tokens:
            continue

        target_counter = pos_df if is_pos else neg_df
        target_method_counters = pos_method_df if is_pos else neg_method_df
        target_examples = pos_examples if is_pos else neg_examples
        query_text = safe_query_text(row)

        for token in union_tokens:
            target_counter[token] += 1
            if sample_query_count > 0 and len(target_examples[token]) < sample_query_count:
                target_examples[token].append(f"{qid}: {query_text}")

        for method, vals in method_tokens.items():
            if not vals:
                continue
            for token in vals:
                target_method_counters[method][token] += 1

    total_pos = len(positive_query_ids)
    total_neg = len(negative_query_ids)

    rows: List[Dict[str, Any]] = []
    all_tokens = set(pos_df.keys()) | set(neg_df.keys())
    for token in sorted(all_tokens):
        pos_q = int(pos_df[token])
        neg_q = int(neg_df[token])
        if pos_q < max(1, int(args.min_positive_query_freq)):
            continue

        pos_rate = pos_q / total_pos if total_pos else 0.0
        neg_rate = neg_q / total_neg if total_neg else 0.0
        smoothed_lift = ((pos_q + 1.0) / (total_pos + 2.0)) / ((neg_q + 1.0) / (total_neg + 2.0))
        precision = pos_q / (pos_q + neg_q) if (pos_q + neg_q) else 0.0
        score = math.log(smoothed_lift) * math.log1p(pos_q)
        in_default = token in DEFAULT_VISUAL_LEXICON

        if not bool(args.include_default_lexicon_tokens) and in_default:
            continue
        if precision < float(args.min_precision):
            continue
        if smoothed_lift < float(args.min_lift):
            continue
        if score < float(args.min_score):
            continue

        row = {
            "token": token,
            "positive_query_df": pos_q,
            "negative_query_df": neg_q,
            "positive_rate": pos_rate,
            "negative_rate": neg_rate,
            "precision": precision,
            "smoothed_lift": smoothed_lift,
            "score": score,
            "in_default_lexicon": in_default,
            "positive_examples": pos_examples.get(token, []),
            "negative_examples": neg_examples.get(token, []),
        }
        for method in methods:
            row[f"{method}_positive_query_df"] = int(pos_method_df[method][token])
            row[f"{method}_negative_query_df"] = int(neg_method_df[method][token])
        rows.append(row)

    rows.sort(
        key=lambda x: (
            -float(x["score"]),
            -int(x["positive_query_df"]),
            int(x["negative_query_df"]),
            x["token"],
        )
    )
    if int(args.top_k) > 0:
        rows = rows[: int(args.top_k)]

    with output_lexicon.open("w") as handle:
        for row in rows:
            handle.write(f"{row['token']}\n")

    fieldnames = [
        "token",
        "positive_query_df",
        "negative_query_df",
        "positive_rate",
        "negative_rate",
        "precision",
        "smoothed_lift",
        "score",
        "in_default_lexicon",
    ]
    for method in methods:
        fieldnames.append(f"{method}_positive_query_df")
        fieldnames.append(f"{method}_negative_query_df")
    fieldnames.extend(["positive_examples", "negative_examples"])

    with output_stats.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row["positive_examples"] = " || ".join(out_row["positive_examples"])
            out_row["negative_examples"] = " || ".join(out_row["negative_examples"])
            writer.writerow(out_row)

    print(f"attribution_jsonl: {args.attribution_jsonl}")
    print(f"label_source: {args.label_source}")
    print(f"methods: {','.join(methods)}")
    print(f"positive_queries: {total_pos}")
    print(f"negative_queries: {total_neg}")
    print(f"selected_tokens: {len(rows)}")
    print(f"output_lexicon_txt: {output_lexicon}")
    print(f"output_stats_tsv: {output_stats}")
    print("Top candidates:")
    for row in rows[:20]:
        print(
            f"  {row['token']}\tpos_df={row['positive_query_df']}\tneg_df={row['negative_query_df']}"
            f"\tprecision={row['precision']:.3f}\tlift={row['smoothed_lift']:.3f}\tscore={row['score']:.3f}"
        )


if __name__ == "__main__":
    main()
