#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
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

DEFAULT_ARTIFACT_TOKENS = {
    "s",
    "t",
    "p",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate token attributions from explain_mmqa_qtype_classifier.py outputs "
            "into class-level and misclassification-focused summaries."
        )
    )
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-txt", type=str, default="")
    parser.add_argument(
        "--label-map-json",
        type=str,
        default="",
        help=(
            "Optional qtype_label_map.json used to map gold_qtype -> gold_label_name when "
            "gold_label_name is missing in attribution rows."
        ),
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--min-token-count",
        type=int,
        default=1,
        help="Drop tokens that appear fewer than this count before ranking.",
    )
    parser.add_argument(
        "--keep-case",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep token case as-is. Default lowercases tokens for aggregation.",
    )
    parser.add_argument(
        "--keep-non-alnum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep punctuation/symbol-only tokens. Default filters them out.",
    )
    parser.add_argument(
        "--print-confusion-topn",
        type=int,
        default=12,
        help="How many confusion pairs to include in text output.",
    )
    parser.add_argument(
        "--drop-stopwords",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop common function words before aggregation.",
    )
    parser.add_argument(
        "--min-token-len",
        type=int,
        default=1,
        help="Drop tokens shorter than this length after normalization.",
    )
    parser.add_argument(
        "--drop-numeric-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop tokens that are digits only.",
    )
    parser.add_argument(
        "--drop-artifact-tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop common tokenizer-fragment artifacts (e.g., s,t,p).",
    )
    parser.add_argument(
        "--artifact-tokens",
        type=str,
        default="s,t,p",
        help="Comma-separated artifact tokens to drop when --drop-artifact-tokens is enabled.",
    )
    parser.add_argument(
        "--require-method-overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only keep tokens that overlap across selected methods on each query row.",
    )
    parser.add_argument(
        "--overlap-methods",
        type=str,
        default="ig,occlusion",
        help=(
            "Comma-separated methods used to define overlap set when "
            "--require-method-overlap is enabled."
        ),
    )
    return parser.parse_args()


def normalize_qtype_name(name: str) -> str:
    return "".join(str(name).lower().split())


def load_type_to_label_name(path: str) -> Dict[str, str]:
    if not path:
        return {}
    payload = json.load(open(path, "r"))
    out = payload.get("type_to_label_name", {})
    return {str(k): str(v) for k, v in out.items()}


def normalize_token(
    token: str,
    keep_case: bool,
    keep_non_alnum: bool,
    stopwords: Optional[set] = None,
    min_token_len: int = 1,
    drop_numeric_only: bool = False,
    artifact_tokens: Optional[set] = None,
) -> Optional[str]:
    t = str(token)
    if t.startswith("##"):
        t = t[2:]
    while t.startswith("Ġ") or t.startswith("▁"):
        t = t[1:]
    t = t.strip()
    if not keep_case:
        t = t.lower()
    if not t:
        return None
    if len(t) < max(int(min_token_len), 1):
        return None
    if not keep_non_alnum and not re.search(r"[a-z0-9]", t):
        return None
    if drop_numeric_only and re.fullmatch(r"\d+", t):
        return None
    if stopwords is not None and t in stopwords:
        return None
    if artifact_tokens is not None and t in artifact_tokens:
        return None
    return t


def update_stats(
    stats: Dict[str, Dict[str, float]],
    items: List[Dict[str, Any]],
    keep_case: bool,
    keep_non_alnum: bool,
    stopwords: Optional[set],
    min_token_len: int,
    drop_numeric_only: bool,
    artifact_tokens: Optional[set],
    allowed_tokens: Optional[set] = None,
) -> None:
    for it in items:
        token = normalize_token(
            it.get("token", ""),
            keep_case=keep_case,
            keep_non_alnum=keep_non_alnum,
            stopwords=stopwords,
            min_token_len=min_token_len,
            drop_numeric_only=drop_numeric_only,
            artifact_tokens=artifact_tokens,
        )
        if not token:
            continue
        if allowed_tokens is not None and token not in allowed_tokens:
            continue
        score = float(it.get("score", 0.0))
        rec = stats.setdefault(token, {"count": 0.0, "score_sum": 0.0, "abs_score_sum": 0.0})
        rec["count"] += 1.0
        rec["score_sum"] += score
        rec["abs_score_sum"] += abs(score)


def top_tokens(
    stats: Dict[str, Dict[str, float]],
    top_k: int,
    prefer_positive: bool,
    min_token_count: int,
) -> List[Dict[str, Any]]:
    rows = []
    for token, v in stats.items():
        count = float(v["count"])
        if count < float(min_token_count):
            continue
        score_sum = float(v["score_sum"])
        abs_sum = float(v["abs_score_sum"])
        mean_score = score_sum / max(count, 1.0)
        rows.append(
            {
                "token": token,
                "count": int(count),
                "score_sum": score_sum,
                "abs_score_sum": abs_sum,
                "mean_score": mean_score,
            }
        )

    if prefer_positive:
        rows.sort(key=lambda x: (x["score_sum"], x["abs_score_sum"], x["count"]), reverse=True)
    else:
        rows.sort(key=lambda x: (x["score_sum"], -x["abs_score_sum"], -x["count"]))
    return rows[:top_k]


def ensure_bucket(
    root: Dict[str, Any],
    key: str,
    class_name: str,
) -> Dict[str, Any]:
    classes = root.setdefault(key, {})
    if class_name not in classes:
        classes[class_name] = {
            "n_rows": 0,
            "positive_stats": {},
            "negative_stats": {},
        }
    return classes[class_name]


def fields_for_method(row: Dict[str, Any], method: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if method == "ig":
        return row.get("ig_top_positive", []) or [], row.get("ig_top_negative", []) or []
    if method == "occlusion":
        return row.get("occlusion_top_positive", []) or [], row.get("occlusion_top_negative", []) or []
    if method == "eg":
        return row.get("eg_top_positive", []) or [], row.get("eg_top_negative", []) or []
    raise ValueError(f"unsupported method: {method}")


def normalized_token_set(
    items: List[Dict[str, Any]],
    keep_case: bool,
    keep_non_alnum: bool,
    stopwords: Optional[set],
    min_token_len: int,
    drop_numeric_only: bool,
    artifact_tokens: Optional[set],
) -> set:
    out = set()
    for it in items:
        token = normalize_token(
            it.get("token", ""),
            keep_case=keep_case,
            keep_non_alnum=keep_non_alnum,
            stopwords=stopwords,
            min_token_len=min_token_len,
            drop_numeric_only=drop_numeric_only,
            artifact_tokens=artifact_tokens,
        )
        if token:
            out.add(token)
    return out


def render_text_report(summary: Dict[str, Any], top_k: int, confusion_topn: int) -> str:
    lines: List[str] = []
    lines.append("=== Attribution Summary ===")
    lines.append(f"input_jsonl: {summary['input_jsonl']}")
    lines.append(f"n_rows: {summary['n_rows']}")
    lines.append(f"n_rows_with_gold_label: {summary['n_rows_with_gold_label']}")
    lines.append(f"n_misclassified: {summary['n_misclassified']}")
    lines.append("")

    conf_pairs = summary.get("confusion_pairs", [])
    if conf_pairs:
        lines.append("=== Top Confusion Pairs ===")
        for row in conf_pairs[:confusion_topn]:
            lines.append(f"{row['pair']}: {row['count']}")
        lines.append("")

    for method in ["ig", "eg", "occlusion"]:
        if method not in summary["methods"]:
            continue
        m = summary["methods"][method]
        lines.append(f"=== Method: {method} ===")

        lines.append("-- Per Predicted Class: Top Positive Tokens --")
        for cls, data in m["per_pred_class"].items():
            lines.append(f"[{cls}] n_rows={data['n_rows']}")
            toks = data["top_positive"][: min(top_k, 12)]
            lines.append(", ".join(f"{x['token']}({x['count']})" for x in toks) if toks else "(none)")
        lines.append("")

        lines.append("-- Per Predicted Class: Top Negative Tokens --")
        for cls, data in m["per_pred_class"].items():
            lines.append(f"[{cls}] n_rows={data['n_rows']}")
            toks = data["top_negative"][: min(top_k, 12)]
            lines.append(", ".join(f"{x['token']}({x['count']})" for x in toks) if toks else "(none)")
        lines.append("")

        mis = m["misclassified"]
        lines.append("-- Misclassified Only (Overall): Top Positive Tokens --")
        lines.append(
            ", ".join(f"{x['token']}({x['count']})" for x in mis["overall"]["top_positive"][: min(top_k, 20)])
            if mis["overall"]["top_positive"]
            else "(none)"
        )
        lines.append("-- Misclassified Only (Overall): Top Negative Tokens --")
        lines.append(
            ", ".join(f"{x['token']}({x['count']})" for x in mis["overall"]["top_negative"][: min(top_k, 20)])
            if mis["overall"]["top_negative"]
            else "(none)"
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_jsonl)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt = Path(args.output_txt) if args.output_txt else None
    if output_txt:
        output_txt.parent.mkdir(parents=True, exist_ok=True)

    type_to_label_name = load_type_to_label_name(args.label_map_json)

    stopwords = DEFAULT_STOPWORDS if args.drop_stopwords else None
    artifact_tokens = None
    if args.drop_artifact_tokens:
        toks = [t.strip().lower() for t in str(args.artifact_tokens).split(",") if t.strip()]
        artifact_tokens = set(toks) if toks else set(DEFAULT_ARTIFACT_TOKENS)
    overlap_methods = [m.strip() for m in str(args.overlap_methods).split(",") if m.strip()]
    valid_methods = {"ig", "eg", "occlusion"}
    overlap_methods = [m for m in overlap_methods if m in valid_methods]

    rows: List[Dict[str, Any]] = []
    with input_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    methods_present = set()
    for row in rows:
        if "ig_top_positive" in row:
            methods_present.add("ig")
        if "eg_top_positive" in row:
            methods_present.add("eg")
        if "occlusion_top_positive" in row:
            methods_present.add("occlusion")

    root: Dict[str, Any] = {
        "input_jsonl": str(input_path),
        "n_rows": len(rows),
        "n_rows_with_gold_label": 0,
        "n_misclassified": 0,
        "options": {
            "top_k": args.top_k,
            "min_token_count": args.min_token_count,
            "drop_stopwords": args.drop_stopwords,
            "min_token_len": args.min_token_len,
            "drop_numeric_only": args.drop_numeric_only,
            "drop_artifact_tokens": args.drop_artifact_tokens,
            "artifact_tokens": sorted(list(artifact_tokens)) if artifact_tokens else [],
            "require_method_overlap": args.require_method_overlap,
            "overlap_methods": overlap_methods,
            "keep_case": args.keep_case,
            "keep_non_alnum": args.keep_non_alnum,
        },
        "methods": {},
        "confusion_pairs": [],
    }

    confusion_counts: Dict[str, int] = defaultdict(int)

    for method in sorted(methods_present):
        root["methods"][method] = {
            "per_pred_class": {},
            "misclassified": {
                "overall": {"n_rows": 0, "positive_stats": {}, "negative_stats": {}},
                "by_pred_class": {},
                "by_confusion_pair": {},
            },
        }

    for row in rows:
        pred_label = str(row.get("pred_label_name", "UNKNOWN"))
        gold_label = row.get("gold_label_name")
        if not gold_label:
            gq = row.get("gold_qtype", "")
            if gq:
                gold_label = type_to_label_name.get(normalize_qtype_name(gq))

        has_gold = gold_label is not None and str(gold_label) != ""
        if has_gold:
            root["n_rows_with_gold_label"] += 1
        misclassified = bool(has_gold and str(gold_label) != pred_label)
        if misclassified:
            root["n_misclassified"] += 1
            pair = f"{gold_label} -> {pred_label}"
            confusion_counts[pair] += 1

        for method in methods_present:
            pos_items, neg_items = fields_for_method(row, method)
            mroot = root["methods"][method]
            allowed_pos = None
            allowed_neg = None
            if args.require_method_overlap and overlap_methods:
                pos_sets = []
                neg_sets = []
                for om in overlap_methods:
                    if om not in methods_present:
                        continue
                    om_pos, om_neg = fields_for_method(row, om)
                    pos_sets.append(
                        normalized_token_set(
                            om_pos,
                            args.keep_case,
                            args.keep_non_alnum,
                            stopwords,
                            args.min_token_len,
                            args.drop_numeric_only,
                            artifact_tokens,
                        )
                    )
                    neg_sets.append(
                        normalized_token_set(
                            om_neg,
                            args.keep_case,
                            args.keep_non_alnum,
                            stopwords,
                            args.min_token_len,
                            args.drop_numeric_only,
                            artifact_tokens,
                        )
                    )
                if len(pos_sets) >= 2:
                    allowed_pos = set.intersection(*pos_sets)
                if len(neg_sets) >= 2:
                    allowed_neg = set.intersection(*neg_sets)

            bucket = ensure_bucket(mroot, "per_pred_class", pred_label)
            bucket["n_rows"] += 1
            update_stats(
                bucket["positive_stats"],
                pos_items,
                args.keep_case,
                args.keep_non_alnum,
                stopwords,
                args.min_token_len,
                args.drop_numeric_only,
                artifact_tokens,
                allowed_pos,
            )
            update_stats(
                bucket["negative_stats"],
                neg_items,
                args.keep_case,
                args.keep_non_alnum,
                stopwords,
                args.min_token_len,
                args.drop_numeric_only,
                artifact_tokens,
                allowed_neg,
            )

            if misclassified:
                mis = mroot["misclassified"]

                overall = mis["overall"]
                overall["n_rows"] += 1
                update_stats(
                    overall["positive_stats"],
                    pos_items,
                    args.keep_case,
                    args.keep_non_alnum,
                    stopwords,
                    args.min_token_len,
                    args.drop_numeric_only,
                    artifact_tokens,
                    allowed_pos,
                )
                update_stats(
                    overall["negative_stats"],
                    neg_items,
                    args.keep_case,
                    args.keep_non_alnum,
                    stopwords,
                    args.min_token_len,
                    args.drop_numeric_only,
                    artifact_tokens,
                    allowed_neg,
                )

                by_pred = ensure_bucket(mis, "by_pred_class", pred_label)
                by_pred["n_rows"] += 1
                update_stats(
                    by_pred["positive_stats"],
                    pos_items,
                    args.keep_case,
                    args.keep_non_alnum,
                    stopwords,
                    args.min_token_len,
                    args.drop_numeric_only,
                    artifact_tokens,
                    allowed_pos,
                )
                update_stats(
                    by_pred["negative_stats"],
                    neg_items,
                    args.keep_case,
                    args.keep_non_alnum,
                    stopwords,
                    args.min_token_len,
                    args.drop_numeric_only,
                    artifact_tokens,
                    allowed_neg,
                )

                pair = f"{gold_label} -> {pred_label}"
                by_pair = ensure_bucket(mis, "by_confusion_pair", pair)
                by_pair["n_rows"] += 1
                update_stats(
                    by_pair["positive_stats"],
                    pos_items,
                    args.keep_case,
                    args.keep_non_alnum,
                    stopwords,
                    args.min_token_len,
                    args.drop_numeric_only,
                    artifact_tokens,
                    allowed_pos,
                )
                update_stats(
                    by_pair["negative_stats"],
                    neg_items,
                    args.keep_case,
                    args.keep_non_alnum,
                    stopwords,
                    args.min_token_len,
                    args.drop_numeric_only,
                    artifact_tokens,
                    allowed_neg,
                )

    root["confusion_pairs"] = [
        {"pair": pair, "count": count}
        for pair, count in sorted(confusion_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    for method in methods_present:
        mroot = root["methods"][method]

        for _, obj in mroot["per_pred_class"].items():
            obj["top_positive"] = top_tokens(
                obj.pop("positive_stats"),
                args.top_k,
                prefer_positive=True,
                min_token_count=args.min_token_count,
            )
            obj["top_negative"] = top_tokens(
                obj.pop("negative_stats"),
                args.top_k,
                prefer_positive=False,
                min_token_count=args.min_token_count,
            )

        mis = mroot["misclassified"]
        overall = mis["overall"]
        overall["top_positive"] = top_tokens(
            overall.pop("positive_stats"),
            args.top_k,
            prefer_positive=True,
            min_token_count=args.min_token_count,
        )
        overall["top_negative"] = top_tokens(
            overall.pop("negative_stats"),
            args.top_k,
            prefer_positive=False,
            min_token_count=args.min_token_count,
        )

        for key in ["by_pred_class", "by_confusion_pair"]:
            for _, obj in mis[key].items():
                obj["top_positive"] = top_tokens(
                    obj.pop("positive_stats"),
                    args.top_k,
                    prefer_positive=True,
                    min_token_count=args.min_token_count,
                )
                obj["top_negative"] = top_tokens(
                    obj.pop("negative_stats"),
                    args.top_k,
                    prefer_positive=False,
                    min_token_count=args.min_token_count,
                )

    with output_json.open("w") as handle:
        json.dump(root, handle, indent=2)
    print(f"Wrote: {output_json}")

    if output_txt:
        text = render_text_report(root, top_k=args.top_k, confusion_topn=args.print_confusion_topn)
        with output_txt.open("w") as handle:
            handle.write(text)
        print(f"Wrote: {output_txt}")


if __name__ == "__main__":
    main()
