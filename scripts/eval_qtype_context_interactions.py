#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Context-pattern analysis for qtype classifiers using span occlusion and "
            "token-pair interaction (synergy) over top-attributed tokens."
        )
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--attribution-jsonl", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-txt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-source",
        type=str,
        default="row_target",
        choices=["row_target", "pred"],
        help="Use row.target_label_idx (default) or row.pred_label_idx as target class.",
    )
    parser.add_argument(
        "--method-source",
        type=str,
        default="ig",
        choices=["ig", "occlusion", "eg"],
        help="Which attribution top-positive list to use when selecting tokens for pair interactions.",
    )
    parser.add_argument("--top-m", type=int, default=8, help="Top attributed tokens considered for pair synergy.")
    parser.add_argument(
        "--require-positive-top-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only consider positive-scored tokens from attribution top list.",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="mask",
        choices=["mask", "pad", "unk", "drop"],
        help="Token perturbation strategy used for span/pair masking.",
    )
    parser.add_argument("--span-min-len", type=int, default=2)
    parser.add_argument("--span-max-len", type=int, default=4)
    parser.add_argument(
        "--max-spans-per-query",
        type=int,
        default=180,
        help="Caps span evaluations per query for runtime control (sampled if exceeded).",
    )
    parser.add_argument(
        "--top-spans-per-query",
        type=int,
        default=3,
        help="How many highest-drop spans to aggregate from each query.",
    )
    parser.add_argument(
        "--top-pairs-per-query",
        type=int,
        default=3,
        help="How many highest-synergy pairs to aggregate from each query.",
    )
    parser.add_argument(
        "--drop-stopwords",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop common stopwords in phrase/pair aggregation.",
    )
    parser.add_argument(
        "--drop-numeric-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop numeric-only tokens in phrase/pair aggregation.",
    )
    parser.add_argument(
        "--min-token-len",
        type=int,
        default=2,
        help="Minimum normalized token length kept in phrase/pair aggregation.",
    )
    parser.add_argument(
        "--min-pattern-count",
        type=int,
        default=2,
        help="Minimum aggregated frequency for a phrase/pair to be reported.",
    )
    parser.add_argument("--top-k-patterns", type=int, default=20)
    return parser.parse_args()


def load_rows(path: str, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def safe_special_mask(tokenizer: Any, input_ids: List[int]) -> List[int]:
    try:
        return tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    except Exception:
        return [0 for _ in input_ids]


def choose_replace_id(tokenizer: Any, mode: str) -> int:
    if mode == "mask" and tokenizer.mask_token_id is not None:
        return int(tokenizer.mask_token_id)
    if mode == "pad" and tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if mode == "unk" and tokenizer.unk_token_id is not None:
        return int(tokenizer.unk_token_id)
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    return 0


@torch.no_grad()
def forward_target(model: Any, ids: torch.Tensor, am: torch.Tensor, target_idx: int) -> Tuple[float, float]:
    logits = model(input_ids=ids, attention_mask=am).logits
    probs = torch.softmax(logits, dim=-1)
    return float(logits[0, target_idx].item()), float(probs[0, target_idx].item())


@torch.no_grad()
def masked_drop(
    model: Any,
    ids: torch.Tensor,
    am: torch.Tensor,
    target_idx: int,
    indices: List[int],
    replace_id: int,
    mode: str,
    base_logit: float,
    base_prob: float,
) -> Tuple[float, float]:
    ids2 = ids.clone()
    am2 = am.clone()
    for idx in indices:
        ids2[0, idx] = replace_id
        if mode == "drop":
            am2[0, idx] = 0
    logit, prob = forward_target(model, ids2, am2, target_idx)
    return base_logit - logit, base_prob - prob


def stable_rng(seed: int, qid: str, suffix: str) -> random.Random:
    digest = hashlib.md5(f"{qid}|{suffix}".encode("utf-8")).hexdigest()
    mix = int(digest[:8], 16)
    return random.Random(seed + mix)


def normalize_token(
    tok: str,
    stopwords: Optional[set],
    drop_numeric_only: bool,
    min_len: int,
) -> Optional[str]:
    t = str(tok)
    if t.startswith("##"):
        t = t[2:]
    while t.startswith("▁") or t.startswith("Ġ"):
        t = t[1:]
    t = t.strip().lower()
    t = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", t)
    if not t:
        return None
    if len(t) < min_len:
        return None
    if drop_numeric_only and re.fullmatch(r"\d+", t):
        return None
    if stopwords is not None and t in stopwords:
        return None
    return t


def select_top_indices(
    items: List[Dict[str, Any]],
    valid_positions: set,
    top_m: int,
    require_positive: bool,
) -> List[int]:
    ranked = sorted(items or [], key=lambda x: float(x.get("score", 0.0)), reverse=True)
    out = []
    used = set()
    for it in ranked:
        idx = int(it.get("index", -1))
        score = float(it.get("score", 0.0))
        if idx in used or idx not in valid_positions:
            continue
        if require_positive and score <= 0.0:
            continue
        out.append(idx)
        used.add(idx)
        if len(out) >= top_m:
            break
    return out


def generate_spans(
    valid_positions: set,
    seq_len: int,
    min_len: int,
    max_len: int,
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for start in range(seq_len):
        if start not in valid_positions:
            continue
        for L in range(min_len, max_len + 1):
            end = start + L - 1
            if end >= seq_len:
                break
            ok = True
            for p in range(start, end + 1):
                if p not in valid_positions:
                    ok = False
                    break
            if ok:
                spans.append((start, end))
    return spans


def add_agg(stats: Dict[str, Dict[str, float]], key: str, value: float) -> None:
    rec = stats.setdefault(key, {"count": 0.0, "sum": 0.0})
    rec["count"] += 1.0
    rec["sum"] += float(value)


def finalize_agg(stats: Dict[str, Dict[str, float]], min_count: int, top_k: int) -> List[Dict[str, Any]]:
    rows = []
    for key, v in stats.items():
        c = int(v["count"])
        if c < min_count:
            continue
        s = float(v["sum"])
        rows.append({"pattern": key, "count": c, "mean_value": s / max(c, 1)})
    rows.sort(key=lambda x: (x["count"], x["mean_value"]), reverse=True)
    return rows[:top_k]


def render_text_report(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=== Context Interaction Summary ===")
    lines.append(f"model_dir: {summary['model_dir']}")
    lines.append(f"attribution_jsonl: {summary['attribution_jsonl']}")
    lines.append(f"method_source: {summary['method_source']}")
    lines.append(f"n_rows_total: {summary['n_rows_total']}")
    lines.append(f"n_rows_used: {summary['n_rows_used']}")
    lines.append(f"n_rows_skipped: {summary['n_rows_skipped']}")
    lines.append("")

    lines.append("=== Global Top Pair Synergies ===")
    for row in summary["global"]["top_pairs"]:
        lines.append(f"{row['pattern']}: count={row['count']} mean_synergy={row['mean_value']:.6f}")
    lines.append("")

    lines.append("=== Global Top Span Drops ===")
    for row in summary["global"]["top_spans"]:
        lines.append(f"{row['pattern']}: count={row['count']} mean_drop={row['mean_value']:.6f}")
    lines.append("")

    lines.append("=== By Predicted Class (Top Pair Synergies) ===")
    for cls, data in summary["by_pred_class"].items():
        lines.append(f"[{cls}] n_rows={data['n_rows']}")
        if not data["top_pairs"]:
            lines.append("(none)")
            continue
        lines.append(", ".join(f"{x['pattern']}({x['count']})" for x in data["top_pairs"][:8]))
    lines.append("")

    lines.append("=== By Confusion Pair (Top Span Drops) ===")
    for cp, data in summary["by_confusion_pair"].items():
        lines.append(f"[{cp}] n_rows={data['n_rows']}")
        if not data["top_spans"]:
            lines.append("(none)")
            continue
        lines.append(", ".join(f"{x['pattern']}({x['count']})" for x in data["top_spans"][:8]))
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt = Path(args.output_txt) if args.output_txt else None
    if out_txt:
        out_txt.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(torch.device(args.device))
    model.eval()

    rows = load_rows(args.attribution_jsonl, args.limit)
    print(f"loaded_rows: {len(rows)}")
    print(f"model_dir: {model_dir}")
    print(f"method_source: {args.method_source}")

    replace_id = choose_replace_id(tokenizer, args.mask_mode)
    device = torch.device(args.device)
    stopwords = DEFAULT_STOPWORDS if args.drop_stopwords else None

    field_name = f"{args.method_source}_top_positive"

    global_pair_stats: Dict[str, Dict[str, float]] = {}
    global_span_stats: Dict[str, Dict[str, float]] = {}
    by_pred: Dict[str, Dict[str, Any]] = {}
    by_conf: Dict[str, Dict[str, Any]] = {}

    n_used = 0
    n_skipped = 0

    for i, row in enumerate(rows, start=1):
        qid = str(row.get("query_id", f"row-{i}"))
        qtext = str(row.get("query_text", "")).strip()
        if not qtext:
            n_skipped += 1
            continue

        if args.target_source == "row_target":
            if "target_label_idx" not in row:
                n_skipped += 1
                continue
            target_idx = int(row["target_label_idx"])
        else:
            if "pred_label_idx" not in row:
                n_skipped += 1
                continue
            target_idx = int(row["pred_label_idx"])

        items = row.get(field_name, []) or []
        if not items:
            n_skipped += 1
            continue

        pred_label = str(row.get("pred_label_name", "UNKNOWN"))
        gold_label = row.get("gold_label_name")
        is_mis = bool(gold_label is not None and pred_label != str(gold_label))
        conf_key = f"{gold_label} -> {pred_label}" if is_mis else None

        enc = tokenizer(
            qtext,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        seq_len = int(ids.shape[1])

        if target_idx < 0 or target_idx >= int(model.config.num_labels):
            n_skipped += 1
            continue

        special_mask = safe_special_mask(tokenizer, ids[0].detach().cpu().tolist())
        valid_positions = {
            pos
            for pos in range(seq_len)
            if int(am[0, pos].item()) == 1 and special_mask[pos] == 0
        }
        if not valid_positions:
            n_skipped += 1
            continue

        base_logit, base_prob = forward_target(model, ids, am, target_idx)
        del base_prob  # Not currently reported in this script.

        tokens = tokenizer.convert_ids_to_tokens(ids[0].detach().cpu().tolist())

        selected = select_top_indices(
            items=items,
            valid_positions=valid_positions,
            top_m=args.top_m,
            require_positive=args.require_positive_top_tokens,
        )
        if len(selected) >= 2:
            single_drop: Dict[int, float] = {}
            for idx in selected:
                d_logit, _ = masked_drop(
                    model=model,
                    ids=ids,
                    am=am,
                    target_idx=target_idx,
                    indices=[idx],
                    replace_id=replace_id,
                    mode=args.mask_mode,
                    base_logit=base_logit,
                    base_prob=0.0,
                )
                single_drop[idx] = d_logit

            pair_rows = []
            for ai in range(len(selected)):
                for bi in range(ai + 1, len(selected)):
                    a = selected[ai]
                    b = selected[bi]
                    d_pair, _ = masked_drop(
                        model=model,
                        ids=ids,
                        am=am,
                        target_idx=target_idx,
                        indices=[a, b],
                        replace_id=replace_id,
                        mode=args.mask_mode,
                        base_logit=base_logit,
                        base_prob=0.0,
                    )
                    synergy = d_pair - single_drop[a] - single_drop[b]
                    pair_rows.append((a, b, d_pair, synergy))

            pair_rows.sort(key=lambda x: x[3], reverse=True)
            keep_pairs = pair_rows[: max(args.top_pairs_per_query, 0)]
            for a, b, _, synergy in keep_pairs:
                ta = normalize_token(tokens[a], stopwords, args.drop_numeric_only, args.min_token_len)
                tb = normalize_token(tokens[b], stopwords, args.drop_numeric_only, args.min_token_len)
                if not ta or not tb:
                    continue
                key = " + ".join(sorted([ta, tb]))
                add_agg(global_pair_stats, key, synergy)

                pred_rec = by_pred.setdefault(pred_label, {"n_rows": 0, "pair_stats": {}, "span_stats": {}})
                add_agg(pred_rec["pair_stats"], key, synergy)
                if conf_key is not None:
                    conf_rec = by_conf.setdefault(conf_key, {"n_rows": 0, "pair_stats": {}, "span_stats": {}})
                    add_agg(conf_rec["pair_stats"], key, synergy)

        spans = generate_spans(
            valid_positions=valid_positions,
            seq_len=seq_len,
            min_len=max(1, args.span_min_len),
            max_len=max(args.span_min_len, args.span_max_len),
        )
        if args.max_spans_per_query > 0 and len(spans) > args.max_spans_per_query:
            rng = stable_rng(args.seed, qid, "span")
            spans = rng.sample(spans, args.max_spans_per_query)

        span_rows = []
        for start, end in spans:
            idxs = list(range(start, end + 1))
            d_logit, _ = masked_drop(
                model=model,
                ids=ids,
                am=am,
                target_idx=target_idx,
                indices=idxs,
                replace_id=replace_id,
                mode=args.mask_mode,
                base_logit=base_logit,
                base_prob=0.0,
            )
            span_rows.append((start, end, d_logit))
        span_rows.sort(key=lambda x: x[2], reverse=True)
        keep_spans = span_rows[: max(args.top_spans_per_query, 0)]

        pred_rec = by_pred.setdefault(pred_label, {"n_rows": 0, "pair_stats": {}, "span_stats": {}})
        pred_rec["n_rows"] += 1
        conf_rec = None
        if conf_key is not None:
            conf_rec = by_conf.setdefault(conf_key, {"n_rows": 0, "pair_stats": {}, "span_stats": {}})
            conf_rec["n_rows"] += 1

        for start, end, drop_val in keep_spans:
            norm_tokens = []
            for p in range(start, end + 1):
                t = normalize_token(tokens[p], stopwords, args.drop_numeric_only, args.min_token_len)
                if t:
                    norm_tokens.append(t)
            if not norm_tokens:
                continue
            phrase = " ".join(norm_tokens)
            add_agg(global_span_stats, phrase, drop_val)
            add_agg(pred_rec["span_stats"], phrase, drop_val)
            if conf_rec is not None:
                add_agg(conf_rec["span_stats"], phrase, drop_val)

        n_used += 1
        if i % 50 == 0:
            print(f"processed: {i}/{len(rows)}")

    summary = {
        "model_dir": str(model_dir),
        "attribution_jsonl": args.attribution_jsonl,
        "method_source": args.method_source,
        "target_source": args.target_source,
        "mask_mode": args.mask_mode,
        "options": {
            "top_m": args.top_m,
            "span_min_len": args.span_min_len,
            "span_max_len": args.span_max_len,
            "max_spans_per_query": args.max_spans_per_query,
            "top_spans_per_query": args.top_spans_per_query,
            "top_pairs_per_query": args.top_pairs_per_query,
            "drop_stopwords": args.drop_stopwords,
            "drop_numeric_only": args.drop_numeric_only,
            "min_token_len": args.min_token_len,
            "min_pattern_count": args.min_pattern_count,
            "top_k_patterns": args.top_k_patterns,
            "require_positive_top_tokens": args.require_positive_top_tokens,
        },
        "n_rows_total": len(rows),
        "n_rows_used": n_used,
        "n_rows_skipped": n_skipped,
        "global": {
            "top_pairs": finalize_agg(global_pair_stats, args.min_pattern_count, args.top_k_patterns),
            "top_spans": finalize_agg(global_span_stats, args.min_pattern_count, args.top_k_patterns),
        },
        "by_pred_class": {},
        "by_confusion_pair": {},
    }

    for cls, rec in sorted(by_pred.items(), key=lambda kv: kv[0]):
        summary["by_pred_class"][cls] = {
            "n_rows": rec["n_rows"],
            "top_pairs": finalize_agg(rec["pair_stats"], args.min_pattern_count, args.top_k_patterns),
            "top_spans": finalize_agg(rec["span_stats"], args.min_pattern_count, args.top_k_patterns),
        }
    for cp, rec in sorted(by_conf.items(), key=lambda kv: kv[0]):
        summary["by_confusion_pair"][cp] = {
            "n_rows": rec["n_rows"],
            "top_pairs": finalize_agg(rec["pair_stats"], args.min_pattern_count, args.top_k_patterns),
            "top_spans": finalize_agg(rec["span_stats"], args.min_pattern_count, args.top_k_patterns),
        }

    with out_json.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote: {out_json}")

    if out_txt:
        text = render_text_report(summary)
        with out_txt.open("w") as handle:
            handle.write(text)
        print(f"Wrote: {out_txt}")


if __name__ == "__main__":
    main()
