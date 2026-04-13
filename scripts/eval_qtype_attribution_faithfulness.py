#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Faithfulness evaluation for qtype attributions. "
            "Compares IG vs Occlusion vs Expected Gradients by masking top-k attributed tokens and measuring "
            "target-score drop."
        )
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--attribution-jsonl", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-txt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument(
        "--ks",
        type=str,
        default="1,3,5",
        help="Comma-separated k values (e.g., '1,3,5,8').",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional query cap for smoke testing.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random baseline sampling.")
    parser.add_argument(
        "--target-source",
        type=str,
        default="row_target",
        choices=["row_target", "pred"],
        help="Use row.target_label_idx (default) or row.pred_label_idx as target class.",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="mask",
        choices=["mask", "pad", "unk", "drop"],
        help="Token perturbation strategy used for faithfulness masking.",
    )
    parser.add_argument(
        "--require-positive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only use tokens with positive attribution score when selecting top-k.",
    )
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
def forward_target(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_idx: int,
) -> Tuple[float, float]:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    probs = torch.softmax(logits, dim=-1)
    return float(logits[0, target_idx].item()), float(probs[0, target_idx].item())


def select_top_indices(
    items: List[Dict[str, Any]],
    valid_positions: set,
    k: int,
    require_positive: bool,
) -> List[int]:
    ranked = sorted(items or [], key=lambda x: float(x.get("score", 0.0)), reverse=True)
    out: List[int] = []
    used = set()
    for it in ranked:
        idx = int(it.get("index", -1))
        score = float(it.get("score", 0.0))
        if idx in used:
            continue
        if idx not in valid_positions:
            continue
        if require_positive and score <= 0.0:
            continue
        out.append(idx)
        used.add(idx)
        if len(out) >= k:
            break
    return out


def stable_rng(seed: int, qid: str, k: int) -> random.Random:
    digest = hashlib.md5(f"{qid}|{k}".encode("utf-8")).hexdigest()
    mix = int(digest[:8], 16)
    return random.Random(seed + mix)


@torch.no_grad()
def masked_drop(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_idx: int,
    indices: List[int],
    replace_id: int,
    mode: str,
    base_logit: float,
    base_prob: float,
) -> Tuple[float, float]:
    ids = input_ids.clone()
    am = attention_mask.clone()
    for idx in indices:
        ids[0, idx] = replace_id
        if mode == "drop":
            am[0, idx] = 0
    logit, prob = forward_target(model, ids, am, target_idx)
    return base_logit - logit, base_prob - prob


def stat(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": len(values),
        "mean": float(mean(values)),
        "std": float(pstdev(values) if len(values) > 1 else 0.0),
    }


def render_text_report(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=== Faithfulness Summary ===")
    lines.append(f"model_dir: {summary['model_dir']}")
    lines.append(f"attribution_jsonl: {summary['attribution_jsonl']}")
    lines.append(f"n_rows_total: {summary['n_rows_total']}")
    lines.append(f"n_rows_used: {summary['n_rows_used']}")
    lines.append(f"ks: {summary['ks']}")
    lines.append("")

    for k_key in sorted(summary["by_k"].keys(), key=lambda x: int(x)):
        row = summary["by_k"][k_key]
        lines.append(f"--- k={k_key} ---")
        has_eg = row.get("eg") is not None
        if has_eg:
            lines.append(
                "logit_drop_mean: "
                f"IG={row['ig']['logit_drop']['mean']:.6f} "
                f"OCC={row['occlusion']['logit_drop']['mean']:.6f} "
                f"EG={row['eg']['logit_drop']['mean']:.6f} "
                f"RAND={row['random']['logit_drop']['mean']:.6f}"
            )
            lines.append(
                "prob_drop_mean:  "
                f"IG={row['ig']['prob_drop']['mean']:.6f} "
                f"OCC={row['occlusion']['prob_drop']['mean']:.6f} "
                f"EG={row['eg']['prob_drop']['mean']:.6f} "
                f"RAND={row['random']['prob_drop']['mean']:.6f}"
            )
            lines.append(
                "paired_win_rate (higher drop wins): "
                f"IG>{row['paired']['ig_beats_occ_rate']:.4f} "
                f"OCC>{row['paired']['occ_beats_ig_rate']:.4f} "
                f"EG>{row['paired']['eg_beats_others_rate']:.4f} "
                f"ties={row['paired']['tie_rate']:.4f} "
                f"(n={row['paired']['n']})"
            )
        else:
            lines.append(
                "logit_drop_mean: "
                f"IG={row['ig']['logit_drop']['mean']:.6f} "
                f"OCC={row['occlusion']['logit_drop']['mean']:.6f} "
                f"RAND={row['random']['logit_drop']['mean']:.6f}"
            )
            lines.append(
                "prob_drop_mean:  "
                f"IG={row['ig']['prob_drop']['mean']:.6f} "
                f"OCC={row['occlusion']['prob_drop']['mean']:.6f} "
                f"RAND={row['random']['prob_drop']['mean']:.6f}"
            )
            lines.append(
                "paired_win_rate (higher drop wins): "
                f"IG>{row['paired']['ig_beats_occ_rate']:.4f} "
                f"OCC>{row['paired']['occ_beats_ig_rate']:.4f} "
                f"ties={row['paired']['tie_rate']:.4f} "
                f"(n={row['paired']['n']})"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"Invalid --ks: {args.ks}")

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
    print(f"ks: {ks}")

    replace_id = choose_replace_id(tokenizer, args.mask_mode)
    device = torch.device(args.device)

    by_k_values: Dict[int, Dict[str, List[float]]] = {
        k: {
            "ig_logit": [],
            "ig_prob": [],
            "occ_logit": [],
            "occ_prob": [],
            "eg_logit": [],
            "eg_prob": [],
            "rand_logit": [],
            "rand_prob": [],
            "paired_ig_minus_occ_logit": [],
            "paired_ig_minus_occ_prob": [],
            "paired_ig_minus_eg_logit": [],
            "paired_occ_minus_eg_logit": [],
        }
        for k in ks
    }

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

        enc = tokenizer(
            qtext,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        seq_len = int(input_ids.shape[1])
        if target_idx < 0 or target_idx >= int(model.config.num_labels):
            n_skipped += 1
            continue

        special_mask = safe_special_mask(tokenizer, input_ids[0].detach().cpu().tolist())
        valid_positions = {
            pos
            for pos in range(seq_len)
            if int(attention_mask[0, pos].item()) == 1 and special_mask[pos] == 0
        }
        if not valid_positions:
            n_skipped += 1
            continue

        base_logit, base_prob = forward_target(model, input_ids, attention_mask, target_idx)

        ig_items = row.get("ig_top_positive", []) or []
        occ_items = row.get("occlusion_top_positive", []) or []
        eg_items = row.get("eg_top_positive", []) or []

        for k in ks:
            ig_idx = select_top_indices(ig_items, valid_positions, k, args.require_positive)
            occ_idx = select_top_indices(occ_items, valid_positions, k, args.require_positive)
            eg_idx = select_top_indices(eg_items, valid_positions, k, args.require_positive)
            has_eg = bool(eg_idx)
            if not ig_idx or not occ_idx:
                continue

            rng = stable_rng(args.seed, qid, k)
            rand_candidates = sorted(valid_positions)
            rng.shuffle(rand_candidates)
            rand_idx = rand_candidates[: min(k, len(rand_candidates))]

            ig_logit_drop, ig_prob_drop = masked_drop(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_idx=target_idx,
                indices=ig_idx,
                replace_id=replace_id,
                mode=args.mask_mode,
                base_logit=base_logit,
                base_prob=base_prob,
            )
            occ_logit_drop, occ_prob_drop = masked_drop(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_idx=target_idx,
                indices=occ_idx,
                replace_id=replace_id,
                mode=args.mask_mode,
                base_logit=base_logit,
                base_prob=base_prob,
            )
            eg_logit_drop = float("nan")
            eg_prob_drop = float("nan")
            if has_eg:
                eg_logit_drop, eg_prob_drop = masked_drop(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_idx=target_idx,
                    indices=eg_idx,
                    replace_id=replace_id,
                    mode=args.mask_mode,
                    base_logit=base_logit,
                    base_prob=base_prob,
                )
            rand_logit_drop, rand_prob_drop = masked_drop(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_idx=target_idx,
                indices=rand_idx,
                replace_id=replace_id,
                mode=args.mask_mode,
                base_logit=base_logit,
                base_prob=base_prob,
            )

            d = by_k_values[k]
            d["ig_logit"].append(float(ig_logit_drop))
            d["ig_prob"].append(float(ig_prob_drop))
            d["occ_logit"].append(float(occ_logit_drop))
            d["occ_prob"].append(float(occ_prob_drop))
            if has_eg:
                d["eg_logit"].append(float(eg_logit_drop))
                d["eg_prob"].append(float(eg_prob_drop))
            d["rand_logit"].append(float(rand_logit_drop))
            d["rand_prob"].append(float(rand_prob_drop))
            d["paired_ig_minus_occ_logit"].append(float(ig_logit_drop - occ_logit_drop))
            d["paired_ig_minus_occ_prob"].append(float(ig_prob_drop - occ_prob_drop))
            if has_eg:
                d["paired_ig_minus_eg_logit"].append(float(ig_logit_drop - eg_logit_drop))
                d["paired_occ_minus_eg_logit"].append(float(occ_logit_drop - eg_logit_drop))

        n_used += 1
        if i % 100 == 0:
            print(f"processed: {i}/{len(rows)}")

    by_k_summary: Dict[str, Any] = {}
    for k in ks:
        d = by_k_values[k]
        paired_logit = d["paired_ig_minus_occ_logit"]
        n_pair = len(paired_logit)
        ig_wins = sum(1 for x in paired_logit if x > 1e-12)
        occ_wins = sum(1 for x in paired_logit if x < -1e-12)
        ties = n_pair - ig_wins - occ_wins
        has_eg = len(d["eg_logit"]) > 0

        eg_section = None
        eg_rate = float("nan")
        if has_eg:
            # Tri-method winner rate for EG: EG beats IG and OCC on same query.
            # We can only compute where EG is present, so align by length via stored paired diffs.
            n_eg = min(len(d["paired_ig_minus_eg_logit"]), len(d["paired_occ_minus_eg_logit"]))
            eg_beats = 0
            tie3 = 0
            for ii in range(n_eg):
                ig_minus_eg = d["paired_ig_minus_eg_logit"][ii]
                occ_minus_eg = d["paired_occ_minus_eg_logit"][ii]
                # EG wins if both IG and OCC drops are lower than EG drop.
                if ig_minus_eg < -1e-12 and occ_minus_eg < -1e-12:
                    eg_beats += 1
                # tie bucket for near-equality cases across both comparisons.
                elif abs(ig_minus_eg) <= 1e-12 and abs(occ_minus_eg) <= 1e-12:
                    tie3 += 1
            eg_rate = (eg_beats / n_eg) if n_eg else float("nan")
            eg_section = {
                "logit_drop": stat(d["eg_logit"]),
                "prob_drop": stat(d["eg_prob"]),
                "n": n_eg,
            }

        by_k_summary[str(k)] = {
            "ig": {
                "logit_drop": stat(d["ig_logit"]),
                "prob_drop": stat(d["ig_prob"]),
            },
            "occlusion": {
                "logit_drop": stat(d["occ_logit"]),
                "prob_drop": stat(d["occ_prob"]),
            },
            "eg": eg_section,
            "random": {
                "logit_drop": stat(d["rand_logit"]),
                "prob_drop": stat(d["rand_prob"]),
            },
            "paired": {
                "n": n_pair,
                "ig_minus_occ_logit": stat(d["paired_ig_minus_occ_logit"]),
                "ig_minus_occ_prob": stat(d["paired_ig_minus_occ_prob"]),
                "ig_minus_eg_logit": stat(d["paired_ig_minus_eg_logit"]) if has_eg else None,
                "occ_minus_eg_logit": stat(d["paired_occ_minus_eg_logit"]) if has_eg else None,
                "ig_beats_occ_rate": (ig_wins / n_pair) if n_pair else float("nan"),
                "occ_beats_ig_rate": (occ_wins / n_pair) if n_pair else float("nan"),
                "eg_beats_others_rate": eg_rate,
                "tie_rate": (ties / n_pair) if n_pair else float("nan"),
            },
        }

    summary = {
        "model_dir": str(model_dir),
        "attribution_jsonl": args.attribution_jsonl,
        "target_source": args.target_source,
        "mask_mode": args.mask_mode,
        "require_positive": args.require_positive,
        "ks": ks,
        "seed": args.seed,
        "n_rows_total": len(rows),
        "n_rows_used": n_used,
        "n_rows_skipped": n_skipped,
        "by_k": by_k_summary,
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
