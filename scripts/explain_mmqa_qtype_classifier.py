#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Token-level attribution for MMQA qtype classifiers using Integrated Gradients "
            "and/or occlusion."
        )
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--label-map-json", type=str, default="")
    parser.add_argument("--method", type=str, default="both", choices=["ig", "occlusion", "both"])
    parser.add_argument("--target", type=str, default="pred", choices=["pred", "gold"])
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument(
        "--occlusion-mode",
        type=str,
        default="mask",
        choices=["mask", "pad", "unk", "drop"],
        help="drop sets attention_mask to 0 at that token; others replace token id.",
    )
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--save-full-token-scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, include full token score arrays in output jsonl.",
    )
    return parser.parse_args()


def normalize_qtype_name(name: str) -> str:
    return "".join(str(name).lower().split())


def get_by_path(record: Dict[str, Any], dot_path: str) -> Any:
    value: Any = record
    for token in dot_path.split("."):
        if not isinstance(value, dict) or token not in value:
            return None
        value = value[token]
    return value


def get_query_text(record: Dict[str, Any]) -> str:
    for key in ["question", "query", "query_text", "text"]:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def get_query_id(record: Dict[str, Any], line_idx: int) -> str:
    for key in ["qid", "query_id", "question_id", "id"]:
        if key in record:
            return str(record[key])
    return f"line-{line_idx}"


def get_question_type(record: Dict[str, Any]) -> str:
    for path in ["metadata.type", "type", "question_type", "qtype", "question_meta.type"]:
        value = get_by_path(record, path)
        if value is not None:
            qtype = str(value).strip()
            if qtype:
                return qtype
    return ""


def read_jsonl(path: str, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qtext = get_query_text(obj)
            if not qtext:
                continue
            rows.append(
                {
                    "query_id": get_query_id(obj, idx),
                    "query_text": qtext,
                    "gold_qtype": get_question_type(obj),
                }
            )
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_type_to_label_name(path: str) -> Dict[str, str]:
    if not path:
        return {}
    payload = json.load(open(path, "r"))
    out = payload.get("type_to_label_name", {})
    return {str(k): str(v) for k, v in out.items()}


def safe_special_mask(tokenizer: Any, input_ids: List[int]) -> List[int]:
    try:
        return tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    except Exception:
        return [0 for _ in input_ids]


def extract_top_tokens(
    tokens: List[str],
    scores: List[float],
    valid_positions: List[int],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items = []
    for pos in valid_positions:
        items.append({"index": pos, "token": tokens[pos], "score": float(scores[pos])})

    pos_sorted = sorted(items, key=lambda x: x["score"], reverse=True)
    neg_sorted = sorted(items, key=lambda x: x["score"])
    return pos_sorted[:top_k], neg_sorted[:top_k]


def integrated_gradients(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_idx: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    embed_layer = model.get_input_embeddings()

    ids = input_ids.to(device)
    am = attention_mask.to(device)
    emb = embed_layer(ids).detach()

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    baseline_ids = torch.full_like(ids, fill_value=pad_id)

    special_mask = safe_special_mask(tokenizer, ids[0].detach().cpu().tolist())
    special_mask_t = torch.tensor(special_mask, dtype=torch.bool, device=device).unsqueeze(0)
    baseline_ids = torch.where(special_mask_t, ids, baseline_ids)
    baseline_emb = embed_layer(baseline_ids).detach()

    total_grad = torch.zeros_like(emb)
    alphas = torch.linspace(0.0, 1.0, steps=steps + 1, device=device)[1:]

    for alpha in alphas:
        x = baseline_emb + alpha * (emb - baseline_emb)
        x.requires_grad_(True)
        logits = model(inputs_embeds=x, attention_mask=am).logits
        target_logit = logits[0, target_idx]
        grad = torch.autograd.grad(target_logit, x, retain_graph=False, create_graph=False)[0]
        total_grad += grad.detach()

    avg_grad = total_grad / max(len(alphas), 1)
    attributions = ((emb - baseline_emb) * avg_grad).sum(dim=-1).squeeze(0)
    return attributions.detach().cpu()


@torch.no_grad()
def occlusion_scores(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_idx: int,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    ids = input_ids.to(device)
    am = attention_mask.to(device)

    base_logits = model(input_ids=ids, attention_mask=am).logits
    base_target = float(base_logits[0, target_idx].item())

    token_count = ids.shape[1]
    out = torch.zeros(token_count, dtype=torch.float32)

    special_mask = safe_special_mask(tokenizer, ids[0].detach().cpu().tolist())

    replace_id = tokenizer.mask_token_id
    if mode == "pad":
        replace_id = tokenizer.pad_token_id
    elif mode == "unk":
        replace_id = tokenizer.unk_token_id
    if replace_id is None:
        replace_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for i in range(token_count):
        if int(am[0, i].item()) == 0 or special_mask[i] == 1:
            continue

        occ_ids = ids.clone()
        occ_am = am.clone()

        if mode == "drop":
            occ_ids[0, i] = replace_id
            occ_am[0, i] = 0
        else:
            occ_ids[0, i] = replace_id

        occ_logits = model(input_ids=occ_ids, attention_mask=occ_am).logits
        occ_target = float(occ_logits[0, target_idx].item())
        out[i] = base_target - occ_target

    return out


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_map_path = args.label_map_json or str(model_dir / "qtype_label_map.json")
    type_to_label_name = load_type_to_label_name(label_map_path) if Path(label_map_path).exists() else {}

    # DebertaV2 fast tokenizer conversion may require protobuf in some HPC envs.
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    label2id = {str(v): int(k) for k, v in id2label.items()}

    rows = read_jsonl(args.input_jsonl, args.limit)
    print(f"loaded_rows: {len(rows)}")
    print(f"model_dir: {model_dir}")
    print(f"method: {args.method}")
    print(f"target: {args.target}")

    with output_path.open("w") as out_f:
        for i, row in enumerate(rows, start=1):
            enc = tokenizer(
                row["query_text"],
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            ids_on_device = input_ids.to(device)
            am_on_device = attention_mask.to(device)

            with torch.no_grad():
                logits = model(input_ids=ids_on_device, attention_mask=am_on_device).logits
                probs = torch.softmax(logits, dim=-1)
                pred_idx = int(torch.argmax(probs, dim=-1)[0].item())
                pred_prob = float(probs[0, pred_idx].item())

            target_idx = pred_idx
            gold_label_name: Optional[str] = None
            if args.target == "gold":
                gq = row.get("gold_qtype", "")
                norm = normalize_qtype_name(gq) if gq else ""
                gold_label_name = type_to_label_name.get(norm)
                if gold_label_name in label2id:
                    target_idx = label2id[gold_label_name]

            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            special_mask = safe_special_mask(tokenizer, input_ids[0].tolist())
            valid_positions = [
                pos
                for pos in range(len(tokens))
                if int(attention_mask[0, pos].item()) == 1 and special_mask[pos] == 0
            ]

            record: Dict[str, Any] = {
                "query_id": row["query_id"],
                "query_text": row["query_text"],
                "gold_qtype": row.get("gold_qtype", ""),
                "gold_label_name": gold_label_name,
                "pred_label_idx": pred_idx,
                "pred_label_name": id2label[pred_idx],
                "pred_prob": pred_prob,
                "target_label_idx": target_idx,
                "target_label_name": id2label[target_idx],
            }

            if args.method in {"ig", "both"}:
                ig_scores = integrated_gradients(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_idx=target_idx,
                    steps=args.ig_steps,
                    device=device,
                )
                ig_list = [float(x) for x in ig_scores.tolist()]
                top_pos, top_neg = extract_top_tokens(tokens, ig_list, valid_positions, args.top_k)
                record["ig_top_positive"] = top_pos
                record["ig_top_negative"] = top_neg
                if args.save_full_token_scores:
                    record["ig_token_scores"] = [
                        {
                            "index": pos,
                            "token": tokens[pos],
                            "score": ig_list[pos],
                            "is_special": bool(special_mask[pos]),
                            "is_active": bool(int(attention_mask[0, pos].item()) == 1),
                        }
                        for pos in range(len(tokens))
                    ]

            if args.method in {"occlusion", "both"}:
                occ_scores = occlusion_scores(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_idx=target_idx,
                    mode=args.occlusion_mode,
                    device=device,
                )
                occ_list = [float(x) for x in occ_scores.tolist()]
                top_pos, top_neg = extract_top_tokens(tokens, occ_list, valid_positions, args.top_k)
                record["occlusion_mode"] = args.occlusion_mode
                record["occlusion_top_positive"] = top_pos
                record["occlusion_top_negative"] = top_neg
                if args.save_full_token_scores:
                    record["occlusion_token_scores"] = [
                        {
                            "index": pos,
                            "token": tokens[pos],
                            "score": occ_list[pos],
                            "is_special": bool(special_mask[pos]),
                            "is_active": bool(int(attention_mask[0, pos].item()) == 1),
                        }
                        for pos in range(len(tokens))
                    ]

            out_f.write(json.dumps(record) + "\n")

            if i % 50 == 0:
                print(f"processed: {i}/{len(rows)}")

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
