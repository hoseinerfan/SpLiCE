#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ID_FIELDS = ["query_id", "qid", "question_id", "id"]
TEXT_FIELDS = ["query_text", "question", "query", "text"]

LABEL_TO_QTYPE = {
    0: "TextQ",
    1: "TableQ",
    2: "ImageQ",
}
QTYPE_TO_ROUTE = {
    "TextQ": "text_only",
    "TableQ": "table_needed",
    "ImageQ": "visual_needed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with fine-tuned 3-class MMQA question-type classifier "
            "(TextQ/TableQ/ImageQ)."
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model dir or HF id for 3-class classifier.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help="Input JSONL containing questions.",
    )
    parser.add_argument(
        "--query-text",
        type=str,
        default=None,
        help="Single query text (alternative to --input-jsonl).",
    )
    parser.add_argument(
        "--query-id",
        type=str,
        default="single_query",
        help="Query id for --query-text mode.",
    )
    parser.add_argument(
        "--id-field",
        type=str,
        default="",
        help="Optional explicit query id field for input JSONL.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="",
        help="Optional explicit query text field for input JSONL.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Output JSONL with predictions and probabilities.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=200,
        help="Progress print interval.",
    )
    return parser.parse_args()


def pick_field(row: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in row:
            return c
    return None


def read_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.query_text:
        return [{"query_id": args.query_id, "query_text": args.query_text}]

    if not args.input_jsonl:
        raise ValueError("Provide --input-jsonl or --query-text.")

    in_path = Path(args.input_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"input-jsonl not found: {in_path}")

    out: List[Dict[str, str]] = []
    with in_path.open("r") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            if args.id_field:
                qid = str(row.get(args.id_field, f"line-{i}"))
            else:
                fid = pick_field(row, ID_FIELDS)
                qid = str(row.get(fid, f"line-{i}")) if fid else f"line-{i}"

            if args.text_field:
                qtext = str(row.get(args.text_field, "")).strip()
            else:
                ft = pick_field(row, TEXT_FIELDS)
                qtext = str(row.get(ft, "")).strip() if ft else ""

            if not qtext:
                continue

            out.append({"query_id": qid, "query_text": qtext})
    return out


def main() -> None:
    args = parse_args()
    rows = read_rows(args)
    if not rows:
        raise ValueError("No valid rows to score.")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device).eval()

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts_qtype = {"TextQ": 0, "TableQ": 0, "ImageQ": 0}
    counts_route = {"text_only": 0, "table_needed": 0, "visual_needed": 0}

    with out_path.open("w") as w:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            texts = [r["query_text"] for r in chunk]
            enc = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.inference_mode():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu()
                preds = torch.argmax(probs, dim=-1).tolist()

            for row, pred_idx, prob_vec in zip(chunk, preds, probs.tolist()):
                pred_qtype = LABEL_TO_QTYPE[int(pred_idx)]
                route = QTYPE_TO_ROUTE[pred_qtype]
                counts_qtype[pred_qtype] += 1
                counts_route[route] += 1

                out_row = {
                    "query_id": row["query_id"],
                    "query_text": row["query_text"],
                    "pred_label": int(pred_idx),
                    "pred_qtype": pred_qtype,
                    "route": route,
                    "prob_textq": float(prob_vec[0]),
                    "prob_tableq": float(prob_vec[1]),
                    "prob_imageq": float(prob_vec[2]),
                }
                w.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            done = min(start + args.batch_size, len(rows))
            if args.print_every > 0 and done % args.print_every == 0:
                print(f"processed {done}/{len(rows)}")

    print(f"Wrote: {out_path}")
    print(
        json.dumps(
            {
                "counts_qtype": counts_qtype,
                "counts_route": counts_route,
                "total": len(rows),
                "model_name": args.model_name,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
