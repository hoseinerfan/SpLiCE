#!/usr/bin/env python3
import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


POS_LABEL = 1  # needs_visual / ImageQ
NEG_LABEL = 0  # text_only / TextQ


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary sequence classifier on MMQA question text "
            "(ImageQ vs TextQ) and evaluate on a balanced dev set."
        )
    )
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--dev-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--model-name",
        type=str,
        default="cross-encoder/nli-deberta-v3-base",
        help="HF model id or local model directory.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--dev-per-class", type=int, default=230)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 autocast if device supports it.",
    )
    parser.add_argument(
        "--save-dev-preds-jsonl",
        type=str,
        default="",
        help="Optional path to write per-query dev predictions.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        if value is not None:
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
            return str(value)
    return ""


def read_mmqa_binary_rows(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    imageq: List[Dict[str, Any]] = []
    textq: List[Dict[str, Any]] = []
    total = 0
    with open(path, "r") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            qtype = get_question_type(record).lower()
            qtext = get_query_text(record)
            if not qtext:
                continue
            row = {
                "query_id": get_query_id(record, line_idx),
                "query_text": qtext,
            }
            if qtype == "imageq":
                row["label"] = POS_LABEL
                imageq.append(row)
            elif qtype == "textq":
                row["label"] = NEG_LABEL
                textq.append(row)
    return imageq, textq, total


def build_balanced(
    imageq: List[Dict[str, Any]],
    textq: List[Dict[str, Any]],
    per_class: int,
    seed: int,
) -> List[Dict[str, Any]]:
    n = min(len(imageq), len(textq))
    if per_class > 0:
        n = min(n, per_class)
    rng = random.Random(seed)
    selected = rng.sample(imageq, n) + rng.sample(textq, n)
    rng.shuffle(selected)
    return selected


class TextClsDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        enc = self.tokenizer(
            row["query_text"],
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "query_id": row["query_id"],
            "query_text": row["query_text"],
            "labels": int(row["label"]),
            **enc,
        }


@dataclass
class Collator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        qids = [f["query_id"] for f in features]
        qtexts = [f["query_text"] for f in features]
        labels = [int(f["labels"]) for f in features]

        model_feats = []
        for f in features:
            model_feats.append(
                {
                    k: v
                    for k, v in f.items()
                    if k not in {"query_id", "query_text", "labels"}
                }
            )

        batch = self.tokenizer.pad(model_feats, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["query_id"] = qids
        batch["query_text"] = qtexts
        return batch


def confusion(labels: List[int], preds: List[int]) -> Dict[str, int]:
    tp = tn = fp = fn = 0
    for y, p in zip(labels, preds):
        tp += int(y == POS_LABEL and p == POS_LABEL)
        tn += int(y == NEG_LABEL and p == NEG_LABEL)
        fp += int(y == NEG_LABEL and p == POS_LABEL)
        fn += int(y == POS_LABEL and p == NEG_LABEL)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_conf(c: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    total = max(tp + tn + fp + fn, 1)
    acc = (tp + tn) / total
    rec_pos = tp / max(tp + fn, 1)
    rec_neg = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (rec_pos + rec_neg)
    prec_pos = tp / max(tp + fp, 1)
    f1_pos = (2 * prec_pos * rec_pos) / max(prec_pos + rec_pos, 1e-12)
    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "precision_pos": prec_pos,
        "recall_pos": rec_pos,
        "recall_neg": rec_neg,
        "f1_pos": f1_pos,
    }


@torch.no_grad()
def evaluate(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
) -> Dict[str, Any]:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_rows: List[Dict[str, Any]] = []
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        labels = batch["labels"].to(device)
        qids = batch.pop("query_id")
        qtexts = batch.pop("query_text")
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(**batch, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        total_loss += float(loss.item())
        total_batches += 1

        labels_cpu = labels.detach().cpu().tolist()
        preds_cpu = preds.detach().cpu().tolist()
        probs_cpu = probs.detach().cpu().tolist()

        all_labels.extend(labels_cpu)
        all_preds.extend(preds_cpu)

        for qid, qtext, y, p, pr in zip(qids, qtexts, labels_cpu, preds_cpu, probs_cpu):
            all_rows.append(
                {
                    "query_id": qid,
                    "query_text": qtext,
                    "gold_label": int(y),
                    "pred_label": int(p),
                    "prob_text_only": float(pr[NEG_LABEL]),
                    "prob_needs_visual": float(pr[POS_LABEL]),
                    "logit_margin_visual_minus_text": float(pr[POS_LABEL] - pr[NEG_LABEL]),
                }
            )

    conf = confusion(all_labels, all_preds)
    m = metrics_from_conf(conf)
    m.update(conf)
    m["loss"] = total_loss / max(total_batches, 1)
    m["n"] = len(all_labels)
    return {"metrics": m, "rows": all_rows}


def train_one_epoch(
    model: Any,
    loader: DataLoader,
    optimizer: Any,
    scheduler: Any,
    device: torch.device,
    grad_accum_steps: int,
    max_grad_norm: float,
    use_bf16: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        labels = batch["labels"].to(device)
        batch.pop("query_id")
        batch.pop("query_text")
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(**batch, labels=labels)
            loss = outputs.loss / grad_accum_steps
        loss.backward()
        total_loss += float(loss.item()) * grad_accum_steps
        total_steps += 1

        if step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    if total_steps % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(total_steps, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_img, train_txt, train_total = read_mmqa_binary_rows(args.train_jsonl)
    dev_img, dev_txt, dev_total = read_mmqa_binary_rows(args.dev_jsonl)
    train_rows = build_balanced(train_img, train_txt, args.train_per_class, args.seed)
    dev_rows = build_balanced(dev_img, dev_txt, args.dev_per_class, args.seed + 7)

    print("=== Data Summary ===")
    print(f"train_jsonl_total_rows: {train_total}")
    print(f"dev_jsonl_total_rows:   {dev_total}")
    print(f"train_imageq_available: {len(train_img)}")
    print(f"train_textq_available:  {len(train_txt)}")
    print(f"dev_imageq_available:   {len(dev_img)}")
    print(f"dev_textq_available:    {len(dev_txt)}")
    print(f"train_balanced_rows:    {len(train_rows)}")
    print(f"dev_balanced_rows:      {len(dev_rows)}")

    device = torch.device(args.device)
    use_bf16 = bool(args.bf16 and device.type == "cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_ds = TextClsDataset(train_rows, tokenizer, args.max_length)
    dev_ds = TextClsDataset(dev_rows, tokenizer, args.max_length)
    collator = Collator(tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum_steps, 1))
    total_steps = max(steps_per_epoch * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_dev_bal_acc = -1.0
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    best_dev_rows: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            use_bf16=use_bf16,
        )
        ev = evaluate(model, dev_loader, device, use_bf16)
        dev_metrics = ev["metrics"]
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        }
        history.append(record)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"dev_acc={dev_metrics['acc']:.4f} dev_bal_acc={dev_metrics['bal_acc']:.4f} "
            f"tp={dev_metrics['tp']} tn={dev_metrics['tn']} fp={dev_metrics['fp']} fn={dev_metrics['fn']}"
        )

        if dev_metrics["bal_acc"] > best_dev_bal_acc:
            best_dev_bal_acc = float(dev_metrics["bal_acc"])
            best_epoch = epoch
            best_dev_rows = ev["rows"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    summary = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_dev_bal_acc": best_dev_bal_acc,
        "history": history,
        "saved_model_dir": str(out_dir),
    }
    summary_path = out_dir / "train_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote: {summary_path}")

    if args.save_dev_preds_jsonl:
        pred_path = Path(args.save_dev_preds_jsonl)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w") as handle:
            for row in best_dev_rows:
                handle.write(json.dumps(row) + "\n")
        print(f"Wrote: {pred_path}")


if __name__ == "__main__":
    main()
