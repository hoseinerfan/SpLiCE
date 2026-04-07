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


TYPE_TO_LABEL = {
    "textq": 0,
    "tableq": 1,
    "imageq": 2,
}
LABEL_TO_NAME = {
    0: "TextQ",
    1: "TableQ",
    2: "ImageQ",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 3-class sequence classifier on MMQA question text "
            "(TextQ/TableQ/ImageQ) and evaluate on balanced dev."
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
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=2000,
        help="Balanced sample size per class for train (0 => use min available).",
    )
    parser.add_argument(
        "--dev-per-class",
        type=int,
        default=230,
        help="Balanced sample size per class for dev (0 => use min available).",
    )
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
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--reinit-classifier",
        action="store_true",
        help="Reinitialize classification head after model load.",
    )
    parser.add_argument(
        "--save-dev-preds-jsonl",
        type=str,
        default="",
        help="Optional path for best-epoch dev predictions.",
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


def read_mmqa_3class_rows(path: str) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    by_type: Dict[str, List[Dict[str, Any]]] = {k: [] for k in TYPE_TO_LABEL.keys()}
    total = 0
    with open(path, "r") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            qtype = get_question_type(record).lower()
            if qtype not in by_type:
                continue
            qtext = get_query_text(record)
            if not qtext:
                continue
            by_type[qtype].append(
                {
                    "query_id": get_query_id(record, line_idx),
                    "query_text": qtext,
                    "label": TYPE_TO_LABEL[qtype],
                    "qtype": qtype,
                }
            )
    return by_type, total


def build_balanced_3class(
    by_type: Dict[str, List[Dict[str, Any]]],
    per_class: int,
    seed: int,
) -> List[Dict[str, Any]]:
    n = min(len(by_type["textq"]), len(by_type["tableq"]), len(by_type["imageq"]))
    if per_class > 0:
        n = min(n, per_class)
    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for key in ["textq", "tableq", "imageq"]:
        selected.extend(rng.sample(by_type[key], n))
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
            "gold_qtype": row["qtype"],
            **enc,
        }


@dataclass
class Collator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        qids = [f["query_id"] for f in features]
        qtexts = [f["query_text"] for f in features]
        gtypes = [f["gold_qtype"] for f in features]
        labels = [int(f["labels"]) for f in features]

        model_feats = []
        for f in features:
            model_feats.append(
                {
                    k: v
                    for k, v in f.items()
                    if k not in {"query_id", "query_text", "labels", "gold_qtype"}
                }
            )

        batch = self.tokenizer.pad(model_feats, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["query_id"] = qids
        batch["query_text"] = qtexts
        batch["gold_qtype"] = gtypes
        return batch


def confusion_matrix(labels: List[int], preds: List[int], num_classes: int = 3) -> List[List[int]]:
    mat = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for y, p in zip(labels, preds):
        mat[y][p] += 1
    return mat


def metrics_from_confusion(conf: List[List[int]]) -> Dict[str, Any]:
    num_classes = len(conf)
    total = sum(sum(row) for row in conf)
    correct = sum(conf[i][i] for i in range(num_classes))
    acc = correct / max(total, 1)

    per_class: Dict[str, Dict[str, float]] = {}
    recalls = []
    f1s = []
    for i in range(num_classes):
        tp = conf[i][i]
        fn = sum(conf[i][j] for j in range(num_classes) if j != i)
        fp = sum(conf[j][i] for j in range(num_classes) if j != i)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
        per_class[LABEL_TO_NAME[i]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(conf[i]),
        }
        recalls.append(recall)
        f1s.append(f1)

    return {
        "acc": acc,
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "per_class": per_class,
        "confusion_matrix": conf,
        "n": total,
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
        gtypes = batch.pop("gold_qtype")
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

        for qid, qtext, gt, y, p, pr in zip(qids, qtexts, gtypes, labels_cpu, preds_cpu, probs_cpu):
            all_rows.append(
                {
                    "query_id": qid,
                    "query_text": qtext,
                    "gold_qtype": gt,
                    "gold_label": int(y),
                    "pred_label": int(p),
                    "pred_qtype": LABEL_TO_NAME[int(p)],
                    "prob_textq": float(pr[TYPE_TO_LABEL["textq"]]),
                    "prob_tableq": float(pr[TYPE_TO_LABEL["tableq"]]),
                    "prob_imageq": float(pr[TYPE_TO_LABEL["imageq"]]),
                }
            )

    conf = confusion_matrix(all_labels, all_preds, num_classes=3)
    m = metrics_from_confusion(conf)
    m["loss"] = total_loss / max(total_batches, 1)
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
        batch.pop("gold_qtype")
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

    train_by_type, train_total = read_mmqa_3class_rows(args.train_jsonl)
    dev_by_type, dev_total = read_mmqa_3class_rows(args.dev_jsonl)
    train_rows = build_balanced_3class(train_by_type, args.train_per_class, args.seed)
    dev_rows = build_balanced_3class(dev_by_type, args.dev_per_class, args.seed + 7)

    print("=== Data Summary ===")
    print(f"train_jsonl_total_rows: {train_total}")
    print(f"dev_jsonl_total_rows:   {dev_total}")
    print(f"train_textq_available:  {len(train_by_type['textq'])}")
    print(f"train_tableq_available: {len(train_by_type['tableq'])}")
    print(f"train_imageq_available: {len(train_by_type['imageq'])}")
    print(f"dev_textq_available:    {len(dev_by_type['textq'])}")
    print(f"dev_tableq_available:   {len(dev_by_type['tableq'])}")
    print(f"dev_imageq_available:   {len(dev_by_type['imageq'])}")
    print(f"train_balanced_rows:    {len(train_rows)}")
    print(f"dev_balanced_rows:      {len(dev_rows)}")

    device = torch.device(args.device)
    use_bf16 = bool(args.bf16 and device.type == "cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label={0: "TextQ", 1: "TableQ", 2: "ImageQ"},
        label2id={"TextQ": 0, "TableQ": 1, "ImageQ": 2},
        ignore_mismatched_sizes=True,
    ).to(device)
    if args.reinit_classifier and hasattr(model, "classifier"):
        model.classifier.reset_parameters()

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

    best_macro_recall = -1.0
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    best_dev_rows: List[Dict[str, Any]] = []
    best_metrics: Optional[Dict[str, Any]] = None

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
            "dev_loss": dev_metrics["loss"],
            "dev_acc": dev_metrics["acc"],
            "dev_macro_recall": dev_metrics["macro_recall"],
            "dev_macro_f1": dev_metrics["macro_f1"],
        }
        history.append(record)

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"dev_acc={dev_metrics['acc']:.4f} "
            f"dev_macro_recall={dev_metrics['macro_recall']:.4f} "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}"
        )

        if dev_metrics["macro_recall"] > best_macro_recall:
            best_macro_recall = float(dev_metrics["macro_recall"])
            best_epoch = epoch
            best_dev_rows = ev["rows"]
            best_metrics = dev_metrics
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    summary = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_dev_macro_recall": best_macro_recall,
        "best_dev_metrics": best_metrics,
        "history": history,
        "saved_model_dir": str(out_dir),
    }
    summary_path = out_dir / "train_summary_3class.json"
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
