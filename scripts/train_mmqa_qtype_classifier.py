#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a question-type classifier on MMQA question text with configurable "
            "label space (including exact 16-class qtype)."
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
        "--qtypes",
        type=str,
        default="all",
        help=(
            "Comma-separated qtypes to include, or 'all' for every type observed in train/dev. "
            "Matching is case-insensitive."
        ),
    )
    parser.add_argument(
        "--train-sampling",
        type=str,
        default="full",
        choices=["full", "balanced"],
        help="Sampling strategy for train split.",
    )
    parser.add_argument(
        "--dev-sampling",
        type=str,
        default="full",
        choices=["full", "balanced"],
        help="Sampling strategy for dev split.",
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=0,
        help=(
            "Optional per-class cap for train. "
            "For balanced mode, this is max balanced size. 0 = no cap."
        ),
    )
    parser.add_argument(
        "--dev-per-class",
        type=int,
        default=0,
        help=(
            "Optional per-class cap for dev. "
            "For balanced mode, this is max balanced size. 0 = no cap."
        ),
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
        "--use-class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inverse-frequency class weights in cross entropy.",
    )
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


def read_mmqa_rows(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, str], int]:
    rows: List[Dict[str, Any]] = []
    display_by_norm: Dict[str, str] = {}
    total = 0

    with open(path, "r") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            qtype_raw = get_question_type(record)
            if not qtype_raw:
                continue
            qtype_norm = qtype_raw.lower()
            qtext = get_query_text(record)
            if not qtext:
                continue
            display_by_norm.setdefault(qtype_norm, qtype_raw)
            rows.append(
                {
                    "query_id": get_query_id(record, line_idx),
                    "query_text": qtext,
                    "qtype_norm": qtype_norm,
                    "qtype_raw": display_by_norm[qtype_norm],
                }
            )
    return rows, display_by_norm, total


def parse_qtypes_arg(qtypes_arg: str) -> Optional[List[str]]:
    if str(qtypes_arg).strip().lower() == "all":
        return None
    out: List[str] = []
    for item in str(qtypes_arg).split(","):
        item = item.strip()
        if item:
            out.append(item.lower())
    if not out:
        raise ValueError("--qtypes cannot be empty")
    return out


def build_label_maps(
    train_rows: List[Dict[str, Any]],
    dev_rows: List[Dict[str, Any]],
    train_display: Dict[str, str],
    dev_display: Dict[str, str],
    qtypes_arg: str,
) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    selected_norms = parse_qtypes_arg(qtypes_arg)
    all_norms = sorted(
        set([r["qtype_norm"] for r in train_rows] + [r["qtype_norm"] for r in dev_rows])
    )
    if selected_norms is None:
        selected_norms = all_norms

    missing = [t for t in selected_norms if t not in all_norms]
    if missing:
        raise ValueError(f"Requested qtypes not found in train/dev: {missing}")

    type_to_label = {t: i for i, t in enumerate(selected_norms)}
    label_to_name: Dict[int, str] = {}
    for t, idx in type_to_label.items():
        display = train_display.get(t) or dev_display.get(t) or t
        label_to_name[idx] = display

    return type_to_label, label_to_name, selected_norms


def build_rows(
    rows: List[Dict[str, Any]],
    selected_norms: List[str],
    type_to_label: Dict[str, int],
    sampling: str,
    per_class_cap: int,
    seed: int,
) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        t = row["qtype_norm"]
        if t in type_to_label:
            by_type[t].append(
                {
                    "query_id": row["query_id"],
                    "query_text": row["query_text"],
                    "label": int(type_to_label[t]),
                    "qtype_norm": t,
                    "qtype_raw": row["qtype_raw"],
                }
            )

    for t in selected_norms:
        if len(by_type[t]) == 0:
            raise ValueError(f"No rows found for selected qtype: {t}")

    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    if sampling == "balanced":
        n = min(len(by_type[t]) for t in selected_norms)
        if per_class_cap > 0:
            n = min(n, per_class_cap)
        for t in selected_norms:
            out.extend(rng.sample(by_type[t], n))
    else:
        for t in selected_norms:
            items = by_type[t]
            if per_class_cap > 0 and len(items) > per_class_cap:
                items = rng.sample(items, per_class_cap)
            out.extend(items)

    rng.shuffle(out)
    return out


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
            "gold_qtype": row["qtype_raw"],
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


def confusion_matrix(labels: List[int], preds: List[int], num_classes: int) -> List[List[int]]:
    mat = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for y, p in zip(labels, preds):
        mat[y][p] += 1
    return mat


def metrics_from_confusion(conf: List[List[int]], label_to_name: Dict[int, str]) -> Dict[str, Any]:
    num_classes = len(conf)
    total = sum(sum(row) for row in conf)
    correct = sum(conf[i][i] for i in range(num_classes))
    acc = correct / max(total, 1)

    per_class: Dict[str, Dict[str, float]] = {}
    recalls = []
    f1s = []
    weighted_f1_num = 0.0
    weighted_f1_den = 0.0

    for i in range(num_classes):
        tp = conf[i][i]
        fn = sum(conf[i][j] for j in range(num_classes) if j != i)
        fp = sum(conf[j][i] for j in range(num_classes) if j != i)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
        support = sum(conf[i])
        per_class[label_to_name[i]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        recalls.append(recall)
        f1s.append(f1)
        weighted_f1_num += f1 * support
        weighted_f1_den += support

    return {
        "acc": acc,
        "macro_recall": sum(recalls) / max(len(recalls), 1),
        "macro_f1": sum(f1s) / max(len(f1s), 1),
        "weighted_f1": weighted_f1_num / max(weighted_f1_den, 1),
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
    label_to_name: Dict[int, str],
    class_weights: Optional[torch.Tensor],
) -> Dict[str, Any]:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_rows: List[Dict[str, Any]] = []
    total_loss = 0.0
    total_batches = 0

    n_classes = len(label_to_name)

    for batch in loader:
        labels = batch["labels"].to(device)
        qids = batch.pop("query_id")
        qtexts = batch.pop("query_text")
        gtypes = batch.pop("gold_qtype")
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(**batch).logits
            loss = F.cross_entropy(logits, labels, weight=class_weights)
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
            out_row = {
                "query_id": qid,
                "query_text": qtext,
                "gold_qtype": gt,
                "gold_label": int(y),
                "pred_label": int(p),
                "pred_qtype": label_to_name[int(p)],
            }
            for idx in range(n_classes):
                out_row[f"prob_{label_to_name[idx]}"] = float(pr[idx])
            all_rows.append(out_row)

    conf = confusion_matrix(all_labels, all_preds, num_classes=n_classes)
    m = metrics_from_confusion(conf, label_to_name)
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
    class_weights: Optional[torch.Tensor],
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
            logits = model(**batch).logits
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss = loss / grad_accum_steps
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


def compute_class_weights(rows: List[Dict[str, Any]], n_classes: int, device: torch.device) -> torch.Tensor:
    counts = [0 for _ in range(n_classes)]
    for row in rows:
        counts[int(row["label"])] += 1
    total = sum(counts)
    weights = []
    for c in counts:
        w = total / max(n_classes * c, 1)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def count_by_qtype(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for row in rows:
        out[row["qtype_raw"]] += 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows_all, train_display, train_total = read_mmqa_rows(args.train_jsonl)
    dev_rows_all, dev_display, dev_total = read_mmqa_rows(args.dev_jsonl)

    type_to_label, label_to_name, selected_norms = build_label_maps(
        train_rows=train_rows_all,
        dev_rows=dev_rows_all,
        train_display=train_display,
        dev_display=dev_display,
        qtypes_arg=args.qtypes,
    )

    train_rows = build_rows(
        rows=train_rows_all,
        selected_norms=selected_norms,
        type_to_label=type_to_label,
        sampling=args.train_sampling,
        per_class_cap=args.train_per_class,
        seed=args.seed,
    )
    dev_rows = build_rows(
        rows=dev_rows_all,
        selected_norms=selected_norms,
        type_to_label=type_to_label,
        sampling=args.dev_sampling,
        per_class_cap=args.dev_per_class,
        seed=args.seed + 7,
    )

    print("=== Data Summary ===")
    print(f"train_jsonl_total_rows: {train_total}")
    print(f"dev_jsonl_total_rows:   {dev_total}")
    print(f"selected_qtypes:        {[label_to_name[type_to_label[t]] for t in selected_norms]}")
    print(f"num_classes:            {len(type_to_label)}")
    print(f"train_sampling:         {args.train_sampling}")
    print(f"dev_sampling:           {args.dev_sampling}")
    print(f"train_rows_used:        {len(train_rows)}")
    print(f"dev_rows_used:          {len(dev_rows)}")
    print(f"train_counts_by_qtype:  {count_by_qtype(train_rows)}")
    print(f"dev_counts_by_qtype:    {count_by_qtype(dev_rows)}")

    device = torch.device(args.device)
    use_bf16 = bool(args.bf16 and device.type == "cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(type_to_label),
        id2label={i: label_to_name[i] for i in range(len(type_to_label))},
        label2id={label_to_name[i]: i for i in range(len(type_to_label))},
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

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_rows, len(type_to_label), device=device)
        print(f"class_weights: {[round(float(w), 6) for w in class_weights.detach().cpu().tolist()]}")

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
            class_weights=class_weights,
        )
        ev = evaluate(
            model=model,
            loader=dev_loader,
            device=device,
            use_bf16=use_bf16,
            label_to_name=label_to_name,
            class_weights=class_weights,
        )
        dev_metrics = ev["metrics"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_acc": dev_metrics["acc"],
            "dev_macro_recall": dev_metrics["macro_recall"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
        }
        history.append(record)

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"dev_acc={dev_metrics['acc']:.4f} "
            f"dev_macro_recall={dev_metrics['macro_recall']:.4f} "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f} "
            f"dev_weighted_f1={dev_metrics['weighted_f1']:.4f}"
        )

        if dev_metrics["macro_recall"] > best_macro_recall:
            best_macro_recall = float(dev_metrics["macro_recall"])
            best_epoch = epoch
            best_dev_rows = ev["rows"]
            best_metrics = dev_metrics
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    label_map_path = out_dir / "qtype_label_map.json"
    with label_map_path.open("w") as handle:
        json.dump(
            {
                "type_to_label_norm": type_to_label,
                "label_to_name": {str(k): v for k, v in label_to_name.items()},
                "selected_qtypes_display": [label_to_name[type_to_label[t]] for t in selected_norms],
            },
            handle,
            indent=2,
        )
    print(f"Wrote: {label_map_path}")

    summary = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_dev_macro_recall": best_macro_recall,
        "best_dev_metrics": best_metrics,
        "history": history,
        "saved_model_dir": str(out_dir),
        "label_map_path": str(label_map_path),
    }
    summary_path = out_dir / "train_summary_qtype.json"
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
