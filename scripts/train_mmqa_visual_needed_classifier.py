#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
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


NEG_LABEL = 0
POS_LABEL = 1
LABEL_TO_NAME = {
    NEG_LABEL: "non_visual_needed",
    POS_LABEL: "visual_needed",
}

# Exact MMQA qtypes normalized with whitespace removed and lowercase.
EXACT_TO_BINARY_LABEL = {
    "textq": NEG_LABEL,
    "tableq": NEG_LABEL,
    "imageq": POS_LABEL,
    "imagelistq": POS_LABEL,
    "compose(textq,tableq)": NEG_LABEL,
    "compose(tableq,textq)": NEG_LABEL,
    "intersect(tableq,textq)": NEG_LABEL,
    "compare(tableq,compose(tableq,textq))": NEG_LABEL,
    "compose(tableq,imagelistq)": POS_LABEL,
    "compose(textq,imagelistq)": POS_LABEL,
    "compose(imageq,tableq)": POS_LABEL,
    "compose(imageq,textq)": POS_LABEL,
    "intersect(imagelistq,tableq)": POS_LABEL,
    "intersect(imagelistq,textq)": POS_LABEL,
    "compare(compose(tableq,imageq),tableq)": POS_LABEL,
    "compare(compose(tableq,imageq),compose(tableq,textq))": POS_LABEL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a supervised binary classifier on MMQA question text where "
            "all qtypes needing image evidence map to visual_needed and the rest "
            "map to non_visual_needed."
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
    parser.add_argument("--epochs", type=int, default=3)
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
        help="Use inverse-frequency class weights for the binary loss.",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Training/eval loss.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma used when --loss-type focal.",
    )
    parser.add_argument(
        "--save-dev-preds-jsonl",
        type=str,
        default="",
        help="Optional path to write best-epoch dev predictions.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def read_mmqa_visual_needed_rows(
    path: str,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, int], List[str]]:
    rows: List[Dict[str, Any]] = []
    counts_by_exact: Dict[str, int] = {}
    total = 0
    unmapped: List[str] = []

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
            qtype_norm = normalize_qtype_name(qtype_raw)
            counts_by_exact[qtype_norm] = counts_by_exact.get(qtype_norm, 0) + 1
            if qtype_norm not in EXACT_TO_BINARY_LABEL:
                if qtype_norm not in unmapped:
                    unmapped.append(qtype_norm)
                continue

            qtext = get_query_text(record)
            if not qtext:
                continue

            rows.append(
                {
                    "query_id": get_query_id(record, line_idx),
                    "query_text": qtext,
                    "label": EXACT_TO_BINARY_LABEL[qtype_norm],
                    "gold_qtype": qtype_raw,
                    "gold_qtype_norm": qtype_norm,
                }
            )

    return rows, total, counts_by_exact, sorted(unmapped)


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
            "gold_qtype": row["gold_qtype"],
            "gold_qtype_norm": row["gold_qtype_norm"],
            **enc,
        }


@dataclass
class Collator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        qids = [f["query_id"] for f in features]
        qtexts = [f["query_text"] for f in features]
        gqtypes = [f["gold_qtype"] for f in features]
        gqtypes_norm = [f["gold_qtype_norm"] for f in features]
        labels = [int(f["labels"]) for f in features]

        model_feats = []
        for f in features:
            model_feats.append(
                {
                    k: v
                    for k, v in f.items()
                    if k not in {"query_id", "query_text", "labels", "gold_qtype", "gold_qtype_norm"}
                }
            )

        batch = self.tokenizer.pad(model_feats, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["query_id"] = qids
        batch["query_text"] = qtexts
        batch["gold_qtype"] = gqtypes
        batch["gold_qtype_norm"] = gqtypes_norm
        return batch


def count_by_binary_label(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter = Counter()
    for row in rows:
        counts[LABEL_TO_NAME[int(row["label"])]] += 1
    return dict(counts)


def count_by_exact_qtype(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter = Counter()
    for row in rows:
        counts[str(row["gold_qtype"])] += 1
    return dict(counts)


def compute_class_weights(rows: List[Dict[str, Any]]) -> torch.Tensor:
    counts = Counter(int(r["label"]) for r in rows)
    total = float(sum(counts.values()))
    weights = []
    for label_idx in [NEG_LABEL, POS_LABEL]:
        cnt = float(counts.get(label_idx, 0))
        if cnt <= 0:
            weights.append(1.0)
        else:
            weights.append(total / (2.0 * cnt))
    return torch.tensor(weights, dtype=torch.float32)


def binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str,
    class_weights: Optional[torch.Tensor],
    focal_gamma: float,
) -> torch.Tensor:
    if loss_type == "ce":
        return F.cross_entropy(logits, labels, weight=class_weights)

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    target_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    target_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    focal_factor = (1.0 - target_probs).pow(float(focal_gamma))
    loss = -focal_factor * target_log_probs
    if class_weights is not None:
        loss = loss * class_weights[labels]
    return loss.mean()


def confusion(labels: List[int], preds: List[int]) -> Dict[str, int]:
    tp = tn = fp = fn = 0
    for y, p in zip(labels, preds):
        tp += int(y == POS_LABEL and p == POS_LABEL)
        tn += int(y == NEG_LABEL and p == NEG_LABEL)
        fp += int(y == NEG_LABEL and p == POS_LABEL)
        fn += int(y == POS_LABEL and p == NEG_LABEL)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_conf(conf: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = conf["tp"], conf["tn"], conf["fp"], conf["fn"]
    total = max(tp + tn + fp + fn, 1)
    acc = (tp + tn) / total
    rec_pos = tp / max(tp + fn, 1)
    rec_neg = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (rec_pos + rec_neg)
    prec_pos = tp / max(tp + fp, 1)
    prec_neg = tn / max(tn + fn, 1)
    f1_pos = (2 * prec_pos * rec_pos) / max(prec_pos + rec_pos, 1e-12)
    f1_neg = (2 * prec_neg * rec_neg) / max(prec_neg + rec_neg, 1e-12)
    macro_recall = 0.5 * (rec_pos + rec_neg)
    macro_f1 = 0.5 * (f1_pos + f1_neg)
    weighted_f1 = ((tp + fn) * f1_pos + (tn + fp) * f1_neg) / max(total, 1)
    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "precision_visual_needed": prec_pos,
        "recall_visual_needed": rec_pos,
        "recall_non_visual_needed": rec_neg,
        "f1_visual_needed": f1_pos,
        "f1_non_visual_needed": f1_neg,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


@torch.no_grad()
def evaluate(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
    loss_type: str,
    class_weights: Optional[torch.Tensor],
    focal_gamma: float,
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
        gqtypes = batch.pop("gold_qtype")
        gqtypes_norm = batch.pop("gold_qtype_norm")
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(**batch)
            logits = outputs.logits
            loss = binary_loss(
                logits=logits,
                labels=labels,
                loss_type=loss_type,
                class_weights=class_weights,
                focal_gamma=focal_gamma,
            )

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        total_loss += float(loss.item())
        total_batches += 1

        labels_cpu = labels.detach().cpu().tolist()
        preds_cpu = preds.detach().cpu().tolist()
        probs_cpu = probs.detach().cpu().tolist()

        all_labels.extend(labels_cpu)
        all_preds.extend(preds_cpu)

        for qid, qtext, gq, gqn, y, p, pr in zip(
            qids,
            qtexts,
            gqtypes,
            gqtypes_norm,
            labels_cpu,
            preds_cpu,
            probs_cpu,
        ):
            all_rows.append(
                {
                    "query_id": qid,
                    "query_text": qtext,
                    "gold_qtype": gq,
                    "gold_qtype_norm": gqn,
                    "gold_label": int(y),
                    "gold_label_name": LABEL_TO_NAME[int(y)],
                    "pred_label": int(p),
                    "pred_label_name": LABEL_TO_NAME[int(p)],
                    "pred_prob": float(pr[int(p)]),
                    "prob_non_visual_needed": float(pr[NEG_LABEL]),
                    "prob_visual_needed": float(pr[POS_LABEL]),
                    "logit_margin_visual_minus_non_visual": float(pr[POS_LABEL] - pr[NEG_LABEL]),
                }
            )

    conf = confusion(all_labels, all_preds)
    metrics = metrics_from_conf(conf)
    metrics.update(conf)
    metrics["loss"] = total_loss / max(total_batches, 1)
    metrics["n"] = len(all_labels)
    return {"metrics": metrics, "rows": all_rows}


def train_one_epoch(
    model: Any,
    loader: DataLoader,
    optimizer: Any,
    scheduler: Any,
    device: torch.device,
    grad_accum_steps: int,
    max_grad_norm: float,
    use_bf16: bool,
    loss_type: str,
    class_weights: Optional[torch.Tensor],
    focal_gamma: float,
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
        batch.pop("gold_qtype_norm")
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(**batch)
            loss = binary_loss(
                logits=outputs.logits,
                labels=labels,
                loss_type=loss_type,
                class_weights=class_weights,
                focal_gamma=focal_gamma,
            ) / grad_accum_steps
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

    train_rows, train_total, train_type_counts, train_unmapped = read_mmqa_visual_needed_rows(args.train_jsonl)
    dev_rows, dev_total, dev_type_counts, dev_unmapped = read_mmqa_visual_needed_rows(args.dev_jsonl)

    if train_unmapped or dev_unmapped:
        missing = sorted(set(train_unmapped + dev_unmapped))
        raise ValueError(
            "Unmapped MMQA qtypes found. Add them to EXACT_TO_BINARY_LABEL: "
            f"{missing}"
        )

    print("=== Data Summary ===")
    print(f"train_jsonl_total_rows:   {train_total}")
    print(f"dev_jsonl_total_rows:     {dev_total}")
    print(f"train_rows_used:          {len(train_rows)}")
    print(f"dev_rows_used:            {len(dev_rows)}")
    print(f"train_binary_distribution:{count_by_binary_label(train_rows)}")
    print(f"dev_binary_distribution:  {count_by_binary_label(dev_rows)}")
    print(f"train_exact_distribution: {train_type_counts}")
    print(f"dev_exact_distribution:   {dev_type_counts}")

    device = torch.device(args.device)
    use_bf16 = bool(args.bf16 and device.type == "cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
        id2label={NEG_LABEL: LABEL_TO_NAME[NEG_LABEL], POS_LABEL: LABEL_TO_NAME[POS_LABEL]},
        label2id={LABEL_TO_NAME[NEG_LABEL]: NEG_LABEL, LABEL_TO_NAME[POS_LABEL]: POS_LABEL},
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

    class_weights: Optional[torch.Tensor] = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_rows).to(device)
        print(f"class_weights:            {class_weights.detach().cpu().tolist()}")

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
            loss_type=args.loss_type,
            class_weights=class_weights,
            focal_gamma=args.focal_gamma,
        )
        ev = evaluate(
            model=model,
            loader=dev_loader,
            device=device,
            use_bf16=use_bf16,
            loss_type=args.loss_type,
            class_weights=class_weights,
            focal_gamma=args.focal_gamma,
        )
        dev_metrics = ev["metrics"]
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        }
        history.append(record)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"dev_acc={dev_metrics['acc']:.4f} "
            f"dev_bal_acc={dev_metrics['bal_acc']:.4f} "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f} "
            f"tp={dev_metrics['tp']} tn={dev_metrics['tn']} "
            f"fp={dev_metrics['fp']} fn={dev_metrics['fn']}"
        )

        if dev_metrics["bal_acc"] > best_dev_bal_acc:
            best_dev_bal_acc = float(dev_metrics["bal_acc"])
            best_epoch = epoch
            best_dev_rows = ev["rows"]
            best_metrics = dev_metrics
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    type_to_label_name = {
        qtype_norm: LABEL_TO_NAME[label_idx]
        for qtype_norm, label_idx in sorted(EXACT_TO_BINARY_LABEL.items())
    }
    label_map_path = out_dir / "qtype_label_map.json"
    with label_map_path.open("w") as handle:
        json.dump(
            {
                "label_space": "visual_needed_binary",
                "type_to_label_norm": EXACT_TO_BINARY_LABEL,
                "type_to_label_name": type_to_label_name,
                "label_to_name": {str(k): v for k, v in LABEL_TO_NAME.items()},
                "selected_labels_display": [LABEL_TO_NAME[NEG_LABEL], LABEL_TO_NAME[POS_LABEL]],
            },
            handle,
            indent=2,
        )
    print(f"Wrote: {label_map_path}")

    summary = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_dev_bal_acc": best_dev_bal_acc,
        "best_dev_metrics": best_metrics,
        "history": history,
        "saved_model_dir": str(out_dir),
        "label_map_path": str(label_map_path),
        "train_rows_used": len(train_rows),
        "dev_rows_used": len(dev_rows),
        "train_binary_distribution": count_by_binary_label(train_rows),
        "dev_binary_distribution": count_by_binary_label(dev_rows),
        "train_exact_distribution": train_type_counts,
        "dev_exact_distribution": dev_type_counts,
        "train_macro_target_ratio": mean([float(r["label"]) for r in train_rows]) if train_rows else 0.0,
    }
    summary_path = out_dir / "train_summary_visual_needed.json"
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
