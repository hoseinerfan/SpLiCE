#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_VISUAL_HYPOTHESIS = (
    "Answering this question requires visual evidence from the document image."
)
DEFAULT_TEXT_HYPOTHESIS = (
    "Answering this question can be done from text alone without looking at the document image."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether a query needs visual evidence using a text-only "
            "NLI/cross-encoder model (no hybrid routing)."
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cross-encoder/nli-deberta-v3-base",
        help="HF model id or local model directory for sequence classification.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help="Optional JSONL file with query rows.",
    )
    parser.add_argument(
        "--query-text",
        type=str,
        default=None,
        help="Single query text. Use this instead of --input-jsonl for one query.",
    )
    parser.add_argument(
        "--query-id",
        type=str,
        default="single_query",
        help="Query id used with --query-text.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Output JSONL with query-level visual evidence decisions.",
    )
    parser.add_argument(
        "--visual-hypothesis",
        type=str,
        default=DEFAULT_VISUAL_HYPOTHESIS,
    )
    parser.add_argument(
        "--text-hypothesis",
        type=str,
        default=DEFAULT_TEXT_HYPOTHESIS,
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.45,
        help="Minimum entailment support required for a confident class.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.05,
        help="Minimum absolute support gap between visual/text hypotheses.",
    )
    parser.add_argument(
        "--binary-only",
        action="store_true",
        help="Force binary decision with no 'uncertain' output.",
    )
    parser.add_argument(
        "--binary-tie-break",
        type=str,
        default="visual",
        choices=["visual", "text"],
        help="When supports are exactly tied in --binary-only mode, pick this class.",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        default="entailment_minus_contradiction",
        choices=["entailment", "entailment_minus_contradiction"],
        help=(
            "How to convert NLI outputs into support values used for decision. "
            "'entailment_minus_contradiction' is usually more stable."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on number of rows from input JSONL (0 = all).",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Progress print interval for JSONL input.",
    )
    return parser.parse_args()


def pick_field(row: Dict, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in row:
            return name
    return None


def read_queries(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.query_text:
        return [{"query_id": args.query_id, "query_text": args.query_text}]

    if not args.input_jsonl:
        raise ValueError("Provide either --query-text or --input-jsonl.")

    path = Path(args.input_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"input-jsonl not found: {path}")

    rows: List[Dict[str, str]] = []
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid_field = pick_field(row, ["qid", "query_id", "id"])
            text_field = pick_field(row, ["question", "query", "query_text"])
            if not text_field:
                continue
            qid = str(row.get(qid_field, f"row_{len(rows)}")) if qid_field else f"row_{len(rows)}"
            qtext = str(row[text_field]).strip()
            if not qtext:
                continue
            rows.append({"query_id": qid, "query_text": qtext})
            if args.max_rows > 0 and len(rows) >= args.max_rows:
                break
    return rows


def get_label_indices(model: Any) -> Tuple[int, Optional[int], Optional[int]]:
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    entail_idx = None
    contra_idx = None
    neutral_idx = None
    for idx, label in id2label.items():
        if "entail" in label:
            entail_idx = idx
        elif "contrad" in label:
            contra_idx = idx
        elif "neutral" in label:
            neutral_idx = idx
    if entail_idx is None:
        raise ValueError(
            f"Could not find entailment label in id2label={model.config.id2label}. "
            "Use an NLI model."
        )
    return entail_idx, contra_idx, neutral_idx


def score_hypothesis(
    model: Any,
    tokenizer: Any,
    device: Any,
    entail_idx: int,
    contra_idx: Optional[int],
    neutral_idx: Optional[int],
    premise: str,
    hypothesis: str,
) -> Dict[str, float]:
    import torch

    with torch.inference_mode():
        encoded = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu()

    out: Dict[str, float] = {"entailment": float(probs[entail_idx].item())}
    if contra_idx is not None:
        out["contradiction"] = float(probs[contra_idx].item())
    if neutral_idx is not None:
        out["neutral"] = float(probs[neutral_idx].item())
    out["logit_entailment"] = float(logits[entail_idx].item())
    return out


def decide_label(
    visual_support: float,
    text_support: float,
    min_support: float,
    min_margin: float,
    binary_only: bool,
    binary_tie_break: str,
) -> str:
    if binary_only:
        if visual_support > text_support:
            return "needs_visual"
        if text_support > visual_support:
            return "text_only"
        return "needs_visual" if binary_tie_break == "visual" else "text_only"

    margin = visual_support - text_support
    if visual_support >= min_support and margin >= min_margin:
        return "needs_visual"
    if text_support >= min_support and (-margin) >= min_margin:
        return "text_only"
    return "uncertain"


def support_from_scores(scores: Dict[str, float], mode: str) -> float:
    if mode == "entailment":
        return float(scores.get("entailment", 0.0))
    contradiction = float(scores.get("contradiction", 0.0))
    entailment = float(scores.get("entailment", 0.0))
    return entailment - contradiction


def main() -> None:
    args = parse_args()
    queries = read_queries(args)
    if not queries:
        raise ValueError("No valid queries found.")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device).eval()
    entail_idx, contra_idx, neutral_idx = get_label_indices(model)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"needs_visual": 0, "text_only": 0, "uncertain": 0}
    with out_path.open("w") as fout:
        for i, row in enumerate(queries, start=1):
            qid = row["query_id"]
            qtext = row["query_text"]

            visual_scores = score_hypothesis(
                model,
                tokenizer,
                device,
                entail_idx,
                contra_idx,
                neutral_idx,
                qtext,
                args.visual_hypothesis,
            )
            text_scores = score_hypothesis(
                model,
                tokenizer,
                device,
                entail_idx,
                contra_idx,
                neutral_idx,
                qtext,
                args.text_hypothesis,
            )

            visual_support = support_from_scores(visual_scores, args.score_mode)
            text_support = support_from_scores(text_scores, args.score_mode)
            margin = visual_support - text_support
            label = decide_label(
                visual_support=visual_support,
                text_support=text_support,
                min_support=args.min_support,
                min_margin=args.min_margin,
                binary_only=args.binary_only,
                binary_tie_break=args.binary_tie_break,
            )
            counts[label] += 1

            out_row = {
                "query_id": qid,
                "query_text": qtext,
                "label": label,
                "needs_visual_evidence": (label == "needs_visual"),
                "visual_support": round(visual_support, 6),
                "text_support": round(text_support, 6),
                "margin_visual_minus_text": round(margin, 6),
                "visual_entailment": round(visual_scores.get("entailment", 0.0), 6),
                "text_entailment": round(text_scores.get("entailment", 0.0), 6),
                "visual_contradiction": round(visual_scores.get("contradiction", 0.0), 6),
                "text_contradiction": round(text_scores.get("contradiction", 0.0), 6),
                "visual_hypothesis": args.visual_hypothesis,
                "text_hypothesis": args.text_hypothesis,
                "visual_scores": {k: round(v, 6) for k, v in visual_scores.items()},
                "text_scores": {k: round(v, 6) for k, v in text_scores.items()},
                "min_support": args.min_support,
                "min_margin": args.min_margin,
                "binary_only": args.binary_only,
                "binary_tie_break": args.binary_tie_break,
                "score_mode": args.score_mode,
                "model_name": args.model_name,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            if args.input_jsonl and args.print_every > 0 and i % args.print_every == 0:
                print(f"processed {i}/{len(queries)}")

    print(f"Wrote: {out_path}")
    print(json.dumps({"counts": counts, "total": len(queries)}, indent=2))


if __name__ == "__main__":
    main()
