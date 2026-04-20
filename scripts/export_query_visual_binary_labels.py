#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


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

DEFAULT_ARTIFACT_TOKENS = {"s", "t", "p", "unc", "ing", "ed", "ly"}

DEFAULT_ATTRIBUTION_FALLBACK_BLOCKLIST = {
    "among",
    "around",
    "inside",
    "outside",
    "across",
    "between",
    "through",
    "toward",
    "towards",
    "into",
    "onto",
    "from",
    "there",
    "here",
}

DEFAULT_VISUAL_LEXICON = {
    "logo",
    "poster",
    "image",
    "photo",
    "picture",
    "face",
    "hair",
    "beard",
    "mustache",
    "glasses",
    "eyeglasses",
    "wear",
    "wearing",
    "wears",
    "holding",
    "holds",
    "color",
    "colour",
    "shape",
    "symbol",
    "flag",
    "jersey",
    "bald",
    "sideburns",
    "visual",
}

NEG_LABEL_NAME = "non_visual_needed"
POS_LABEL_NAME = "visual_needed"

VISUAL_GOLD_QTYPES = {
    "imageq",
    "imagelistq",
    "compose(tableq,imagelistq)",
    "compose(textq,imagelistq)",
    "compose(imageq,tableq)",
    "compose(imageq,textq)",
    "intersect(imagelistq,tableq)",
    "intersect(imagelistq,textq)",
    "compare(compose(tableq,imageq),tableq)",
    "compare(compose(tableq,imageq),compose(tableq,textq))",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export strict binary visual/non-visual labels for query words and "
            "important phrases using attribution overlap and optional span-audit files."
        )
    )
    parser.add_argument("--attribution-jsonl", type=str, required=True)
    parser.add_argument(
        "--context-json",
        action="append",
        default=[],
        help=(
            "Optional eval_qtype_context_interactions.py JSON output. "
            "Repeat to intersect/union important phrase spans across multiple sources."
        ),
    )
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--qids-file", type=str, default="")
    parser.add_argument(
        "--methods",
        type=str,
        default="ig,occlusion",
        help="Comma-separated attribution methods: ig,occlusion,eg",
    )
    parser.add_argument(
        "--require-method-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, important words are intersection across methods; else union.",
    )
    parser.add_argument(
        "--require-phrase-source-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, important phrases must appear in every provided --context-json source.",
    )
    parser.add_argument("--min-token-len", type=int, default=3)
    parser.add_argument("--drop-stopwords", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-artifact-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--artifact-tokens", type=str, default="")
    parser.add_argument("--visual-lexicon", type=str, default="")
    parser.add_argument("--visual-lexicon-file", type=str, default="")
    parser.add_argument(
        "--augment-visual-tokens-from-attribution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, also emit attribution-augmented visual token fields for target visual queries. "
            "This does not overwrite the strict label field."
        ),
    )
    parser.add_argument(
        "--augment-target",
        type=str,
        default="pred",
        choices=["pred", "gold", "any"],
        help="Which rows are eligible for attribution-based token augmentation.",
    )
    parser.add_argument(
        "--augment-max-tokens",
        type=int,
        default=2,
        help="Max number of additional attribution-driven visual tokens to surface per query.",
    )
    return parser.parse_args()


def normalize_token(raw: str) -> str:
    token = str(raw)
    if token.startswith("##"):
        token = token[2:]
    while token.startswith("▁") or token.startswith("Ġ"):
        token = token[1:]
    token = token.strip().lower()
    token = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", token)
    return token


def parse_csv_set(raw: str) -> Set[str]:
    out = set()
    for item in str(raw).split(","):
        token = normalize_token(item)
        if token:
            out.add(token)
    return out


def normalize_qtype_name(raw: Any) -> str:
    return "".join(str(raw).strip().lower().split())


def read_list_file(path: str) -> Set[str]:
    if not path:
        return set()
    out = set()
    with open(path, "r") as handle:
        for line in handle:
            token = normalize_token(line.strip())
            if token:
                out.add(token)
    return out


def read_qid_filter(qids: Iterable[str], qids_file: str) -> Optional[Set[str]]:
    out = {str(x).strip() for x in qids if str(x).strip()}
    if qids_file:
        with open(qids_file, "r") as handle:
            for line in handle:
                qid = line.strip()
                if qid:
                    out.add(qid)
    return out if out else None


def query_word_tokens(query: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", str(query), flags=re.UNICODE)


def field_name(method: str) -> str:
    method_name = method.strip().lower()
    if method_name not in {"ig", "occlusion", "eg"}:
        raise ValueError(f"Unsupported method: {method}")
    return f"{method_name}_top_positive"


def visual_lexicon_hit(token: str, visual_lexicon: Set[str]) -> bool:
    if not token:
        return False
    candidates = {token}
    if len(token) > 3 and token.endswith("s"):
        candidates.add(token[:-1])
    if len(token) > 4 and token.endswith("es"):
        candidates.add(token[:-2])
    if len(token) > 4 and token.endswith("ies"):
        candidates.add(token[:-3] + "y")
    return any(x in visual_lexicon for x in candidates)


def collect_method_tokens(
    row: Dict[str, Any],
    methods: Sequence[str],
    min_token_len: int,
    stopwords: Optional[Set[str]],
    artifact_tokens: Optional[Set[str]],
) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for method in methods:
        vals: Set[str] = set()
        for item in row.get(field_name(method), []) or []:
            token = normalize_token(item.get("token", ""))
            if not token:
                continue
            if len(token) < max(1, min_token_len):
                continue
            if stopwords is not None and token in stopwords:
                continue
            if artifact_tokens is not None and token in artifact_tokens:
                continue
            vals.add(token)
        out[method] = vals
    return out


def collect_method_token_scores(
    row: Dict[str, Any],
    methods: Sequence[str],
    min_token_len: int,
    stopwords: Optional[Set[str]],
    artifact_tokens: Optional[Set[str]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for method in methods:
        vals: Dict[str, float] = {}
        for item in row.get(field_name(method), []) or []:
            token = normalize_token(item.get("token", ""))
            if not token:
                continue
            if len(token) < max(1, min_token_len):
                continue
            if stopwords is not None and token in stopwords:
                continue
            if artifact_tokens is not None and token in artifact_tokens:
                continue
            score = float(item.get("score", 0.0))
            prev = vals.get(token)
            if prev is None or score > prev:
                vals[token] = score
        out[method] = vals
    return out


def build_influential_set(tokens_by_method: Dict[str, Set[str]], require_overlap: bool) -> Set[str]:
    sets = [vals for vals in tokens_by_method.values() if vals]
    if not sets:
        return set()
    if require_overlap:
        out = set(sets[0])
        for vals in sets[1:]:
            out &= vals
        return out
    out: Set[str] = set()
    for vals in sets:
        out |= vals
    return out


def classify_query_tokens_binary(
    query: str,
    influential: Set[str],
    visual_lexicon: Set[str],
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    raw_tokens = query_word_tokens(query)
    token_rows: List[Dict[str, Any]] = []
    visual_indices: List[int] = []
    non_visual_indices: List[int] = []

    for idx, raw in enumerate(raw_tokens):
        norm = normalize_token(raw)
        important = bool(norm and norm in influential)
        visual_hit = bool(norm and visual_lexicon_hit(norm, visual_lexicon))
        label = POS_LABEL_NAME if important and visual_hit else NEG_LABEL_NAME

        row = {
            "index": idx,
            "token": raw,
            "norm": norm,
            "label": label,
            "important": important,
            "visual_lexicon_hit": visual_hit,
            "reasons": [],
        }
        if important:
            row["reasons"].append("attribution_overlap")
        if visual_hit:
            row["reasons"].append("visual_lexicon")

        token_rows.append(row)
        if label == POS_LABEL_NAME:
            visual_indices.append(idx)
        else:
            non_visual_indices.append(idx)

    return token_rows, visual_indices, non_visual_indices


def row_is_target_visual(row: Dict[str, Any], target: str) -> bool:
    pred_visual = str(row.get("pred_label_name", "")).strip() == POS_LABEL_NAME
    gold_visual = normalize_qtype_name(row.get("gold_qtype", "")) in VISUAL_GOLD_QTYPES
    if target == "pred":
        return pred_visual
    if target == "gold":
        return gold_visual
    return pred_visual or gold_visual


def build_clause_ids(raw_tokens: Sequence[str]) -> List[int]:
    clause_ids: List[int] = []
    clause_idx = 0
    for tok in raw_tokens:
        clause_ids.append(clause_idx)
        if tok in {",", ";", ":"}:
            clause_idx += 1
    return clause_ids


def select_attribution_augmented_indices(
    query: str,
    token_rows: Sequence[Dict[str, Any]],
    token_scores_by_method: Dict[str, Dict[str, float]],
    target_visual: bool,
    max_tokens: int,
) -> List[int]:
    if not target_visual or max_tokens <= 0:
        return []

    raw_tokens = query_word_tokens(query)
    clause_ids = build_clause_ids(raw_tokens)
    strict_visual_indices = [int(row["index"]) for row in token_rows if row.get("label") == POS_LABEL_NAME]
    anchor_clauses = {clause_ids[idx] for idx in strict_visual_indices if 0 <= idx < len(clause_ids)}

    candidates: List[Dict[str, Any]] = []
    for row in token_rows:
        idx = int(row["index"])
        norm = str(row.get("norm", ""))
        if not row.get("important"):
            continue
        if row.get("label") == POS_LABEL_NAME:
            continue
        if not norm or norm in DEFAULT_ATTRIBUTION_FALLBACK_BLOCKLIST:
            continue
        method_scores = {
            method: float(scores.get(norm, 0.0))
            for method, scores in token_scores_by_method.items()
            if float(scores.get(norm, 0.0)) > 0.0
        }
        if not method_scores:
            continue
        candidates.append(
            {
                "index": idx,
                "max_score": max(method_scores.values()),
                "method_scores": method_scores,
                "same_clause_anchor": bool(anchor_clauses) and idx < len(clause_ids) and clause_ids[idx] in anchor_clauses,
            }
        )

    if not candidates:
        return []

    if anchor_clauses:
        pool = [c for c in candidates if c["same_clause_anchor"]]
        if not pool:
            return []
    else:
        pool = candidates
    selected: List[int] = []
    selected_set: Set[int] = set()

    for method in token_scores_by_method:
        method_pool = [c for c in pool if method in c["method_scores"] and c["index"] not in selected_set]
        if not method_pool:
            continue
        method_pool.sort(key=lambda c: (-c["method_scores"][method], -c["max_score"], c["index"]))
        chosen = method_pool[0]
        selected.append(chosen["index"])
        selected_set.add(chosen["index"])
        if len(selected) >= max_tokens:
            return sorted(selected)

    remaining = [c for c in pool if c["index"] not in selected_set]
    remaining.sort(key=lambda c: (not c["same_clause_anchor"], -c["max_score"], c["index"]))
    for cand in remaining:
        selected.append(cand["index"])
        selected_set.add(cand["index"])
        if len(selected) >= max_tokens:
            break

    return sorted(selected)


def normalize_phrase(
    raw_phrase: str,
    min_token_len: int,
    stopwords: Optional[Set[str]],
    artifact_tokens: Optional[Set[str]],
) -> Tuple[str, List[str]]:
    norm_tokens: List[str] = []
    for raw in query_word_tokens(raw_phrase):
        token = normalize_token(raw)
        if not token:
            continue
        if len(token) < max(1, min_token_len):
            continue
        if stopwords is not None and token in stopwords:
            continue
        if artifact_tokens is not None and token in artifact_tokens:
            continue
        norm_tokens.append(token)
    return " ".join(norm_tokens), norm_tokens


def load_phrase_sources(
    context_paths: Sequence[str],
    qid_filter: Optional[Set[str]],
    min_token_len: int,
    stopwords: Optional[Set[str]],
    artifact_tokens: Optional[Set[str]],
) -> List[Dict[str, Dict[str, Dict[str, Any]]]]:
    phrase_sources: List[Dict[str, Dict[str, Dict[str, Any]]]] = []
    for raw_path in context_paths:
        path = Path(raw_path)
        payload = json.load(open(path, "r"))
        raw_rows = (((payload.get("raw_span_audit") or {}).get("rows")) or [])
        by_qid: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for row in raw_rows:
            qid = str(row.get("query_id", "")).strip()
            if not qid:
                continue
            if qid_filter is not None and qid not in qid_filter:
                continue
            raw_phrase = str(row.get("raw_span", "")).strip()
            norm_phrase, norm_tokens = normalize_phrase(
                raw_phrase=raw_phrase,
                min_token_len=min_token_len,
                stopwords=stopwords,
                artifact_tokens=artifact_tokens,
            )
            if not norm_phrase:
                continue
            rec = by_qid[qid].get(norm_phrase)
            drop_val = float(row.get("drop_logit", 0.0))
            if rec is None:
                by_qid[qid][norm_phrase] = {
                    "norm_phrase": norm_phrase,
                    "raw_examples": [raw_phrase] if raw_phrase else [],
                    "norm_tokens": norm_tokens,
                    "drop_values": [drop_val],
                    "source_paths": [str(path)],
                }
            else:
                if raw_phrase and raw_phrase not in rec["raw_examples"]:
                    rec["raw_examples"].append(raw_phrase)
                rec["drop_values"].append(drop_val)
                if str(path) not in rec["source_paths"]:
                    rec["source_paths"].append(str(path))
        phrase_sources.append(by_qid)
    return phrase_sources


def merge_phrase_sources(
    phrase_sources: Sequence[Dict[str, Dict[str, Dict[str, Any]]]],
    require_overlap: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    if not phrase_sources:
        return {}

    all_qids: Set[str] = set()
    for src in phrase_sources:
        all_qids |= set(src.keys())

    out: Dict[str, List[Dict[str, Any]]] = {}
    for qid in sorted(all_qids):
        if require_overlap:
            phrase_keys: Optional[Set[str]] = None
            for src in phrase_sources:
                cur = set(src.get(qid, {}).keys())
                if phrase_keys is None:
                    phrase_keys = cur
                else:
                    phrase_keys &= cur
            keep = phrase_keys or set()
        else:
            keep = set()
            for src in phrase_sources:
                keep |= set(src.get(qid, {}).keys())

        merged_rows: List[Dict[str, Any]] = []
        for phrase in sorted(keep):
            drop_values: List[float] = []
            raw_examples: List[str] = []
            norm_tokens: List[str] = []
            source_paths: List[str] = []
            for src in phrase_sources:
                rec = src.get(qid, {}).get(phrase)
                if rec is None:
                    continue
                drop_values.extend(rec["drop_values"])
                for raw in rec["raw_examples"]:
                    if raw not in raw_examples:
                        raw_examples.append(raw)
                if not norm_tokens:
                    norm_tokens = list(rec["norm_tokens"])
                for sp in rec["source_paths"]:
                    if sp not in source_paths:
                        source_paths.append(sp)
            merged_rows.append(
                {
                    "norm_phrase": phrase,
                    "raw_examples": raw_examples,
                    "norm_tokens": norm_tokens,
                    "mean_drop_logit": float(mean(drop_values)) if drop_values else 0.0,
                    "n_occurrences": len(drop_values),
                    "source_paths": source_paths,
                }
            )
        out[qid] = sorted(merged_rows, key=lambda x: (-x["mean_drop_logit"], x["norm_phrase"]))
    return out


def classify_phrases_binary(
    phrase_rows: Sequence[Dict[str, Any]],
    visual_lexicon: Set[str],
) -> Tuple[List[Dict[str, Any]], int, int]:
    out: List[Dict[str, Any]] = []
    visual_count = 0
    non_visual_count = 0
    for row in phrase_rows:
        hits = [tok for tok in row["norm_tokens"] if visual_lexicon_hit(tok, visual_lexicon)]
        label = POS_LABEL_NAME if hits else NEG_LABEL_NAME
        out_row = {
            "label": label,
            "important": True,
            "visual_lexicon_hits": hits,
            **row,
        }
        out.append(out_row)
        if label == POS_LABEL_NAME:
            visual_count += 1
        else:
            non_visual_count += 1
    return out, visual_count, non_visual_count


def main() -> None:
    args = parse_args()
    methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    for method in methods:
        field_name(method)

    qid_filter = read_qid_filter(args.qid, args.qids_file)
    stopwords = DEFAULT_STOPWORDS if args.drop_stopwords else None

    artifact_tokens: Optional[Set[str]] = None
    if args.drop_artifact_tokens:
        artifact_tokens = set(DEFAULT_ARTIFACT_TOKENS)
        artifact_tokens |= parse_csv_set(args.artifact_tokens)

    visual_lexicon = set(DEFAULT_VISUAL_LEXICON)
    visual_lexicon |= parse_csv_set(args.visual_lexicon)
    visual_lexicon |= read_list_file(args.visual_lexicon_file)

    phrase_sources = load_phrase_sources(
        context_paths=args.context_json,
        qid_filter=qid_filter,
        min_token_len=args.min_token_len,
        stopwords=stopwords,
        artifact_tokens=artifact_tokens,
    )
    phrases_by_qid = merge_phrase_sources(
        phrase_sources=phrase_sources,
        require_overlap=bool(args.require_phrase_source_overlap),
    )

    in_path = Path(args.attribution_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with in_path.open("r") as in_f, out_path.open("w") as out_f:
        for line in in_f:
            row = json.loads(line)
            qid = str(row.get("query_id", "")).strip()
            if qid_filter is not None and qid not in qid_filter:
                continue

            query = str(row.get("query_text", ""))
            tokens_by_method = collect_method_tokens(
                row=row,
                methods=methods,
                min_token_len=args.min_token_len,
                stopwords=stopwords,
                artifact_tokens=artifact_tokens,
            )
            token_scores_by_method = collect_method_token_scores(
                row=row,
                methods=methods,
                min_token_len=args.min_token_len,
                stopwords=stopwords,
                artifact_tokens=artifact_tokens,
            )
            influential = build_influential_set(
                tokens_by_method=tokens_by_method,
                require_overlap=bool(args.require_method_overlap),
            )
            token_rows, visual_token_indices, non_visual_token_indices = classify_query_tokens_binary(
                query=query,
                influential=influential,
                visual_lexicon=visual_lexicon,
            )
            target_visual = row_is_target_visual(row=row, target=args.augment_target)
            attribution_augmented_indices = select_attribution_augmented_indices(
                query=query,
                token_rows=token_rows,
                token_scores_by_method=token_scores_by_method,
                target_visual=bool(args.augment_visual_tokens_from_attribution) and target_visual,
                max_tokens=int(args.augment_max_tokens),
            )
            visual_token_indices_augmented = sorted(set(visual_token_indices) | set(attribution_augmented_indices))
            visual_token_indices_augmented_set = set(visual_token_indices_augmented)
            phrase_rows, visual_phrase_count, non_visual_phrase_count = classify_phrases_binary(
                phrase_rows=phrases_by_qid.get(qid, []),
                visual_lexicon=visual_lexicon,
            )
            for token_row in token_rows:
                idx = int(token_row["index"])
                token_row["augmented_label"] = (
                    POS_LABEL_NAME if idx in visual_token_indices_augmented_set else NEG_LABEL_NAME
                )
                token_row["augmented_reasons"] = list(token_row.get("reasons", []))
                if idx in attribution_augmented_indices and "attribution_fallback" not in token_row["augmented_reasons"]:
                    token_row["augmented_reasons"].append("attribution_fallback")

            out_row = {
                "query_id": qid,
                "query_text": query,
                "gold_qtype": row.get("gold_qtype", ""),
                "pred_label_name": row.get("pred_label_name", ""),
                "source_attribution_file": str(in_path),
                "source_context_files": list(args.context_json),
                "methods_used": methods,
                "require_method_overlap": bool(args.require_method_overlap),
                "require_phrase_source_overlap": bool(args.require_phrase_source_overlap),
                "important_token_overlap": sorted(influential),
                "important_tokens_by_method": {k: sorted(v) for k, v in tokens_by_method.items()},
                "token_labels": token_rows,
                "visual_token_indices": visual_token_indices,
                "non_visual_token_indices": non_visual_token_indices,
                "visual_token_count": len(visual_token_indices),
                "non_visual_token_count": len(non_visual_token_indices),
                "augment_visual_tokens_from_attribution": bool(args.augment_visual_tokens_from_attribution),
                "augment_target": str(args.augment_target),
                "attribution_augmented_token_indices": attribution_augmented_indices,
                "visual_token_indices_augmented": visual_token_indices_augmented,
                "visual_token_count_augmented": len(visual_token_indices_augmented),
                "phrase_labels": phrase_rows,
                "visual_phrase_count": visual_phrase_count,
                "non_visual_phrase_count": non_visual_phrase_count,
            }
            out_f.write(json.dumps(out_row) + "\n")
            written += 1

    print(f"Wrote: {out_path}")
    print(f"rows: {written}")


if __name__ == "__main__":
    main()
