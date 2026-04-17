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
        visual_hit = bool(norm and norm in visual_lexicon)
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
        hits = [tok for tok in row["norm_tokens"] if tok in visual_lexicon]
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
            influential = build_influential_set(
                tokens_by_method=tokens_by_method,
                require_overlap=bool(args.require_method_overlap),
            )
            token_rows, visual_token_indices, non_visual_token_indices = classify_query_tokens_binary(
                query=query,
                influential=influential,
                visual_lexicon=visual_lexicon,
            )
            phrase_rows, visual_phrase_count, non_visual_phrase_count = classify_phrases_binary(
                phrase_rows=phrases_by_qid.get(qid, []),
                visual_lexicon=visual_lexicon,
            )

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
