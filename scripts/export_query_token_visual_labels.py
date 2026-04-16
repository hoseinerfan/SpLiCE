#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export query-token visual/non-visual labels from attribution JSONL "
            "using method-overlap tokens plus a visual lexicon."
        )
    )
    parser.add_argument("--attribution-jsonl", type=str, required=True)
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
        help="If true, influential tokens are intersection across methods; else union.",
    )
    parser.add_argument("--min-token-len", type=int, default=3)
    parser.add_argument("--drop-stopwords", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-artifact-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--artifact-tokens", type=str, default="")
    parser.add_argument("--visual-lexicon", type=str, default="")
    parser.add_argument("--visual-lexicon-file", type=str, default="")
    return parser.parse_args()


def normalize_token(raw: str) -> str:
    t = str(raw)
    if t.startswith("##"):
        t = t[2:]
    while t.startswith("▁") or t.startswith("Ġ"):
        t = t[1:]
    t = t.strip().lower()
    t = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", t)
    return t


def query_word_tokens(query: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", str(query), flags=re.UNICODE)


def parse_csv_set(raw: str) -> Set[str]:
    out = set()
    for x in str(raw).split(","):
        x = normalize_token(x)
        if x:
            out.add(x)
    return out


def read_list_file(path: str) -> Set[str]:
    if not path:
        return set()
    out = set()
    with open(path, "r") as handle:
        for line in handle:
            v = normalize_token(line.strip())
            if v:
                out.add(v)
    return out


def read_qid_filter(qids: Iterable[str], qids_file: str) -> Optional[Set[str]]:
    out = {str(x).strip() for x in qids if str(x).strip()}
    if qids_file:
        with open(qids_file, "r") as handle:
            for line in handle:
                q = line.strip()
                if q:
                    out.add(q)
    return out if out else None


def field_name(method: str) -> str:
    m = method.strip().lower()
    if m not in {"ig", "occlusion", "eg"}:
        raise ValueError(f"Unsupported method: {method}")
    return f"{m}_top_positive"


def collect_method_tokens(
    row: Dict[str, Any],
    methods: List[str],
    min_token_len: int,
    stopwords: Optional[Set[str]],
    artifact_tokens: Optional[Set[str]],
) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for m in methods:
        field = field_name(m)
        vals = set()
        for item in row.get(field, []) or []:
            tok = normalize_token(item.get("token", ""))
            if not tok:
                continue
            if len(tok) < max(1, min_token_len):
                continue
            if stopwords is not None and tok in stopwords:
                continue
            if artifact_tokens is not None and tok in artifact_tokens:
                continue
            vals.add(tok)
        out[m] = vals
    return out


def build_influential_set(tokens_by_method: Dict[str, Set[str]], require_overlap: bool) -> Set[str]:
    sets = [v for v in tokens_by_method.values() if v]
    if not sets:
        return set()
    if require_overlap:
        result = set(sets[0])
        for s in sets[1:]:
            result &= s
        return result
    result = set()
    for s in sets:
        result |= s
    return result


def classify_query_tokens(
    query: str,
    influential: Set[str],
    visual_lexicon: Set[str],
) -> Tuple[List[Dict[str, Any]], List[int], List[int], List[str], List[str]]:
    raw_tokens = query_word_tokens(query)
    token_rows: List[Dict[str, Any]] = []
    visual_idx: List[int] = []
    non_visual_idx: List[int] = []
    visual_toks: List[str] = []
    non_visual_toks: List[str] = []

    for idx, raw in enumerate(raw_tokens):
        norm = normalize_token(raw)
        klass = "neutral"
        reasons: List[str] = []

        if norm and norm in visual_lexicon:
            klass = "visual"
            reasons.append("visual_lexicon")
            if norm in influential:
                reasons.append("attribution_overlap")
        elif norm and norm in influential:
            klass = "non_visual"
            reasons.append("attribution_overlap")

        token_rows.append(
            {
                "index": idx,
                "token": raw,
                "norm": norm,
                "class": klass,
                "reasons": reasons,
            }
        )
        if klass == "visual":
            visual_idx.append(idx)
            visual_toks.append(raw)
        elif klass == "non_visual":
            non_visual_idx.append(idx)
            non_visual_toks.append(raw)

    return token_rows, visual_idx, non_visual_idx, visual_toks, non_visual_toks


def main() -> None:
    args = parse_args()
    methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    for m in methods:
        field_name(m)

    qid_filter = read_qid_filter(args.qid, args.qids_file)
    stopwords = DEFAULT_STOPWORDS if args.drop_stopwords else None

    artifact_tokens: Optional[Set[str]] = None
    if args.drop_artifact_tokens:
        artifact_tokens = set(DEFAULT_ARTIFACT_TOKENS)
        artifact_tokens |= parse_csv_set(args.artifact_tokens)

    visual_lexicon = set(DEFAULT_VISUAL_LEXICON)
    visual_lexicon |= parse_csv_set(args.visual_lexicon)
    visual_lexicon |= read_list_file(args.visual_lexicon_file)

    in_path = Path(args.attribution_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with in_path.open("r") as in_f, out_path.open("w") as out_f:
        for line in in_f:
            row = json.loads(line)
            qid = str(row.get("query_id", ""))
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
                require_overlap=args.require_method_overlap,
            )
            token_rows, vis_idx, non_vis_idx, vis_toks, non_vis_toks = classify_query_tokens(
                query=query,
                influential=influential,
                visual_lexicon=visual_lexicon,
            )

            out_row = {
                "query_id": qid,
                "query_text": query,
                "gold_qtype": row.get("gold_qtype", ""),
                "pred_label_name": row.get("pred_label_name", ""),
                "source_attribution_file": str(in_path),
                "methods_used": methods,
                "require_method_overlap": bool(args.require_method_overlap),
                "influential_overlap_tokens": sorted(influential),
                "influential_tokens_by_method": {k: sorted(v) for k, v in tokens_by_method.items()},
                "query_token_classes": token_rows,
                "visual_token_indices": vis_idx,
                "non_visual_token_indices": non_vis_idx,
                "visual_tokens": vis_toks,
                "non_visual_tokens": non_vis_toks,
            }
            out_f.write(json.dumps(out_row) + "\n")
            written += 1

    print(f"Wrote: {out_path}")
    print(f"rows: {written}")


if __name__ == "__main__":
    main()

