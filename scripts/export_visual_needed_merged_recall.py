#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


POS_LABEL_NAME = "visual_needed"

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

DEFAULT_VISUAL_LEXICON = {
    "animal",
    "animals",
    "banner",
    "beard",
    "billboard",
    "bald",
    "blanket",
    "color",
    "colour",
    "cover",
    "covers",
    "emblem",
    "face",
    "faces",
    "film",
    "flag",
    "flower",
    "flowers",
    "glasses",
    "hair",
    "holding",
    "holds",
    "horse",
    "horses",
    "icon",
    "icons",
    "image",
    "images",
    "jersey",
    "logo",
    "logos",
    "movie",
    "movies",
    "mule",
    "mules",
    "mustache",
    "photo",
    "photos",
    "photograph",
    "photographs",
    "picture",
    "pictures",
    "poster",
    "posters",
    "racehorse",
    "racehorses",
    "rose",
    "roses",
    "shape",
    "shapes",
    "sign",
    "signs",
    "sideburns",
    "sport",
    "sports",
    "symbol",
    "symbols",
    "title",
    "titles",
    "visual",
    "wear",
    "wearing",
    "wears",
}

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
            "Create a merged visual cue file with recall-oriented backfill so "
            "target visual queries do not end up with empty token/phrase lists."
        )
    )
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument(
        "--target",
        type=str,
        default="gold",
        choices=["pred", "gold", "pred_or_gold"],
        help="Which label source decides whether a query must receive visual cues.",
    )
    parser.add_argument("--max-fallback-tokens", type=int, default=2)
    parser.add_argument("--max-fallback-phrases", type=int, default=2)
    parser.add_argument("--visual-lexicon", type=str, default="")
    parser.add_argument("--visual-lexicon-file", type=str, default="")
    parser.add_argument(
        "--keep-non-target-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, keep non-target rows in the output as well.",
    )
    return parser.parse_args()


def normalize_qtype_name(name: str) -> str:
    return "".join(str(name).lower().split())


def normalize_token(raw: str) -> str:
    token = str(raw)
    if token.startswith("##"):
        token = token[2:]
    while token.startswith("▁") or token.startswith("Ġ"):
        token = token[1:]
    token = token.strip().lower()
    token = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", token)
    return token


def query_word_tokens(query: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", str(query), flags=re.UNICODE)


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


def uniq_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def token_variants(token: str) -> Set[str]:
    token = normalize_token(token)
    if not token:
        return set()
    out = {token}
    if token.endswith("ies") and len(token) > 4:
        out.add(token[:-3] + "y")
    if token.endswith("es") and len(token) > 4:
        out.add(token[:-1])
        out.add(token[:-2])
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        out.add(token[:-1])
    return {x for x in out if x}


def visual_hits(token: str, visual_lexicon: Set[str]) -> List[str]:
    return sorted({variant for variant in token_variants(token) if variant in visual_lexicon})


def is_target_visual(row: Dict[str, Any], target: str) -> bool:
    pred_visual = str(row.get("pred_label_name", "")) == POS_LABEL_NAME
    gold_visual = normalize_qtype_name(row.get("gold_qtype", "")) in VISUAL_GOLD_QTYPES
    if target == "pred":
        return pred_visual
    if target == "gold":
        return gold_visual
    return pred_visual or gold_visual


def collect_visual_phrase_texts(
    phrase_rows: Sequence[Dict[str, Any]],
    visual_lexicon: Set[str],
) -> Tuple[List[str], Set[str]]:
    phrases: List[str] = []
    phrase_norm_tokens: Set[str] = set()
    for row in phrase_rows:
        hit_tokens = []
        for tok in row.get("norm_tokens", []) or []:
            if visual_hits(str(tok), visual_lexicon):
                hit_tokens.append(str(tok))
        if not hit_tokens:
            continue
        raw_examples = row.get("raw_examples") or []
        if raw_examples:
            phrases.extend(str(x).strip() for x in raw_examples if str(x).strip())
        else:
            norm_phrase = str(row.get("norm_phrase", "")).strip()
            if norm_phrase:
                phrases.append(norm_phrase)
        phrase_norm_tokens.update(normalize_token(tok) for tok in row.get("norm_tokens", []) or [])
    return uniq_keep_order(phrases), {x for x in phrase_norm_tokens if x}


def backfill_visual_phrases(
    phrase_rows: Sequence[Dict[str, Any]],
    max_fallback_phrases: int,
) -> Tuple[List[str], Set[str], bool]:
    phrases: List[str] = []
    phrase_norm_tokens: Set[str] = set()
    for row in phrase_rows:
        raw_examples = row.get("raw_examples") or []
        if raw_examples:
            candidate = str(raw_examples[0]).strip()
        else:
            candidate = str(row.get("norm_phrase", "")).strip()
        if not candidate:
            continue
        phrases.append(candidate)
        phrase_norm_tokens.update(normalize_token(tok) for tok in row.get("norm_tokens", []) or [])
        if len(phrases) >= max(1, max_fallback_phrases):
            break
    phrases = uniq_keep_order(phrases)
    return phrases, {x for x in phrase_norm_tokens if x}, bool(phrases)


def collect_visual_tokens(
    token_rows: Sequence[Dict[str, Any]],
    visual_lexicon: Set[str],
) -> List[str]:
    out: List[str] = []
    for row in token_rows:
        raw = str(row.get("token", "")).strip()
        norm = normalize_token(row.get("norm", raw))
        if not raw or not norm:
            continue
        if not bool(row.get("important")):
            continue
        if visual_hits(norm, visual_lexicon):
            out.append(raw)
    return uniq_keep_order(out)


def backfill_visual_tokens(
    token_rows: Sequence[Dict[str, Any]],
    query_text: str,
    preferred_norms: Set[str],
    max_fallback_tokens: int,
) -> Tuple[List[str], bool]:
    tokens: List[str] = []
    preferred_norms = {normalize_token(x) for x in preferred_norms if normalize_token(x)}

    if preferred_norms:
        for row in token_rows:
            raw = str(row.get("token", "")).strip()
            norm = normalize_token(row.get("norm", raw))
            if not raw or not norm or not bool(row.get("important")):
                continue
            if norm in preferred_norms:
                tokens.append(raw)
                if len(tokens) >= max(1, max_fallback_tokens):
                    return uniq_keep_order(tokens), True

    for row in token_rows:
        raw = str(row.get("token", "")).strip()
        norm = normalize_token(row.get("norm", raw))
        if not raw or not norm or not bool(row.get("important")):
            continue
        tokens.append(raw)
        if len(tokens) >= max(1, max_fallback_tokens):
            return uniq_keep_order(tokens), True

    for raw in query_word_tokens(query_text):
        norm = normalize_token(raw)
        if not norm or norm in DEFAULT_STOPWORDS:
            continue
        tokens.append(str(raw).strip())
        if len(tokens) >= max(1, max_fallback_tokens):
            return uniq_keep_order(tokens), True

    return uniq_keep_order(tokens), bool(tokens)


def main() -> None:
    args = parse_args()

    visual_lexicon = set(DEFAULT_VISUAL_LEXICON)
    visual_lexicon |= parse_csv_set(args.visual_lexicon)
    visual_lexicon |= read_list_file(args.visual_lexicon_file)

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    target_rows = 0
    written_rows = 0
    rows_with_any_cues = 0
    rows_missing_any_cues = 0
    token_backfill_rows = 0
    phrase_backfill_rows = 0

    with in_path.open("r") as in_f, out_path.open("w") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)

            target_visual = is_target_visual(row, args.target)
            if target_visual:
                target_rows += 1
            elif not args.keep_non_target_rows:
                continue

            query_text = str(row.get("query_text", ""))
            token_rows = row.get("token_labels", []) or []
            phrase_rows = row.get("phrase_labels", []) or []

            visual_needed_phrases, phrase_norm_tokens = collect_visual_phrase_texts(
                phrase_rows=phrase_rows,
                visual_lexicon=visual_lexicon,
            )
            phrase_backfill_applied = False
            if target_visual and not visual_needed_phrases:
                visual_needed_phrases, phrase_norm_tokens, phrase_backfill_applied = backfill_visual_phrases(
                    phrase_rows=phrase_rows,
                    max_fallback_phrases=args.max_fallback_phrases,
                )
                if phrase_backfill_applied:
                    phrase_backfill_rows += 1

            visual_needed_tokens = collect_visual_tokens(
                token_rows=token_rows,
                visual_lexicon=visual_lexicon,
            )
            token_backfill_applied = False
            if target_visual and not visual_needed_tokens:
                visual_needed_tokens, token_backfill_applied = backfill_visual_tokens(
                    token_rows=token_rows,
                    query_text=query_text,
                    preferred_norms=phrase_norm_tokens,
                    max_fallback_tokens=args.max_fallback_tokens,
                )
                if token_backfill_applied:
                    token_backfill_rows += 1

            merged = {
                "query_id": row.get("query_id"),
                "query_text": query_text,
                "gold_qtype": row.get("gold_qtype", ""),
                "pred_label_name": row.get("pred_label_name", ""),
                "target_visual_query": bool(target_visual),
                "visual_needed_tokens": uniq_keep_order(visual_needed_tokens),
                "visual_needed_phrases": uniq_keep_order(visual_needed_phrases),
                "token_backfill_applied": bool(token_backfill_applied),
                "phrase_backfill_applied": bool(phrase_backfill_applied),
                "source_export_file": str(in_path),
            }

            has_any_cues = bool(merged["visual_needed_tokens"] or merged["visual_needed_phrases"])
            if has_any_cues:
                rows_with_any_cues += 1
            else:
                rows_missing_any_cues += 1

            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            written_rows += 1

    print(f"Wrote: {out_path}")
    print(f"total_rows: {total_rows}")
    print(f"target_rows: {target_rows}")
    print(f"written_rows: {written_rows}")
    print(f"rows_with_any_cues: {rows_with_any_cues}")
    print(f"rows_missing_any_cues: {rows_missing_any_cues}")
    print(f"token_backfill_rows: {token_backfill_rows}")
    print(f"phrase_backfill_rows: {phrase_backfill_rows}")


if __name__ == "__main__":
    main()
