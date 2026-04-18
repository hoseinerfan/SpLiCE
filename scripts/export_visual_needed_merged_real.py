#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


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

DEFAULT_VISUAL_LEXICON = {
    "ad",
    "advertisement",
    "advertisements",
    "amusement",
    "animal",
    "animals",
    "arch",
    "archway",
    "artwork",
    "banner",
    "beard",
    "billboard",
    "billboards",
    "bald",
    "ball",
    "balls",
    "blanket",
    "blankets",
    "body",
    "building",
    "buildings",
    "clothing",
    "coat",
    "collage",
    "color",
    "colors",
    "colour",
    "cover",
    "covers",
    "covering",
    "column",
    "columns",
    "dress",
    "dressed",
    "emblem",
    "entrance",
    "face",
    "faces",
    "facial",
    "flag",
    "flags",
    "floating",
    "flower",
    "flowers",
    "forehead",
    "front",
    "glasses",
    "hair",
    "hand",
    "hands",
    "holding",
    "holds",
    "horse",
    "horses",
    "icon",
    "icons",
    "image",
    "images",
    "indoor",
    "jacket",
    "jersey",
    "left",
    "live",
    "location",
    "locations",
    "logo",
    "logos",
    "man",
    "metal",
    "mountain",
    "mountains",
    "movie",
    "movies",
    "mule",
    "mules",
    "mustache",
    "object",
    "objects",
    "outdoor",
    "park",
    "parks",
    "person",
    "people",
    "photo",
    "photos",
    "photograph",
    "photographs",
    "picture",
    "pictures",
    "player",
    "players",
    "poster",
    "posters",
    "race",
    "racehorse",
    "racehorses",
    "reddish",
    "rectangle",
    "rectangular",
    "right",
    "river",
    "rivers",
    "rose",
    "roses",
    "scene",
    "scenes",
    "shape",
    "shaped",
    "shapes",
    "shirt",
    "shore",
    "shiny",
    "show",
    "shown",
    "shows",
    "sign",
    "signs",
    "sitting",
    "sleeve",
    "sleeves",
    "sport",
    "sports",
    "stage",
    "standing",
    "statue",
    "statues",
    "stiped",
    "striped",
    "structure",
    "structures",
    "symbol",
    "symbols",
    "theatre",
    "title",
    "titles",
    "top",
    "tree",
    "trees",
    "visible",
    "visual",
    "water",
    "wear",
    "wearing",
    "wears",
    "white",
    "zipped",
    "unzipped",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a merged visual cue file using only attribution-derived tokens and "
            "phrases, without fallback filling from arbitrary query text."
        )
    )
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument(
        "--target",
        type=str,
        default="gold",
        choices=["gold", "pred", "pred_or_gold", "all"],
        help="Which rows are considered target visual rows.",
    )
    parser.add_argument(
        "--keep-empty-target-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, keep target rows even when no real visual cues are extracted.",
    )
    parser.add_argument("--visual-lexicon", type=str, default="")
    parser.add_argument("--visual-lexicon-file", type=str, default="")
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


def visual_hits(token: str, visual_lexicon: Set[str]) -> bool:
    for variant in token_variants(token):
        if variant in visual_lexicon:
            return True
    return False


def is_target_visual(row: Dict[str, Any], target: str) -> bool:
    pred_visual = str(row.get("pred_label_name", "")) == POS_LABEL_NAME
    gold_visual = normalize_qtype_name(row.get("gold_qtype", "")) in VISUAL_GOLD_QTYPES
    if target == "gold":
        return gold_visual
    if target == "pred":
        return pred_visual
    if target == "pred_or_gold":
        return pred_visual or gold_visual
    return True


def collect_visual_phrases_and_tokens(
    row: Dict[str, Any],
    visual_lexicon: Set[str],
) -> Dict[str, List[str]]:
    visual_phrases: List[str] = []
    phrase_token_norms: List[str] = []

    for phrase_row in row.get("phrase_labels", []) or []:
        norm_tokens = [normalize_token(tok) for tok in (phrase_row.get("norm_tokens") or [])]
        norm_tokens = [tok for tok in norm_tokens if tok and tok not in DEFAULT_STOPWORDS]
        hit_tokens = [tok for tok in norm_tokens if visual_hits(tok, visual_lexicon)]
        if not hit_tokens:
            continue

        raw_examples = phrase_row.get("raw_examples") or []
        if raw_examples:
            visual_phrases.extend(str(x).strip() for x in raw_examples if str(x).strip())
        else:
            norm_phrase = str(phrase_row.get("norm_phrase", "")).strip()
            if norm_phrase:
                visual_phrases.append(norm_phrase)

        # Once a phrase is genuinely visual, propagate all of its normalized tokens.
        phrase_token_norms.extend(norm_tokens)

    phrase_token_norms = uniq_keep_order(phrase_token_norms)
    phrase_token_norm_set = set(phrase_token_norms)

    visual_tokens: List[str] = []
    for token_row in row.get("token_labels", []) or []:
        raw = str(token_row.get("token", "")).strip()
        norm = normalize_token(token_row.get("norm", raw))
        important = bool(token_row.get("important"))
        if not raw or not norm or not important:
            continue
        if visual_hits(norm, visual_lexicon) or norm in phrase_token_norm_set:
            visual_tokens.append(raw)

    # Add propagated phrase tokens as tokens too, keeping them real and attribution-derived.
    visual_tokens.extend(phrase_token_norms)

    surface_by_norm: Dict[str, str] = {}
    for token_row in row.get("token_labels", []) or []:
        raw = str(token_row.get("token", "")).strip()
        norm = normalize_token(token_row.get("norm", raw))
        if raw and norm and norm not in surface_by_norm:
            surface_by_norm[norm] = raw
    surfaced = [surface_by_norm.get(normalize_token(tok), str(tok).strip()) for tok in visual_tokens]

    return {
        "visual_needed_tokens": uniq_keep_order(surfaced),
        "visual_needed_phrases": uniq_keep_order(visual_phrases),
    }


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

    with in_path.open("r") as in_f, out_path.open("w") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)

            target_visual = is_target_visual(row, args.target)
            if target_visual:
                target_rows += 1
            elif args.target != "all":
                continue

            cues = collect_visual_phrases_and_tokens(row=row, visual_lexicon=visual_lexicon)
            has_any = bool(cues["visual_needed_tokens"] or cues["visual_needed_phrases"])
            if has_any:
                rows_with_any_cues += 1
            else:
                rows_missing_any_cues += 1
                if target_visual and not args.keep_empty-target-rows:
                    continue

            out_row = {
                "query_id": row.get("query_id"),
                "query_text": row.get("query_text"),
                "gold_qtype": row.get("gold_qtype", ""),
                "pred_label_name": row.get("pred_label_name", ""),
                "target_visual_query": bool(target_visual),
                **cues,
                "source_export_file": str(in_path),
            }
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written_rows += 1

    print(f"Wrote: {out_path}")
    print(f"total_rows: {total_rows}")
    print(f"target_rows: {target_rows}")
    print(f"written_rows: {written_rows}")
    print(f"rows_with_any_cues: {rows_with_any_cues}")
    print(f"rows_missing_any_cues: {rows_missing_any_cues}")


if __name__ == "__main__":
    main()
