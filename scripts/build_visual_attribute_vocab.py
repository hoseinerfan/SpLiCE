#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set


DEFAULT_STRONG_VISUAL_TOKENS: Set[str] = {
    "bald",
    "balding",
    "hair",
    "haired",
    "beard",
    "bearded",
    "mustache",
    "moustache",
    "glasses",
    "eyeglasses",
    "sunglasses",
    "spectacles",
    "goggles",
    "wear",
    "wearing",
    "wears",
    "worn",
    "holding",
    "holds",
    "hold",
    "face",
    "facial",
    "eyes",
    "eye",
    "nose",
    "mouth",
    "ears",
    "ear",
    "neck",
    "hand",
    "hands",
    "logo",
    "poster",
    "cover",
    "image",
    "photo",
    "picture",
    "flag",
    "symbol",
    "icon",
    "portrait",
    "silhouette",
    "smile",
    "smiling",
    "standing",
    "sitting",
    "pose",
    "posed",
    "tattoo",
    "hat",
    "cap",
    "helmet",
    "shirt",
    "jacket",
    "dress",
    "skirt",
    "uniform",
    "color",
    "coloured",
    "colored",
}


DEFAULT_COLOR_TOKENS: Set[str] = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "pink",
    "purple",
    "brown",
    "gray",
    "grey",
    "gold",
    "silver",
}


DEFAULT_COLOR_CONTEXT_TOKENS: Set[str] = {
    "hair",
    "beard",
    "mustache",
    "face",
    "eyes",
    "shirt",
    "dress",
    "jacket",
    "hat",
    "cap",
    "uniform",
    "logo",
    "poster",
    "cover",
    "flag",
    "background",
    "car",
    "animal",
    "lion",
    "castle",
    "symbol",
}


DEFAULT_WEAK_BLOCKLIST: Set[str] = {
    "born",
    "died",
    "married",
    "album",
    "song",
    "film",
    "movie",
    "season",
    "episode",
    "league",
    "club",
    "team",
    "company",
    "university",
    "county",
    "city",
    "country",
    "career",
    "founded",
    "founder",
    "director",
    "producer",
    "actor",
    "actress",
    "writer",
    "novel",
    "draft",
    "billionaire",
    "billionaires",
    "lgbt",
}


TOKEN_RE = re.compile(r"[a-z0-9']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a visual-only concept vocabulary from an existing concept list."
    )
    parser.add_argument("--input-concepts-txt", type=str, required=True)
    parser.add_argument("--output-concepts-txt", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument(
        "--max-words",
        type=int,
        default=4,
        help="Maximum words allowed in a kept concept.",
    )
    parser.add_argument(
        "--keep-colors-with-context",
        action="store_true",
        help="Keep color concepts only when paired with visual context tokens.",
    )
    parser.add_argument(
        "--disable-weak-blocklist",
        action="store_true",
        help="Disable weak blocklist filtering.",
    )
    return parser.parse_args()


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def is_visual_concept(
    concept: str,
    max_words: int,
    keep_colors_with_context: bool,
    use_weak_blocklist: bool,
) -> bool:
    tokens = tokenize(concept)
    if not tokens:
        return False
    if len(tokens) > max_words:
        return False

    token_set = set(tokens)
    has_strong = len(token_set.intersection(DEFAULT_STRONG_VISUAL_TOKENS)) > 0
    has_color = len(token_set.intersection(DEFAULT_COLOR_TOKENS)) > 0
    has_color_context = len(token_set.intersection(DEFAULT_COLOR_CONTEXT_TOKENS)) > 0
    has_weak_block = len(token_set.intersection(DEFAULT_WEAK_BLOCKLIST)) > 0

    if has_strong:
        # Allow strong visual concepts even if they include weak block words
        # such as "movie poster".
        return True

    if keep_colors_with_context and has_color and has_color_context:
        return True

    if use_weak_blocklist and has_weak_block:
        return False

    return False


def main() -> None:
    args = parse_args()

    in_path = Path(args.input_concepts_txt)
    out_path = Path(args.output_concepts_txt)
    summary_path = Path(args.summary_json) if args.summary_json else None

    if not in_path.is_file():
        raise FileNotFoundError(f"Input not found: {in_path}")

    concepts: List[str] = []
    seen: Set[str] = set()
    with open(in_path, "r") as handle:
        for line in handle:
            concept = line.strip()
            if not concept:
                continue
            if concept in seen:
                continue
            seen.add(concept)
            concepts.append(concept)

    kept: List[str] = []
    dropped: List[str] = []
    use_weak_blocklist = not args.disable_weak_blocklist

    for concept in concepts:
        if is_visual_concept(
            concept=concept,
            max_words=args.max_words,
            keep_colors_with_context=args.keep_colors_with_context,
            use_weak_blocklist=use_weak_blocklist,
        ):
            kept.append(concept)
        else:
            dropped.append(concept)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        for concept in kept:
            handle.write(concept + "\n")

    summary: Dict[str, object] = {
        "input_path": str(in_path),
        "output_path": str(out_path),
        "input_count": len(concepts),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "keep_ratio": round(len(kept) / max(1, len(concepts)), 6),
        "max_words": args.max_words,
        "keep_colors_with_context": bool(args.keep_colors_with_context),
        "weak_blocklist_enabled": use_weak_blocklist,
        "examples_kept": kept[:25],
        "examples_dropped": dropped[:25],
    }

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")

    print(f"Input concepts: {len(concepts)}")
    print(f"Kept visual concepts: {len(kept)}")
    print(f"Dropped concepts: {len(dropped)}")
    print(f"Wrote visual vocab: {out_path}")
    if summary_path is not None:
        print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
