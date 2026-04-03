#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract OCR word boxes from page images using Tesseract."
    )
    parser.add_argument("--doc-id", type=str, required=True, help="Document id.")
    parser.add_argument("--num-pages", type=int, required=True, help="Number of pages to scan.")
    parser.add_argument(
        "--page-image-template",
        type=str,
        required=True,
        help=(
            "Image path template with {doc_id} and {page_index}. "
            "Example: /path/{doc_id}_{page_index}.png"
        ),
    )
    parser.add_argument("--output-jsonl", type=str, required=True, help="Output OCR JSONL.")
    parser.add_argument("--lang", type=str, default="eng", help="Tesseract language.")
    parser.add_argument("--min-conf", type=float, default=35.0, help="Min OCR confidence to keep a word.")
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default="",
        help="Optional explicit tesseract binary path.",
    )
    parser.add_argument(
        "--tesseract-config",
        type=str,
        default="--oem 3 --psm 6",
        help="Raw config string passed to pytesseract.image_to_data.",
    )
    parser.add_argument(
        "--keep-empty-pages",
        action="store_true",
        help="Write records for pages with zero OCR words.",
    )
    return parser.parse_args()


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def parse_conf(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return -1.0


def ocr_image_words(
    image_path: Path,
    lang: str,
    min_conf: float,
    config: str,
) -> Dict:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    words: List[Dict] = []
    n = len(data.get("text", []))
    for i in range(n):
        raw = str(data["text"][i]).strip()
        if not raw:
            continue
        conf = parse_conf(str(data.get("conf", ["-1"] * n)[i]))
        if conf < min_conf:
            continue

        try:
            left = int(data["left"][i])
            top = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue

        x0 = max(0, min(width, left))
        y0 = max(0, min(height, top))
        x1 = max(0, min(width, left + w))
        y1 = max(0, min(height, top + h))
        if x1 <= x0 or y1 <= y0:
            continue

        words.append(
            {
                "text": raw,
                "norm": normalize_token(raw),
                "conf": round(conf, 3),
                "bbox_xyxy": [x0, y0, x1, y1],
            }
        )

    return {
        "width": width,
        "height": height,
        "words": words,
    }


def main() -> None:
    args = parse_args()

    if Image is None:
        raise RuntimeError("Pillow is required. Install with: pip install pillow")
    if pytesseract is None:
        raise RuntimeError("pytesseract is required. Install with: pip install pytesseract")
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pages_written = 0
    words_total = 0
    pages_missing = 0

    with open(out_path, "w") as out:
        for page_index in range(args.num_pages):
            image_path = Path(
                args.page_image_template.format(doc_id=args.doc_id, page_index=page_index)
            )
            if not image_path.is_file():
                pages_missing += 1
                print(f"Missing image: {image_path}")
                continue

            record = ocr_image_words(
                image_path=image_path,
                lang=args.lang,
                min_conf=args.min_conf,
                config=args.tesseract_config,
            )
            n_words = len(record["words"])
            if n_words == 0 and not args.keep_empty_pages:
                print(f"page {page_index}: words=0 (skipped)")
                continue

            row = {
                "doc_id": args.doc_id,
                "page_id": f"{args.doc_id}:{page_index}",
                "page_index": page_index,
                "image_path": str(image_path),
                "width": record["width"],
                "height": record["height"],
                "words": record["words"],
            }
            out.write(json.dumps(row) + "\n")
            pages_written += 1
            words_total += n_words
            print(f"page {page_index}: words={n_words}")

    summary = {
        "doc_id": args.doc_id,
        "num_pages_requested": args.num_pages,
        "pages_written": pages_written,
        "pages_missing_images": pages_missing,
        "words_total": words_total,
        "output_jsonl": str(out_path),
        "min_conf": args.min_conf,
        "lang": args.lang,
    }
    print("\n=== OCR Extraction Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
