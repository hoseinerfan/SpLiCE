#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Finalize labels for one page with strict layout policy: "
            "table_text/table_structure->table_all, "
            "image_region from PDF image boxes, "
            "and remove ocr_text only on image patches."
        )
    )
    p.add_argument("--labels-jsonl", type=str, required=True, help="Input labels JSONL.")
    p.add_argument("--output-jsonl", type=str, required=True, help="Output labels JSONL.")
    p.add_argument("--pdf-path", type=str, required=True, help="PDF path for image-box extraction.")
    p.add_argument("--page-id", type=str, required=True, help="Target page id in doc_id:page_index format.")
    p.add_argument("--grid-size", type=int, default=32, help="Grid size per side.")
    p.add_argument("--image-token-start", type=int, default=0, help="Image token start.")
    p.add_argument("--image-token-count", type=int, default=1024, help="Image token count.")
    p.add_argument(
        "--image-hit-mode",
        type=str,
        default="center",
        choices=["center", "overlap_or_center"],
        help="Image patch hit rule from PDF image boxes.",
    )
    p.add_argument(
        "--image-overlap-threshold",
        type=float,
        default=0.12,
        help="Used only with overlap_or_center mode.",
    )
    p.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional summary JSON output.",
    )
    p.add_argument(
        "--overlay-image",
        type=str,
        default="",
        help="Optional page image to render a flat overlay (no heatmaps).",
    )
    p.add_argument(
        "--overlay-output",
        type=str,
        default="",
        help="Output flat overlay PNG path.",
    )
    p.add_argument(
        "--masks-dir",
        type=str,
        default="",
        help="Optional directory to write binary masks and patch lists.",
    )
    return p.parse_args()


def parse_page_id(page_id: str) -> Tuple[str, int]:
    if ":" not in page_id:
        raise ValueError(f"Invalid page-id '{page_id}', expected doc_id:page_index")
    doc_id, page_str = page_id.rsplit(":", 1)
    return doc_id, int(page_str)


def overlap_fraction_of_patch(a: List[float], b: List[float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    patch_area = (a[2] - a[0]) * (a[3] - a[1])
    return inter / patch_area if patch_area > 0 else 0.0


def center_in_box(patch_box: List[float], box: List[float]) -> bool:
    cx = 0.5 * (patch_box[0] + patch_box[2])
    cy = 0.5 * (patch_box[1] + patch_box[3])
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def patch_box_from_index(
    patch_index: int,
    grid_size: int,
    image_token_start: int,
) -> Tuple[int, int, List[float]]:
    rel = patch_index - image_token_start
    row = rel // grid_size
    col = rel % grid_size
    x0 = col / grid_size
    y0 = row / grid_size
    x1 = (col + 1) / grid_size
    y1 = (row + 1) / grid_size
    return row, col, [x0, y0, x1, y1]


def box_area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: List[float], b: List[float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = box_area(a) + box_area(b) - inter
    return inter / ua if ua > 0 else 0.0


def dedupe_boxes(boxes: List[List[float]], iou_thr: float = 0.95) -> List[List[float]]:
    out: List[List[float]] = []
    for b in boxes:
        merged = False
        for i, c in enumerate(out):
            if iou(b, c) >= iou_thr:
                out[i] = [
                    min(c[0], b[0]),
                    min(c[1], b[1]),
                    max(c[2], b[2]),
                    max(c[3], b[3]),
                ]
                merged = True
                break
        if not merged:
            out.append(list(b))
    return out


def normalize_box_xyxy(x0: float, y0: float, x1: float, y1: float, w: float, h: float) -> List[float]:
    nx0 = max(0.0, min(1.0, x0 / w))
    ny0 = max(0.0, min(1.0, y0 / h))
    nx1 = max(0.0, min(1.0, x1 / w))
    ny1 = max(0.0, min(1.0, y1 / h))
    return [nx0, ny0, nx1, ny1]


def load_pdf_image_boxes(pdf_path: Path, page_index: int) -> List[List[float]]:
    if fitz is None:
        raise RuntimeError("pymupdf is required. Install with: pip install pymupdf")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    if page_index < 0 or page_index >= len(doc):
        raise ValueError(f"Page index {page_index} out of range for {pdf_path} (pages={len(doc)})")
    page = doc[page_index]
    w = float(page.rect.width)
    h = float(page.rect.height)
    if w <= 0 or h <= 0:
        return []

    out: List[List[float]] = []

    # Path 1: image blocks from text dict.
    for block in page.get_text("dict").get("blocks", []):
        if int(block.get("type", -1)) != 1:  # image block
            continue
        bbox = block.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox]
        if x1 <= x0 or y1 <= y0:
            continue
        nx0, ny0, nx1, ny1 = normalize_box_xyxy(x0, y0, x1, y1, w, h)
        if nx1 > nx0 and ny1 > ny0:
            out.append([nx0, ny0, nx1, ny1])

    # Path 2: image xrefs/rects catches pages where text dict has no image blocks.
    try:
        for img in page.get_images(full=True):
            if not img:
                continue
            xref = int(img[0])
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            for r in rects:
                x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                nx0, ny0, nx1, ny1 = normalize_box_xyxy(x0, y0, x1, y1, w, h)
                if nx1 > nx0 and ny1 > ny0:
                    out.append([nx0, ny0, nx1, ny1])
    except Exception:
        pass

    return dedupe_boxes(out, iou_thr=0.95)


def write_masks(
    masks_dir: Path,
    grid_size: int,
    image_token_start: int,
    cell_sets: Dict[str, Set[Tuple[int, int]]],
) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)
    for concept, cells in cell_sets.items():
        mask_path = masks_dir / f"{concept}_mask.txt"
        idx_path = masks_dir / f"{concept}_patches.txt"
        with open(mask_path, "w") as f:
            for r in range(grid_size):
                f.write("".join("1" if (r, c) in cells else "0" for c in range(grid_size)) + "\n")
        indices = sorted(image_token_start + r * grid_size + c for (r, c) in cells)
        with open(idx_path, "w") as f:
            for x in indices:
                f.write(f"{x}\n")


def render_flat_overlay(
    page_image: Path,
    overlay_output: Path,
    grid_size: int,
    cell_sets: Dict[str, Set[Tuple[int, int]]],
) -> None:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required. Install with: pip install pillow")
    if not page_image.is_file():
        raise FileNotFoundError(f"Overlay image not found: {page_image}")

    img = Image.open(page_image).convert("RGBA")
    w, h = img.size
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov, "RGBA")

    palette = {
        "table_all": (0, 200, 0, 100),      # green
        "ocr_text": (40, 120, 255, 85),     # blue
        "image_region": (255, 80, 80, 110), # red
    }
    for concept in ["table_all", "ocr_text", "image_region"]:
        color = palette[concept]
        for rr, cc in cell_sets.get(concept, set()):
            x0 = int(cc * w / grid_size)
            x1 = int((cc + 1) * w / grid_size)
            y0 = int(rr * h / grid_size)
            y1 = int((rr + 1) * h / grid_size)
            dr.rectangle([x0, y0, x1, y1], fill=color)

    overlay_output.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(img, ov).save(overlay_output)


def main() -> None:
    args = parse_args()

    labels_path = Path(args.labels_jsonl)
    out_path = Path(args.output_jsonl)
    pdf_path = Path(args.pdf_path)

    if not labels_path.is_file():
        raise FileNotFoundError(f"labels-jsonl not found: {labels_path}")
    if args.image_token_count != args.grid_size * args.grid_size:
        raise ValueError("--image-token-count must equal grid-size^2 for this script.")

    _, page_index = parse_page_id(args.page_id)
    image_boxes = load_pdf_image_boxes(pdf_path=pdf_path, page_index=page_index)
    print(f"pdf image boxes: {len(image_boxes)} {image_boxes}")

    rows: List[Dict] = []
    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # First pass: compute image patch set for the target page.
    image_cells: Set[Tuple[int, int]] = set()
    for row in rows:
        if str(row.get("page_id", "")) != args.page_id:
            continue
        patch_index = int(row.get("patch_index", -1))
        if patch_index < args.image_token_start or patch_index >= args.image_token_start + args.image_token_count:
            continue
        rr, cc, pb = patch_box_from_index(
            patch_index=patch_index,
            grid_size=args.grid_size,
            image_token_start=args.image_token_start,
        )
        hit = False
        for b in image_boxes:
            if args.image_hit_mode == "center":
                if center_in_box(pb, b):
                    hit = True
                    break
            else:
                if center_in_box(pb, b) or overlap_fraction_of_patch(pb, b) >= args.image_overlap_threshold:
                    hit = True
                    break
        if hit:
            image_cells.add((rr, cc))

    # Second pass: rewrite concepts for target page.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    concept_cells: Dict[str, Set[Tuple[int, int]]] = {
        "table_all": set(),
        "ocr_text": set(),
        "image_region": set(),
    }

    with open(out_path, "w") as g:
        for row in rows:
            if str(row.get("page_id", "")) == args.page_id:
                patch_index = int(row.get("patch_index", -1))
                table_w = 0.0
                ocr_w = 0.0

                for item in row.get("top_concepts", []):
                    c = str(item.get("concept", "")).strip().lower()
                    w = float(item.get("weight", 0.0))
                    if c in {"table_text", "table_structure", "table_all"}:
                        table_w = max(table_w, w)
                    elif c == "ocr_text":
                        ocr_w = max(ocr_w, w)

                in_image = False
                rr = cc = -1
                if (
                    patch_index >= args.image_token_start
                    and patch_index < args.image_token_start + args.image_token_count
                ):
                    rr, cc, _ = patch_box_from_index(
                        patch_index=patch_index,
                        grid_size=args.grid_size,
                        image_token_start=args.image_token_start,
                    )
                    in_image = (rr, cc) in image_cells

                # Strict precedence: image only removes OCR on the same patch.
                if in_image:
                    ocr_w = 0.0

                new_tc: List[Dict] = []
                if table_w > 0:
                    new_tc.append({"concept": "table_all", "weight": round(table_w, 6)})
                    counts["table_all"] += 1
                    if rr >= 0:
                        concept_cells["table_all"].add((rr, cc))
                if ocr_w > 0:
                    new_tc.append({"concept": "ocr_text", "weight": round(ocr_w, 6)})
                    counts["ocr_text"] += 1
                    if rr >= 0:
                        concept_cells["ocr_text"].add((rr, cc))
                if in_image:
                    new_tc.append({"concept": "image_region", "weight": 1.0})
                    counts["image_region"] += 1
                    concept_cells["image_region"].add((rr, cc))

                row["top_concepts"] = new_tc

            g.write(json.dumps(row) + "\n")

    print(f"Wrote: {out_path}")
    print(
        "page counts -> "
        f"image_region: {counts['image_region']} "
        f"ocr_text: {counts['ocr_text']} "
        f"table_all: {counts['table_all']}"
    )

    summary = {
        "labels_jsonl": str(labels_path),
        "output_jsonl": str(out_path),
        "pdf_path": str(pdf_path),
        "page_id": args.page_id,
        "grid_size": args.grid_size,
        "image_token_start": args.image_token_start,
        "image_token_count": args.image_token_count,
        "image_hit_mode": args.image_hit_mode,
        "image_overlap_threshold": args.image_overlap_threshold,
        "pdf_image_boxes": image_boxes,
        "counts": {
            "image_region": int(counts["image_region"]),
            "ocr_text": int(counts["ocr_text"]),
            "table_all": int(counts["table_all"]),
        },
    }

    if args.summary_json:
        sp = Path(args.summary_json)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"Wrote: {sp}")

    if args.masks_dir:
        write_masks(
            masks_dir=Path(args.masks_dir),
            grid_size=args.grid_size,
            image_token_start=args.image_token_start,
            cell_sets=concept_cells,
        )
        print(f"Wrote masks: {args.masks_dir}")

    if args.overlay_image and args.overlay_output:
        render_flat_overlay(
            page_image=Path(args.overlay_image),
            overlay_output=Path(args.overlay_output),
            grid_size=args.grid_size,
            cell_sets=concept_cells,
        )
        print(f"Wrote overlay: {args.overlay_output}")


if __name__ == "__main__":
    main()
