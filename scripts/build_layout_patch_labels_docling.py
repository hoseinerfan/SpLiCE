#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import fitz  # pymupdf
except ImportError:
    fitz = None


TEXT_LABEL_KEYWORDS = {
    "text",
    "title",
    "paragraph",
    "list",
    "caption",
    "header",
    "footer",
    "section",
    "footnote",
    "equation",
    "formula",
    "code",
}

TABLE_LABEL_KEYWORDS = {
    "table",
}

VISUAL_LABEL_KEYWORDS = {
    "picture",
    "figure",
    "image",
    "chart",
    "diagram",
    "graphic",
    "logo",
    "illustration",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build layout-aware patch labels from Docling zones. "
            "Output schema matches build_layout_patch_labels.py."
        )
    )
    p.add_argument("--labels-jsonl", type=str, required=True, help="Base patch labels JSONL (single doc).")
    p.add_argument("--pdf-path", type=str, required=True, help="Input PDF path for Docling.")
    p.add_argument("--output-jsonl", type=str, required=True, help="Output JSONL path.")
    p.add_argument("--grid-size", type=int, default=32, help="Patch grid size.")
    p.add_argument("--image-token-start", type=int, default=0, help="Image token start index.")
    p.add_argument("--image-token-count", type=int, default=1024, help="Image token count.")
    p.add_argument("--min-overlap-text", type=float, default=0.08, help="Min patch overlap for text zone hit.")
    p.add_argument("--min-overlap-table", type=float, default=0.08, help="Min patch overlap for table zone hit.")
    p.add_argument(
        "--min-overlap-visual",
        type=float,
        default=0.10,
        help="Min patch overlap for visual zone hit.",
    )
    p.add_argument(
        "--include-visual-region",
        action="store_true",
        help="Emit visual_region from Docling figure/picture-like zones.",
    )
    p.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional summary JSON path.",
    )
    p.add_argument(
        "--summary-tsv",
        type=str,
        default="",
        help="Optional per-page summary TSV path.",
    )
    p.add_argument(
        "--zones-debug-json",
        type=str,
        default="",
        help="Optional extracted Docling zones dump (debug).",
    )
    p.add_argument(
        "--docling-device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
        help="Docling accelerator device.",
    )
    p.add_argument(
        "--docling-num-threads",
        type=int,
        default=8,
        help="Thread count passed to Docling accelerator options.",
    )
    p.add_argument(
        "--docling-do-ocr",
        action="store_true",
        help="Enable OCR stage in Docling conversion.",
    )
    p.add_argument(
        "--docling-do-table-structure",
        action="store_true",
        help="Enable table-structure stage in Docling conversion.",
    )
    p.add_argument(
        "--docling-fallback-to-cpu",
        action="store_true",
        help="On accelerator failure, retry conversion on CPU.",
    )
    p.add_argument(
        "--docling-page-number-base",
        type=str,
        default="auto",
        choices=["auto", "zero", "one"],
        help=(
            "Interpret Docling page_no numbering. "
            "'zero' means page_no is 0-based, 'one' means 1-based, 'auto' infers from document."
        ),
    )
    return p.parse_args()


def infer_page_id(row: Dict[str, Any]) -> str:
    page_id = str(row.get("page_id", "")).strip()
    if page_id:
        return page_id
    rid = str(row.get("id", "")).strip()
    if "#patch" in rid:
        return rid.split("#patch", 1)[0]
    return rid


def parse_page_id(page_id: str) -> Tuple[str, int]:
    if ":" not in page_id:
        raise ValueError(f"Invalid page_id (expected doc:page): {page_id}")
    doc_id, page_str = page_id.rsplit(":", 1)
    return doc_id, int(page_str)


def patch_box_from_index(patch_index: int, grid_size: int, image_token_start: int) -> Tuple[int, int, List[float]]:
    rel = patch_index - image_token_start
    row = rel // grid_size
    col = rel % grid_size
    x0 = col / grid_size
    y0 = row / grid_size
    x1 = (col + 1) / grid_size
    y1 = (row + 1) / grid_size
    return row, col, [x0, y0, x1, y1]


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


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_label(label: Any) -> str:
    s = str(label).strip().lower()
    s = s.replace("docitemlabel.", "")
    s = s.replace("_", " ")
    return s


def classify_layout_label(label: str) -> str:
    ls = normalize_label(label)
    if any(k in ls for k in TABLE_LABEL_KEYWORDS):
        return "table"
    if any(k in ls for k in VISUAL_LABEL_KEYWORDS):
        return "visual"
    if any(k in ls for k in TEXT_LABEL_KEYWORDS):
        return "text"
    return "other"


def get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def extract_bbox_xyxy_raw(bbox_obj: Any) -> Optional[Tuple[float, float, float, float]]:
    if bbox_obj is None:
        return None

    if isinstance(bbox_obj, dict):
        # Common dict patterns
        for keys in (
            ("x0", "y0", "x1", "y1"),
            ("left", "top", "right", "bottom"),
            ("l", "t", "r", "b"),
        ):
            vals = [safe_float(bbox_obj.get(k)) for k in keys]
            if all(v is not None for v in vals):
                x0, y0, x1, y1 = vals  # type: ignore[misc]
                # Docling-style top/bottom coordinates may come in either order.
                x0, x1 = min(float(x0), float(x1)), max(float(x0), float(x1))
                y0, y1 = min(float(y0), float(y1)), max(float(y0), float(y1))
                if x1 > x0 and y1 > y0:
                    return x0, y0, x1, y1
        return None

    # Object attrs
    for keys in (
        ("x0", "y0", "x1", "y1"),
        ("left", "top", "right", "bottom"),
        ("l", "t", "r", "b"),
    ):
        vals = [safe_float(getattr(bbox_obj, k, None)) for k in keys]
        if all(v is not None for v in vals):
            x0, y0, x1, y1 = vals  # type: ignore[misc]
            x0, x1 = min(float(x0), float(x1)), max(float(x0), float(x1))
            y0, y1 = min(float(y0), float(y1)), max(float(y0), float(y1))
            if x1 > x0 and y1 > y0:
                return x0, y0, x1, y1
    return None


def extract_bbox_and_origin(obj: Any) -> Tuple[Optional[Tuple[float, float, float, float]], str]:
    # obj may itself be bbox or contain bbox
    direct = extract_bbox_xyxy_raw(obj)
    if direct is not None:
        origin = normalize_label(get_attr_or_key(obj, "coord_origin", "")) or normalize_label(
            get_attr_or_key(obj, "origin", "")
        )
        return direct, origin

    bbox = get_attr_or_key(obj, "bbox", None)
    bb = extract_bbox_xyxy_raw(bbox)
    if bb is not None:
        origin = normalize_label(get_attr_or_key(bbox, "coord_origin", "")) or normalize_label(
            get_attr_or_key(bbox, "origin", "")
        )
        return bb, origin

    return None, ""


def normalize_bbox_to_page(
    bbox_xyxy: Tuple[float, float, float, float],
    page_w: float,
    page_h: float,
    origin_hint: str,
) -> Optional[List[float]]:
    x0, y0, x1, y1 = bbox_xyxy
    if x1 <= x0 or y1 <= y0:
        return None

    # Already normalized case.
    if x1 <= 1.5 and y1 <= 1.5 and page_w > 0 and page_h > 0:
        nx0, ny0, nx1, ny1 = clamp01(x0), clamp01(y0), clamp01(x1), clamp01(y1)
    else:
        if page_w <= 0 or page_h <= 0:
            return None
        nx0, ny0, nx1, ny1 = x0 / page_w, y0 / page_h, x1 / page_w, y1 / page_h

    # Convert bottom-left origin into top-left normalized coordinates.
    if "bottom" in origin_hint:
        ny0, ny1 = 1.0 - ny1, 1.0 - ny0

    nx0, ny0, nx1, ny1 = clamp01(nx0), clamp01(ny0), clamp01(nx1), clamp01(ny1)
    if nx1 <= nx0 or ny1 <= ny0:
        return None
    return [nx0, ny0, nx1, ny1]


def load_page_sizes_from_pdf(pdf_path: Path) -> Dict[int, Tuple[float, float]]:
    if fitz is None:
        raise RuntimeError("pymupdf is required for robust page normalization. Install with: pip install pymupdf")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(str(pdf_path))
    out: Dict[int, Tuple[float, float]] = {}
    for i in range(len(doc)):
        page = doc[i]
        out[i] = (float(page.rect.width), float(page.rect.height))
    return out


def resolve_page_index(raw_page_no: Any, valid_page_indices: Set[int], page_offset: int) -> Optional[int]:
    f = safe_float(raw_page_no)
    if f is None:
        return None
    v = int(f)
    v2 = v + int(page_offset)
    if v2 in valid_page_indices:
        return v2
    return None


def iter_item_provenances(item: Any) -> Iterable[Any]:
    # Most common in Docling: item.prov list
    prov = get_attr_or_key(item, "prov", None)
    if prov is None:
        prov = get_attr_or_key(item, "provenance", None)
    if prov is None:
        return []
    if isinstance(prov, list) or isinstance(prov, tuple):
        return prov
    return [prov]


def iter_docling_items(doc: Any) -> Iterable[Any]:
    if hasattr(doc, "iterate_items"):
        # Docling's common API: yields (item, level)
        for pair in doc.iterate_items():
            if isinstance(pair, tuple) and len(pair) >= 1:
                yield pair[0]
            else:
                yield pair
        return

    # Fallback fields
    for attr_name in ("items", "main_text", "body"):
        val = get_attr_or_key(doc, attr_name, None)
        if isinstance(val, list):
            for x in val:
                yield x


def infer_docling_page_offset(
    doc: Any,
    valid_page_indices: Set[int],
    mode: str,
) -> Tuple[int, List[int]]:
    if mode == "zero":
        return 0, []
    if mode == "one":
        return -1, []

    observed: List[int] = []
    for item in iter_docling_items(doc):
        prov_list = list(iter_item_provenances(item))
        if not prov_list:
            prov_list = [item]
        for prov in prov_list:
            raw_page_no = get_attr_or_key(prov, "page_no", None)
            if raw_page_no is None:
                raw_page_no = get_attr_or_key(item, "page_no", None)
            f = safe_float(raw_page_no)
            if f is None:
                continue
            observed.append(int(f))

    if not observed:
        return 0, observed

    score_zero = sum(1 for v in observed if v in valid_page_indices)
    score_one = sum(1 for v in observed if (v - 1) in valid_page_indices)

    if score_one > score_zero:
        return -1, observed
    if score_one == score_zero:
        if 0 in observed:
            return 0, observed
        if min(observed) >= 1:
            return -1, observed
    return 0, observed


def build_docling_converter(
    device: str,
    num_threads: int,
    do_ocr: bool,
    do_table_structure: bool,
) -> Any:
    try:
        from docling.datamodel.base_models import InputFormat
        try:
            from docling.datamodel.accelerator_options import AcceleratorOptions
        except Exception:
            from docling.datamodel.pipeline_options import AcceleratorOptions  # type: ignore[attr-defined]
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter
        from docling.document_converter import PdfFormatOption
    except Exception as exc:
        raise RuntimeError(
            "Docling is not installed in this environment. "
            "Install in your target env with: pip install docling"
        ) from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=int(num_threads),
        device=str(device),
    )
    pipeline_options.do_ocr = bool(do_ocr)
    pipeline_options.do_table_structure = bool(do_table_structure)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    return converter


def run_docling_convert(
    pdf_path: Path,
    device: str,
    num_threads: int,
    do_ocr: bool,
    do_table_structure: bool,
    fallback_to_cpu: bool,
) -> Tuple[Any, str]:
    converter = build_docling_converter(
        device=device,
        num_threads=num_threads,
        do_ocr=do_ocr,
        do_table_structure=do_table_structure,
    )
    try:
        result = converter.convert(str(pdf_path))
        used_device = str(device).lower()
    except Exception as exc:
        emsg = str(exc).lower()
        accel_error = (
            "cuda" in emsg
            or "xpu" in emsg
            or "mps" in emsg
            or "driver error" in emsg
            or "conversionstatus.failure" in emsg
        )
        if not fallback_to_cpu or str(device).lower() == "cpu" or not accel_error:
            raise
        print(
            "Docling conversion failed on accelerator "
            f"'{device}' ({exc}). Retrying on CPU..."
        )
        converter = build_docling_converter(
            device="cpu",
            num_threads=num_threads,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
        )
        result = converter.convert(str(pdf_path))
        used_device = "cpu"

    doc = get_attr_or_key(result, "document", None)
    if doc is None:
        raise RuntimeError("Docling conversion returned no document object.")
    return doc, used_device


def extract_docling_zone_boxes(
    doc: Any,
    page_sizes: Dict[int, Tuple[float, float]],
    valid_page_indices: Set[int],
    page_offset: int,
) -> Tuple[Dict[int, List[List[float]]], Dict[int, List[List[float]]], Dict[int, List[List[float]]], List[Dict[str, Any]]]:
    page_text_boxes: Dict[int, List[List[float]]] = defaultdict(list)
    page_table_boxes: Dict[int, List[List[float]]] = defaultdict(list)
    page_visual_boxes: Dict[int, List[List[float]]] = defaultdict(list)
    debug_rows: List[Dict[str, Any]] = []

    for item in iter_docling_items(doc):
        raw_label = get_attr_or_key(item, "label", None)
        if raw_label is None:
            raw_label = get_attr_or_key(item, "type", None)
        if raw_label is None:
            raw_label = item.__class__.__name__
        label = normalize_label(raw_label)
        zone_type = classify_layout_label(label)
        if zone_type not in {"text", "table", "visual"}:
            continue

        prov_list = list(iter_item_provenances(item))
        if not prov_list:
            prov_list = [item]  # fallback: item may directly carry bbox/page info

        for prov in prov_list:
            raw_page_no = get_attr_or_key(prov, "page_no", None)
            if raw_page_no is None:
                raw_page_no = get_attr_or_key(item, "page_no", None)
            page_index = resolve_page_index(raw_page_no, valid_page_indices, page_offset=page_offset)
            if page_index is None:
                continue

            page_w, page_h = page_sizes[page_index]
            bbox_raw, origin = extract_bbox_and_origin(prov)
            if bbox_raw is None:
                bbox_raw, origin = extract_bbox_and_origin(item)
            if bbox_raw is None:
                continue
            bbox_norm = normalize_bbox_to_page(
                bbox_xyxy=bbox_raw,
                page_w=page_w,
                page_h=page_h,
                origin_hint=origin,
            )
            if bbox_norm is None:
                continue

            if zone_type == "table":
                page_table_boxes[page_index].append(bbox_norm)
            elif zone_type == "text":
                page_text_boxes[page_index].append(bbox_norm)
            elif zone_type == "visual":
                page_visual_boxes[page_index].append(bbox_norm)

            debug_rows.append(
                {
                    "page_index": page_index,
                    "zone_type": zone_type,
                    "label": label,
                    "bbox_norm_xyxy": [round(float(v), 6) for v in bbox_norm],
                    "bbox_raw_xyxy": [float(v) for v in bbox_raw],
                    "origin_hint": origin,
                }
            )

    return page_text_boxes, page_table_boxes, page_visual_boxes, debug_rows


def main() -> None:
    args = parse_args()

    labels_path = Path(args.labels_jsonl)
    pdf_path = Path(args.pdf_path)
    out_path = Path(args.output_jsonl)

    if not labels_path.is_file():
        raise FileNotFoundError(f"labels-jsonl not found: {labels_path}")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"pdf-path not found: {pdf_path}")
    if args.image_token_count != args.grid_size * args.grid_size:
        raise ValueError("--image-token-count must equal grid-size^2.")

    rows: List[Dict[str, Any]] = []
    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {labels_path}")

    page_ids = sorted({infer_page_id(r) for r in rows if infer_page_id(r)})
    if not page_ids:
        raise ValueError("No page_ids found in labels-jsonl.")

    doc_ids = {parse_page_id(pid)[0] for pid in page_ids}
    if len(doc_ids) != 1:
        raise ValueError(
            f"labels-jsonl appears to contain multiple docs ({sorted(doc_ids)}); expected one doc per run."
        )

    page_index_to_id: Dict[int, str] = {}
    for pid in page_ids:
        _, pidx = parse_page_id(pid)
        page_index_to_id[pidx] = pid
    valid_page_indices = set(page_index_to_id.keys())

    page_sizes = load_page_sizes_from_pdf(pdf_path=pdf_path)
    missing = sorted(valid_page_indices - set(page_sizes.keys()))
    if missing:
        raise ValueError(f"Page indices in labels not present in PDF: {missing}")

    doc, docling_device_used = run_docling_convert(
        pdf_path=pdf_path,
        device=args.docling_device,
        num_threads=args.docling_num_threads,
        do_ocr=args.docling_do_ocr,
        do_table_structure=args.docling_do_table_structure,
        fallback_to_cpu=args.docling_fallback_to_cpu,
    )
    page_offset, observed_page_numbers = infer_docling_page_offset(
        doc=doc,
        valid_page_indices=valid_page_indices,
        mode=args.docling_page_number_base,
    )
    page_text_boxes, page_table_boxes, page_visual_boxes, debug_rows = extract_docling_zone_boxes(
        doc=doc,
        page_sizes=page_sizes,
        valid_page_indices=valid_page_indices,
        page_offset=page_offset,
    )

    page_text_hits: Dict[str, Set[int]] = {}
    page_table_hits: Dict[str, Set[int]] = {}
    page_visual_hits: Dict[str, Set[int]] = {}

    for pidx, page_id in page_index_to_id.items():
        text_boxes = page_text_boxes.get(pidx, [])
        table_boxes = page_table_boxes.get(pidx, [])
        visual_boxes = page_visual_boxes.get(pidx, [])

        text_hits: Set[int] = set()
        table_hits: Set[int] = set()
        visual_hits: Set[int] = set()

        for patch_index in range(args.image_token_start, args.image_token_start + args.image_token_count):
            _, _, pb = patch_box_from_index(
                patch_index=patch_index,
                grid_size=args.grid_size,
                image_token_start=args.image_token_start,
            )

            for b in table_boxes:
                if overlap_fraction_of_patch(pb, b) >= args.min_overlap_table:
                    table_hits.add(patch_index)
                    break
            for b in text_boxes:
                if overlap_fraction_of_patch(pb, b) >= args.min_overlap_text:
                    text_hits.add(patch_index)
                    break
            if args.include_visual_region:
                for b in visual_boxes:
                    if overlap_fraction_of_patch(pb, b) >= args.min_overlap_visual:
                        visual_hits.add(patch_index)
                        break

        page_text_hits[page_id] = text_hits
        page_table_hits[page_id] = table_hits
        page_visual_hits[page_id] = visual_hits

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    per_page = defaultdict(lambda: defaultdict(int))
    total_rows = 0

    with open(out_path, "w") as out:
        for row in rows:
            page_id = infer_page_id(row)
            patch_index = int(row.get("patch_index", -1))
            labels: List[Dict[str, Any]] = []

            in_image_range = (
                patch_index >= args.image_token_start
                and patch_index < (args.image_token_start + args.image_token_count)
            )
            if in_image_range and page_id:
                in_text = patch_index in page_text_hits.get(page_id, set())
                in_table = patch_index in page_table_hits.get(page_id, set())
                in_visual = args.include_visual_region and (patch_index in page_visual_hits.get(page_id, set()))

                if in_table and in_text:
                    labels = [{"concept": "table_text", "weight": 1.0}]
                    counts["table_text"] += 1
                    per_page[page_id]["table_text"] += 1
                elif in_table:
                    labels = [{"concept": "table_structure", "weight": 1.0}]
                    counts["table_structure"] += 1
                    per_page[page_id]["table_structure"] += 1
                elif in_text:
                    labels = [{"concept": "ocr_text", "weight": 1.0}]
                    counts["ocr_text"] += 1
                    per_page[page_id]["ocr_text"] += 1
                elif in_visual:
                    labels = [{"concept": "visual_region", "weight": 1.0}]
                    counts["visual_region"] += 1
                    per_page[page_id]["visual_region"] += 1
                else:
                    counts["unlabeled"] += 1
                    per_page[page_id]["unlabeled"] += 1
            else:
                counts["out_of_image_range"] += 1
                if page_id:
                    per_page[page_id]["out_of_image_range"] += 1

            row["top_concepts"] = labels
            out.write(json.dumps(row) + "\n")
            total_rows += 1

    summary = {
        "backend": "docling",
        "labels_jsonl": str(labels_path),
        "pdf_path": str(pdf_path),
        "output_jsonl": str(out_path),
        "pages": len(page_ids),
        "rows": total_rows,
        "grid_size": args.grid_size,
        "image_token_start": args.image_token_start,
        "image_token_count": args.image_token_count,
        "min_overlap_text": args.min_overlap_text,
        "min_overlap_table": args.min_overlap_table,
        "min_overlap_visual": args.min_overlap_visual,
        "include_visual_region": bool(args.include_visual_region),
        "docling_device_requested": args.docling_device,
        "docling_device_used": docling_device_used,
        "docling_num_threads": int(args.docling_num_threads),
        "docling_do_ocr": bool(args.docling_do_ocr),
        "docling_do_table_structure": bool(args.docling_do_table_structure),
        "docling_fallback_to_cpu": bool(args.docling_fallback_to_cpu),
        "docling_page_number_base": args.docling_page_number_base,
        "docling_page_offset_used": int(page_offset),
        "docling_observed_page_numbers_minmax": (
            [int(min(observed_page_numbers)), int(max(observed_page_numbers))]
            if observed_page_numbers
            else []
        ),
        "docling_zone_counts": {
            "text_boxes_total": int(sum(len(v) for v in page_text_boxes.values())),
            "table_boxes_total": int(sum(len(v) for v in page_table_boxes.values())),
            "visual_boxes_total": int(sum(len(v) for v in page_visual_boxes.values())),
        },
        "counts": dict(counts),
    }

    print("=== Docling Layout Label Summary ===")
    print(json.dumps(summary, indent=2))

    if args.summary_json:
        sp = Path(args.summary_json)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"Wrote: {sp}")

    if args.summary_tsv:
        tp = Path(args.summary_tsv)
        tp.parent.mkdir(parents=True, exist_ok=True)
        with open(tp, "w") as f:
            f.write("page_id\tocr_text\ttable_text\ttable_structure\tvisual_region\tunlabeled\n")
            for page_id in page_ids:
                rec = per_page[page_id]
                f.write(
                    f"{page_id}\t"
                    f"{int(rec.get('ocr_text', 0))}\t"
                    f"{int(rec.get('table_text', 0))}\t"
                    f"{int(rec.get('table_structure', 0))}\t"
                    f"{int(rec.get('visual_region', 0))}\t"
                    f"{int(rec.get('unlabeled', 0))}\n"
                )
        print(f"Wrote: {tp}")

    if args.zones_debug_json:
        zp = Path(args.zones_debug_json)
        zp.parent.mkdir(parents=True, exist_ok=True)
        with open(zp, "w") as f:
            for rec in debug_rows:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote: {zp}")


if __name__ == "__main__":
    main()
