#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
except ImportError:
    AutoImageProcessor = None
    TableTransformerForObjectDetection = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create layout-aware patch labels using OCR + table detection with priority:\n"
            "table_text > table_structure > ocr_text > visual_region."
        )
    )
    parser.add_argument("--labels-jsonl", type=str, required=True, help="Base patch labels JSONL (one doc).")
    parser.add_argument("--ocr-jsonl", type=str, required=True, help="OCR words JSONL for the same doc.")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--page-image-template",
        type=str,
        required=True,
        help="Template: /path/{doc_id}_{page_index}.png",
    )
    parser.add_argument(
        "--visual-labels-jsonl",
        type=str,
        default="",
        help="Optional visual labels JSONL for fallback visual_region labeling.",
    )
    parser.add_argument(
        "--visual-min-score",
        type=float,
        default=0.20,
        help="Min max-concept score in visual labels to mark visual_region.",
    )

    parser.add_argument("--grid-size", type=int, default=32, help="Patch grid size.")
    parser.add_argument("--image-token-start", type=int, default=0, help="Image token start index.")
    parser.add_argument("--image-token-count", type=int, default=1024, help="Image token count.")
    parser.add_argument("--min-word-conf", type=float, default=45.0, help="OCR confidence threshold.")
    parser.add_argument("--min-word-len", type=int, default=2, help="Minimum normalized OCR token length.")
    parser.add_argument(
        "--max-word-box-area-frac",
        type=float,
        default=0.02,
        help="Drop OCR word boxes larger than this fraction of page area.",
    )
    parser.add_argument("--min-overlap-text", type=float, default=0.08, help="Min patch overlap with OCR word box.")
    parser.add_argument("--min-overlap-table", type=float, default=0.10, help="Min patch overlap with table box.")
    parser.add_argument(
        "--table-expand-x",
        type=float,
        default=0.02,
        help="Expand detected table boxes horizontally by this normalized margin per side.",
    )
    parser.add_argument(
        "--table-expand-y",
        type=float,
        default=0.02,
        help="Expand detected table boxes vertically by this normalized margin per side.",
    )
    parser.add_argument(
        "--table-dilate-cells",
        type=int,
        default=1,
        help="Dilate table-hit patch cells by this radius.",
    )
    parser.add_argument(
        "--text-dilate-cells",
        type=int,
        default=1,
        help="Dilate OCR-hit patch cells by this radius to better cover tables.",
    )
    parser.add_argument(
        "--text-neighbor-radius",
        type=int,
        default=1,
        help="Neighborhood radius for suppressing isolated OCR text hits outside tables.",
    )
    parser.add_argument(
        "--min-text-neighbors",
        type=int,
        default=1,
        help="Min neighboring OCR-hit patches (outside tables) required to keep a text patch.",
    )
    parser.add_argument(
        "--text-max-visual-score",
        type=float,
        default=0.30,
        help="If visual score exceeds this, do not assign ocr_text outside tables.",
    )

    parser.add_argument(
        "--table-model-name",
        type=str,
        default="microsoft/table-transformer-detection",
        help="HF model name for table detection.",
    )
    parser.add_argument(
        "--table-score-threshold",
        type=float,
        default=0.85,
        help="Table detector confidence threshold.",
    )
    parser.add_argument(
        "--min-table-area-frac",
        type=float,
        default=0.01,
        help="Ignore detected table boxes smaller than this normalized page area.",
    )
    parser.add_argument(
        "--max-table-area-frac",
        type=float,
        default=0.70,
        help="Ignore detected table boxes larger than this normalized page area.",
    )
    parser.add_argument(
        "--enable-ocr-table-fallback",
        action="store_true",
        help="Infer table boxes from OCR row/column regularity and union with detector boxes.",
    )
    parser.add_argument(
        "--ocr-table-row-tol",
        type=float,
        default=0.015,
        help="Row clustering tolerance in normalized Y for OCR-table fallback.",
    )
    parser.add_argument(
        "--ocr-table-col-tol",
        type=float,
        default=0.040,
        help="Column bin width in normalized X for OCR-table fallback.",
    )
    parser.add_argument(
        "--ocr-table-min-rows",
        type=int,
        default=4,
        help="Minimum OCR rows required to form a fallback table box.",
    )
    parser.add_argument(
        "--ocr-table-min-cols",
        type=int,
        default=3,
        help="Minimum OCR columns required to form a fallback table box.",
    )
    parser.add_argument(
        "--ocr-table-min-words-per-row",
        type=int,
        default=3,
        help="Minimum OCR words per row to keep the row for fallback table detection.",
    )
    parser.add_argument(
        "--ocr-table-min-words",
        type=int,
        default=18,
        help="Minimum OCR words on page before attempting fallback table detection.",
    )
    parser.add_argument(
        "--ocr-table-expand",
        type=float,
        default=0.01,
        help="Expand fallback OCR-derived table boxes by this normalized margin.",
    )
    parser.add_argument(
        "--disable-table-detector",
        action="store_true",
        help="Skip table detection and rely only on OCR/visual.",
    )
    default_device = "cpu"
    if torch is not None and torch.cuda.is_available():
        default_device = "cuda"
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Torch device for table detector.",
    )
    parser.add_argument("--summary-json", type=str, default="", help="Optional summary JSON output.")
    parser.add_argument("--summary-tsv", type=str, default="", help="Optional per-page concept count TSV.")
    return parser.parse_args()


def infer_page_id(row: Dict) -> str:
    page_id = str(row.get("page_id", "")).strip()
    if page_id:
        return page_id
    rid = str(row.get("id", "")).strip()
    if "#patch" in rid:
        return rid.split("#patch", 1)[0]
    return rid


def parse_page_components(page_id: str) -> Tuple[str, int]:
    if ":" not in page_id:
        raise ValueError(f"Invalid page_id (expected doc:page): {page_id}")
    doc_id, page_str = page_id.rsplit(":", 1)
    return doc_id, int(page_str)


def patch_bbox(
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


def center_in_box(a: List[float], b: List[float]) -> bool:
    cx = 0.5 * (a[0] + a[2])
    cy = 0.5 * (a[1] + a[3])
    return (b[0] <= cx <= b[2]) and (b[1] <= cy <= b[3])


def expand_box_norm(box: List[float], dx: float, dy: float) -> List[float]:
    x0 = max(0.0, box[0] - dx)
    y0 = max(0.0, box[1] - dy)
    x1 = min(1.0, box[2] + dx)
    y1 = min(1.0, box[3] + dy)
    if x1 <= x0 or y1 <= y0:
        return box
    return [x0, y0, x1, y1]


def box_area(box: List[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


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


def merge_boxes(boxes: List[List[float]], iou_thr: float = 0.25) -> List[List[float]]:
    if not boxes:
        return []
    out: List[List[float]] = []
    for box in boxes:
        merged = False
        for j, cur in enumerate(out):
            if iou(box, cur) >= iou_thr:
                out[j] = [
                    min(cur[0], box[0]),
                    min(cur[1], box[1]),
                    max(cur[2], box[2]),
                    max(cur[3], box[3]),
                ]
                merged = True
                break
        if not merged:
            out.append(list(box))
    return out


def expand_allowed(allowed: Set[int], grid_size: int, image_token_start: int, radius: int) -> Set[int]:
    if radius <= 0 or not allowed:
        return allowed
    out: Set[int] = set(allowed)
    for patch_index in list(allowed):
        rel = patch_index - image_token_start
        row = rel // grid_size
        col = rel % grid_size
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = row + dr
                cc = col + dc
                if 0 <= rr < grid_size and 0 <= cc < grid_size:
                    out.add(image_token_start + rr * grid_size + cc)
    return out


def count_neighbors(
    patch_index: int,
    mask: Set[int],
    grid_size: int,
    image_token_start: int,
    radius: int,
) -> int:
    rel = patch_index - image_token_start
    row = rel // grid_size
    col = rel % grid_size
    cnt = 0
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            rr = row + dr
            cc = col + dc
            if 0 <= rr < grid_size and 0 <= cc < grid_size:
                p = image_token_start + rr * grid_size + cc
                if p in mask:
                    cnt += 1
    return cnt


def filter_isolated_hits(
    hits: Set[int],
    table_hits: Set[int],
    grid_size: int,
    image_token_start: int,
    radius: int,
    min_neighbors: int,
) -> Set[int]:
    if min_neighbors <= 0 or radius <= 0 or not hits:
        return set(hits)
    out: Set[int] = set()
    for p in hits:
        if p in table_hits:
            out.add(p)
            continue
        n = count_neighbors(
            patch_index=p,
            mask=hits,
            grid_size=grid_size,
            image_token_start=image_token_start,
            radius=radius,
        )
        if n >= min_neighbors:
            out.add(p)
    return out


def load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_ocr_boxes(
    ocr_jsonl: Path,
    min_conf: float,
    min_word_len: int,
    max_word_box_area_frac: float,
) -> Dict[str, List[List[float]]]:
    page_boxes: Dict[str, List[List[float]]] = {}
    with open(ocr_jsonl, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            page_id = str(row.get("page_id", "")).strip()
            if not page_id:
                continue
            width = float(row.get("width", 0) or 0)
            height = float(row.get("height", 0) or 0)
            boxes: List[List[float]] = []
            for word in row.get("words", []):
                if not isinstance(word, dict):
                    continue
                conf = float(word.get("conf", -1.0))
                if conf < min_conf:
                    continue
                norm = str(word.get("norm", "")).strip()
                if len(norm) < max(0, min_word_len):
                    continue
                box = word.get("bbox_xyxy", [])
                if not isinstance(box, list) or len(box) != 4:
                    continue
                if width <= 0 or height <= 0:
                    continue
                x0, y0, x1, y1 = [float(v) for v in box]
                if x1 <= x0 or y1 <= y0:
                    continue
                area_frac = ((x1 - x0) * (y1 - y0)) / (width * height)
                if area_frac > max_word_box_area_frac:
                    continue
                boxes.append([x0 / width, y0 / height, x1 / width, y1 / height])
            page_boxes[page_id] = boxes
    return page_boxes


def infer_table_boxes_from_ocr(
    boxes: List[List[float]],
    row_tol: float,
    col_tol: float,
    min_rows: int,
    min_cols: int,
    min_words_per_row: int,
    min_words: int,
    expand: float,
    min_area_frac: float,
    max_area_frac: float,
) -> List[List[float]]:
    if len(boxes) < max(1, min_words):
        return []

    words = []
    for b in boxes:
        x0, y0, x1, y1 = b
        if x1 <= x0 or y1 <= y0:
            continue
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        words.append((x0, y0, x1, y1, cx, cy))
    if len(words) < max(1, min_words):
        return []

    words.sort(key=lambda t: t[5])
    rows: List[List[Tuple[float, float, float, float, float, float]]] = []
    for w in words:
        if not rows:
            rows.append([w])
            continue
        prev_cy = sum(x[5] for x in rows[-1]) / len(rows[-1])
        if abs(w[5] - prev_cy) <= row_tol:
            rows[-1].append(w)
        else:
            rows.append([w])

    rows = [r for r in rows if len(r) >= max(1, min_words_per_row)]
    if len(rows) < max(1, min_rows):
        return []

    row_bins: List[Set[int]] = []
    for r in rows:
        bins = {int(round(w[4] / max(col_tol, 1e-6))) for w in r}
        row_bins.append(bins)

    col_counts = defaultdict(int)
    for bins in row_bins:
        for b in bins:
            col_counts[b] += 1
    min_col_support = max(2, int(round(0.5 * len(rows))))
    good_cols = {b for b, c in col_counts.items() if c >= min_col_support}
    if len(good_cols) < max(1, min_cols):
        return []

    kept = []
    for r, bins in zip(rows, row_bins):
        if len(good_cols.intersection(bins)) < max(1, min_cols):
            continue
        kept.extend(r)
    if len(kept) < max(1, min_words):
        return []

    x0 = min(w[0] for w in kept)
    y0 = min(w[1] for w in kept)
    x1 = max(w[2] for w in kept)
    y1 = max(w[3] for w in kept)
    box = expand_box_norm([x0, y0, x1, y1], expand, expand)
    area = box_area(box)
    if area < min_area_frac or area > max_area_frac:
        return []
    return [box]


def load_visual_max_map(path: Optional[Path]) -> Dict[Tuple[str, int], float]:
    if path is None:
        return {}
    out: Dict[Tuple[str, int], float] = {}
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            page_id = infer_page_id(row)
            if not page_id:
                continue
            patch_index = int(row.get("patch_index", -1))
            if patch_index < 0:
                continue
            mx = 0.0
            for item in row.get("top_concepts", []):
                try:
                    w = float(item.get("weight", 0.0))
                except Exception:
                    w = 0.0
                if w > mx:
                    mx = w
            out[(page_id, patch_index)] = mx
    return out


class TableDetector:
    def __init__(
        self,
        model_name: str,
        device: str,
        score_thr: float,
        min_area_frac: float,
        max_area_frac: float,
    ) -> None:
        if torch is None:
            raise RuntimeError("torch is required for table detection. Install with: pip install torch")
        if Image is None:
            raise RuntimeError("Pillow is required. Install with: pip install pillow")
        if AutoImageProcessor is None or TableTransformerForObjectDetection is None:
            raise RuntimeError(
                "transformers is required for table detection. Install with: pip install transformers"
            )
        self.score_thr = score_thr
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}

    def detect_normalized_boxes(self, image_path: Path) -> List[List[float]]:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        if w <= 0 or h <= 0:
            return []

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=self.device)
        processed = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.score_thr,
        )[0]

        boxes: List[List[float]] = []
        scores = processed["scores"].detach().cpu().tolist()
        labels = processed["labels"].detach().cpu().tolist()
        bboxes = processed["boxes"].detach().cpu().tolist()
        for score, label_id, box in zip(scores, labels, bboxes):
            label = self.id2label.get(int(label_id), "")
            if "table" not in label:
                continue
            x0, y0, x1, y1 = [float(v) for v in box]
            x0 = max(0.0, min(float(w), x0))
            y0 = max(0.0, min(float(h), y0))
            x1 = max(0.0, min(float(w), x1))
            y1 = max(0.0, min(float(h), y1))
            if x1 <= x0 or y1 <= y0:
                continue
            nx0, ny0, nx1, ny1 = [x0 / w, y0 / h, x1 / w, y1 / h]
            area = max(0.0, nx1 - nx0) * max(0.0, ny1 - ny0)
            if area < self.min_area_frac or area > self.max_area_frac:
                continue
            boxes.append([nx0, ny0, nx1, ny1])
        return boxes


def main() -> None:
    args = parse_args()
    if torch is None:
        raise RuntimeError("torch is required. Run in the splice environment with torch installed.")
    labels_path = Path(args.labels_jsonl)
    ocr_path = Path(args.ocr_jsonl)
    out_path = Path(args.output_jsonl)
    vis_path = Path(args.visual_labels_jsonl) if args.visual_labels_jsonl else None

    if not labels_path.is_file():
        raise FileNotFoundError(f"labels-jsonl not found: {labels_path}")
    if not ocr_path.is_file():
        raise FileNotFoundError(f"ocr-jsonl not found: {ocr_path}")
    if vis_path is not None and not vis_path.is_file():
        raise FileNotFoundError(f"visual-labels-jsonl not found: {vis_path}")
    if args.image_token_count != args.grid_size * args.grid_size:
        raise ValueError("--image-token-count must equal grid-size^2 for this script.")

    rows = load_rows(labels_path)
    page_ids = sorted({infer_page_id(r) for r in rows if infer_page_id(r)})
    if not page_ids:
        raise ValueError(f"No page rows found in {labels_path}")

    ocr_boxes = load_ocr_boxes(
        ocr_jsonl=ocr_path,
        min_conf=args.min_word_conf,
        min_word_len=args.min_word_len,
        max_word_box_area_frac=args.max_word_box_area_frac,
    )
    visual_max = load_visual_max_map(vis_path)

    detector: Optional[TableDetector] = None
    if not args.disable_table_detector:
        detector = TableDetector(
            model_name=args.table_model_name,
            device=args.device,
            score_thr=args.table_score_threshold,
            min_area_frac=args.min_table_area_frac,
            max_area_frac=args.max_table_area_frac,
        )

    page_text_hits: Dict[str, Set[int]] = {}
    page_table_hits: Dict[str, Set[int]] = {}
    page_table_boxes: Dict[str, List[List[float]]] = {}

    for page_id in page_ids:
        doc_id, page_index = parse_page_components(page_id)
        image_path = Path(args.page_image_template.format(doc_id=doc_id, page_index=page_index))
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing page image: {image_path}")

        table_boxes = detector.detect_normalized_boxes(image_path) if detector else []
        if table_boxes:
            table_boxes = [
                expand_box_norm(box=b, dx=args.table_expand_x, dy=args.table_expand_y)
                for b in table_boxes
            ]
        if args.enable_ocr_table_fallback:
            table_boxes.extend(
                infer_table_boxes_from_ocr(
                    boxes=ocr_boxes.get(page_id, []),
                    row_tol=args.ocr_table_row_tol,
                    col_tol=args.ocr_table_col_tol,
                    min_rows=args.ocr_table_min_rows,
                    min_cols=args.ocr_table_min_cols,
                    min_words_per_row=args.ocr_table_min_words_per_row,
                    min_words=args.ocr_table_min_words,
                    expand=args.ocr_table_expand,
                    min_area_frac=args.min_table_area_frac,
                    max_area_frac=args.max_table_area_frac,
                )
            )
        table_boxes = merge_boxes(table_boxes, iou_thr=0.25)
        page_table_boxes[page_id] = table_boxes

        text_hits_raw: Set[int] = set()
        table_hits: Set[int] = set()

        for patch_index in range(args.image_token_start, args.image_token_start + args.image_token_count):
            _, _, pb = patch_bbox(
                patch_index=patch_index,
                grid_size=args.grid_size,
                image_token_start=args.image_token_start,
            )

            for box in ocr_boxes.get(page_id, []):
                if overlap_fraction_of_patch(pb, box) >= args.min_overlap_text:
                    text_hits_raw.add(patch_index)
                    break

            for box in table_boxes:
                if (
                    overlap_fraction_of_patch(pb, box) >= args.min_overlap_table
                    or center_in_box(pb, box)
                ):
                    table_hits.add(patch_index)
                    break

        if args.table_dilate_cells > 0:
            table_hits = expand_allowed(
                allowed=table_hits,
                grid_size=args.grid_size,
                image_token_start=args.image_token_start,
                radius=args.table_dilate_cells,
            )

        text_hits_clean = filter_isolated_hits(
            hits=text_hits_raw,
            table_hits=table_hits,
            grid_size=args.grid_size,
            image_token_start=args.image_token_start,
            radius=args.text_neighbor_radius,
            min_neighbors=args.min_text_neighbors,
        )

        if args.text_dilate_cells > 0:
            text_hits_clean = expand_allowed(
                allowed=text_hits_clean,
                grid_size=args.grid_size,
                image_token_start=args.image_token_start,
                radius=args.text_dilate_cells,
            )

        page_text_hits[page_id] = text_hits_clean
        page_table_hits[page_id] = table_hits

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    counts = defaultdict(int)
    per_page = defaultdict(lambda: defaultdict(int))

    with open(out_path, "w") as out:
        for row in rows:
            page_id = infer_page_id(row)
            patch_index = int(row.get("patch_index", -1))
            labels: List[Dict] = []

            in_image_range = (
                patch_index >= args.image_token_start
                and patch_index < (args.image_token_start + args.image_token_count)
            )
            if in_image_range and page_id:
                in_text = patch_index in page_text_hits.get(page_id, set())
                in_table = patch_index in page_table_hits.get(page_id, set())
                vis_score = float(visual_max.get((page_id, patch_index), 0.0))

                if in_table and in_text:
                    labels = [{"concept": "table_text", "weight": 1.0}]
                    counts["table_text"] += 1
                    per_page[page_id]["table_text"] += 1
                elif in_table:
                    labels = [{"concept": "table_structure", "weight": 1.0}]
                    counts["table_structure"] += 1
                    per_page[page_id]["table_structure"] += 1
                elif in_text:
                    if vis_score <= args.text_max_visual_score:
                        labels = [{"concept": "ocr_text", "weight": 1.0}]
                        counts["ocr_text"] += 1
                        per_page[page_id]["ocr_text"] += 1
                    elif vis_score >= args.visual_min_score:
                        labels = [{"concept": "visual_region", "weight": round(vis_score, 6)}]
                        counts["visual_region"] += 1
                        per_page[page_id]["visual_region"] += 1
                    else:
                        counts["unlabeled"] += 1
                        per_page[page_id]["unlabeled"] += 1
                elif vis_score >= args.visual_min_score:
                    labels = [{"concept": "visual_region", "weight": round(vis_score, 6)}]
                    counts["visual_region"] += 1
                    per_page[page_id]["visual_region"] += 1
                else:
                    counts["unlabeled"] += 1
                    per_page[page_id]["unlabeled"] += 1
            else:
                counts["out_of_image_range"] += 1
                per_page[page_id]["out_of_image_range"] += 1

            row["top_concepts"] = labels
            out.write(json.dumps(row) + "\n")
            total_rows += 1

    summary = {
        "labels_jsonl": str(labels_path),
        "ocr_jsonl": str(ocr_path),
        "visual_labels_jsonl": str(vis_path) if vis_path else "",
        "output_jsonl": str(out_path),
        "pages": len(page_ids),
        "rows": total_rows,
        "grid_size": args.grid_size,
        "image_token_start": args.image_token_start,
        "image_token_count": args.image_token_count,
        "table_detector_enabled": not args.disable_table_detector,
        "table_model_name": args.table_model_name if not args.disable_table_detector else "",
        "table_score_threshold": args.table_score_threshold,
        "enable_ocr_table_fallback": bool(args.enable_ocr_table_fallback),
        "visual_min_score": args.visual_min_score,
        "counts": dict(counts),
    }

    print("=== Layout Label Summary ===")
    print(json.dumps(summary, indent=2))

    if args.summary_json:
        p = Path(args.summary_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        print(f"Wrote: {p}")

    if args.summary_tsv:
        p = Path(args.summary_tsv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as handle:
            handle.write("page_id\tocr_text\ttable_text\ttable_structure\tvisual_region\tunlabeled\n")
            for page_id in page_ids:
                rec = per_page[page_id]
                handle.write(
                    f"{page_id}\t"
                    f"{int(rec.get('ocr_text', 0))}\t"
                    f"{int(rec.get('table_text', 0))}\t"
                    f"{int(rec.get('table_structure', 0))}\t"
                    f"{int(rec.get('visual_region', 0))}\t"
                    f"{int(rec.get('unlabeled', 0))}\n"
                )
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
