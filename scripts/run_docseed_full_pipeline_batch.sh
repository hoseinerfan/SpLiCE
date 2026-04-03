#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run full doc-seed pipeline for a list of doc IDs:
1) build doc-seed text dictionaries
2) label text patches
3) merge text + visual patch labels
4) render PDF pages to PNG
5) OCR extraction
6) OCR-constrained label cleanup (v6b-style)

Usage:
  scripts/run_docseed_full_pipeline_batch.sh \
    --doc-ids-file /path/to/doc_ids.txt \
    [--output-root /path/to/embeddings] \
    [--embeddings-root /path/to/colpali_embeddings] \
    [--visual-labels-root /path/to/visual_labels] \
    [--mean-path /path/to/colpali_mean_max.pt] \
    [--pdf-root /path/to/pdfs_dev] \
    [--mmqa-dev-jsonl /path/to/MMQA_dev.jsonl] \
    [--mmqa-texts-jsonl /path/to/MMQA_texts.jsonl] \
    [--mmqa-tables-jsonl /path/to/MMQA_tables.jsonl] \
    [--tesseract-cmd /path/to/tesseract] \
    [--device cuda:0] \
    [--model-name vidore/colpali-v1.2-hf] \
    [--backend transformers] \
    [--batch-size 16] \
    [--dtype bfloat16] \
    [--skip-existing]
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DOC_IDS_FILE=""
OUTPUT_ROOT="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings"
EMBEDDINGS_ROOT="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali-v1.2_m3-docvqa_dev"
VISUAL_LABELS_ROOT="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/patch_labels_visual_attr_human_strict_v1_t10_w005"
MEAN_PATH="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/means/colpali_mean_max.pt"
PDF_ROOT="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/splits/pdfs_dev"
MMQA_DEV_JSONL="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl"
MMQA_TEXTS_JSONL="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_texts.jsonl"
MMQA_TABLES_JSONL="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_tables.jsonl"
TESS_CMD="tesseract"

MODEL_NAME="vidore/colpali-v1.2-hf"
BACKEND="transformers"
BATCH_SIZE="16"
DTYPE="bfloat16"
DEVICE="cuda:0"
SKIP_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --doc-ids-file) DOC_IDS_FILE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --embeddings-root) EMBEDDINGS_ROOT="$2"; shift 2 ;;
    --visual-labels-root) VISUAL_LABELS_ROOT="$2"; shift 2 ;;
    --mean-path) MEAN_PATH="$2"; shift 2 ;;
    --pdf-root) PDF_ROOT="$2"; shift 2 ;;
    --mmqa-dev-jsonl) MMQA_DEV_JSONL="$2"; shift 2 ;;
    --mmqa-texts-jsonl) MMQA_TEXTS_JSONL="$2"; shift 2 ;;
    --mmqa-tables-jsonl) MMQA_TABLES_JSONL="$2"; shift 2 ;;
    --tesseract-cmd) TESS_CMD="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${DOC_IDS_FILE}" ]]; then
  echo "ERROR: --doc-ids-file is required" >&2
  usage
  exit 2
fi
if [[ ! -f "${DOC_IDS_FILE}" ]]; then
  echo "ERROR: doc ids file not found: ${DOC_IDS_FILE}" >&2
  exit 2
fi

mapfile -t DOC_IDS < <(grep -v '^[[:space:]]*$' "${DOC_IDS_FILE}" | sed 's/[[:space:]]*$//')
if [[ "${#DOC_IDS[@]}" -eq 0 ]]; then
  echo "ERROR: no doc ids found in ${DOC_IDS_FILE}" >&2
  exit 2
fi

echo "Docs: ${#DOC_IDS[@]}"
echo "Output root: ${OUTPUT_ROOT}"

# Stage 1: Build doc-seed dictionaries for all docs in one pass.
BUILD_CMD=(
  python scripts/build_docseed_text_dictionary.py
  --doc-ids-file "${DOC_IDS_FILE}"
  --mmqa-dev-jsonl "${MMQA_DEV_JSONL}"
  --mmqa-texts-jsonl "${MMQA_TEXTS_JSONL}"
  --mmqa-tables-jsonl "${MMQA_TABLES_JSONL}"
  --output-root "${OUTPUT_ROOT}"
  --build-dictionary
  --model-name "${MODEL_NAME}"
  --backend "${BACKEND}"
  --batch-size "${BATCH_SIZE}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
)
echo "+ ${BUILD_CMD[*]}"
"${BUILD_CMD[@]}"

for DOC in "${DOC_IDS[@]}"; do
  echo
  echo "===================="
  echo "DOC: ${DOC}"
  echo "===================="

  CTX="${OUTPUT_ROOT}/debug_${DOC}_linked_context_auto"
  TXT_DICT="${CTX}/doc_seed_text_dict.pt"
  TXT_VOC="${CTX}/doc_seed_concepts_text_strict.txt"
  TXT_OUT_DIR="${CTX}/doc_seed_text_patch_labels_dir"
  TXT_LAB="${CTX}/doc_seed_text_patch_labels.jsonl"
  MERGED="${CTX}/doc_seed_text_visual_merged.jsonl"
  OCR_JSONL="${CTX}/ocr_words.jsonl"
  OCR_OUT="${CTX}/doc_seed_text_visual_merged_ocr_v6b.jsonl"
  OCR_SUM_TSV="${CTX}/doc_seed_text_visual_merged_ocr_v6b_summary.tsv"
  OCR_SUM_JSON="${CTX}/doc_seed_text_visual_merged_ocr_v6b_summary.json"

  EMB_PATH="${EMBEDDINGS_ROOT}/${DOC}.safetensors"
  VIS_LAB="${VISUAL_LABELS_ROOT}/${DOC}.jsonl"
  PDF_PATH="${PDF_ROOT}/${DOC}.pdf"
  IMG_DIR="${OUTPUT_ROOT}/debug_${DOC}_page_pngs_144"

  if [[ ! -f "${TXT_DICT}" || ! -f "${TXT_VOC}" ]]; then
    echo "ERROR: missing doc-seed text dictionary for ${DOC} under ${CTX}" >&2
    exit 1
  fi
  if [[ ! -f "${EMB_PATH}" ]]; then
    echo "ERROR: missing embeddings file: ${EMB_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${VIS_LAB}" ]]; then
    echo "ERROR: missing visual labels file: ${VIS_LAB}" >&2
    exit 1
  fi
  if [[ ! -f "${PDF_PATH}" ]]; then
    echo "ERROR: missing PDF file: ${PDF_PATH}" >&2
    exit 1
  fi

  # Stage 2: text patch labeling
  if [[ "${SKIP_EXISTING}" -eq 1 && -f "${TXT_LAB}" ]]; then
    echo "skip text labeling (exists): ${TXT_LAB}"
  else
    echo "+ python scripts/label_page_patches_bulk.py --embeddings-path ${EMB_PATH} ..."
    python scripts/label_page_patches_bulk.py \
      --embeddings-path "${EMB_PATH}" \
      --dictionary-path "${TXT_DICT}" \
      --vocab-path "${TXT_VOC}" \
      --mean-path "${MEAN_PATH}" \
      --output-dir "${TXT_OUT_DIR}" \
      --method cosine \
      --topk 8 \
      --min-weight 0.03 \
      --batch-size 4096 \
      --device "${DEVICE%%:*}"
    mv "${TXT_OUT_DIR}/${DOC}.jsonl" "${TXT_LAB}"
  fi

  # Stage 3: merge text+visual
  if [[ "${SKIP_EXISTING}" -eq 1 && -f "${MERGED}" ]]; then
    echo "skip merge (exists): ${MERGED}"
  else
    echo "+ python scripts/merge_patch_labels_dualpass.py --text-labels-path ${TXT_LAB} ..."
    python scripts/merge_patch_labels_dualpass.py \
      --text-labels-path "${TXT_LAB}" \
      --visual-labels-path "${VIS_LAB}" \
      --output-path "${MERGED}" \
      --merge-mode concat \
      --text-weight 1.0 \
      --visual-weight 1.2 \
      --topk 10 \
      --topk-text 8 \
      --topk-visual 8 \
      --min-weight 0.0 \
      --min-weight-text 0.03 \
      --min-weight-visual 0.0 \
      --preserve-component-scores \
      --include-visual-only
  fi

  # Stage 4: render pages for OCR
  mkdir -p "${IMG_DIR}"
  if [[ "${SKIP_EXISTING}" -eq 1 && -f "${IMG_DIR}/${DOC}_0.png" ]]; then
    echo "skip PDF render (images exist): ${IMG_DIR}"
  else
    echo "+ render PDF pages -> ${IMG_DIR}"
    python - <<PY
from pdf2image import convert_from_path
doc="${DOC}"
pdf="${PDF_PATH}"
img_dir="${IMG_DIR}"
pages = convert_from_path(pdf, dpi=144)
for i, im in enumerate(pages):
    im.save(f"{img_dir}/{doc}_{i}.png")
print("rendered_pages", len(pages))
PY
  fi

  # Number of pages from merged labels.
  NUM_PAGES="$(python - <<PY
import json
pages=set()
for line in open("${MERGED}"):
    row=json.loads(line)
    pid=str(row.get("page_id","")).strip()
    if pid:
        pages.add(pid)
print(len(pages))
PY
)"
  if [[ -z "${NUM_PAGES}" || "${NUM_PAGES}" -le 0 ]]; then
    echo "ERROR: could not infer page count from ${MERGED}" >&2
    exit 1
  fi

  # Stage 5: OCR extraction
  if [[ "${SKIP_EXISTING}" -eq 1 && -f "${OCR_JSONL}" ]]; then
    echo "skip OCR extraction (exists): ${OCR_JSONL}"
  else
    echo "+ python scripts/extract_ocr_word_boxes.py --doc-id ${DOC} ..."
    python scripts/extract_ocr_word_boxes.py \
      --doc-id "${DOC}" \
      --num-pages "${NUM_PAGES}" \
      --page-image-template "${IMG_DIR}/{doc_id}_{page_index}.png" \
      --output-jsonl "${OCR_JSONL}" \
      --min-conf 35 \
      --keep-empty-pages \
      --tesseract-cmd "${TESS_CMD}"
  fi

  # Stage 6: OCR-constrained cleanup (v6b policy)
  if [[ "${SKIP_EXISTING}" -eq 1 && -f "${OCR_OUT}" ]]; then
    echo "skip OCR-constrained filtering (exists): ${OCR_OUT}"
  else
    echo "+ python scripts/apply_ocr_box_constrained_labels.py --labels-path ${MERGED} ..."
    python scripts/apply_ocr_box_constrained_labels.py \
      --labels-path "${MERGED}" \
      --ocr-jsonl "${OCR_JSONL}" \
      --output-path "${OCR_OUT}" \
      --concepts "domenico,dolce,gabbana,dolce gabbana" \
      --concept-min-score "domenico=0.16" \
      --concept-min-score "dolce=0.16" \
      --concept-min-score "gabbana=0.168" \
      --concept-min-score "dolce gabbana=0.166" \
      --concept-terms "dolce=dolce|gabbana" \
      --concept-terms "gabbana=dolce|gabbana" \
      --concept-terms "dolce gabbana=dolce|gabbana" \
      --term-match-mode all \
      --min-word-conf 45 \
      --min-overlap 0.08 \
      --expand-cells 0 \
      --keep-other-concepts \
      --summary-tsv "${OCR_SUM_TSV}" \
      --summary-json "${OCR_SUM_JSON}"
  fi

  wc -l "${TXT_LAB}" "${MERGED}" "${OCR_OUT}"
  column -t -s $'\t' "${OCR_SUM_TSV}" | head -n 30 || true
done

echo
echo "Pipeline finished for ${#DOC_IDS[@]} docs."
