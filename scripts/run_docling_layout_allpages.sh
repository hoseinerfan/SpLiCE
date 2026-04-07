#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run Docling layout labeling + per-page finalization/overlays for one or more docs.

Defaults are tuned to the latest "table fill" settings:
- Docling labels with visual regions enabled
- min-overlap-table=0.06
- min-overlap-visual=0.06
- Finalize with table filling/rect fill enabled

Usage (single doc):
  scripts/run_docling_layout_allpages.sh \
    --doc-id becae20d9ea3609df085dd316335cefc

Usage (multiple docs):
  scripts/run_docling_layout_allpages.sh \
    --doc-ids-file /tmp/doc_ids.txt

Optional:
  --output-root /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings
  --pdf-root /mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/splits/pdfs_dev
  --run-tag allpages_docling_tfill_YYYYMMDD_HHMMSS
  --docling-device cpu
  --docling-page-number-base one
  --image-source docling
  --rectangularize-selected-regions
  --rectangularize-min-cells 1
  --rectangularize-max-area-frac 1.0
  --skip-missing
  --no-include-visual-region
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DOC_ID=""
DOC_IDS_FILE=""
OUTPUT_ROOT="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings"
PDF_ROOT="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/splits/pdfs_dev"
RUN_TAG="allpages_docling_tfill_$(date +%Y%m%d_%H%M%S)"

DOCLING_DEVICE="cpu"
DOCLING_PAGE_NUMBER_BASE="one"
INCLUDE_VISUAL_REGION=1

MIN_OVERLAP_TEXT="0.08"
MIN_OVERLAP_TABLE="0.06"
MIN_OVERLAP_VISUAL="0.06"

IMAGE_SOURCE="docling"
IMAGE_TOKEN_START="0"
IMAGE_TOKEN_COUNT="1024"
GRID_SIZE="32"

TABLE_EXPAND_FROM_STRUCTURE="1"
TABLE_GAP_FILL_PASSES="2"
TABLE_GAP_FILL_MIN_NEIGHBORS="2"
TABLE_RECT_FILL_MIN_STRUCTURE_CELLS="25"
TABLE_RECT_FILL_MAX_AREA_FRAC="0.60"
TABLE_COMPONENT_RECT_MIN_CELLS="40"
TABLE_COMPONENT_RECT_MAX_AREA_FRAC="0.60"
RECTANGULARIZE_SELECTED_REGIONS=0
RECTANGULARIZE_MIN_CELLS="1"
RECTANGULARIZE_MAX_AREA_FRAC="1.0"

SKIP_MISSING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --doc-id) DOC_ID="$2"; shift 2 ;;
    --doc-ids-file) DOC_IDS_FILE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --pdf-root) PDF_ROOT="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --docling-device) DOCLING_DEVICE="$2"; shift 2 ;;
    --docling-page-number-base) DOCLING_PAGE_NUMBER_BASE="$2"; shift 2 ;;
    --image-source) IMAGE_SOURCE="$2"; shift 2 ;;
    --rectangularize-selected-regions) RECTANGULARIZE_SELECTED_REGIONS=1; shift ;;
    --rectangularize-min-cells) RECTANGULARIZE_MIN_CELLS="$2"; shift 2 ;;
    --rectangularize-max-area-frac) RECTANGULARIZE_MAX_AREA_FRAC="$2"; shift 2 ;;
    --skip-missing) SKIP_MISSING=1; shift ;;
    --no-include-visual-region) INCLUDE_VISUAL_REGION=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${DOC_ID}" && -z "${DOC_IDS_FILE}" ]]; then
  echo "ERROR: provide --doc-id or --doc-ids-file" >&2
  usage
  exit 2
fi
if [[ -n "${DOC_ID}" && -n "${DOC_IDS_FILE}" ]]; then
  echo "ERROR: use only one of --doc-id or --doc-ids-file" >&2
  exit 2
fi

DOC_IDS=()
if [[ -n "${DOC_ID}" ]]; then
  DOC_IDS+=("${DOC_ID}")
else
  if [[ ! -f "${DOC_IDS_FILE}" ]]; then
    echo "ERROR: doc ids file not found: ${DOC_IDS_FILE}" >&2
    exit 2
  fi
  mapfile -t DOC_IDS < <(grep -v '^[[:space:]]*$' "${DOC_IDS_FILE}" | sed 's/[[:space:]]*$//')
  if [[ "${#DOC_IDS[@]}" -eq 0 ]]; then
    echo "ERROR: no doc ids in ${DOC_IDS_FILE}" >&2
    exit 2
  fi
fi

echo "Docs: ${#DOC_IDS[@]}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run tag: ${RUN_TAG}"

for DOC in "${DOC_IDS[@]}"; do
  echo
  echo "===================="
  echo "DOC: ${DOC}"
  echo "===================="

  CTX="${OUTPUT_ROOT}/debug_${DOC}_linked_context_auto"
  PDF="${PDF_ROOT}/${DOC}.pdf"
  IMG_ROOT="${OUTPUT_ROOT}/debug_${DOC}_page_pngs_144"
  BASE_LABELS="${CTX}/doc_seed_text_patch_labels.jsonl"

  OUTDIR="${CTX}/${RUN_TAG}"
  mkdir -p "${OUTDIR}"

  if [[ ! -f "${BASE_LABELS}" || ! -f "${PDF}" ]]; then
    msg="Missing prereq(s): labels=${BASE_LABELS} pdf=${PDF}"
    if [[ "${SKIP_MISSING}" -eq 1 ]]; then
      echo "skip: ${msg}"
      continue
    else
      echo "ERROR: ${msg}" >&2
      exit 1
    fi
  fi

  DOCLING_LABELS="${OUTDIR}/doc_seed_layout_patch_labels_docling_tfill.jsonl"
  DOCLING_SUM_JSON="${OUTDIR}/doc_seed_layout_patch_labels_docling_tfill_summary.json"
  DOCLING_SUM_TSV="${OUTDIR}/doc_seed_layout_patch_labels_docling_tfill_summary.tsv"
  DOCLING_ZONES_JSONL="${OUTDIR}/docling_tfill_zones.jsonl"

  BUILD_CMD=(
    python scripts/build_layout_patch_labels_docling.py
    --labels-jsonl "${BASE_LABELS}"
    --pdf-path "${PDF}"
    --output-jsonl "${DOCLING_LABELS}"
    --min-overlap-text "${MIN_OVERLAP_TEXT}"
    --min-overlap-table "${MIN_OVERLAP_TABLE}"
    --min-overlap-visual "${MIN_OVERLAP_VISUAL}"
    --docling-device "${DOCLING_DEVICE}"
    --docling-page-number-base "${DOCLING_PAGE_NUMBER_BASE}"
    --summary-json "${DOCLING_SUM_JSON}"
    --summary-tsv "${DOCLING_SUM_TSV}"
    --zones-debug-json "${DOCLING_ZONES_JSONL}"
  )
  if [[ "${INCLUDE_VISUAL_REGION}" -eq 1 ]]; then
    BUILD_CMD+=(--include-visual-region)
  fi
  echo "+ ${BUILD_CMD[*]}"
  "${BUILD_CMD[@]}"

  PAGES="$(
    python - <<PY
import json
print(int(json.load(open("${DOCLING_SUM_JSON}"))["pages"]))
PY
  )"
  if [[ -z "${PAGES}" || "${PAGES}" -le 0 ]]; then
    echo "ERROR: invalid page count from ${DOCLING_SUM_JSON}" >&2
    exit 1
  fi

  for PAGE_IDX in $(seq 0 $((PAGES - 1))); do
    PAGE_ID="${DOC}:${PAGE_IDX}"
    IMG="${IMG_ROOT}/${DOC}_${PAGE_IDX}.png"

    if [[ ! -f "${IMG}" ]]; then
      msg="missing page image: ${IMG}"
      if [[ "${SKIP_MISSING}" -eq 1 ]]; then
        echo "skip page ${PAGE_IDX}: ${msg}"
        continue
      else
        echo "ERROR: ${msg}" >&2
        exit 1
      fi
    fi

    cp "${IMG}" "${OUTDIR}/${DOC}_page${PAGE_IDX}_raw.png"

    FIN_CMD=(
      python scripts/finalize_layout_page_strict.py
      --labels-jsonl "${DOCLING_LABELS}"
      --output-jsonl "${OUTDIR}/${DOC}_page${PAGE_IDX}_labels_tfill.jsonl"
      --pdf-path "${PDF}"
      --page-id "${PAGE_ID}"
      --grid-size "${GRID_SIZE}"
      --image-token-start "${IMAGE_TOKEN_START}"
      --image-token-count "${IMAGE_TOKEN_COUNT}"
      --image-source "${IMAGE_SOURCE}"
      --table-expand-from-structure "${TABLE_EXPAND_FROM_STRUCTURE}"
      --table-gap-fill-passes "${TABLE_GAP_FILL_PASSES}"
      --table-gap-fill-min-neighbors "${TABLE_GAP_FILL_MIN_NEIGHBORS}"
      --table-rect-fill-from-structure
      --table-rect-fill-min-structure-cells "${TABLE_RECT_FILL_MIN_STRUCTURE_CELLS}"
      --table-rect-fill-max-area-frac "${TABLE_RECT_FILL_MAX_AREA_FRAC}"
      --table-component-rect-fill
      --table-component-rect-min-cells "${TABLE_COMPONENT_RECT_MIN_CELLS}"
      --table-component-rect-max-area-frac "${TABLE_COMPONENT_RECT_MAX_AREA_FRAC}"
      --summary-json "${OUTDIR}/${DOC}_page${PAGE_IDX}_summary_tfill.json"
      --overlay-image "${IMG}"
      --overlay-output "${OUTDIR}/${DOC}_page${PAGE_IDX}_overlay_tfill.png"
    )
    if [[ "${RECTANGULARIZE_SELECTED_REGIONS}" -eq 1 ]]; then
      FIN_CMD+=(
        --rectangularize-selected-regions
        --rectangularize-min-cells "${RECTANGULARIZE_MIN_CELLS}"
        --rectangularize-max-area-frac "${RECTANGULARIZE_MAX_AREA_FRAC}"
      )
    fi
    echo "+ ${FIN_CMD[*]}"
    "${FIN_CMD[@]}"
  done

  echo "Wrote all outputs to: ${OUTDIR}"
  ls -lh "${OUTDIR}"/*overlay_tfill.png || true
done
