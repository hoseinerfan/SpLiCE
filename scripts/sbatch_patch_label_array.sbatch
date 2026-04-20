#!/bin/bash
#SBATCH --job-name=splice_patch_batches
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

usage() {
  cat <<'EOF'
Run batched docseed + Docling patch labeling in a Slurm array.

Expected batch root contents:
- batch_00.txt
- batch_01.txt
- ...

Each batch file should contain one doc id per line.

Typical usage:
  sbatch --array=0-9%4 scripts/sbatch_patch_label_array.sbatch \
    --batch-root /mmfs1/scratch/jacks.local/aerfanshekooh/custom/slurm/patch_labels_remaining \
    --repo-root /mmfs1/scratch/jacks.local/aerfanshekooh/repos/SpLiCE

Optional overrides:
  --batch-pattern 'batch_*.txt'
  --output-root /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs
  --embeddings-output-root /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings
  --embeddings-root /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali-v1.2_m3-docvqa_dev
  --visual-labels-root /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/patch_labels_visual_attr_human_strict_v1_t10_w005
  --mean-path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/means/colpali_mean_max.pt
  --pdf-root /mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/splits/pdfs_dev
  --mmqa-dev-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl
  --mmqa-texts-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_texts.jsonl
  --mmqa-tables-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_tables.jsonl
  --local-colpali-model /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/model/colpali-v1.2
  --docling-device cpu
  --device cuda:0
  --logs-dir /mmfs1/scratch/jacks.local/aerfanshekooh/custom/slurm/patch_labels_remaining/logs

Notes:
- The script uses the current repo checkout to run the pipeline.
- It assumes the repo-relative virtualenv exists at m3docrag/.venv.
- It always uses --skip-ocr-stages and --skip-existing for the docseed stage.
- Under Slurm, the script prefers SLURM_SUBMIT_DIR and also accepts --repo-root to avoid resolving the repo from the staged spool copy.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${DEFAULT_REPO_ROOT}}"

BATCH_ROOT=""
BATCH_PATTERN="batch_*.txt"
EMBEDDINGS_OUTPUT_ROOT="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings"
OUTPUT_ROOT="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs"
EMBEDDINGS_ROOT="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali-v1.2_m3-docvqa_dev"
VISUAL_LABELS_ROOT="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/patch_labels_visual_attr_human_strict_v1_t10_w005"
MEAN_PATH="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/means/colpali_mean_max.pt"
PDF_ROOT="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/splits/pdfs_dev"
MMQA_DEV_JSONL="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl"
MMQA_TEXTS_JSONL="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_texts.jsonl"
MMQA_TABLES_JSONL="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_tables.jsonl"
LOCAL_COLPALI_MODEL="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/model/colpali-v1.2"
DOCLING_DEVICE="cpu"
DEVICE="cuda:0"
LOGS_DIR=""
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-root) BATCH_ROOT="$2"; shift 2 ;;
    --batch-pattern) BATCH_PATTERN="$2"; shift 2 ;;
    --embeddings-output-root) EMBEDDINGS_OUTPUT_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --embeddings-root) EMBEDDINGS_ROOT="$2"; shift 2 ;;
    --visual-labels-root) VISUAL_LABELS_ROOT="$2"; shift 2 ;;
    --mean-path) MEAN_PATH="$2"; shift 2 ;;
    --pdf-root) PDF_ROOT="$2"; shift 2 ;;
    --mmqa-dev-jsonl) MMQA_DEV_JSONL="$2"; shift 2 ;;
    --mmqa-texts-jsonl) MMQA_TEXTS_JSONL="$2"; shift 2 ;;
    --mmqa-tables-jsonl) MMQA_TABLES_JSONL="$2"; shift 2 ;;
    --local-colpali-model) LOCAL_COLPALI_MODEL="$2"; shift 2 ;;
    --docling-device) DOCLING_DEVICE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --logs-dir) LOGS_DIR="$2"; shift 2 ;;
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --task-index) TASK_INDEX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${BATCH_ROOT}" ]]; then
  echo "ERROR: --batch-root is required" >&2
  usage
  exit 2
fi
if [[ ! -d "${REPO_ROOT}/scripts" ]]; then
  echo "ERROR: repo root is invalid: ${REPO_ROOT}" >&2
  exit 2
fi
cd "${REPO_ROOT}"
if [[ ! -d "${BATCH_ROOT}" ]]; then
  echo "ERROR: batch root not found: ${BATCH_ROOT}" >&2
  exit 2
fi

mapfile -t BATCH_FILES < <(find "${BATCH_ROOT}" -maxdepth 1 -name "${BATCH_PATTERN}" | sort)
if [[ "${#BATCH_FILES[@]}" -eq 0 ]]; then
  echo "ERROR: no batch files found in ${BATCH_ROOT} matching ${BATCH_PATTERN}" >&2
  exit 1
fi
if ! [[ "${TASK_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: task index must be an integer, got: ${TASK_INDEX}" >&2
  exit 2
fi
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#BATCH_FILES[@]} )); then
  echo "ERROR: task index ${TASK_INDEX} is out of range for ${#BATCH_FILES[@]} batch files" >&2
  exit 2
fi

BATCH_FILE="${BATCH_FILES[${TASK_INDEX}]}"
BATCH_NAME="$(basename "${BATCH_FILE%.txt}")"

if [[ -n "${LOGS_DIR}" ]]; then
  mkdir -p "${LOGS_DIR}"
  LOG_BASE="${LOGS_DIR}/${BATCH_NAME}"
  exec > >(tee -a "${LOG_BASE}.out") 2> >(tee -a "${LOG_BASE}.err" >&2)
fi

if [[ -f "${REPO_ROOT}/m3docrag/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/m3docrag/.venv/bin/activate"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/.cache"
export HF_HOME="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/.cache/huggingface"
export HF_HUB_CACHE="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/.cache/huggingface/hub"
export TORCH_HOME="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/.cache/torch"
export TMPDIR="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/.cache/tmp"

mkdir -p "${XDG_CACHE_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" "${TMPDIR}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "TASK_INDEX=${TASK_INDEX}"
echo "BATCH_FILE=${BATCH_FILE}"
echo "BATCH_NAME=${BATCH_NAME}"
echo "DOC_COUNT=$(grep -vc '^[[:space:]]*$' "${BATCH_FILE}")"

bash scripts/run_docseed_full_pipeline_batch.sh \
  --doc-ids-file "${BATCH_FILE}" \
  --output-root "${EMBEDDINGS_OUTPUT_ROOT}" \
  --embeddings-root "${EMBEDDINGS_ROOT}" \
  --visual-labels-root "${VISUAL_LABELS_ROOT}" \
  --mean-path "${MEAN_PATH}" \
  --pdf-root "${PDF_ROOT}" \
  --mmqa-dev-jsonl "${MMQA_DEV_JSONL}" \
  --mmqa-texts-jsonl "${MMQA_TEXTS_JSONL}" \
  --mmqa-tables-jsonl "${MMQA_TABLES_JSONL}" \
  --model-name "${LOCAL_COLPALI_MODEL}" \
  --device "${DEVICE}" \
  --skip-ocr-stages \
  --skip-existing

bash scripts/run_docling_layout_allpages.sh \
  --doc-ids-file "${BATCH_FILE}" \
  --input-root "${EMBEDDINGS_OUTPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --pdf-root "${PDF_ROOT}" \
  --docling-device "${DOCLING_DEVICE}" \
  --image-source docling \
  --rectangularize-selected-regions \
  --no-assign-background-class \
  --skip-missing

python scripts/export_layout_patch_assignments.py \
  --doc-ids-file "${BATCH_FILE}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag latest \
  --output-json "${OUTPUT_ROOT}/layout_patch_assignments_${BATCH_NAME}.json"

python - <<PY
import json
from pathlib import Path

doc_ids = [x.strip() for x in open(${BATCH_FILE@Q}) if x.strip()]
input_root = Path(${EMBEDDINGS_OUTPUT_ROOT@Q})
output_root = Path(${OUTPUT_ROOT@Q})

def latest_run_dir(ctx, doc_id):
    runs = []
    if not ctx.is_dir():
        return None
    for child in ctx.iterdir():
        if child.is_dir() and any(child.glob(f"{doc_id}_page*_labels_tfill.jsonl")):
            runs.append(child)
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]

prereq_ready = 0
layout_ready = 0
pages_expected = 0
pages_done = 0

for doc_id in doc_ids:
    base_labels = input_root / f"debug_{doc_id}_linked_context_auto" / "doc_seed_text_patch_labels.jsonl"
    img_dir = input_root / f"debug_{doc_id}_page_pngs_144"
    has_prereq = base_labels.is_file() and img_dir.is_dir() and any(img_dir.glob(f"{doc_id}_*.png"))
    prereq_ready += int(has_prereq)

    run_dir = latest_run_dir(output_root / f"debug_{doc_id}_linked_context_auto", doc_id)
    if run_dir is None:
        continue

    done = len(list(run_dir.glob(f"{doc_id}_page*_labels_tfill.jsonl")))
    exp = done
    summ = run_dir / "doc_seed_layout_patch_labels_docling_tfill_summary.json"
    if summ.is_file():
        exp = int(json.load(open(summ))["pages"])

    if done == exp and exp > 0:
        layout_ready += 1
    pages_expected += exp
    pages_done += done

print("batch_docs =", len(doc_ids))
print("prereq_ready =", prereq_ready)
print("layout_ready =", layout_ready)
print("pages_expected =", pages_expected)
print("pages_done =", pages_done)
print("pages_missing =", pages_expected - pages_done)
PY

echo "Finished batch ${BATCH_NAME}"
