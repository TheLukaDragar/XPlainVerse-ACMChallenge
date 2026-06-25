#!/usr/bin/env bash
# Build v2 ms-swift training JSONL on Ljubljana (elixir-lj) inside Apptainer.
#
# Stage-2 (prompt/data v2) dataset prep. Produces, under dataset/:
#   train_vlm_v2.jsonl        filtered + ratio-enforced + 50%-hypothetical train
#   train_vlm_infer_v2.jsonl  same rows, user-only
#   val_vlm_v2.jsonl          val rows (primary prompt, for eval_steps)
#   val_vlm_infer_v2.jsonl    ALL val rows, user-only (submission inference)
#
# The build is pure-stdlib Python (no torch) but reads ~560k manifest rows +
# explanation JSON files, so run it inside the container on a compute node —
# never on the Slurm login node (see .cursor/rules/xplainverse-lj-node.mdc).
#
# === Launch from the Slurm login node ===
#   LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=02:00:00 \
#     LJ_APPTAINER_IMAGE=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge-lj:latest \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/build_train_v2_lj.sh
#
# === Already inside the container (GPU node) ===
#   bash scripts/build_train_v2_lj.sh
#
# Override knobs:
#   HYPOTHETICAL_RATIO=0.6 FAKE_REAL_RATIO=3 bash scripts/build_train_v2_lj.sh
#   MEASURE_ONLY=1 MAX_ROWS=2000 bash scripts/build_train_v2_lj.sh   # dry-run stats
set -euo pipefail

# --- Code root: prefer the repo that contains this script (bind-mounted HOME),
# so output JSONL never lands in a baked-in /workspace inside the GHCR image. ---
_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CODE_ROOT:-}" ]]; then
  :
elif [[ -d "${_SCRIPT_ROOT}/dataset" ]]; then
  CODE_ROOT="${_SCRIPT_ROOT}"
elif [[ -d "${HOME}/luka/code/XPlainVerse-ACMChallenge/dataset" ]]; then
  CODE_ROOT="${HOME}/luka/code/XPlainVerse-ACMChallenge"
else
  CODE_ROOT="/workspace/XPlainVerse-ACMChallenge"
fi

# --- Data root (images + per-split manifest.jsonl) ---
if [[ -n "${XPLAINVERSE_DATA_ROOT:-}" ]]; then
  DATA_ROOT="${XPLAINVERSE_DATA_ROOT}"
elif [[ -d /primoz/luka/XPlainVerse/data/XPlainVerse/train ]]; then
  DATA_ROOT="/primoz/luka/XPlainVerse/data/XPlainVerse"
else
  DATA_ROOT="${HOME}/luka/data/XPlainVerse"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${CODE_ROOT}/dataset}"
PROMPT_FILE="${PROMPT_FILE:-${CODE_ROOT}/dataset/prompt_v2.txt}"

HYPOTHETICAL_RATIO="${HYPOTHETICAL_RATIO:-0.50}"
FAKE_REAL_RATIO="${FAKE_REAL_RATIO:-4.0}"
FILTER_MIN_SENTENCES="${FILTER_MIN_SENTENCES:-3}"
FILTER_MIN_WORDS="${FILTER_MIN_WORDS:-80}"
FILTER_MIN_CONNECTIVES="${FILTER_MIN_CONNECTIVES:-0}"
SEED="${SEED:-42}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

EXTRA_ARGS=()
if [[ -n "${MEASURE_ONLY:-}" && "${MEASURE_ONLY}" != "0" ]]; then
  EXTRA_ARGS+=(--measure-only)
fi
if [[ -n "${MAX_ROWS:-}" && "${MAX_ROWS}" != "0" ]]; then
  EXTRA_ARGS+=(--max-rows "${MAX_ROWS}")
fi
if [[ -n "${SAMPLE_EXAMPLES:-}" && "${SAMPLE_EXAMPLES}" != "0" ]]; then
  EXTRA_ARGS+=(--sample-examples "${SAMPLE_EXAMPLES}")
fi

if [[ ! -d "${DATA_ROOT}/train" ]]; then
  echo "error: data root missing train split: ${DATA_ROOT}/train" >&2
  echo "  set XPLAINVERSE_DATA_ROOT or place data at ${HOME}/luka/data/XPlainVerse" >&2
  exit 1
fi
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "error: prompt file missing: ${PROMPT_FILE}" >&2
  exit 1
fi

echo "=== Build v2 training JSONL (lj / Apptainer) ==="
echo "code_root:        ${CODE_ROOT}"
echo "data_root:        ${DATA_ROOT}"
echo "output_dir:       ${OUTPUT_DIR}"
echo "prompt_file:      ${PROMPT_FILE}"
echo "hypothetical:     ${HYPOTHETICAL_RATIO}"
echo "fake_real_ratio:  ${FAKE_REAL_RATIO}"
echo "filter:           sentences>=${FILTER_MIN_SENTENCES} words>=${FILTER_MIN_WORDS} connectives>=${FILTER_MIN_CONNECTIVES}"
echo "extra:            ${EXTRA_ARGS[*]:-<none>}"
echo "started:          $(date -Is)"
echo

python3 "${CODE_ROOT}/dataset/build_swift_jsonl_v2.py" \
  --splits train val \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --prompt-file "${PROMPT_FILE}" \
  --hypothetical-ratio "${HYPOTHETICAL_RATIO}" \
  --fake-real-ratio "${FAKE_REAL_RATIO}" \
  --filter-min-sentences "${FILTER_MIN_SENTENCES}" \
  --filter-min-words "${FILTER_MIN_WORDS}" \
  --filter-min-connectives "${FILTER_MIN_CONNECTIVES}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

echo
echo "=== done at $(date -Is) ==="
if [[ -z "${MEASURE_ONLY:-}" || "${MEASURE_ONLY}" == "0" ]]; then
  ls -la "${OUTPUT_DIR}/train_vlm_v2.jsonl" "${OUTPUT_DIR}/val_vlm_v2.jsonl" 2>&1 || true
  echo "row counts:"
  wc -l "${OUTPUT_DIR}/train_vlm_v2.jsonl" "${OUTPUT_DIR}/val_vlm_v2.jsonl" 2>&1 || true
fi
