#!/usr/bin/env bash
# End-to-end checkpoint eval: infer → submission → official evaluate_val.py
#
# Prerequisites:
#   - LoRA checkpoint from train_vlm_sanity / train_vlm_full
#   - Reward judge: either remote vLLM (REWARD_BASE_URL) or local GPU (EVAL_BACKEND=transformers)
#
# Usage:
#   ADAPTERS=runs/vlm_sanity/v1-*/checkpoint-100 ./scripts/eval_checkpoint.sh
#   NUM_SAMPLES=100 ADAPTERS=... ./scripts/eval_checkpoint.sh
#
# Remote judge (recommended on fast machine):
#   REWARD_BASE_URL=http://REWARD_HOST:8000/v1 ./scripts/eval_checkpoint.sh ...
#
# Skip infer (reuse existing infer JSONL):
#   INFER_JSONL=runs/eval/foo/infer.jsonl SKIP_INFER=1 ./scripts/eval_checkpoint.sh

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge}"
CODE_ROOT="${CODE_ROOT:-${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge}"

export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

# --- infer ---
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
USE_HF="${USE_HF:-true}"
VAL_JSONL="${VAL_JSONL:-${CODE_ROOT}/dataset/val_vlm_infer.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
INFER_BACKEND="${INFER_BACKEND:-transformers}"
SKIP_INFER="${SKIP_INFER:-0}"

# --- submission ---
SIMPLE_MODE="${SIMPLE_MODE:-match_verdict}"

# --- evaluate_val.py ---
GROUND_TRUTH="${GROUND_TRUTH:-${CODE_ROOT}/evaluation/data/val_ground_truth.jsonl}"
EVAL_BACKEND="${EVAL_BACKEND:-openai_compatible}"
REWARD_BASE_URL="${REWARD_BASE_URL:-http://localhost:8000/v1}"
REWARD_MODEL="${REWARD_MODEL:-Qwen/Qwen3.5-4B}"
QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-8}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"

ADAPTERS="${ADAPTERS:-}"
if [[ "${SKIP_INFER}" != "1" && -z "${ADAPTERS}" ]]; then
  echo "error: set ADAPTERS to a checkpoint dir, or SKIP_INFER=1 with INFER_JSONL" >&2
  exit 1
fi

if [[ -n "${ADAPTERS}" && "${ADAPTERS}" == *"*"* ]]; then
  shopt -s nullglob
  _matches=(${ADAPTERS})
  shopt -u nullglob
  if [[ ${#_matches[@]} -eq 0 ]]; then
    echo "error: no checkpoint matched: ${ADAPTERS}" >&2
    exit 1
  fi
  ADAPTERS="${_matches[-1]}"
fi

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
CKPT_NAME="${CKPT_NAME:-$(basename "${ADAPTERS:-manual}")}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/runs/eval/${CKPT_NAME}_${RUN_TAG}}"
INFER_JSONL="${INFER_JSONL:-${OUTPUT_DIR}/infer.jsonl}"
SUBMISSION_JSONL="${SUBMISSION_JSONL:-${OUTPUT_DIR}/submission.jsonl}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_DIR}/eval_results}"

mkdir -p "${OUTPUT_DIR}"

echo "=== XPlainVerse checkpoint eval ==="
echo "adapters:     ${ADAPTERS:-<skip>}"
echo "val infer:    ${VAL_JSONL}#${NUM_SAMPLES}"
echo "output dir:   ${OUTPUT_DIR}"
echo "eval backend: ${EVAL_BACKEND}"
if [[ "${EVAL_BACKEND}" == "openai_compatible" ]]; then
  echo "reward url:   ${REWARD_BASE_URL}"
  echo "reward model: ${REWARD_MODEL}"
fi
echo

# --- Step 1: infer ---
if [[ "${SKIP_INFER}" != "1" ]]; then
  if [[ ! -d "${ADAPTERS}" ]]; then
    echo "error: adapters dir not found: ${ADAPTERS}" >&2
    exit 1
  fi
  echo ">>> [1/3] Infer (${NUM_SAMPLES} val rows)..."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  swift infer \
    --model "${MODEL}" \
    --model_type "${MODEL_TYPE}" \
    --use_hf "${USE_HF}" \
    --adapters "${ADAPTERS}" \
    --val_dataset "${VAL_JSONL}#${NUM_SAMPLES}" \
    --infer_backend "${INFER_BACKEND}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --result_path "${INFER_JSONL}"
else
  if [[ ! -f "${INFER_JSONL}" ]]; then
    echo "error: SKIP_INFER=1 but INFER_JSONL not found: ${INFER_JSONL}" >&2
    exit 1
  fi
  echo ">>> [1/3] Infer skipped — using ${INFER_JSONL}"
fi

# --- Step 2: submission ---
echo ">>> [2/3] Build submission (simple_mode=${SIMPLE_MODE})..."
(
  cd "${CODE_ROOT}/evaluation"
  python3 build_submission.py \
    --infer "${INFER_JSONL}" \
    --output "${SUBMISSION_JSONL}" \
    --simple-mode "${SIMPLE_MODE}" \
    --errors-json "${OUTPUT_DIR}/submission_errors.json"
)

# --- Step 3: official eval ---
echo ">>> [3/3] Official evaluate_val.py..."
EVAL_ARGS=(
  --submission "${SUBMISSION_JSONL}"
  --ground-truth "${GROUND_TRUTH}"
  --output-dir "${EVAL_DIR}"
  --backend "${EVAL_BACKEND}"
  --model-name "${REWARD_MODEL}"
  --qwen-batch-size "${QWEN_BATCH_SIZE}"
)

if [[ "${EVAL_BACKEND}" == "openai_compatible" ]]; then
  EVAL_ARGS+=(--base-url "${REWARD_BASE_URL}")
else
  EVAL_ARGS+=(--device-map "${EVAL_DEVICE}")
fi

(
  cd "${CODE_ROOT}/evaluation"
  python3 evaluate_val.py "${EVAL_ARGS[@]}"
)

echo
echo "=== Done ==="
echo "infer:       ${INFER_JSONL}"
echo "submission:  ${SUBMISSION_JSONL}"
echo "scores:      ${EVAL_DIR}/final_scores.json"
echo
if command -v python3 >/dev/null 2>&1; then
  python3 - "${EVAL_DIR}/final_scores.json" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    s = json.load(f)
keys = [
    "samples_completed",
    "detection_f1",
    "complex_overall_score",
    "complex_entity_f1",
    "complex_evidence_f1",
    "complex_bert_f1",
    "simple_overall_score",
    "explanation_score",
]
for k in keys:
    if k in s:
        print(f"  {k}: {s[k]}")
PY
fi
