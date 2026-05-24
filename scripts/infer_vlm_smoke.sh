#!/usr/bin/env bash
# Smoke infer on a few val rows with a LoRA checkpoint from train_vlm_sanity/full.
#
# IMPORTANT: pass --use_hf true so infer reuses the HuggingFace cache from training.
# Without it, ms-swift defaults to ModelScope and re-downloads ~17GB.
#
# Usage:
#   ADAPTERS=runs/vlm_sanity/v0-*/checkpoint-100 ./scripts/infer_vlm_smoke.sh
#   NUM_SAMPLES=8 ADAPTERS=runs/vlm_full/v0-*/checkpoint-4000 ./scripts/infer_vlm_smoke.sh

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge}"
CODE_ROOT="${CODE_ROOT:-${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge}"

export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
USE_HF="${USE_HF:-true}"
VAL_JSONL="${VAL_JSONL:-${CODE_ROOT}/dataset/val_vlm.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
INFER_BACKEND="${INFER_BACKEND:-transformers}"
RESULT_PATH="${RESULT_PATH:-${WORKSPACE_ROOT}/runs/vlm_infer_smoke.jsonl}"

ADAPTERS="${ADAPTERS:-}"
if [[ -z "${ADAPTERS}" ]]; then
  echo "error: set ADAPTERS to a checkpoint dir, e.g.:" >&2
  echo "  ADAPTERS=runs/vlm_sanity/v0-*/checkpoint-100 ./scripts/infer_vlm_smoke.sh" >&2
  exit 1
fi

# Expand globs if user passed vx-*/checkpoint-* .
if [[ "${ADAPTERS}" == *"*"* ]]; then
  shopt -s nullglob
  _matches=(${ADAPTERS})
  shopt -u nullglob
  if [[ ${#_matches[@]} -eq 0 ]]; then
    echo "error: no checkpoint matched: ${ADAPTERS}" >&2
    exit 1
  fi
  ADAPTERS="${_matches[-1]}"
fi

if [[ ! -d "${ADAPTERS}" ]]; then
  echo "error: adapters dir not found: ${ADAPTERS}" >&2
  exit 1
fi

mkdir -p "$(dirname "${RESULT_PATH}")"

echo "=== VLM infer smoke ==="
echo "model:       ${MODEL} (use_hf=${USE_HF})"
echo "adapters:    ${ADAPTERS}"
echo "val:         ${VAL_JSONL}#${NUM_SAMPLES}"
echo "result:      ${RESULT_PATH}"
echo "gpu:         ${CUDA_VISIBLE_DEVICES}"
echo

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift infer \
  --model "${MODEL}" \
  --model_type "${MODEL_TYPE}" \
  --use_hf "${USE_HF}" \
  --adapters "${ADAPTERS}" \
  --val_dataset "${VAL_JSONL}#${NUM_SAMPLES}" \
  --infer_backend "${INFER_BACKEND}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --result_path "${RESULT_PATH}"

echo
echo "Done. Inspect: ${RESULT_PATH}"
