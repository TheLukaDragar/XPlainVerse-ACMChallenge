#!/usr/bin/env bash
# Single-GPU checkpoint eval — no remote vLLM judge server required.
#
# Runs sequentially on one GPU:
#   1. Qwen3-VL infer (transformers, ~20 GB)
#   2. build_submission.py (CPU)
#   3. evaluate_val.py with Qwen3.5-4B via transformers (~8 GB)
#
# Step 3 unloads after step 1 exits, so both fit on one A100 80GB.
# Do NOT run serve_reward_judge.sh on the same GPU at the same time.
#
# Usage:
#   ./scripts/eval_checkpoint_one_gpu.sh
#   ADAPTERS=runs/vlm_sanity/v1-*/checkpoint-100 NUM_SAMPLES=16 ./scripts/eval_checkpoint_one_gpu.sh
#
# Reuse infer only:
#   SKIP_INFER=1 INFER_JSONL=runs/eval/foo/infer.jsonl ./scripts/eval_checkpoint_one_gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EVAL_BACKEND="${EVAL_BACKEND:-transformers}"
export EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
export INFER_BACKEND="${INFER_BACKEND:-transformers}"
export NUM_SAMPLES="${NUM_SAMPLES:-8}"
export QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-4}"

# Default checkpoint: latest sanity run if ADAPTERS unset.
if [[ -z "${ADAPTERS:-}" ]]; then
  WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge}"
  shopt -s nullglob
  _candidates=("${WORKSPACE_ROOT}"/runs/vlm_sanity/v*/checkpoint-*)
  shopt -u nullglob
  if [[ ${#_candidates[@]} -gt 0 ]]; then
    export ADAPTERS="${_candidates[-1]}"
    echo "Auto-selected checkpoint: ${ADAPTERS}"
  fi
fi

echo "=== One-GPU eval (sequential VLM infer → local Qwen3.5-4B judge) ==="
echo "GPU:          ${CUDA_VISIBLE_DEVICES}"
echo "eval backend: ${EVAL_BACKEND} on ${EVAL_DEVICE}"
echo "num samples:  ${NUM_SAMPLES}"
echo

exec "${SCRIPT_DIR}/eval_checkpoint.sh"
