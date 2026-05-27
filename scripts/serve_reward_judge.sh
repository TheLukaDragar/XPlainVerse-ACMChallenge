#!/usr/bin/env bash
# Serve Qwen3.5-4B as OpenAI-compatible API for official eval + GRPO reward calls.
#
# Used by:
#   evaluation/evaluate_val.py  (--backend openai_compatible)
#   future external_plugins/xplainverse_rewards.py (AsyncORM)
#
# Usage (on the fast GPU machine):
#   ./scripts/serve_reward_judge.sh
#   HOST=0.0.0.0 PORT=8000 ./scripts/serve_reward_judge.sh
#
# Smoke test from another host:
#   curl -s http://HOST:8000/v1/models | python3 -m json.tool

set -euo pipefail

export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

MODEL="${REWARD_MODEL:-Qwen/Qwen3.5-4B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "error: vllm not found on PATH" >&2
  exit 1
fi

echo "=== XPlainVerse reward judge (vLLM OpenAI server) ==="
echo "model:              ${MODEL}"
echo "served_model_name:  ${SERVED_MODEL_NAME}"
echo "listen:             ${HOST}:${PORT}"
echo "gpu_memory_util:    ${GPU_MEMORY_UTIL}"
echo "max_model_len:      ${MAX_MODEL_LEN}"
echo
echo "Evaluate from train machine:"
echo "  REWARD_BASE_URL=http://$(hostname -f 2>/dev/null || hostname):${PORT}/v1 \\"
echo "  REWARD_MODEL=${SERVED_MODEL_NAME} ./scripts/eval_checkpoint.sh ..."
echo

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}"
