#!/usr/bin/env bash
# GRPO for the Qwen3-VL-8B compressor (complex -> simple), optimising the
# official simple metric directly:  reward = 0.7*BERT(vs GT simple) + 0.3*SLE_norm.
#
# Local rewards only (BERTScore + SLE) — no remote judge needed.
# Warm-starts from the SFT compressor LoRA so GRPO stays on-distribution.
#
# Smoke test (1 GPU, a few steps):
#   SMOKE=1 LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=01:00:00 \
#     ./scripts/lj_gpu_exec.sh bash scripts/train_compressor_grpo_lj.sh
#
# Full run (4 GPU):
#   LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=48:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/train_compressor_grpo_lj.sh
#
# Another cluster: run inside a container with ms-swift + vLLM. BERTScore
# (microsoft/deberta-xlarge-mnli) and SLE (liamcripwell/sle-base) must be
# reachable/cached (set HF_HOME or pre-download).
set -euo pipefail

# Enter the training container if a wrapper exists and we are not already inside.
if [[ -z "${_GRPO_IN_CONTAINER:-}" && -x "${HOME}/xplainverse_exec.sh" ]]; then
  export _GRPO_IN_CONTAINER=1
  exec "${HOME}/xplainverse_exec.sh" bash "$0" "$@"
fi

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CODE_ROOT:-}" ]]; then :;
elif [[ -d "${_SCRIPT_ROOT}/research" ]]; then CODE_ROOT="${_SCRIPT_ROOT}";
elif [[ -d /workspace/XPlainVerse-ACMChallenge/research ]]; then CODE_ROOT="/workspace/XPlainVerse-ACMChallenge";
else CODE_ROOT="${HOME}/luka/code/XPlainVerse-ACMChallenge"; fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for _cuda_lib in cu13 cu12 cu121; do
  _nv="/usr/local/lib/python3.10/dist-packages/nvidia/${_cuda_lib}/lib"
  [[ -d "${_nv}" ]] && { export LD_LIBRARY_PATH="${_nv}:${LD_LIBRARY_PATH:-}"; break; }
done

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
USE_HF="${USE_HF:-true}"
SFT_ADAPTERS="${SFT_ADAPTERS:-/home/jakob/luka/runs/compressor_vl/checkpoint-10000}"

PLUGIN="${PLUGIN:-${CODE_ROOT}/research/experiments/03_grpo/compressor_reward.py}"
REWARD_FUNCS="${REWARD_FUNCS:-xpv_simple_bert xpv_simple_sle}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.7 0.3}"

TRAIN_JSONL="${TRAIN_JSONL:-${CODE_ROOT}/dataset/train_compressor_grpo.jsonl}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
PER_DEVICE_BS="${PER_DEVICE_BS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BETA="${BETA:-0.04}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.4}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-2560}"
SEED="${SEED:-42}"

_TS="$(date -u +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jakob/luka/runs/compressor_grpo/${_TS}}"

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-compressor_grpo_${_TS}}"

# --- Smoke overrides: tiny, fast, single GPU, no external services ---
if [[ "${SMOKE:-0}" == "1" ]]; then
  NPROC_PER_NODE=1
  NUM_GENERATIONS="${NUM_GENERATIONS_SMOKE:-4}"
  PER_DEVICE_BS="${PER_DEVICE_BS_SMOKE:-4}"
  GRAD_ACCUM=1
  MAX_STEPS="${MAX_STEPS_SMOKE:-4}"
  NUM_EPOCHS=1
  SAVE_STEPS=4
  TRAIN_SLICE="${TRAIN_SLICE:-64}"
  REPORT_TO=none
  VLLM_GPU_UTIL="${VLLM_GPU_UTIL_SMOKE:-0.35}"
  OUTPUT_DIR="${OUTPUT_DIR_SMOKE:-/home/jakob/luka/runs/compressor_grpo_smoke/${_TS}}"
  echo ">>> SMOKE MODE: 1 GPU, ${MAX_STEPS} steps, slice ${TRAIN_SLICE}, report=none"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC_PER_NODE-1)))}"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "error: ${TRAIN_JSONL} missing. Run:" >&2
  echo "  python3 dataset/build_compressor_grpo_jsonl.py" >&2
  exit 1
fi
TRAIN_DATASET="${TRAIN_JSONL}"
if [[ -n "${TRAIN_SLICE:-}" ]]; then
  TRAIN_DATASET="${TRAIN_JSONL}#${TRAIN_SLICE}"
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "error: 'swift' not on PATH inside the container." >&2
  exit 1
fi
if [[ "${ATTN_IMPL}" == "flash_attn" ]] && ! python3 -c "import flash_attn" 2>/dev/null; then
  echo "warning: flash_attn not importable; using sdpa." >&2
  ATTN_IMPL=sdpa
fi

ADAPTER_FLAG=()
if [[ -n "${SFT_ADAPTERS}" && -d "${SFT_ADAPTERS}" ]]; then
  ADAPTER_FLAG=(--adapters "${SFT_ADAPTERS}")
  echo "warm-start adapters: ${SFT_ADAPTERS}"
else
  echo "warning: SFT_ADAPTERS not found (${SFT_ADAPTERS}); training fresh LoRA." >&2
fi

SCHEDULE_ARGS=(--num_train_epochs "${NUM_EPOCHS}")
if [[ -n "${MAX_STEPS}" ]]; then
  SCHEDULE_ARGS=(--max_steps "${MAX_STEPS}")
fi

mkdir -p "${OUTPUT_DIR}"
EFF=$((PER_DEVICE_BS * NPROC_PER_NODE * GRAD_ACCUM))
echo "=== Compressor GRPO ==="
echo "  model:        ${MODEL} (${MODEL_TYPE})"
echo "  train:        ${TRAIN_DATASET}"
echo "  reward:       ${REWARD_FUNCS}  weights=[${REWARD_WEIGHTS}]"
echo "  plugin:       ${PLUGIN}"
echo "  gpus:         ${NPROC_PER_NODE}  (CUDA=${CUDA_VISIBLE_DEVICES})"
echo "  group/batch:  num_generations=${NUM_GENERATIONS} per_device=${PER_DEVICE_BS} accum=${GRAD_ACCUM} eff=${EFF}"
echo "  lr/beta:      ${LEARNING_RATE} / ${BETA}"
echo "  vllm:         colocate util=${VLLM_GPU_UTIL} max_len=${VLLM_MAX_LEN}"
echo "  output:       ${OUTPUT_DIR}"
echo

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL}" \
  --model_type "${MODEL_TYPE}" \
  --use_hf "${USE_HF}" \
  "${ADAPTER_FLAG[@]}" \
  --train_type lora \
  --torch_dtype bfloat16 \
  --attn_impl "${ATTN_IMPL}" \
  --dataset "${TRAIN_DATASET}" \
  --external_plugins "${PLUGIN}" \
  --reward_funcs ${REWARD_FUNCS} \
  --reward_weights ${REWARD_WEIGHTS} \
  --num_generations "${NUM_GENERATIONS}" \
  --max_completion_length "${MAX_COMPLETION_LEN}" \
  --max_length "${MAX_LENGTH}" \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_GPU_UTIL}" \
  --vllm_max_model_len "${VLLM_MAX_LEN}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --beta "${BETA}" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --freeze_llm false \
  "${SCHEDULE_ARGS[@]}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 5 \
  --logging_steps "${LOGGING_STEPS}" \
  --report_to "${REPORT_TO}" \
  --seed "${SEED}" \
  --add_version false \
  --output_dir "${OUTPUT_DIR}"

echo
echo "=== Done ==="
echo "  output: ${OUTPUT_DIR}"
