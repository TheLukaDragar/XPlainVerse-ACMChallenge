#!/usr/bin/env bash
# Frida ms-swift GRPO for Qwen3-VL-8B compressor (complex -> simple).
#
# Reward = 0.7·BERT(vs GT simple) + 0.3·SLE_norm — see
# research/experiments/03_grpo/compressor_reward.py
#
# Warm-starts from compressor_vl SFT LoRA. Requires train_compressor_grpo.jsonl
# (python3 dataset/build_compressor_grpo_jsonl.py).
#
# Smoke (inside container on 1 GPU):
#   SMOKE=1 NPROC_PER_NODE=1 bash scripts/train_compressor_grpo_frida.sh

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge}"
CODE_ROOT="${CODE_ROOT:-${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${SLURM_NNODES:-}" ]]; then
  NNODES="${NNODES:-${SLURM_NNODES}}"
else
  NNODES="${NNODES:-1}"
fi

if [[ -n "${SLURM_PROCID:-}" ]]; then
  NODE_RANK="${NODE_RANK:-${SLURM_PROCID}}"
else
  NODE_RANK="${NODE_RANK:-0}"
fi

# shellcheck source=frida_resources.sh
source "${CODE_ROOT}/scripts/frida_resources.sh"
MASTER_ADDR="$(frida_resolve_master_addr "${NNODES}")"
frida_export_nccl_env "${NNODES}"

if [[ -z "${MASTER_PORT:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MASTER_PORT="$((20000 + (SLURM_JOB_ID % 40000)))"
  else
    MASTER_PORT="29500"
  fi
fi

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE}}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(bash -c "source '${CODE_ROOT}/scripts/frida_resources.sh'; frida_cuda_visible_devices '${NPROC_PER_NODE}'")}"

export NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE CUDA_VISIBLE_DEVICES

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
USE_HF="${USE_HF:-true}"
SFT_ADAPTERS="${SFT_ADAPTERS:-${WORKSPACE_ROOT}/runs/compressor_vl_sft/job_93030/checkpoint-10000}"

PLUGIN="${PLUGIN:-${CODE_ROOT}/research/experiments/03_grpo/compressor_reward.py}"
REWARD_FUNCS="${REWARD_FUNCS:-xpv_simple_bert xpv_simple_sle}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.7 0.3}"

TRAIN_JSONL="${TRAIN_JSONL:-${CODE_ROOT}/dataset/train_compressor_grpo.jsonl}"
TRAIN_SLICE="${TRAIN_SLICE:-}"

NPROC_PER_NODE="${NPROC_PER_NODE}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
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
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.35}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-2560}"
SEED="${SEED:-42}"
ADD_VERSION="${ADD_VERSION:-false}"

OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/runs/compressor_grpo}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}/job_${SLURM_JOB_ID}"
fi

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-compressor_grpo_frida}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  NPROC_PER_NODE=1
  export CUDA_VISIBLE_DEVICES=0
  NUM_GENERATIONS="${NUM_GENERATIONS_SMOKE:-4}"
  PER_DEVICE_BS="${PER_DEVICE_BS_SMOKE:-4}"
  GRAD_ACCUM=1
  MAX_STEPS="${MAX_STEPS_SMOKE:-4}"
  NUM_EPOCHS=1
  SAVE_STEPS=4
  TRAIN_SLICE="${TRAIN_SLICE:-64}"
  REPORT_TO=none
  VLLM_GPU_UTIL="${VLLM_GPU_UTIL_SMOKE:-0.35}"
  OUTPUT_DIR="${OUTPUT_DIR_SMOKE:-${WORKSPACE_ROOT}/runs/compressor_grpo_smoke/manual}"
  echo ">>> SMOKE MODE: 1 GPU, ${MAX_STEPS} steps, slice ${TRAIN_SLICE}"
fi

frida_apply_cpu_defaults "${NPROC_PER_NODE}"

FRIDA_USER_HOME="${FRIDA_USER_HOME:-/shared/home/${SLURM_JOB_USER:-${USER:-luka.dragar}}}"
if [[ -d "${FRIDA_USER_HOME}" ]]; then
  export HF_HOME="${HF_HOME:-${FRIDA_USER_HOME}/.cache/huggingface}"
  export WANDB_DIR="${WANDB_DIR:-${FRIDA_USER_HOME}/.wandb}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${FRIDA_USER_HOME}/.cache/wandb}"
  export FRIDA_NETRC="${FRIDA_NETRC:-${FRIDA_USER_HOME}/.netrc}"
fi

if [[ "${REPORT_TO}" == *wandb* ]]; then
  if frida_load_wandb_key; then
    echo "wandb: loaded API key" >&2
  elif wandb status 2>/dev/null | grep -q '"api_key": null'; then
    echo "error: wandb not configured." >&2
    exit 1
  fi
fi

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "error: ${TRAIN_JSONL} missing. Run: python3 dataset/build_compressor_grpo_jsonl.py" >&2
  exit 1
fi

TRAIN_DATASET="${TRAIN_JSONL}"
if [[ -n "${TRAIN_SLICE}" ]]; then
  TRAIN_DATASET="${TRAIN_JSONL}#${TRAIN_SLICE}"
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "error: 'swift' not on PATH inside the training container." >&2
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
  echo "error: SFT adapter not found: ${SFT_ADAPTERS}" >&2
  exit 1
fi

SCHEDULE_ARGS=(--num_train_epochs "${NUM_EPOCHS}")
if [[ -n "${MAX_STEPS}" ]]; then
  SCHEDULE_ARGS=(--max_steps "${MAX_STEPS}")
fi

mkdir -p "${OUTPUT_DIR}"

echo "prefetch: policy + reward models..." >&2
frida_warm_hf_model "${MODEL}"
frida_warm_hf_model "${XPV_BERT_MODEL:-microsoft/deberta-xlarge-mnli}"
frida_warm_hf_model "${XPV_SLE_MODEL:-liamcripwell/sle-base}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

EFF=$((PER_DEVICE_BS * NPROC_PER_NODE * GRAD_ACCUM))
echo "=== Compressor GRPO (Frida) ==="
echo "  model:        ${MODEL} (${MODEL_TYPE})"
echo "  train:        ${TRAIN_DATASET}"
echo "  reward:       ${REWARD_FUNCS}  weights=[${REWARD_WEIGHTS}]"
echo "  plugin:       ${PLUGIN}"
echo "  topology:     NNODES=${NNODES} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "  gpus:         ${NPROC_PER_NODE}  (CUDA=${CUDA_VISIBLE_DEVICES})"
echo "  group/batch:  num_generations=${NUM_GENERATIONS} per_device=${PER_DEVICE_BS} accum=${GRAD_ACCUM} eff=${EFF}"
echo "  lr/beta:      ${LEARNING_RATE} / ${BETA}"
echo "  vllm:         colocate util=${VLLM_GPU_UTIL} max_len=${VLLM_MAX_LEN}"
echo "  output:       ${OUTPUT_DIR}"
echo

swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL}" \
  --model_type "${MODEL_TYPE}" \
  --use_hf "${USE_HF}" \
  "${ADAPTER_FLAG[@]}" \
  --tuner_type lora \
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
  --add_version "${ADD_VERSION}" \
  --output_dir "${OUTPUT_DIR}"

echo
echo "=== Done ==="
echo "  output: ${OUTPUT_DIR}"
