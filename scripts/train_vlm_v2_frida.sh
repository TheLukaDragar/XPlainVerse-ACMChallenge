#!/usr/bin/env bash
# Frida ms-swift SFT for Qwen3-VL on XPlainVerse prompt/data v2.
#
# Default target: one Frida H100 node (8 × H100 80GB) using torchrun via
# ms-swift's distributed environment variables:
#   NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, NPROC_PER_NODE.
#
# This intentionally starts from the base Qwen model. It does not resume or
# load adapters from the cancelled v1 run unless the caller explicitly adds
# such support later.

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge}"
CODE_ROOT="${CODE_ROOT:-${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge}"

export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

# --- Slurm / torchrun topology --------------------------------------------
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
  NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(bash -c "source '${CODE_ROOT}/scripts/frida_resources.sh'; frida_cuda_visible_devices '${NPROC_PER_NODE}'")}"

export NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE CUDA_VISIBLE_DEVICES

# --- Model -----------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
USE_HF="${USE_HF:-true}"

# --- Data v2 ---------------------------------------------------------------
TRAIN_JSONL="${TRAIN_JSONL:-${CODE_ROOT}/dataset/train_vlm_v2.jsonl}"
VAL_JSONL="${VAL_JSONL:-${CODE_ROOT}/dataset/val_vlm_v2.jsonl}"
TRAIN_SLICE="${TRAIN_SLICE:-}"
VAL_SLICE="${VAL_SLICE:-1000}"

if [[ -n "${TRAIN_SLICE}" ]]; then
  TRAIN_DATASET="${TRAIN_JSONL}#${TRAIN_SLICE}"
else
  TRAIN_DATASET="${TRAIN_JSONL}"
fi

# --- Hyperparams -----------------------------------------------------------
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
LEARNING_RATE="${LEARNING_RATE:-1.5e-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/runs/vlm_v2_h100}"
SEED="${SEED:-42}"

# 8 × H100 default: global/effective batch = 4 * 8 * 1 = 32.
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

SAVE_STEPS="${SAVE_STEPS:-1000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

# Frida shared FS default favors no expensive multimodal packing prepass.
# Override PACKING=true if a packing cache has already been warmed.
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
PACKING="${PACKING:-false}"
PADDING_FREE="${PADDING_FREE:-false}"
LAZY_TOKENIZE="${LAZY_TOKENIZE:-true}"
PACKING_CACHE="${PACKING_CACHE:-}"
DEEPSPEED="${DEEPSPEED:-zero2}"

# Keep generation eval so the W&B callback logs 16 example predictions.
PREDICT_WITH_GENERATE="${PREDICT_WITH_GENERATE:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export WANDB_SAMPLE_N="${WANDB_SAMPLE_N:-16}"

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-vlm_v2_h100_${NUM_EPOCHS}ep}"

frida_apply_cpu_defaults "${NPROC_PER_NODE}"

# Pyxis containers often run as root; point W&B/HF caches at the mounted home.
FRIDA_USER_HOME="${FRIDA_USER_HOME:-/shared/home/${SLURM_JOB_USER:-${USER:-luka.dragar}}}"
if [[ -d "${FRIDA_USER_HOME}" ]]; then
  export HF_HOME="${HF_HOME:-${FRIDA_USER_HOME}/.cache/huggingface}"
  export WANDB_DIR="${WANDB_DIR:-${FRIDA_USER_HOME}/.wandb}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${FRIDA_USER_HOME}/.cache/wandb}"
  export FRIDA_NETRC="${FRIDA_NETRC:-${FRIDA_USER_HOME}/.netrc}"
fi

if [[ "${REPORT_TO}" == *wandb* ]]; then
  if frida_load_wandb_key; then
    echo "wandb: loaded API key from NETRC=${NETRC:-?}" >&2
  elif wandb status 2>/dev/null | grep -q '"api_key": null'; then
    echo "error: wandb not configured. Set WANDB_API_KEY or ~/.netrc on the login node." >&2
    exit 1
  fi
fi

EXTERNAL_PLUGIN="${CODE_ROOT}/external_plugins/wandb_predictions.py"
USE_PRED_CALLBACK=false
if [[ "${PREDICT_WITH_GENERATE}" == "true" && "${REPORT_TO}" == *wandb* ]]; then
  if [[ -f "${EXTERNAL_PLUGIN}" ]]; then
    USE_PRED_CALLBACK=true
  else
    echo "warning: plugin not found at ${EXTERNAL_PLUGIN}; W&B preds disabled." >&2
  fi
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "error: 'swift' not on PATH inside the training container." >&2
  exit 1
fi

# Cache model before spawning multi-GPU torchrun (rank-0-only download + DDP barrier breaks NCCL).
frida_warm_hf_model "${MODEL}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "error: ${TRAIN_JSONL} missing. Build v2 JSONL first with scripts/sbatch_build_train_v2.sbatch." >&2
  exit 1
fi
if [[ ! -f "${VAL_JSONL}" ]]; then
  echo "error: ${VAL_JSONL} missing. Build v2 JSONL first with scripts/sbatch_build_train_v2.sbatch." >&2
  exit 1
fi

if [[ "${ATTN_IMPL}" == "flash_attn" ]] && ! python3 -c "import flash_attn" 2>/dev/null; then
  echo "warning: flash_attn not importable; falling back to sdpa and disabling packing/padding_free." >&2
  ATTN_IMPL=sdpa
  PACKING=false
  PADDING_FREE=false
  LAZY_TOKENIZE=true
fi

if [[ "${PACKING}" == "true" ]]; then
  LAZY_TOKENIZE=false
  if [[ -n "${PACKING_CACHE}" ]]; then
    mkdir -p "${PACKING_CACHE}" 2>/dev/null || true
  fi
fi

if [[ -n "${DEEPSPEED:-}" ]] && ! python3 -c "import importlib.metadata as m; m.version('deepspeed')" 2>/dev/null; then
  echo "warning: deepspeed distribution not found; using plain DDP." >&2
  DEEPSPEED=""
fi

# Avoid ms-swift add_version DDP broadcast (rank desync on some Frida nodes).
ADD_VERSION="${ADD_VERSION:-false}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}/job_${SLURM_JOB_ID}"
fi
mkdir -p "${OUTPUT_DIR}"

TRAIN_SCHEDULE_ARGS=(--num_train_epochs "${NUM_EPOCHS}")
if [[ -n "${MAX_STEPS}" ]]; then
  TRAIN_SCHEDULE_ARGS=(--max_steps "${MAX_STEPS}")
fi

DEEPSPEED_FLAG=()
if [[ "${NPROC_PER_NODE}" -gt 1 && -n "${DEEPSPEED}" ]]; then
  DEEPSPEED_FLAG=(--deepspeed "${DEEPSPEED}")
fi

PLUGIN_FLAG=()
CALLBACK_FLAG=()
if [[ "${USE_PRED_CALLBACK}" == "true" ]]; then
  PLUGIN_FLAG=(--external_plugins "${EXTERNAL_PLUGIN}")
  CALLBACK_FLAG=(--callbacks wandb_predictions)
fi

PACKING_CACHE_FLAG=()
if [[ -n "${PACKING_CACHE}" ]]; then
  PACKING_CACHE_FLAG=(--packing_cache "${PACKING_CACHE}")
fi

EFF_BATCH=$((PER_DEVICE_BS * NPROC_PER_NODE * GRAD_ACCUM * NNODES))
TRAIN_ROWS=363602
APPROX_STEPS=$(((TRAIN_ROWS + EFF_BATCH - 1) / EFF_BATCH))

echo "=== XPlainVerse VLM v2 SFT on Frida ==="
echo "model:               ${MODEL}"
echo "train:               ${TRAIN_DATASET}"
echo "val:                 ${VAL_JSONL}#${VAL_SLICE} (W&B examples: ${WANDB_SAMPLE_N})"
echo "topology:            NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "gpus/rank:           NPROC_PER_NODE=${NPROC_PER_NODE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "batch:               per_device=${PER_DEVICE_BS} grad_accum=${GRAD_ACCUM} -> effective=${EFF_BATCH}"
echo "approx steps/epoch:  ${APPROX_STEPS} for ${TRAIN_ROWS} v2 rows"
echo "schedule:            ${TRAIN_SCHEDULE_ARGS[*]}"
echo "eval/save:           eval_steps=${EVAL_STEPS} save_steps=${SAVE_STEPS} val_slice=${VAL_SLICE}"
echo "generation eval:     ${PREDICT_WITH_GENERATE} max_new_tokens=${MAX_NEW_TOKENS}"
echo "max_length/lr/lora:  ${MAX_LENGTH} / ${LEARNING_RATE} / r=${LORA_RANK} alpha=${LORA_ALPHA}"
echo "attn/deepspeed:      ${ATTN_IMPL} / ${DEEPSPEED:-<off>}"
echo "packing:             ${PACKING} padding_free=${PADDING_FREE} lazy_tokenize=${LAZY_TOKENIZE} cache=${PACKING_CACHE:-<off>}"
echo "workers:             cpus=${FRIDA_CPUS_TOTAL:-?} dataset_num_proc=${DATASET_NUM_PROC} dataloader_workers/rank=${DATALOADER_NUM_WORKERS}"
echo "output:              ${OUTPUT_DIR}"
echo "add_version:         ${ADD_VERSION}"
echo "report_to:           ${REPORT_TO}"
if [[ "${REPORT_TO}" == *wandb* ]]; then
  echo "wandb:               ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_RUN_NAME})"
fi
echo

swift sft \
  --model "${MODEL}" \
  --model_type "${MODEL_TYPE}" \
  --use_hf "${USE_HF}" \
  --dataset "${TRAIN_DATASET}" \
  --val_dataset "${VAL_JSONL}#${VAL_SLICE}" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --attn_impl "${ATTN_IMPL}" \
  --packing "${PACKING}" \
  --padding_free "${PADDING_FREE}" \
  --lazy_tokenize "${LAZY_TOKENIZE}" \
  "${PACKING_CACHE_FLAG[@]}" \
  --max_length "${MAX_LENGTH}" \
  "${TRAIN_SCHEDULE_ARGS[@]}" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --freeze_llm false \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 5 \
  --logging_steps "${LOGGING_STEPS}" \
  --predict_with_generate "${PREDICT_WITH_GENERATE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --report_to ${REPORT_TO} \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --load_from_cache_file true \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  --add_version "${ADD_VERSION}" \
  "${DEEPSPEED_FLAG[@]}" \
  "${PLUGIN_FLAG[@]}" \
  "${CALLBACK_FLAG[@]}"
