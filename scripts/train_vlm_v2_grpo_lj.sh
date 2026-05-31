#!/usr/bin/env bash
# GRPO RL for Pass-2 complex explanation on Ljubljana (elixir-lj, 4×A100 80GB).
#
# Continues from the v2 SFT LoRA (ckpt-1655) and optimises the leaderboard
# complex metric directly: reward = BERTScore-F1 (same DeBERTa config as the
# official scorer) + a light verdict/format term. Reward code lives in
# research/experiments/03_grpo/grpo_reward.py (registered into ms-swift orms).
#
# Runs in the single lj container (GHCR -lj). From the Slurm login node:
#   LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=24:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/train_vlm_v2_grpo_lj.sh
#
# Build the GRPO dataset once first (prompt-only rows + reward columns):
#   ./scripts/lj_ghcr_image_exec.sh bash -c \
#     'python3 research/experiments/03_grpo/build_grpo_jsonl.py \
#        --in dataset/train_vlm_v2.jsonl --out dataset/train_grpo.jsonl'
#
# NOTE: the lj image has no vLLM, so rollouts use HF generate (USE_VLLM=false) —
# correct but slow. Keep NUM_GENERATIONS / batch modest, or add vLLM later.
set -euo pipefail

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CODE_ROOT:-}" ]]; then :; elif [[ -d "${_SCRIPT_ROOT}/scripts" && -d "${_SCRIPT_ROOT}/dataset" ]]; then CODE_ROOT="${_SCRIPT_ROOT}";
elif [[ -d /workspace/XPlainVerse-ACMChallenge ]]; then CODE_ROOT="/workspace/XPlainVerse-ACMChallenge";
else CODE_ROOT="/home/jakob/luka/code/XPlainVerse-ACMChallenge"; fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
# ms-swift main imports FSDP2 on torch 2.4.1 — keep the shim on PYTHONPATH.
_SHIM="${CODE_ROOT}/scripts/lj_swift_compat"
[[ -f "${_SHIM}/sitecustomize.py" ]] && export PYTHONPATH="${_SHIM}:${PYTHONPATH:-}"
for _cuda_lib in cu121 cu12 cu13; do
  _nv="/usr/local/lib/python3.10/dist-packages/nvidia/${_cuda_lib}/lib"
  [[ -d "${_nv}" ]] && { export LD_LIBRARY_PATH="${_nv}:${LD_LIBRARY_PATH:-}"; break; }
done
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

LJ_RUNS_ROOT="${LJ_RUNS_ROOT:-/home/jakob/luka/runs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
USE_HF="${USE_HF:-true}"
# Start from the v2 SFT LoRA (our best complex model).
ADAPTERS="${ADAPTERS:-/home/jakob/luka/runs/vlm_v2/run-20260529-104845/checkpoint-1655}"

TRAIN_JSONL="${TRAIN_JSONL:-${CODE_ROOT}/dataset/train_grpo.jsonl}"
REWARD_PLUGIN="${REWARD_PLUGIN:-${CODE_ROOT}/research/experiments/03_grpo/grpo_reward.py}"
REWARD_FUNCS="${REWARD_FUNCS:-xpv_complex_bert xpv_verdict_format}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0 0.3}"

NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-400}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BETA="${BETA:-0.001}"            # KL to the SFT reference model
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LOG_STEPS="${LOG_STEPS:-1}"
# The cu130 lj image ships vLLM — use it for fast rollouts. Colocate shares the
# training GPUs (no separate server); tune memory split via VLLM_GPU_MEM_UTIL.
USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.4}"
DEEPSPEED="${DEEPSPEED:-zero2}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"

# shellcheck source=lj_resources.sh
source "${CODE_ROOT}/scripts/lj_resources.sh"
lj_apply_cpu_defaults "${NPROC_PER_NODE}"

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
_TS="$(date -u +%Y%m%d-%H%M%S)"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-vlm_v2_grpo_${_TS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${LJ_RUNS_ROOT}/vlm_v2_grpo/run-${_TS}}"

if ! command -v swift >/dev/null 2>&1; then
  echo "error: 'swift' not on PATH. Run via ./scripts/lj_ghcr_image_exec.sh." >&2
  exit 1
fi
if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "error: ${TRAIN_JSONL} missing. Build it first:" >&2
  echo "  python3 ${CODE_ROOT}/research/experiments/03_grpo/build_grpo_jsonl.py \\" >&2
  echo "    --in ${CODE_ROOT}/dataset/train_vlm_v2.jsonl --out ${TRAIN_JSONL}" >&2
  exit 1
fi
if [[ "${ATTN_IMPL}" == "flash_attn" ]] && ! python3 -c "import flash_attn" 2>/dev/null; then
  echo "warning: flash_attn not importable; using sdpa." >&2; ATTN_IMPL=sdpa
fi
if [[ -n "${DEEPSPEED:-}" ]] && ! python3 -c "import importlib.metadata as m; m.version('deepspeed')" 2>/dev/null; then
  echo "warning: deepspeed not found; DDP without ZeRO." >&2; DEEPSPEED=""
fi

mkdir -p "${OUTPUT_DIR}"
TRAIN_SCHEDULE=(--num_train_epochs "${NUM_EPOCHS}")
[[ -n "${MAX_STEPS}" ]] && TRAIN_SCHEDULE=(--max_steps "${MAX_STEPS}")
ADAPTER_FLAG=(); [[ -n "${ADAPTERS}" ]] && ADAPTER_FLAG=(--adapters "${ADAPTERS}")
DEEPSPEED_FLAG=(); [[ "${NPROC_PER_NODE}" -gt 1 && -n "${DEEPSPEED}" ]] && DEEPSPEED_FLAG=(--deepspeed "${DEEPSPEED}")
VLLM_FLAGS=()
if [[ "${USE_VLLM}" == "true" ]]; then
  VLLM_FLAGS=(--vllm_mode "${VLLM_MODE}" --vllm_gpu_memory_utilization "${VLLM_GPU_MEM_UTIL}")
fi

echo "=== VLM v2 GRPO (lj) ==="
echo "  model/adapters : ${MODEL}  <- ${ADAPTERS:-<none>}"
echo "  reward         : ${REWARD_FUNCS}  weights=[${REWARD_WEIGHTS}]"
echo "  plugin         : ${REWARD_PLUGIN}"
echo "  rollouts       : num_generations=${NUM_GENERATIONS} temp=${TEMPERATURE} max_completion=${MAX_COMPLETION_LENGTH} vllm=${USE_VLLM}"
echo "  batch          : per_device=${PER_DEVICE_BS} grad_accum=${GRAD_ACCUM} gpus=${NPROC_PER_NODE}"
echo "  lr/beta/lora   : ${LEARNING_RATE} / KL=${BETA} / r=${LORA_RANK} a=${LORA_ALPHA}"
echo "  train          : ${TRAIN_JSONL}"
echo "  output         : ${OUTPUT_DIR}"
echo

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NPROC_PER_NODE="${NPROC_PER_NODE}" \
swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL}" \
  --model_type "${MODEL_TYPE}" \
  --use_hf "${USE_HF}" \
  "${ADAPTER_FLAG[@]}" \
  --tuner_type lora \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --freeze_vit true \
  --torch_dtype bfloat16 \
  --attn_impl "${ATTN_IMPL}" \
  --dataset "${TRAIN_JSONL}" \
  --external_plugins "${REWARD_PLUGIN}" \
  --reward_funcs ${REWARD_FUNCS} \
  --reward_weights ${REWARD_WEIGHTS} \
  --num_generations "${NUM_GENERATIONS}" \
  --temperature "${TEMPERATURE}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --max_length "${MAX_LENGTH}" \
  --use_vllm "${USE_VLLM}" \
  "${VLLM_FLAGS[@]}" \
  --beta "${BETA}" \
  "${TRAIN_SCHEDULE[@]}" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 5 \
  --logging_steps "${LOG_STEPS}" \
  --log_completions true \
  --report_to "${REPORT_TO}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --seed "${SEED:-42}" \
  --output_dir "${OUTPUT_DIR}" \
  --add_version "${ADD_VERSION:-false}" \
  "${DEEPSPEED_FLAG[@]}"

echo
echo "=== GRPO done ==="
echo "ckpt: ${OUTPUT_DIR}"
