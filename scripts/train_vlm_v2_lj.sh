#!/usr/bin/env bash
# Stage-2 VLM LoRA SFT (prompt/data v2) for Qwen3-VL — Ljubljana (elixir-lj-gpu-01).
#
# lj/Apptainer adaptation of scripts/train_vlm_v2_frida.sh. Same ms-swift recipe,
# but uses the lj container conventions from scripts/train_vlm_full_lj.sh
# (lj_resources.sh CPU defaults, Apptainer LD_LIBRARY_PATH, /primoz data) and the
# v2 dataset built by scripts/build_train_v2_lj.sh.
#
# === Quick start (from Slurm login node) ===
#   Runs inside the single lj training container (GHCR -lj image, built from
#   docker/Dockerfile.lj) via scripts/lj_ghcr_image_exec.sh — the same container
#   used for Pass-1, so the lj stack stays aligned. The devel-base image ships
#   nvcc (deepspeed) and flash_attn (cu121/torch2.4); this script adds a small
#   FSDP2 shim on PYTHONPATH so ms-swift `main` imports on torch 2.4.1.
#
#   # 1) Build v2 JSONL once (if dataset/*_v2.jsonl missing):
#   LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=02:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/build_train_v2_lj.sh
#
#   # 2) Full 4-GPU v2 SFT (background-safe: prefer sbatch_train_vlm_v2_lj.sbatch):
#   LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=48:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh env REPORT_TO=wandb bash scripts/train_vlm_v2_lj.sh
#
# Smoke (tiny steps, 1 GPU):
#   LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=01:00:00 ./scripts/lj_ghcr_image_exec.sh \
#     env REPORT_TO=tensorboard MAX_STEPS=4 NPROC_PER_NODE=1 TRAIN_SLICE=64 VAL_SLICE=8 \
#     SAVE_STEPS=999999 EVAL_STEPS=999999 PREDICT_WITH_GENERATE=false \
#     OUTPUT_DIR=/home/jakob/luka/runs/vlm_v2_smoke bash scripts/train_vlm_v2_lj.sh
#
# Hyperparams match scripts/train_vlm_v2_frida.sh. Defaults: 4 GPUs, eff batch 32.
set -euo pipefail

# --- Code root: prefer the repo that contains this script (bind-mounted HOME). ---
_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CODE_ROOT:-}" ]]; then
  :
elif [[ -d "${_SCRIPT_ROOT}/scripts" && -d "${_SCRIPT_ROOT}/dataset" ]]; then
  CODE_ROOT="${_SCRIPT_ROOT}"
elif [[ -d /workspace/XPlainVerse-ACMChallenge ]]; then
  CODE_ROOT="/workspace/XPlainVerse-ACMChallenge"
else
  CODE_ROOT="/home/jakob/luka/code/XPlainVerse-ACMChallenge"
fi
if [[ -n "${XPLAINVERSE_DATA_ROOT:-}" ]]; then
  LJ_DATA_ROOT="${XPLAINVERSE_DATA_ROOT}"
elif [[ -d /primoz/luka/XPlainVerse/data/XPlainVerse/train ]]; then
  LJ_DATA_ROOT="/primoz/luka/XPlainVerse/data/XPlainVerse"
else
  LJ_DATA_ROOT="/home/jakob/luka/data/XPlainVerse"
fi
LJ_RUNS_ROOT="${LJ_RUNS_ROOT:-/home/jakob/luka/runs}"

# --- Container Python (HOME bind-mount must not shadow torch/ms-swift) ---
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# ms-swift in the lj images imports FSDP2 (torch>=2.6) at trainer-factory import
# time, but the images ship torch 2.4.1. This sitecustomize shim adds a harmless
# FSDPModule placeholder so `swift sft` imports. Drop once images ship torch>=2.6.
_SWIFT_COMPAT_DIR="${CODE_ROOT}/scripts/lj_swift_compat"
if [[ -f "${_SWIFT_COMPAT_DIR}/sitecustomize.py" ]]; then
  export PYTHONPATH="${_SWIFT_COMPAT_DIR}:${PYTHONPATH:-}"
fi

# --- CUDA / PyTorch env (lj SIF: torch +cu121; prefer cu121/cu12 before cu13) ---
for _cuda_lib in cu121 cu12 cu13; do
  _nv_lib="/usr/local/lib/python3.10/dist-packages/nvidia/${_cuda_lib}/lib"
  if [[ -d "${_nv_lib}" ]]; then
    export LD_LIBRARY_PATH="${_nv_lib}:${LD_LIBRARY_PATH:-}"
    break
  fi
done
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

# --- Multi-GPU (lj default: 4× A100) ---
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
  GRAD_ACCUM="${GRAD_ACCUM:-4}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
  GRAD_ACCUM="${GRAD_ACCUM:-16}"
fi

# --- Model ---
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
USE_HF="${USE_HF:-true}"

# --- Data (v2) ---
TRAIN_JSONL="${TRAIN_JSONL:-${CODE_ROOT}/dataset/train_vlm_v2.jsonl}"
VAL_JSONL="${VAL_JSONL:-${CODE_ROOT}/dataset/val_vlm_v2.jsonl}"
TRAIN_SLICE="${TRAIN_SLICE:-}"
VAL_SLICE="${VAL_SLICE:-1000}"
MAX_STEPS="${MAX_STEPS:-}"

# --- Hyperparams (same as train_vlm_v2_frida.sh) ---
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.5e-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-${LJ_RUNS_ROOT}/vlm_v2}"
SEED="${SEED:-42}"

SAVE_STEPS="${SAVE_STEPS:-400}"
EVAL_STEPS="${EVAL_STEPS:-400}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

# --- Component freezing / grounding (defaults: LoRA-LLM only, vision frozen) ---
# Set FREEZE_VIT=false (+ FREEZE_ALIGNER=false) to fine-tune the vision tower so
# the model can ground explanations on the actual evidence regions. When the ViT
# is trained, also use a small VIT_LR/ALIGNER_LR (≈1e-5) and turn on
# VIT_GRAD_CKPT to keep memory in check; see sbatch_train_vlm_v2_unfrozen_lj.sbatch.
FREEZE_VIT="${FREEZE_VIT:-true}"
FREEZE_ALIGNER="${FREEZE_ALIGNER:-true}"
FREEZE_LLM="${FREEZE_LLM:-false}"
VIT_GRAD_CKPT="${VIT_GRAD_CKPT:-false}"
VIT_LR="${VIT_LR:-}"
ALIGNER_LR="${ALIGNER_LR:-}"

PACKING="${PACKING:-true}"
PADDING_FREE="${PADDING_FREE:-true}"
LAZY_TOKENIZE="${LAZY_TOKENIZE:-false}"
PACKING_CACHE="${PACKING_CACHE:-}"
DEEPSPEED="${DEEPSPEED:-zero2}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"

# shellcheck source=lj_resources.sh
source "${CODE_ROOT}/scripts/lj_resources.sh"
lj_apply_cpu_defaults "${NPROC_PER_NODE}"

PREDICT_WITH_GENERATE="${PREDICT_WITH_GENERATE:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export WANDB_SAMPLE_N="${WANDB_SAMPLE_N:-16}"

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-vlm_v2_lj_${NUM_EPOCHS}ep}"

if [[ "${REPORT_TO}" == *wandb* ]] && [[ -z "${WANDB_API_KEY:-}" ]]; then
  if wandb status 2>/dev/null | grep -q '"api_key": null'; then
    echo "warning: wandb not logged in. Run: wandb login  (or set WANDB_API_KEY, or REPORT_TO=tensorboard)" >&2
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
  echo "error: 'swift' (ms-swift) not on PATH. Run via ./scripts/lj_ghcr_image_exec.sh." >&2
  exit 1
fi

_TRAIN_JSONL_PATH="${TRAIN_JSONL%%#*}"
if [[ ! -f "${_TRAIN_JSONL_PATH}" ]]; then
  echo "error: ${_TRAIN_JSONL_PATH} missing. Build the v2 dataset first:" >&2
  echo "    LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=02:00:00 \\" >&2
  echo "      ./scripts/lj_ghcr_image_exec.sh bash scripts/build_train_v2_lj.sh" >&2
  exit 1
fi

if [[ -n "${TRAIN_SLICE}" ]]; then
  TRAIN_DATASET="${TRAIN_JSONL}#${TRAIN_SLICE}"
else
  TRAIN_DATASET="${TRAIN_JSONL}"
fi

TRAIN_SCHEDULE_ARGS=(--num_train_epochs "${NUM_EPOCHS}")
if [[ -n "${MAX_STEPS}" ]]; then
  TRAIN_SCHEDULE_ARGS=(--max_steps "${MAX_STEPS}")
fi

if [[ "${ATTN_IMPL}" == "flash_attn" ]] && ! python3 -c "import flash_attn" 2>/dev/null; then
  echo "warning: flash_attn not importable; falling back to ATTN_IMPL=sdpa PACKING=false PADDING_FREE=false LAZY_TOKENIZE=true" >&2
  ATTN_IMPL=sdpa
  PACKING=false
  PADDING_FREE=false
  LAZY_TOKENIZE=true
fi

if [[ "${PACKING}" == "true" ]]; then
  LAZY_TOKENIZE=false
  if [[ -z "${PACKING_CACHE}" ]] && [[ -d /primoz ]]; then
    PACKING_CACHE="/primoz/luka/cache/ms_swift_packing_v2"
    mkdir -p "${PACKING_CACHE}" 2>/dev/null || PACKING_CACHE=""
  fi
fi

# ms-swift's `swift sft --help` only prints a stub (args are dataclass fields),
# so probe argument support by grepping the installed ms-swift source instead.
_SWIFT_DIR="$(python3 -c 'import os,swift; print(os.path.dirname(swift.__file__))' 2>/dev/null || true)"
swift_supports_arg() {
  # Conservative: if we cannot locate the source, assume the arg IS supported
  # (the pinned lj container ships a recent ms-swift with all of these).
  [[ -z "${_SWIFT_DIR}" ]] && return 0
  grep -rqoE "\b$1\b" "${_SWIFT_DIR}" 2>/dev/null
}

# Some ms-swift builds (e.g. the torch 2.11 SIF) do not expose --packing_cache.
# Drop it rather than crash with "remaining_argv: ['--packing_cache', ...]".
if [[ -n "${PACKING_CACHE}" ]] && ! swift_supports_arg packing_cache; then
  echo "note: this ms-swift build has no --packing_cache; disabling cache flag." >&2
  PACKING_CACHE=""
fi

# ms-swift calls transformers.require_version('deepspeed') when --deepspeed is set.
if [[ -n "${DEEPSPEED:-}" ]] && ! python3 -c "import importlib.metadata as m; m.version('deepspeed')" 2>/dev/null; then
  echo "warning: deepspeed distribution not found; multi-GPU will use DDP without ZeRO." >&2
  DEEPSPEED=""
fi

mkdir -p "${OUTPUT_DIR}"

EFF_BATCH=$(( PER_DEVICE_BS * NPROC_PER_NODE * GRAD_ACCUM ))
# Approx steps/epoch for progress display (v2 train rows; cheap line count).
TRAIN_ROWS="$(wc -l < "${_TRAIN_JSONL_PATH}" 2>/dev/null || echo 0)"
if [[ "${TRAIN_ROWS}" -gt 0 && "${EFF_BATCH}" -gt 0 ]]; then
  APPROX_STEPS=$(( (TRAIN_ROWS + EFF_BATCH - 1) / EFF_BATCH ))
else
  APPROX_STEPS="?"
fi

echo "=== VLM v2 SFT (lj / Apptainer) ==="
echo "code_root:           ${CODE_ROOT}"
echo "data_root (build):   ${LJ_DATA_ROOT}"
echo "model:               ${MODEL}"
echo "train:               ${TRAIN_DATASET}  (${TRAIN_ROWS} rows)"
echo "val (eval):          ${VAL_JSONL}#${VAL_SLICE}  (wandb table: ${WANDB_SAMPLE_N} samples)"
echo "gpus:                NPROC=${NPROC_PER_NODE}  CUDA=${CUDA_VISIBLE_DEVICES}"
echo "per_device_bs:       ${PER_DEVICE_BS}  grad_accum: ${GRAD_ACCUM}  → eff batch ${EFF_BATCH}"
if [[ -n "${MAX_STEPS}" ]]; then
  echo "train_schedule:      max_steps=${MAX_STEPS}  (NUM_EPOCHS=${NUM_EPOCHS} ignored)"
else
  echo "train_schedule:      num_train_epochs=${NUM_EPOCHS}  (~${APPROX_STEPS} steps/epoch)"
fi
echo "max_length:          ${MAX_LENGTH}  lr: ${LEARNING_RATE}  rank: ${LORA_RANK} alpha: ${LORA_ALPHA}"
echo "freeze:              vit=${FREEZE_VIT} aligner=${FREEZE_ALIGNER} llm=${FREEZE_LLM}  vit_grad_ckpt=${VIT_GRAD_CKPT}"
echo "vision_lr:           vit_lr=${VIT_LR:-<=lr>} aligner_lr=${ALIGNER_LR:-<=lr>}"
echo "attn_impl:           ${ATTN_IMPL}"
echo "deepspeed:           ${DEEPSPEED:-<off>}"
echo "packing:             ${PACKING}  padding_free: ${PADDING_FREE}  lazy_tokenize: ${LAZY_TOKENIZE}"
echo "packing_cache:       ${PACKING_CACHE:-<off>}"
echo "cpus (alloc/total):  ${LJ_CPUS_TOTAL:-?}  dataset_num_proc: ${DATASET_NUM_PROC}  dataloader_workers/rank: ${DATALOADER_NUM_WORKERS}"
echo "predict_with_gen:    ${PREDICT_WITH_GENERATE}  (max_new_tokens=${MAX_NEW_TOKENS})"
echo "wandb_pred_callback: ${USE_PRED_CALLBACK}  (sample_n=${WANDB_SAMPLE_N})"
echo "output:              ${OUTPUT_DIR}"
echo "report_to:           ${REPORT_TO}"
if [[ "${REPORT_TO}" == *wandb* ]]; then
  echo "wandb:               ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_RUN_NAME})"
fi
echo

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

# Separate LR for the vision tower / aligner when unfrozen. Guard the flags in
# case the installed ms-swift build does not expose them (older releases).
VIT_LR_FLAG=()
if [[ -n "${VIT_LR}" ]]; then
  if swift_supports_arg vit_lr; then
    VIT_LR_FLAG+=(--vit_lr "${VIT_LR}")
  else
    echo "note: ms-swift has no --vit_lr; ViT will train at --learning_rate=${LEARNING_RATE}." >&2
  fi
fi
if [[ -n "${ALIGNER_LR}" ]] && swift_supports_arg aligner_lr; then
  VIT_LR_FLAG+=(--aligner_lr "${ALIGNER_LR}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
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
  --vit_gradient_checkpointing "${VIT_GRAD_CKPT}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --freeze_vit "${FREEZE_VIT}" \
  --freeze_aligner "${FREEZE_ALIGNER}" \
  --freeze_llm "${FREEZE_LLM}" \
  "${VIT_LR_FLAG[@]}" \
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
  --add_version "${ADD_VERSION:-false}" \
  "${DEEPSPEED_FLAG[@]}" \
  "${PLUGIN_FLAG[@]}" \
  "${CALLBACK_FLAG[@]}"

echo
echo "Done. Merge LoRA for faster infer:"
echo "  swift export --adapters ${OUTPUT_DIR}/checkpoint-XXXX --merge_lora true"
