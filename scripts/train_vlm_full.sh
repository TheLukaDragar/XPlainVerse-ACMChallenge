#!/usr/bin/env bash
# Full VLM LoRA SFT on XPlainVerse train_vlm.jsonl (450k rows: 320k fake + 130k real).
#
# Hardware targets (effective batch reflects --packing reducing sample count):
#   1× A100 80GB  — default below (batch 1, accum 16 → eff batch 16 packed)
#   4× A100       — NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 PER_DEVICE_BS=2 GRAD_ACCUM=4
#
# Usage (from repo root):
#   ./scripts/train_vlm_full.sh
#   NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_vlm_full.sh
#   REPORT_TO=tensorboard ./scripts/train_vlm_full.sh                          # disable wandb
#   PACKING=false GRAD_ACCUM=32 ./scripts/train_vlm_full.sh                    # disable packing
#   PREDICT_WITH_GENERATE=false VAL_SLICE=2000 ./scripts/train_vlm_full.sh     # loss eval mode
#
# Tuning aligned with ms-swift Qwen3-VL Best Practice:
#   --packing true + --padding_free true + flash_attn for ~30-50% speedup
#   --max_length 4096 to avoid truncating long fake explanations
#   --vit_gradient_checkpointing false (ViT is frozen)
#   --deepspeed zero2 on multi-GPU
#
# Eval mode (default `PREDICT_WITH_GENERATE=true`):
#   Full eval runs on VAL_SLICE samples (rouge/bleu/token_acc). The W&B table
#   only logs WANDB_SAMPLE_N rows from predict.jsonl — same 16 images every
#   eval (deterministic via SEED=42), independent of VAL_SLICE.
#   Set PREDICT_WITH_GENERATE=false for classic loss-based eval.
#
# Prerequisites:
#   python3 dataset/build_swift_jsonl.py   # produces dataset/train_vlm.jsonl

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge}"
CODE_ROOT="${CODE_ROOT:-${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge}"

export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

# --- Multi-GPU (optional) ---
# Packing typically halves the sample count -> halve grad_accum vs un-packed defaults.
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
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

# --- Data ---
TRAIN_JSONL="${TRAIN_JSONL:-${CODE_ROOT}/dataset/train_vlm.jsonl}"
VAL_JSONL="${VAL_JSONL:-${CODE_ROOT}/dataset/val_vlm.jsonl}"
# Subset for eval during training (full val is 110k). Rouge/loss computed on all
# VAL_SLICE rows; W&B table only logs WANDB_SAMPLE_N of them.
VAL_SLICE="${VAL_SLICE:-2000}"

# --- Hyperparams (Qwen3-VL Best Practice + XPlainVerse baseline) ---
NUM_EPOCHS="${NUM_EPOCHS:-1}"
# Lowered from 2e-4: rank 16 has ~2x params of official rank-8 example (lr 1e-4).
LEARNING_RATE="${LEARNING_RATE:-1.5e-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
# Bumped from 2048: long fake GT (~300+ words) + 1024 image tokens + prompt
# overflows 2048 and truncates assistant target -> kills entity coverage.
MAX_LENGTH="${MAX_LENGTH:-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/runs/vlm_full}"
SEED="${SEED:-42}"

SAVE_STEPS="${SAVE_STEPS:-400}"
EVAL_STEPS="${EVAL_STEPS:-400}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

# --- Performance flags (Qwen3-VL Best Practice) ---
# NOTE: packing + lazy_tokenize are mutually exclusive in ms-swift 4.x (two-pass
# map for multimodal packing is extremely slow on 450k images over shared FS).
# We disable packing and use lazy_tokenize instead — tokenizes on-the-fly per
# batch, zero upfront Map cost. Re-enable packing if you pre-build a packing
# cache on $SCRATCH first (--packing_cache /local/scratch/...).
PACKING="${PACKING:-false}"
PADDING_FREE="${PADDING_FREE:-false}"
LAZY_TOKENIZE="${LAZY_TOKENIZE:-true}"
DEEPSPEED="${DEEPSPEED:-zero2}"

# --- Eval-time generation + W&B prediction table ---
# `true`  → generates per eval, writes predict.jsonl, callback logs preds to W&B.
#           eval_loss replaced by eval_rouge*/eval_token_acc.
# `false` → classic loss-based eval (no per-step prediction inspection).
PREDICT_WITH_GENERATE="${PREDICT_WITH_GENERATE:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
# W&B table only — first N rows from predict.jsonl per eval (same 16 images
# every time thanks to deterministic val sampling). Does NOT shrink VAL_SLICE.
export WANDB_SAMPLE_N="${WANDB_SAMPLE_N:-16}"

# --- Weights & Biases (optional; disable with REPORT_TO=tensorboard) ---
REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-vlm_full_${NUM_EPOCHS}ep}"

if [[ "${REPORT_TO}" == *wandb* ]] && [[ -z "${WANDB_API_KEY:-}" ]]; then
  if wandb status 2>/dev/null | grep -q '"api_key": null'; then
    echo "warning: wandb not logged in. Run: wandb login  (or set WANDB_API_KEY)" >&2
  fi
fi

# Custom callback to log image+gt+pred to W&B during each eval.
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
  echo "error: 'swift' not on PATH." >&2
  exit 1
fi

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "error: ${TRAIN_JSONL} missing. Run: python3 dataset/build_swift_jsonl.py" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

EFF_BATCH=$(( PER_DEVICE_BS * NPROC_PER_NODE * GRAD_ACCUM ))
echo "=== VLM full SFT ==="
echo "model:               ${MODEL}"
echo "train:               ${TRAIN_JSONL} (450k, ~2.5:1 fake:real)"
echo "val (eval):          ${VAL_JSONL}#${VAL_SLICE}  (wandb table: ${WANDB_SAMPLE_N} samples)"
echo "gpus:                NPROC=${NPROC_PER_NODE}  CUDA=${CUDA_VISIBLE_DEVICES}"
echo "per_device_bs:       ${PER_DEVICE_BS}  grad_accum: ${GRAD_ACCUM}  → eff batch ${EFF_BATCH}"
echo "epochs:              ${NUM_EPOCHS}"
echo "max_length:          ${MAX_LENGTH}  lr: ${LEARNING_RATE}  rank: ${LORA_RANK}"
echo "packing:             ${PACKING}  padding_free: ${PADDING_FREE}  lazy_tokenize: ${LAZY_TOKENIZE}"
echo "predict_with_gen:    ${PREDICT_WITH_GENERATE}  (max_new_tokens=${MAX_NEW_TOKENS})"
echo "wandb_pred_callback: ${USE_PRED_CALLBACK}  (sample_n=${WANDB_SAMPLE_N})"
echo "output:              ${OUTPUT_DIR}"
echo "report_to:           ${REPORT_TO}"
if [[ "${REPORT_TO}" == *wandb* ]]; then
  echo "wandb:               ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_RUN_NAME})"
fi
echo

# Build optional deepspeed flag (only on multi-GPU).
DEEPSPEED_FLAG=()
if [[ "${NPROC_PER_NODE}" -gt 1 && -n "${DEEPSPEED}" ]]; then
  DEEPSPEED_FLAG=(--deepspeed "${DEEPSPEED}")
fi

# Custom callback flags (only added when prerequisites are satisfied).
PLUGIN_FLAG=()
CALLBACK_FLAG=()
if [[ "${USE_PRED_CALLBACK}" == "true" ]]; then
  PLUGIN_FLAG=(--external_plugins "${EXTERNAL_PLUGIN}")
  CALLBACK_FLAG=(--callbacks wandb_predictions)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
swift sft \
  --model "${MODEL}" \
  --model_type "${MODEL_TYPE}" \
  --use_hf "${USE_HF}" \
  --dataset "${TRAIN_JSONL}" \
  --val_dataset "${VAL_JSONL}#${VAL_SLICE}" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --packing "${PACKING}" \
  --padding_free "${PADDING_FREE}" \
  --lazy_tokenize "${LAZY_TOKENIZE}" \
  --max_length "${MAX_LENGTH}" \
  --num_train_epochs "${NUM_EPOCHS}" \
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
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --load_from_cache_file true \
  --lazy_tokenize true \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  "${DEEPSPEED_FLAG[@]}" \
  "${PLUGIN_FLAG[@]}" \
  "${CALLBACK_FLAG[@]}"

echo
echo "Done. Merge LoRA for faster infer:"
echo "  swift export --adapters ${OUTPUT_DIR}/vx-*/checkpoint-XXXX --merge_lora true"
