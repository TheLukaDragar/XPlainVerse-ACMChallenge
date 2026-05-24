#!/usr/bin/env bash
# Sanity VLM LoRA SFT on XPlainVerse (Qwen3-VL-8B-Instruct).
#
# Goal: confirm the dataset/prompt format trains and loss drops in ~15–30 min
# on 1× A100 before launching the full 260k run.
#
# Usage (from repo root):
#   ./scripts/train_vlm_sanity.sh
#   MAX_STEPS=50 ./scripts/train_vlm_sanity.sh
#   CUDA_VISIBLE_DEVICES=0 ./scripts/train_vlm_sanity.sh
#   REPORT_TO=tensorboard ./scripts/train_vlm_sanity.sh   # disable wandb
#   PREDICT_WITH_GENERATE=false ./scripts/train_vlm_sanity.sh   # keep eval_loss, no W&B preds
#
# Eval runs on VAL_SLICE val rows (rouge/loss). W&B table logs only WANDB_SAMPLE_N
# rows from predict.jsonl — same 16 images every eval (SEED=42), independent of
# VAL_SLICE.
#
# After training, infer a few val rows:
#   ADAPTERS=runs/vlm_sanity/vx-*/checkpoint-* ./scripts/infer_vlm_smoke.sh

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge}"
CODE_ROOT="${CODE_ROOT:-${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge}"

# --- ms-swift / CUDA env (match workspace rules) ---
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Qwen3-VL image token budget (~1024 tokens ≈ 1M pixels default)
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Model ---
# Base instruct checkpoint (NOT the challenge baseline XPlainVerse merge).
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
USE_HF="${USE_HF:-true}"

# --- Data (small slice; rebuild JSONL first if missing) ---
TRAIN_JSONL="${TRAIN_JSONL:-${CODE_ROOT}/dataset/train_vlm.jsonl}"
VAL_JSONL="${VAL_JSONL:-${CODE_ROOT}/dataset/val_vlm.jsonl}"
TRAIN_SLICE="${TRAIN_SLICE:-500}"   # ms-swift: path.jsonl#N
# Eval metrics on this many val rows. W&B table size is WANDB_SAMPLE_N only.
VAL_SLICE="${VAL_SLICE:-100}"

# --- Run config ---
MAX_STEPS="${MAX_STEPS:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/runs/vlm_sanity}"
SEED="${SEED:-42}"

# --- Eval-time generation + W&B prediction table ---
# `true`  → generates per eval, writes predict.jsonl, custom callback logs
#           image+gt+pred to W&B. eval_loss is replaced by eval_token_acc/rouge.
# `false` → classic loss-based eval, no W&B prediction table.
PREDICT_WITH_GENERATE="${PREDICT_WITH_GENERATE:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
# W&B table only — first N rows from predict.jsonl per eval. Does NOT shrink VAL_SLICE.
export WANDB_SAMPLE_N="${WANDB_SAMPLE_N:-16}"

# --- Weights & Biases (optional; disable with REPORT_TO=tensorboard) ---
REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-vlm_sanity_${MAX_STEPS}steps}"

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
  echo "error: 'swift' not on PATH (install ms-swift)." >&2
  exit 1
fi

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "error: ${TRAIN_JSONL} not found. Run: python3 dataset/build_swift_jsonl.py" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=== VLM sanity SFT ==="
echo "model:               ${MODEL}"
echo "train:               ${TRAIN_JSONL}#${TRAIN_SLICE}"
echo "val:                 ${VAL_JSONL}#${VAL_SLICE}  (wandb table: ${WANDB_SAMPLE_N} samples)"
echo "max_steps:           ${MAX_STEPS}"
echo "predict_with_gen:    ${PREDICT_WITH_GENERATE}  (max_new_tokens=${MAX_NEW_TOKENS})"
echo "wandb_pred_callback: ${USE_PRED_CALLBACK}  (sample_n=${WANDB_SAMPLE_N})"
echo "output:              ${OUTPUT_DIR}"
echo "gpu:                 ${CUDA_VISIBLE_DEVICES}"
echo "report_to:           ${REPORT_TO}"
if [[ "${REPORT_TO}" == *wandb* ]]; then
  echo "wandb:               ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_RUN_NAME})"
fi
echo

# Custom callback flags (only added when prerequisites are satisfied).
PLUGIN_FLAG=()
CALLBACK_FLAG=()
if [[ "${USE_PRED_CALLBACK}" == "true" ]]; then
  PLUGIN_FLAG=(--external_plugins "${EXTERNAL_PLUGIN}")
  CALLBACK_FLAG=(--callbacks wandb_predictions)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
  --model "${MODEL}" \
  --model_type "${MODEL_TYPE}" \
  --use_hf "${USE_HF}" \
  --dataset "${TRAIN_JSONL}#${TRAIN_SLICE}" \
  --val_dataset "${VAL_JSONL}#${VAL_SLICE}" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --max_length 2048 \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --freeze_llm false \
  --eval_strategy steps \
  --eval_steps 25 \
  --save_strategy steps \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --predict_with_generate "${PREDICT_WITH_GENERATE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --report_to ${REPORT_TO} \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --load_from_cache_file true \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  "${PLUGIN_FLAG[@]}" \
  "${CALLBACK_FLAG[@]}"

echo
echo "Done. Checkpoints under: ${OUTPUT_DIR}/vx-*/checkpoint-*"
echo "Next: swift infer with --adapters on val_vlm_infer.jsonl (see scripts/infer_vlm_smoke.sh when added)."
