#!/usr/bin/env bash
# Pass-1 binary classifier on Lj (elixir-lj-gpu-01, 2× GPU DDP by default).
#
# Default backbone: DINOv3-7B (Simplicity Prevails DINOv3-Linear) — A100 80GB.
# Alternate: BACKBONE=baseline_models/pass1/siglip2-giant for SigLIP2-Linear.
#
# Prefer GHCR training image (pandas/sklearn/peft): scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_lj.sh
# Local SIF fallback: ./scripts/lj_gpu_exec.sh bash scripts/run_pass1_lj.sh
#
# Slurm (2 GPUs + wandb):
#   LJ_GPU_GRES=gpu:2 LJ_GPU_TIME=12:00:00 \
#     LJ_APPTAINER_IMAGE=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge-lj:latest \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_lj.sh
#
# SigLIP2-giant:
#   BACKBONE=baseline_models/pass1/siglip2-giant OUTPUT_DIR=/home/jakob/luka/runs/pass1_siglip2_giant \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_lj.sh
#
# === Smoke (2 GPU, no wandb) ===
#   LJ_GPU_GRES=gpu:2 LJ_GPU_TIME=02:00:00 REPORT_TO=none \
#     TRAIN_SLICE=512 VAL_SLICE=1024 EPOCHS=1 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_lj.sh

set -euo pipefail

if [[ -d "${HOME}/luka/code/XPlainVerse-ACMChallenge/research" ]]; then
  _CODE_DEFAULT="${HOME}/luka/code/XPlainVerse-ACMChallenge"
elif [[ -d /workspace/XPlainVerse-ACMChallenge ]]; then
  _CODE_DEFAULT="/workspace/XPlainVerse-ACMChallenge"
else
  _CODE_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

CODE_ROOT="${CODE_ROOT:-${_CODE_DEFAULT}}"
EXP_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier"
MANIFEST_DIR="${MANIFEST_DIR:-${EXP_DIR}/manifests}"
LJ_RUNS_ROOT="${LJ_RUNS_ROOT:-/home/jakob/luka/runs}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
export CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

_PASS1="${CODE_ROOT}/baseline_models/pass1"
if [[ -n "${BACKBONE:-}" ]]; then
  if [[ "${BACKBONE}" != /* ]] && { [[ -e "${CODE_ROOT}/${BACKBONE}" ]] || [[ "${BACKBONE}" == baseline_models/* ]]; }; then
    BACKBONE="${CODE_ROOT}/${BACKBONE}"
  fi
elif [[ -f "${_PASS1}/dinov3-7b/config.json" ]]; then
  BACKBONE="${_PASS1}/dinov3-7b"
elif [[ -f "${_PASS1}/dinov3-large/config.json" ]]; then
  BACKBONE="${_PASS1}/dinov3-large"
else
  BACKBONE="facebook/dinov3-vit7b16-pretrain-lvd1689m"
fi

_BACKBONE_TAG="${BACKBONE##*/}"
OUTPUT_DIR="${OUTPUT_DIR:-${LJ_RUNS_ROOT}/pass1_${_BACKBONE_TAG}/v1-$(date -u +%Y%m%d-%H%M%S)}"

# Paper (Simplicity Prevails): AdamW lr=1e-3, batch=128, 2 epochs, constant lr, no aug.
# SigLIP LoRA tuned recipe (web / Bombek / HF deepfake): see scripts/run_pass1_siglip_lora_tuned_lj.sh
# BATCH_SIZE is per GPU; effective batch = BATCH_SIZE × NPROC_PER_NODE.
LORA="${LORA:-0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
EPOCHS="${EPOCHS:-2}"
LR_HEAD="${LR_HEAD:-1e-3}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
WARMUP_RATIO="${WARMUP_RATIO:-0}"
if [[ -z "${BATCH_SIZE:-}" ]]; then
  if [[ "${BACKBONE}" == *"7b"* ]] || [[ "${BACKBONE}" == *"vit7b"* ]]; then
    BATCH_SIZE=80
  elif [[ "${BACKBONE}" == *"giant"* ]] || [[ "${BACKBONE}" == *"siglip2"* ]]; then
    BATCH_SIZE=128
  else
    BATCH_SIZE=64
  fi
fi
NUM_WORKERS="${NUM_WORKERS:-8}"
TRAIN_SLICE="${TRAIN_SLICE:-0}"
# Training-time val: random subset (0 = full 110k, slow). Full val → eval.py after training.
VAL_SLICE="${VAL_SLICE:-10000}"
# IMAGE_SIZE: override processor resize. 0 = backbone default (DINOv3=224, SigLIP2=384).
IMAGE_SIZE="${IMAGE_SIZE:-0}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${MANIFEST_DIR}/manifest_train_balanced.parquet}"

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-pass1_${_BACKBONE_TAG}_${EPOCHS}ep_${NPROC_PER_NODE}gpu}"
export WANDB_TAGS="${WANDB_TAGS:-pass1,${NPROC_PER_NODE}gpu}"

if [[ "${REPORT_TO}" == *wandb* ]] && [[ -z "${WANDB_API_KEY:-}" ]]; then
  if wandb status 2>/dev/null | grep -q '"api_key": null'; then
    echo "warning: wandb not logged in. Run: wandb login  (or REPORT_TO=none)" >&2
  fi
fi

if [[ "${REPORT_TO}" == *wandb* ]]; then
  PASS1_REPORT_TO=wandb
else
  PASS1_REPORT_TO=none
fi

# Step 1 — manifests
if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  echo "=== building manifests ==="
  CODE_ROOT="${CODE_ROOT}" MANIFEST_DIR="${MANIFEST_DIR}" \
    python3 "${EXP_DIR}/build_manifest.py"
fi

mkdir -p "${OUTPUT_DIR}"
EFFECTIVE_BATCH=$((BATCH_SIZE * NPROC_PER_NODE))
echo "=== Pass-1 training (Lj) ==="
echo "  code root  : ${CODE_ROOT}"
echo "  backbone   : ${BACKBONE}"
echo "  train      : ${TRAIN_MANIFEST}"
echo "  val        : ${MANIFEST_DIR}/manifest_val.parquet"
echo "  gpus       : ${NPROC_PER_NODE} (CUDA ${CUDA_VISIBLE_DEVICES})"
echo "  batch/gpu  : ${BATCH_SIZE}  (effective ${EFFECTIVE_BATCH})"
echo "  LoRA       : ${LORA}  (r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT})"
echo "  epochs     : ${EPOCHS}"
echo "  lr_head    : ${LR_HEAD}"
echo "  lr_backbone: ${LR_BACKBONE}"
echo "  lr_schedule: ${LR_SCHEDULE}  warmup_ratio=${WARMUP_RATIO}"
echo "  output     : ${OUTPUT_DIR}"
echo "  train_slice: ${TRAIN_SLICE:-all}"
echo "  val_slice  : ${VAL_SLICE:-all}"
echo "  image_size : ${IMAGE_SIZE} (0 = backbone default)"
if [[ "${REPORT_TO}" == *wandb* ]]; then
  echo "  wandb      : ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_RUN_NAME})"
else
  echo "  wandb      : off (REPORT_TO=${REPORT_TO})"
fi
echo

cd "${EXP_DIR}"
_TRAIN_ARGS=(
  --train "${TRAIN_MANIFEST}"
  --val "${MANIFEST_DIR}/manifest_val.parquet"
  --out "${OUTPUT_DIR}"
  --backbone "${BACKBONE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr-head "${LR_HEAD}"
  --lr-backbone "${LR_BACKBONE}"
  --lr-schedule "${LR_SCHEDULE}"
  --warmup-ratio "${WARMUP_RATIO}"
  --lora "${LORA}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --num-workers "${NUM_WORKERS}"
  --train-slice "${TRAIN_SLICE}"
  --val-slice "${VAL_SLICE}"
  --image-size "${IMAGE_SIZE}"
  --report-to "${PASS1_REPORT_TO}"
)

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  # Match ms-swift (swift/cli/main.py): torch.distributed.run + NPROC_PER_NODE only.
  # Do NOT use torchrun --standalone or --rdzv_* — those publish FQDN and hang on Lj.
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
  _TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE}")
  if [[ -n "${MASTER_ADDR:-}" ]]; then
    _TORCHRUN_ARGS+=(--master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
  fi
  python3 -m torch.distributed.run "${_TORCHRUN_ARGS[@]}" train.py "${_TRAIN_ARGS[@]}"
else
  python3 train.py "${_TRAIN_ARGS[@]}"
fi

echo
echo "=== Done ==="
echo "metrics : ${OUTPUT_DIR}/metrics.json"
echo "preds   : ${OUTPUT_DIR}/val_predictions.parquet"
echo "ckpt    : ${OUTPUT_DIR}/best_ckpt/ckpt.pt"
