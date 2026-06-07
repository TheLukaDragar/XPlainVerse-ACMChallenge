#!/usr/bin/env bash
# Pass-1 ensemble: SigLIP2-Giant + DINOv2-Large (dual-tower fusion).
#
# Replaces SO400M SigLIP tower with Giant in the same Bombek-style recipe.
# NO Bombek init — weights are SO400M-only and incompatible with Giant.
#
# Memory: Giant + DINOv2 is ~2× heavier than SO400M + DINOv2. Default batch 8/GPU
# (vs 16 for SO400M). Increase GRAD_ACCUM to keep effective batch ~256.
#
# Launch (4× A100):
#   LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=24:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_ensemble_giant_lj.sh
#
# Smoke (OOM check, 1 GPU):
#   TRAIN_SLICE=256 VAL_SLICE=256 EPOCHS=1 REPORT_TO=none NPROC_PER_NODE=1 \
#     CUDA_VISIBLE_DEVICES=0 LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=00:45:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_ensemble_giant_lj.sh

set -euo pipefail

if [[ -d "${HOME}/luka/code/XPlainVerse-ACMChallenge/research" ]]; then
  CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
elif [[ -d /workspace/XPlainVerse-ACMChallenge ]]; then
  CODE_ROOT="/workspace/XPlainVerse-ACMChallenge"
else
  CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

EXP_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier"
MANIFEST_DIR="${MANIFEST_DIR:-${EXP_DIR}/manifests}"
LJ_RUNS_ROOT="${LJ_RUNS_ROOT:-/home/jakob/luka/runs}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

_PASS1="${CODE_ROOT}/baseline_models/pass1"
if [[ -f "${_PASS1}/siglip2-giant/config.json" ]]; then
  SIGLIP="${SIGLIP:-${_PASS1}/siglip2-giant}"
else
  SIGLIP="${SIGLIP:-google/siglip2-giant-opt-patch16-384}"
fi
DINOV2="${DINOV2:-vit_large_patch14_dinov2.lvd142m}"
IMAGE_SIZE="${IMAGE_SIZE:-392}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR_HEAD="${LR_HEAD:-2e-4}"
LR_LORA="${LR_LORA:-5e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
VAL_SLICE="${VAL_SLICE:-10000}"
SELECT_METRIC="${SELECT_METRIC:-macro_f1}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${MANIFEST_DIR}/manifest_train_full_balanced.parquet}"

_RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${LJ_RUNS_ROOT}/pass1_ensemble/giant_dinov2_${_RUN_TS}}"

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-pass1_ensemble_giant_${_RUN_TS}}"
export WANDB_TAGS="${WANDB_TAGS:-pass1,ensemble,siglip-giant,dinov2,${NPROC_PER_NODE}gpu}"

if [[ "${REPORT_TO}" == *wandb* ]]; then
  ENSEMBLE_REPORT_TO=wandb
else
  ENSEMBLE_REPORT_TO=none
fi

if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  echo "=== building manifests (missing ${TRAIN_MANIFEST}) ==="
  CODE_ROOT="${CODE_ROOT}" MANIFEST_DIR="${MANIFEST_DIR}" python3 "${EXP_DIR}/build_manifest.py"
fi

mkdir -p "${OUTPUT_DIR}"
EFF=$((BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM))
echo "=== Pass-1 ensemble SigLIP2-Giant + DINOv2 (Lj) ==="
echo "  siglip      : ${SIGLIP}"
echo "  dinov2      : ${DINOV2}"
echo "  image_size  : ${IMAGE_SIZE}"
echo "  init        : scratch (no Bombek — Giant incompatible)"
echo "  train       : ${TRAIN_MANIFEST}"
echo "  gpus        : ${NPROC_PER_NODE}"
echo "  batch/gpu   : ${BATCH_SIZE}  grad_accum=${GRAD_ACCUM}  (effective ${EFF})"
echo "  epochs      : ${EPOCHS}  lr_head=${LR_HEAD}  lr_lora=${LR_LORA}"
echo "  lora        : r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "  select      : ${SELECT_METRIC}  val_slice=${VAL_SLICE}"
echo "  output      : ${OUTPUT_DIR}"
echo

cd "${EXP_DIR}"
_ARGS=(
  --train "${TRAIN_MANIFEST}"
  --val "${MANIFEST_DIR}/manifest_val.parquet"
  --out "${OUTPUT_DIR}"
  --siglip "${SIGLIP}"
  --dinov2 "${DINOV2}"
  --image-size "${IMAGE_SIZE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --lr-head "${LR_HEAD}"
  --lr-lora "${LR_LORA}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --val-slice "${VAL_SLICE}"
  --select-metric "${SELECT_METRIC}"
  --report-to "${ENSEMBLE_REPORT_TO}"
)
if [[ -n "${TRAIN_SLICE:-}" ]]; then
  _ARGS+=(--train-slice "${TRAIN_SLICE}")
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  python3 -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train_ensemble.py "${_ARGS[@]}"
else
  python3 train_ensemble.py "${_ARGS[@]}"
fi

echo
echo "=== Done ==="
echo "ckpt : ${OUTPUT_DIR}/best_ckpt/ckpt.pt"
