#!/usr/bin/env bash
# Pass-1 timm full fine-tune @ 512×512 (ConvNeXt default; all backbone params trainable).
#
# Requires GHCR Lj image with timm (docker/Dockerfile.lj):
#   LJ_APPTAINER_IMAGE=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge-lj:latest
#
# Launch (2× A100):
#   LJ_GPU_GRES=gpu:2 LJ_GPU_TIME=08:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_timm_fullft_lj.sh
#
# Native 512 CSATv2 (smaller, ~21M params):
#   TIMM_MODEL=csatv2_21m.sw_r512_in1k ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_timm_fullft_lj.sh

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

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

TIMM_MODEL="${TIMM_MODEL:-convnext_small.fb_in22k_ft_in1k}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-24}"
LR="${LR:-5e-5}"
VAL_SLICE="${VAL_SLICE:-10000}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${MANIFEST_DIR}/manifest_train_balanced.parquet}"

_RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
_MODEL_TAG="${TIMM_MODEL//\//_}"
OUTPUT_DIR="${OUTPUT_DIR:-${LJ_RUNS_ROOT}/pass1_timm/${_MODEL_TAG}_512_fullft_${_RUN_TS}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-pass1_timm_${_MODEL_TAG}_512_${_RUN_TS}}"

if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  CODE_ROOT="${CODE_ROOT}" MANIFEST_DIR="${MANIFEST_DIR}" python3 "${EXP_DIR}/build_manifest.py"
fi

mkdir -p "${OUTPUT_DIR}"
echo "=== Pass-1 timm full fine-tune (Lj) ==="
echo "  model      : ${TIMM_MODEL}"
echo "  image_size : ${IMAGE_SIZE}"
echo "  gpus       : ${NPROC_PER_NODE}"
echo "  batch/gpu  : ${BATCH_SIZE}  (eff $((BATCH_SIZE * NPROC_PER_NODE)))"
echo "  epochs     : ${EPOCHS}  lr=${LR}"
echo "  output     : ${OUTPUT_DIR}"
echo

cd "${EXP_DIR}"
_TRAIN_ARGS=(
  --train "${TRAIN_MANIFEST}"
  --val "${MANIFEST_DIR}/manifest_val.parquet"
  --out "${OUTPUT_DIR}"
  --model "${TIMM_MODEL}"
  --image-size "${IMAGE_SIZE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --val-slice "${VAL_SLICE}"
  --report-to "${REPORT_TO:-wandb}"
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  python3 -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train_timm.py "${_TRAIN_ARGS[@]}"
else
  python3 train_timm.py "${_TRAIN_ARGS[@]}"
fi

echo
echo "=== Done ==="
echo "ckpt : ${OUTPUT_DIR}/best_ckpt/ckpt.pt"
