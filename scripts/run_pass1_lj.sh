#!/usr/bin/env bash
# Pass-1 binary classifier on Lj (elixir-lj-gpu-01, single GPU).
#
# Default backbone: local SigLIP2 (DINOv3 is Meta-gated on HF).
#
# === From Slurm login ===
#   LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=08:00:00 \
#     ./scripts/lj_gpu_exec.sh bash scripts/run_pass1_lj.sh
#
# === Smoke (512 train / 1k val, 1 epoch) ===
#   LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=01:00:00 \
#     ./scripts/lj_gpu_exec.sh bash -lc \
#     'TRAIN_SLICE=512 VAL_SLICE=1024 EPOCHS=1 BATCH_SIZE=32 \
#      OUTPUT_DIR=/home/jakob/luka/runs/pass1_smoke bash scripts/run_pass1_lj.sh'
#
# === Full run (260k balanced, 2 epochs, ~2–3 h on A100) ===
#   LJ_GPU_GRES=gpu:1 LJ_GPU_MEM=64G LJ_GPU_TIME=08:00:00 \
#     ./scripts/lj_gpu_exec.sh bash scripts/run_pass1_lj.sh
#
# When DINOv3 access is approved:
#   BACKBONE=baseline_models/pass1/dinov3-large ./scripts/lj_gpu_exec.sh bash scripts/run_pass1_lj.sh

set -euo pipefail

if [[ -d /workspace/XPlainVerse-ACMChallenge ]]; then
  _CODE_DEFAULT="/workspace/XPlainVerse-ACMChallenge"
else
  _CODE_DEFAULT="/home/jakob/luka/code/XPlainVerse-ACMChallenge"
fi

CODE_ROOT="${CODE_ROOT:-${_CODE_DEFAULT}}"
EXP_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier"
MANIFEST_DIR="${MANIFEST_DIR:-${EXP_DIR}/manifests}"
LJ_RUNS_ROOT="${LJ_RUNS_ROOT:-/home/jakob/luka/runs}"
OUTPUT_DIR="${OUTPUT_DIR:-${LJ_RUNS_ROOT}/pass1_siglip2/v1-$(date -u +%Y%m%d-%H%M%S)}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

_SIGLIP_LOCAL="${CODE_ROOT}/baseline_models/pass1/siglip2-so400m"
_DINO_LOCAL="${CODE_ROOT}/baseline_models/pass1/dinov3-large"
if [[ -n "${BACKBONE:-}" ]]; then
  :
elif [[ -f "${_SIGLIP_LOCAL}/config.json" ]]; then
  BACKBONE="${_SIGLIP_LOCAL}"
elif [[ -f "${_DINO_LOCAL}/config.json" ]]; then
  BACKBONE="${_DINO_LOCAL}"
else
  BACKBONE="google/siglip2-so400m-patch14-384"
fi

LORA="${LORA:-0}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TRAIN_SLICE="${TRAIN_SLICE:-0}"
VAL_SLICE="${VAL_SLICE:-0}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${MANIFEST_DIR}/manifest_train_balanced.parquet}"

# Step 1 — manifests
if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  echo "=== building manifests ==="
  CODE_ROOT="${CODE_ROOT}" MANIFEST_DIR="${MANIFEST_DIR}" \
    python3 "${EXP_DIR}/build_manifest.py"
fi

mkdir -p "${OUTPUT_DIR}"
echo "=== Pass-1 training (Lj) ==="
echo "  code root  : ${CODE_ROOT}"
echo "  backbone   : ${BACKBONE}"
echo "  train      : ${TRAIN_MANIFEST}"
echo "  val        : ${MANIFEST_DIR}/manifest_val.parquet"
echo "  LoRA       : ${LORA}"
echo "  epochs     : ${EPOCHS}"
echo "  batch_size : ${BATCH_SIZE}"
echo "  output     : ${OUTPUT_DIR}"
echo "  train_slice: ${TRAIN_SLICE:-all}"
echo "  val_slice  : ${VAL_SLICE:-all}"
echo

cd "${EXP_DIR}"
python3 train.py \
  --train "${TRAIN_MANIFEST}" \
  --val "${MANIFEST_DIR}/manifest_val.parquet" \
  --out "${OUTPUT_DIR}" \
  --backbone "${BACKBONE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lora "${LORA}" \
  --num-workers "${NUM_WORKERS}" \
  --train-slice "${TRAIN_SLICE}" \
  --val-slice "${VAL_SLICE}"

echo
echo "=== Done ==="
echo "metrics : ${OUTPUT_DIR}/metrics.json"
echo "preds   : ${OUTPUT_DIR}/val_predictions.parquet"
echo "ckpt    : ${OUTPUT_DIR}/best_ckpt/ckpt.pt"
