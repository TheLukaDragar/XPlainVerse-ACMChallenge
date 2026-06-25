#!/usr/bin/env bash
# Pass-1 DINOv3-Large + MAC head (DINO-MAC reproduction) on all_v3 (+NTIRE later).
#
# Decorrelated backbone to LOGIT-FUSE with the SigLIP2+DINOv2 ensemble.
#
# Launch (4x A100):
#   LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=120:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_dinov3_mac_lj.sh
#
# Smoke (1 GPU, tiny):
#   TRAIN_SLICE=512 VAL_SLICE=200 EPOCHS=1 REPORT_TO=none NPROC_PER_NODE=1 \
#     CUDA_VISIBLE_DEVICES=0 LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=00:40:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_dinov3_mac_lj.sh

set -euo pipefail

# Gated DINOv3 weights need HF cache/token (container runs --cleanenv --no-home).
export HOME="${HOME:-/home/jakob}"
export HF_HOME="${HF_HOME:-/home/jakob/.cache/huggingface}"

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

DINOV3="${DINOV3:-vit_large_patch16_dinov3.lvd1689m}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NUM_WORKERS="${NUM_WORKERS:-12}"
LR_HEAD="${LR_HEAD:-2e-4}"
LR_LORA="${LR_LORA:-5e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
DEEP_SUPERVISION="${DEEP_SUPERVISION:-1}"
FOCAL_ALPHA="${FOCAL_ALPHA:-0.25}"
AUGMENT="${AUGMENT:-1}"
VAL_SLICE="${VAL_SLICE:-0}"
SELECT_METRIC="${SELECT_METRIC:-macro_f1}"
INIT_CKPT="${INIT_CKPT:-}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${MANIFEST_DIR}/external/manifest_all_v3.parquet}"
VAL_MANIFEST="${VAL_MANIFEST:-${MANIFEST_DIR}/manifest_val_holdout.parquet}"

_RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${LJ_RUNS_ROOT}/pass1_dinov3_mac/dinov3_mac_${_RUN_TS}}"

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-pass1_dinov3_mac_${_RUN_TS}}"
export WANDB_TAGS="${WANDB_TAGS:-pass1,dinov3,mac,all_v3,${NPROC_PER_NODE}gpu}"

if [[ "${REPORT_TO}" == *wandb* ]]; then
  TRAIN_REPORT_TO=wandb
else
  TRAIN_REPORT_TO=none
fi

if [[ -n "${INIT_CKPT}" && ! -f "${INIT_CKPT}" ]]; then
  echo "ERROR: init ckpt not found: ${INIT_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  echo "ERROR: train manifest not found: ${TRAIN_MANIFEST}" >&2
  exit 1
fi
if [[ ! -f "${VAL_MANIFEST}" ]]; then
  echo "ERROR: val holdout not found: ${VAL_MANIFEST}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
EFF=$((BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM))
echo "=== Pass-1 DINOv3-MAC (all_v3) ==="
echo "  dinov3      : ${DINOV3}"
echo "  train       : ${TRAIN_MANIFEST}"
echo "  val(holdout): ${VAL_MANIFEST}  val_slice=${VAL_SLICE}"
echo "  image_size  : ${IMAGE_SIZE}  epochs=${EPOCHS}"
echo "  lora        : r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "  eff_batch   : ${EFF}"
echo "  output      : ${OUTPUT_DIR}"
echo

cd "${EXP_DIR}"
_ARGS=(
  --train "${TRAIN_MANIFEST}"
  --val "${VAL_MANIFEST}"
  --out "${OUTPUT_DIR}"
  --dinov3 "${DINOV3}"
  --image-size "${IMAGE_SIZE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --lr-head "${LR_HEAD}"
  --lr-lora "${LR_LORA}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --deep-supervision "${DEEP_SUPERVISION}"
  --focal-alpha "${FOCAL_ALPHA}"
  --augment "${AUGMENT}"
  --val-slice "${VAL_SLICE}"
  --select-metric "${SELECT_METRIC}"
  --init-ckpt "${INIT_CKPT}"
  --num-workers "${NUM_WORKERS}"
  --report-to "${TRAIN_REPORT_TO}"
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
    train_dinov3_mac.py "${_ARGS[@]}"
else
  python3 train_dinov3_mac.py "${_ARGS[@]}"
fi

echo
echo "=== Done ==="
echo "ckpt : ${OUTPUT_DIR}/best_ckpt/ckpt.pt"
