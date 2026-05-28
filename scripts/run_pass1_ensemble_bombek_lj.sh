#!/usr/bin/env bash
# Bombek1 exact recipe: SigLIP2-SO400M + DINOv2-Large ensemble + LoRA on XPlainVerse.
#
# Reference: refs/ai-image-detector-siglip-dinov2/ (HF Bombek1/ai-image-detector-siglip-dinov2)
#
# Hyperparams (OpenFake):
#   DINOv2 @ 392×392, SigLIP2 @ native 384 (patch14-384), 5 epochs, batch 16/GPU,
#   grad_accum 4 (2 GPU → eff 128; use GRAD_ACCUM=9 on 1 GPU → 144)
#   lr_head=2e-4, lr_lora=5e-5, cosine+warmup, LoRA r32/α64, focal γ=2 α=0.25
#   quality-agnostic aug (JPEG/blur/noise/resize jitter)
#
# Launch (4× A100 default):
#   LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=12:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_ensemble_bombek_lj.sh
#
# 10k eval after training:
#   CKPT=~/luka/runs/pass1_ensemble/.../best_ckpt/ckpt.pt \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_ensemble_eval_10k.sh

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

_PASS1="${CODE_ROOT}/baseline_models/pass1"
if [[ -f "${_PASS1}/siglip2-so400m/config.json" ]]; then
  SIGLIP="${SIGLIP:-${_PASS1}/siglip2-so400m}"
else
  SIGLIP="${SIGLIP:-google/siglip2-so400m-patch14-384}"
fi
DINOV2="${DINOV2:-vit_large_patch14_dinov2.lvd142m}"
IMAGE_SIZE="${IMAGE_SIZE:-392}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR_HEAD="${LR_HEAD:-2e-4}"
LR_LORA="${LR_LORA:-5e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
VAL_SLICE="${VAL_SLICE:-10000}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${MANIFEST_DIR}/manifest_train_balanced.parquet}"

_RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${LJ_RUNS_ROOT}/pass1_ensemble/bombek_so400m_dinov2_${_RUN_TS}}"

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-pass1_ensemble_bombek_${_RUN_TS}}"
export WANDB_TAGS="${WANDB_TAGS:-pass1,ensemble,bombek,${NPROC_PER_NODE}gpu}"

if [[ "${REPORT_TO}" == *wandb* ]] && [[ -z "${WANDB_API_KEY:-}" ]]; then
  if wandb status 2>/dev/null | grep -q '"api_key": null'; then
    echo "warning: wandb not logged in. Run: wandb login  (or set WANDB_API_KEY, or REPORT_TO=none)" >&2
  fi
fi

if [[ "${REPORT_TO}" == *wandb* ]]; then
  ENSEMBLE_REPORT_TO=wandb
else
  ENSEMBLE_REPORT_TO=none
fi

if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  CODE_ROOT="${CODE_ROOT}" MANIFEST_DIR="${MANIFEST_DIR}" python3 "${EXP_DIR}/build_manifest.py"
fi

mkdir -p "${OUTPUT_DIR}"
EFF=$((BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM))
echo "=== Pass-1 Bombek ensemble (Lj) ==="
echo "  siglip     : ${SIGLIP}"
echo "  dinov2     : ${DINOV2}"
echo "  image_size : ${IMAGE_SIZE}"
echo "  gpus       : ${NPROC_PER_NODE}"
echo "  batch/gpu  : ${BATCH_SIZE}  grad_accum=${GRAD_ACCUM}  (effective ${EFF})"
echo "  epochs     : ${EPOCHS}  lr_head=${LR_HEAD}  lr_lora=${LR_LORA}"
echo "  lora       : r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "  output     : ${OUTPUT_DIR}"
if [[ "${REPORT_TO}" == *wandb* ]]; then
  echo "  wandb      : ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_RUN_NAME})"
else
  echo "  wandb      : off (REPORT_TO=${REPORT_TO})"
fi
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
  --report-to "${ENSEMBLE_REPORT_TO}"
)

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
