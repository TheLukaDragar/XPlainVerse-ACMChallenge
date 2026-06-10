#!/usr/bin/env bash
# Pass-1 ensemble: combined fine-tune on XP pooled + OpenFake 0-14 + DFBench.
#
# Warm-start from trainval_resume best checkpoint. 1 epoch default (~2M images).
#
# Prereq: ./scripts/prepare_external_training_lj.sh finished successfully.
#
# Launch (4x A100):
#   LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=120:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_ensemble_external_all_lj.sh
#
# Smoke:
#   TRAIN_SLICE=1024 VAL_SLICE=512 EPOCHS=1 REPORT_TO=none NPROC_PER_NODE=1 \
#     CUDA_VISIBLE_DEVICES=0 LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=01:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_ensemble_external_all_lj.sh

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
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR_HEAD="${LR_HEAD:-5e-5}"
LR_LORA="${LR_LORA:-2e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
AUGMENT="${AUGMENT:-1}"
VAL_SLICE="${VAL_SLICE:-0}"
SELECT_METRIC="${SELECT_METRIC:-macro_f1}"
INIT_BOMBEK="${INIT_BOMBEK:-/home/jakob/luka/runs/pass1_ensemble/trainval_resume_e2-4_20260610-181626/best_ckpt/ckpt.pt}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${MANIFEST_DIR}/external/manifest_all_v1.parquet}"
VAL_MANIFEST="${VAL_MANIFEST:-${MANIFEST_DIR}/manifest_val_holdout.parquet}"

_RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${LJ_RUNS_ROOT}/pass1_ensemble/external_all_v1_${_RUN_TS}}"

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-pass1_external_all_v1_${_RUN_TS}}"
export WANDB_TAGS="${WANDB_TAGS:-pass1,ensemble,external-all-v1,macro-f1,${NPROC_PER_NODE}gpu}"

if [[ "${REPORT_TO}" == *wandb* ]]; then
  ENSEMBLE_REPORT_TO=wandb
else
  ENSEMBLE_REPORT_TO=none
fi

if [[ ! -f "${INIT_BOMBEK}" ]]; then
  echo "ERROR: init checkpoint not found: ${INIT_BOMBEK}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  echo "ERROR: train manifest not found: ${TRAIN_MANIFEST}" >&2
  echo "Run: ./scripts/prepare_external_training_lj.sh" >&2
  exit 1
fi
if [[ ! -f "${VAL_MANIFEST}" ]]; then
  echo "ERROR: val holdout not found: ${VAL_MANIFEST}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
EFF=$((BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM))
echo "=== Pass-1 ensemble EXTERNAL ALL v1 ==="
echo "  train       : ${TRAIN_MANIFEST}"
echo "  val(holdout): ${VAL_MANIFEST}"
echo "  init        : ${INIT_BOMBEK}"
echo "  epochs      : ${EPOCHS}  lr_head=${LR_HEAD}  lr_lora=${LR_LORA}"
echo "  eff_batch   : ${EFF}"
echo "  output      : ${OUTPUT_DIR}"
echo

cd "${EXP_DIR}"
_ARGS=(
  --train "${TRAIN_MANIFEST}"
  --val "${VAL_MANIFEST}"
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
  --augment "${AUGMENT}"
  --val-slice "${VAL_SLICE}"
  --select-metric "${SELECT_METRIC}"
  --init-bombek "${INIT_BOMBEK}"
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
