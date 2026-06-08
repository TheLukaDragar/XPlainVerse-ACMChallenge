#!/usr/bin/env bash
# Pass-1 ensemble FROM-SCRATCH train: pooled train+val, no resampling, focal loss.
#
# vs run_pass1_ensemble_retrain_lj.sh, this:
#   - trains FROM SCRATCH (no Bombek init, no warm-start from any checkpoint) so we
#     measure whether more data alone lifts ranking on a held-out val slice
#   - trains on manifest_trainval_pooled.parquet = ALL train + (val minus a 1000-row
#     holdout), natural distribution, EVERY image used once (no under/oversampling);
#     class imbalance handled by focal loss only
#   - validates on manifest_val_holdout.parquet (1000 stratified val rows) and reports
#     AUC / macro F1 / fake F1 / real F1 so we can see if ranking improved
#   - keeps quality-agnostic augmentation (--augment 1)
#   - 1 epoch over ~560k images
#
# Launch (4x A100 default):
#   LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=16:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_ensemble_trainval_scratch_lj.sh
#
# Smoke (tiny slice, no wandb):
#   TRAIN_SLICE=512 VAL_SLICE=512 REPORT_TO=none NPROC_PER_NODE=1 \
#     CUDA_VISIBLE_DEVICES=0 LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=00:40:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_ensemble_trainval_scratch_lj.sh
#
# NOTE: val is folded into training — the 1000-row holdout is for monitoring only, NOT
# a clean calibration set. For a submission threshold you no longer have a full clean
# val; calibrate against test-prior assumptions instead.

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
LR_HEAD="${LR_HEAD:-2e-4}"
LR_LORA="${LR_LORA:-5e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
AUGMENT="${AUGMENT:-1}"
HOLDOUT_N="${HOLDOUT_N:-1000}"
# holdout is only 1000 rows — validate on all of it (no slice).
VAL_SLICE="${VAL_SLICE:-0}"
SELECT_METRIC="${SELECT_METRIC:-macro_f1}"
# from scratch: no Bombek init, no checkpoint warm-start.
INIT_BOMBEK="${INIT_BOMBEK:-}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${MANIFEST_DIR}/manifest_trainval_pooled.parquet}"
VAL_MANIFEST="${VAL_MANIFEST:-${MANIFEST_DIR}/manifest_val_holdout.parquet}"

_RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${LJ_RUNS_ROOT}/pass1_ensemble/trainval_scratch_${_RUN_TS}}"

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-luka_borut}"
export WANDB_PROJECT="${WANDB_PROJECT:-XPlainVerse-ACMChallenge}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-pass1_trainval_scratch_${_RUN_TS}}"
export WANDB_TAGS="${WANDB_TAGS:-pass1,ensemble,from-scratch,trainval-pooled,macro-f1,${NPROC_PER_NODE}gpu}"

if [[ "${REPORT_TO}" == *wandb* ]]; then
  ENSEMBLE_REPORT_TO=wandb
else
  ENSEMBLE_REPORT_TO=none
fi

if [[ ! -f "${TRAIN_MANIFEST}" || ! -f "${VAL_MANIFEST}" ]]; then
  echo "=== building manifests (missing pooled/holdout) ==="
  CODE_ROOT="${CODE_ROOT}" MANIFEST_DIR="${MANIFEST_DIR}" HOLDOUT_N="${HOLDOUT_N}" \
    python3 "${EXP_DIR}/build_manifest.py"
fi

mkdir -p "${OUTPUT_DIR}"
EFF=$((BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM))
echo "=== Pass-1 ensemble FROM-SCRATCH (pooled train+val, no resampling) ==="
echo "  siglip      : ${SIGLIP}"
echo "  dinov2      : ${DINOV2}"
echo "  image_size  : ${IMAGE_SIZE}"
echo "  train       : ${TRAIN_MANIFEST}"
echo "  val(holdout): ${VAL_MANIFEST}"
echo "  init        : from scratch (no warm-start)"
echo "  augment     : ${AUGMENT}"
echo "  epochs      : ${EPOCHS}  lr_head=${LR_HEAD}  lr_lora=${LR_LORA}"
echo "  lora        : r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "  select      : ${SELECT_METRIC}  val_slice=${VAL_SLICE}  eff_batch=${EFF}"
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
echo "metrics (holdout) in best_ckpt; AUC / macro_f1_at_best / real_f1_at_best / fake_f1_at_macro"
