#!/usr/bin/env bash
# Pass-1 holdout TTA for external warmstart ckpt — honest threshold calibration (1k rows).
#
#   LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=00:30:00 ./scripts/run_pass1_holdout_tta_warmstart_lj.sh
set -euo pipefail

if [[ -z "${_PASS1_HOLDOUT_IN_CONTAINER:-}" ]]; then
  export _PASS1_HOLDOUT_IN_CONTAINER=1
  _INNER="export _PASS1_HOLDOUT_IN_CONTAINER=1 HOME=/home/jakob"
  for _v in ENS_CKPT OUT_DIR BATCH_SIZE NUM_WORKERS MANIFEST_DIR CUDA_VISIBLE_DEVICES; do
    if [[ -n "${!_v:-}" ]]; then
      _INNER+=" ${_v}=$(printf '%q' "${!_v}")"
    fi
  done
  _INNER+="; exec bash scripts/run_pass1_holdout_tta_warmstart_lj.sh"
  exec ./scripts/lj_ghcr_image_exec.sh bash -c "${_INNER}"
fi

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-${_SCRIPT_ROOT}}"
EXP_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier"
MANIFEST_DIR="${MANIFEST_DIR:-${EXP_DIR}/manifests}"
ENS_CKPT="${ENS_CKPT:-/home/jakob/luka/runs/pass1_ensemble/external_all_warmstart_20260612-105218/best_ckpt/ckpt.pt}"
HOLDOUT_MANIFEST="${HOLDOUT_MANIFEST:-${MANIFEST_DIR}/manifest_val_holdout.parquet}"
HOLDOUT_TTA="${HOLDOUT_TTA:-${MANIFEST_DIR}/manifest_val_holdout_tta.parquet}"
TEST_PRED="${TEST_PRED:-/home/jakob/luka/runs/pass1_test_tta_warmstart/predictions.parquet}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

_TS="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/pass1_holdout_tta_warmstart/${_TS}}"
CAL_OUT="${CAL_OUT:-/home/jakob/luka/runs/pass1_calibration/warmstart_holdout_${_TS}.json}"
mkdir -p "${OUT_DIR}" "$(dirname "${CAL_OUT}")"

echo "=== Pass-1 holdout TTA + calibration (warmstart) ==="
echo "  ckpt:           ${ENS_CKPT}"
echo "  holdout:        ${HOLDOUT_MANIFEST}"
echo "  test_preds:     ${TEST_PRED}"
echo "  out_dir:        ${OUT_DIR}"
echo "  calibration:    ${CAL_OUT}"
echo

if [[ ! -f "${HOLDOUT_TTA}" ]]; then
  echo "=== build manifest_val_holdout_tta (1k orig + 1k flip) ==="
  python3 "${EXP_DIR}/build_manifest_tta.py" \
    --input "${HOLDOUT_MANIFEST}" \
    --output "${HOLDOUT_TTA}"
else
  echo "=== reusing ${HOLDOUT_TTA} ==="
fi

echo "=== ensemble inference (2k forwards) ==="
python3 "${EXP_DIR}/eval_ensemble_tta.py" \
  --ckpt "${ENS_CKPT}" \
  --manifest "${HOLDOUT_TTA}" \
  --labels-manifest "${HOLDOUT_MANIFEST}" \
  --out "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"

echo
echo "=== calibrate threshold (holdout labels, test prior) ==="
python3 "${EXP_DIR}/calibrate_macrof1.py" \
  --val-pred "${OUT_DIR}/predictions.parquet" \
  --test-pred "${TEST_PRED}" \
  --deployed-score-col p_fake_mean \
  --deployed-threshold 0.47 \
  --out "${CAL_OUT}"

echo
echo "=== Done ==="
echo "  holdout preds: ${OUT_DIR}/predictions.parquet"
echo "  holdout metrics: ${OUT_DIR}/metrics.json"
echo "  calibration: ${CAL_OUT}"
