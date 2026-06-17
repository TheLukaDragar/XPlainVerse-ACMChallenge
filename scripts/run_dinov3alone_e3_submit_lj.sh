#!/usr/bin/env bash
# DINOv3-alone Path A: test TTA (epoch-3 ckpt) → flip-patch submission.
#
# Usage (login node):
#   ./scripts/lj_ghcr_image_exec.sh bash scripts/run_dinov3alone_e3_submit_lj.sh
#
# Env overrides:
#   CKPT, OUT_TTA, NEW_THRESHOLD (default 0.338 = 2-epoch holdout macro-F1), OUT_DIR, START_STEP

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
CKPT="${CKPT:-/home/jakob/luka/runs/pass1_dinov3_mac/dinov3_mac_20260617-123523/best_ckpt/ckpt.pt}"
OUT_TTA="${OUT_TTA:-/home/jakob/luka/runs/pass1_dinov3_test_tta/e3_full}"
# Holdout macro-F1 optimal from 2-epoch DINOv3 calibration (same as dinov3alone_338).
NEW_THRESHOLD="${NEW_THRESHOLD:-0.338}"
BASE_COMPLEX="${BASE_COMPLEX:-/home/jakob/luka/runs/pass2_test_complex/20260606-195337/complex_explanations.jsonl}"
OLD_COMPRESSOR_INFER="${OLD_COMPRESSOR_INFER:-/home/jakob/luka/runs/compressor_test/20260607-080830/compressor_infer.jsonl}"
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/submission_dinov3alone_e3}"
START_STEP="${START_STEP:-1}"

echo "=== DINOv3-alone e3 submission pipeline ==="
echo "  ckpt       : ${CKPT}"
echo "  out_tta    : ${OUT_TTA}"
echo "  threshold  : ${NEW_THRESHOLD} (holdout macro-F1, 2-epoch DINO)"
echo "  out_dir    : ${OUT_DIR}"
echo "  start_step : ${START_STEP}"
echo

if [[ "${START_STEP}" -le 1 ]]; then
  echo "=== [1/2] DINOv3 test TTA (GHCR image, ~20 min) ==="
  CKPT="${CKPT}" OUT="${OUT_TTA}" \
    bash "${CODE_ROOT}/scripts/lj_ghcr_image_exec.sh" \
    bash "${CODE_ROOT}/scripts/run_dinov3_test_tta_lj.sh"
fi

if [[ ! -f "${OUT_TTA}/predictions.parquet" ]]; then
  echo "error: missing ${OUT_TTA}/predictions.parquet" >&2
  exit 1
fi

echo "=== [2/2] Flip-patch @ ${NEW_THRESHOLD} ==="
export _CALIBRATED_IN_CONTAINER=1
OUT_DIR="${OUT_DIR}" START_STEP=1 \
  TEST_TTA="${OUT_TTA}/predictions.parquet" \
  NEW_THRESHOLD="${NEW_THRESHOLD}" \
  NEW_SCORE_COL="p_fake_mean" \
  BASE_COMPLEX="${BASE_COMPLEX}" \
  OLD_COMPLEX="${BASE_COMPLEX}" \
  OLD_COMPRESSOR_INFER="${OLD_COMPRESSOR_INFER}" \
  bash "${CODE_ROOT}/scripts/run_calibrated_resubmit_lj.sh"

echo "=== Done ==="
echo "  submission.zip: ${OUT_DIR}/submission.zip"
