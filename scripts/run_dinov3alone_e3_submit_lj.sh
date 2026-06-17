#!/usr/bin/env bash
# DINOv3-alone Path A: test TTA (epoch-3 ckpt) → threshold pick → flip-patch submission.
#
# Usage (login node):
#   ./scripts/lj_ghcr_image_exec.sh bash scripts/run_dinov3alone_e3_submit_lj.sh
#
# Env overrides:
#   CKPT, OUT_TTA, NEW_THRESHOLD (skip auto-pick), OUT_DIR, START_STEP (1-6)

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
CKPT="${CKPT:-/home/jakob/luka/runs/pass1_dinov3_mac/dinov3_mac_20260617-123523/best_ckpt/ckpt.pt}"
OUT_TTA="${OUT_TTA:-/home/jakob/luka/runs/pass1_dinov3_test_tta/e3_full}"
OLD_TTA="${OLD_TTA:-/home/jakob/luka/runs/pass1_dinov3_test_tta/20260617-091636/predictions.parquet}"
BASE_COMPLEX="${BASE_COMPLEX:-/home/jakob/luka/runs/pass2_test_complex/20260606-195337/complex_explanations.jsonl}"
OLD_COMPRESSOR_INFER="${OLD_COMPRESSOR_INFER:-/home/jakob/luka/runs/compressor_test/20260607-080830/compressor_infer.jsonl}"
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/submission_dinov3alone_e3}"
START_STEP="${START_STEP:-1}"

echo "=== DINOv3-alone e3 submission pipeline ==="
echo "  ckpt      : ${CKPT}"
echo "  out_tta   : ${OUT_TTA}"
echo "  out_dir   : ${OUT_DIR}"
echo "  start_step: ${START_STEP}"
echo

if [[ "${START_STEP}" -le 1 ]]; then
  echo "=== [1/2] DINOv3 test TTA ==="
  CKPT="${CKPT}" OUT="${OUT_TTA}" \
    bash "${CODE_ROOT}/scripts/run_dinov3_test_tta_lj.sh"
fi

if [[ ! -f "${OUT_TTA}/predictions.parquet" ]]; then
  echo "error: missing ${OUT_TTA}/predictions.parquet" >&2
  exit 1
fi

if [[ -z "${NEW_THRESHOLD:-}" ]]; then
  echo "=== [2a] Auto-pick threshold (match old e2 fake-rate @ 0.338) ==="
  NEW_THRESHOLD="$(python3 - <<PY
import numpy as np
import pandas as pd

old = pd.read_parquet("${OLD_TTA}")
new = pd.read_parquet("${OUT_TTA}/predictions.parquet")
ref_thr = 0.338
ref_rate = float((old["p_fake_mean"] >= ref_thr).mean())
scores = new["p_fake_mean"].astype(float).values
# Binary search threshold to match reference fake rate.
lo, hi = 0.0, 1.0
for _ in range(64):
    mid = (lo + hi) / 2
    if float((scores >= mid).mean()) > ref_rate:
        lo = mid
    else:
        hi = mid
thr = (lo + hi) / 2
rate = float((scores >= thr).mean())
print(f"ref_rate={ref_rate:.5f} thr={thr:.4f} new_rate={rate:.5f}", flush=True)
print(f"{thr:.4f}")
PY
)"
  echo "  picked threshold: ${NEW_THRESHOLD}"
else
  echo "=== [2a] Using fixed threshold: ${NEW_THRESHOLD} ==="
fi

echo "=== [2b] Flip-patch submission ==="
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
