#!/usr/bin/env bash
# Standalone 10k val eval for Bombek ensemble checkpoint.
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
EXP_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier"
CKPT="${CKPT:?set CKPT=path/to/best_ckpt/ckpt.pt}"
OUT_DIR="${OUT_DIR:?set OUT_DIR=path/to/eval_out}"
VAL_SLICE="${VAL_SLICE:-10000}"

exec ./scripts/lj_ghcr_image_exec.sh python3 "${EXP_DIR}/eval_ensemble.py" \
  --ckpt "${CKPT}" \
  --manifest "${EXP_DIR}/manifests/manifest_val.parquet" \
  --out "${OUT_DIR}" \
  --slice "${VAL_SLICE}" \
  --batch-size "${BATCH_SIZE:-32}"
