#!/usr/bin/env bash
# Standalone Pass-1 eval on 10k val samples (SigLIP LoRA best checkpoint).
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/jakob/luka/code/XPlainVerse-ACMChallenge}"
EXP_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier"
CKPT="${CKPT:-/home/jakob/luka/runs/pass1_siglip2_giant/lora-r16-bs32-v1-20260528-071134/best_ckpt/ckpt.pt}"
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/pass1_siglip2_giant/lora-r16-bs32-v1-20260528-071134/eval_10k}"
VAL_SLICE="${VAL_SLICE:-10000}"
LOG="${LOG:-/home/jakob/luka/runs/logs/pass1_eval/siglip_lora_eval_10k.log}"

mkdir -p "$(dirname "${LOG}")" "${OUT_DIR}"
export PYTHONNOUSERSITE=1

cd "${EXP_DIR}"
exec python3 eval.py \
  --ckpt "${CKPT}" \
  --manifest "${EXP_DIR}/manifests/manifest_val.parquet" \
  --out "${OUT_DIR}" \
  --slice "${VAL_SLICE}" \
  --batch-size 64 \
  --num-workers 8 \
  2>&1 | tee "${LOG}"
