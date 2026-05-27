#!/usr/bin/env bash
# Pass-1 classifier: build manifests → train → eval on val.
#
# Usage:
#   ./run.sh                 # 2 epochs, frozen backbone
#   LORA=1 ./run.sh          # LoRA the backbone too
#
# Prefer Lj launcher: scripts/run_pass1_lj.sh

set -euo pipefail

if [[ -d /workspace/XPlainVerse-ACMChallenge ]]; then
  CODE_ROOT="/workspace/XPlainVerse-ACMChallenge"
else
  CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

EXP_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier"
RUN_OUT="${OUTPUT_DIR:-${CODE_ROOT%/code/*}/runs/pass1_dinov3/v1-$(date -u +%Y%m%d-%H%M%S)}"
MANIFEST_DIR="${MANIFEST_DIR:-${EXP_DIR}/manifests}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LORA="${LORA:-0}"
BACKBONE="${BACKBONE:-facebook/dinov3-vitl16-pretrain-lvd1689m}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"

if [[ ! -f "${MANIFEST_DIR}/manifest_train_balanced.parquet" ]]; then
  echo "=== building manifests ==="
  CODE_ROOT="${CODE_ROOT}" MANIFEST_DIR="${MANIFEST_DIR}" python3 "${EXP_DIR}/build_manifest.py"
fi

mkdir -p "${RUN_OUT}"
echo "=== Pass-1 training ==="
echo "  backbone   : ${BACKBONE}"
echo "  LoRA       : ${LORA}"
echo "  epochs     : ${EPOCHS}"
echo "  batch_size : ${BATCH_SIZE}"
echo "  output dir : ${RUN_OUT}"
echo

cd "${EXP_DIR}"
python3 train.py \
  --train "${MANIFEST_DIR}/manifest_train_balanced.parquet" \
  --val "${MANIFEST_DIR}/manifest_val.parquet" \
  --out "${RUN_OUT}" \
  --backbone "${BACKBONE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lora "${LORA}"

echo
echo "=== Done ==="
echo "metrics: ${RUN_OUT}/metrics.json"
echo "preds:   ${RUN_OUT}/val_predictions.parquet"
