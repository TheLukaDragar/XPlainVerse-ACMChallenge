#!/usr/bin/env bash
# Pass-1 classifier: build manifests → train DINOv3 + linear → eval on val.
#
# DO NOT LAUNCH UNTIL research/experiments/01_gold_verdict/ shows
#   conditioned complex_overall ≥ baseline + 0.03
# (see ../01_gold_verdict/README.md for decision rule).
#
# Usage:
#   ./run.sh                 # builds manifests if missing, trains 2 epochs frozen-backbone
#   LORA=1 ./run.sh          # LoRA the backbone too (slower, +capacity)

set -euo pipefail

WORKSPACE_ROOT=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge
EXP_DIR="${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge/research/experiments/02_pass1_classifier"
RUN_OUT="${WORKSPACE_ROOT}/runs/pass1_dinov3/v1-$(date -u +%Y%m%d-%H%M%S)"
MANIFEST_DIR="${EXP_DIR}/manifests"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

LORA="${LORA:-0}"
BACKBONE="${BACKBONE:-facebook/dinov3-vitl16-pretrain-lvd1689m}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Step 1 — build manifests if missing
if [[ ! -f "${MANIFEST_DIR}/manifest_train_balanced.parquet" ]]; then
  echo "=== building manifests ==="
  python3 "${EXP_DIR}/build_manifest.py"
fi

mkdir -p "${RUN_OUT}"
echo "=== Pass-1 training ==="
echo "  backbone   : ${BACKBONE}"
echo "  LoRA       : ${LORA}"
echo "  epochs     : ${EPOCHS}"
echo "  batch_size : ${BATCH_SIZE}"
echo "  output dir : ${RUN_OUT}"
echo

python3 "${EXP_DIR}/train_pass1.py" \
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
