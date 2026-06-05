#!/usr/bin/env bash
# Pass-1 ensemble on full test split with horizontal-flip TTA (2 preds per image).
#
# From Slurm login:
#   LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=08:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_test_tta_lj.sh
#
# On elixir-lj-gpu-01 directly:
#   ~/xplainverse_exec.sh bash scripts/run_pass1_test_tta_lj.sh
#
# Env:
#   ENS_CKPT          ensemble checkpoint (default: bombek best)
#   TEST_IMAGES_DIR   flat test images dir on primoz NVMe
#   OUT_DIR           output dir for predictions
#   THRESHOLD         decision threshold for pred_label_mean (default 0.129)
#   BATCH_SIZE        inference batch size (default 32)
set -euo pipefail

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CODE_ROOT:-}" ]]; then :; elif [[ -d "${_SCRIPT_ROOT}/evaluation" ]]; then CODE_ROOT="${_SCRIPT_ROOT}";
elif [[ -d /workspace/XPlainVerse-ACMChallenge/evaluation ]]; then CODE_ROOT="/workspace/XPlainVerse-ACMChallenge";
else CODE_ROOT="${HOME}/luka/code/XPlainVerse-ACMChallenge"; fi

EXP_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier"
MANIFEST_DIR="${MANIFEST_DIR:-${EXP_DIR}/manifests}"
TEST_IMAGES_DIR="${TEST_IMAGES_DIR:-/primoz/luka/XPlainVerse/data/XPlainVerse/test/images}"
ENS_CKPT="${ENS_CKPT:-/home/jakob/luka/runs/pass1_ensemble/bombek_so400m_dinov2_20260528-225201/best_ckpt/ckpt.pt}"
THRESHOLD="${THRESHOLD:-0.129}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSITE:-1}"

_TS="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/pass1_test_tta/${_TS}}"
mkdir -p "${OUT_DIR}"

MANIFEST_TEST="${MANIFEST_DIR}/manifest_test.parquet"
MANIFEST_TTA="${MANIFEST_DIR}/manifest_test_tta.parquet"

echo "=== Pass-1 test TTA (lj) ==="
echo "  code_root:       ${CODE_ROOT}"
echo "  test_images_dir: ${TEST_IMAGES_DIR}"
echo "  ensemble ckpt:   ${ENS_CKPT}"
echo "  threshold:       ${THRESHOLD}"
echo "  out_dir:         ${OUT_DIR}"
echo

if [[ ! -f "${MANIFEST_TTA}" ]]; then
  echo "=== building test manifests (orig + flip rows) ==="
  python3 "${EXP_DIR}/build_test_manifest.py" \
    --test-images-dir "${TEST_IMAGES_DIR}" \
    --out-dir "${MANIFEST_DIR}" \
    --tta
else
  echo "=== reusing manifests in ${MANIFEST_DIR} ==="
fi

echo "=== ensemble inference (400k forwards = 200k orig + 200k flip) ==="
python3 "${EXP_DIR}/eval_ensemble_test.py" \
  --ckpt "${ENS_CKPT}" \
  --manifest "${MANIFEST_TTA}" \
  --out "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --threshold "${THRESHOLD}"

echo
echo "=== Done ==="
echo "  wide preds : ${OUT_DIR}/predictions.parquet"
echo "  long preds : ${OUT_DIR}/predictions_long.parquet"
echo "  summary    : ${OUT_DIR}/metrics.json"
