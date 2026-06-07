#!/usr/bin/env bash
# Eval Bombek1 pretrained ensemble (SigLIP2+DINOv2+LoRA+head) on XPlainVerse val.
#
# Requires timm — use GHCR -lj image (docker/Dockerfile.lj), NOT ~/xplainverse_exec.sh.
#
# Login node:
#   ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_bombek_pretrained_val_lj.sh
#
# GPU node (inside GHCR -lj shell):
#   bash scripts/run_pass1_bombek_pretrained_val_lj.sh
set -euo pipefail

ensure_pass1_deps() {
  if python3 -c "import timm" 2>/dev/null; then
    return 0
  fi
  local target="${PASS1_PYTARGET:-/tmp/pass1_pkgs}"
  echo "=== installing Pass-1 deps (timm/peft/sklearn) into ${target} ==="
  mkdir -p "${target}"
  pip install --target "${target}" "timm>=1.0.15" peft scikit-learn -q
  export PYTHONPATH="${target}${PYTHONPATH:+:${PYTHONPATH}}"
  python3 -c "import timm; print('timm', timm.__version__)"
}

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-${_SCRIPT_ROOT}}"
EXP_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier"
MANIFEST="${VAL_MANIFEST:-${EXP_DIR}/manifests/manifest_val.parquet}"
CKPT="${CKPT:-bombek}"
VAL_SLICE="${VAL_SLICE:-0}"
_TS="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/pass1_bombek_pretrained/val_${_TS}}"

mkdir -p "${OUT_DIR}"
echo "=== Bombek pretrained Pass-1 val eval ==="
echo "  ckpt:     ${CKPT}"
echo "  manifest: ${MANIFEST} (slice=${VAL_SLICE})"
echo "  out:      ${OUT_DIR}"

ensure_pass1_deps
python3 "${EXP_DIR}/load_ensemble_ckpt.py" --download
python3 "${EXP_DIR}/eval_ensemble.py" \
  --ckpt "${CKPT}" \
  --manifest "${MANIFEST}" \
  --out "${OUT_DIR}" \
  --slice "${VAL_SLICE}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --num-workers "${NUM_WORKERS:-8}"

echo "=== Done ==="
echo "  metrics:     ${OUT_DIR}/metrics.json"
echo "  predictions: ${OUT_DIR}/predictions.parquet"
