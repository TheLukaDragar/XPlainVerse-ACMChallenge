#!/usr/bin/env bash
# Pass-1 ensemble on FULL 110k val with horizontal-flip TTA (orig + flip).
# Computes thr_best_f1 on all 110k val labels from p_fake_mean.
#
#   sbatch scripts/sbatch_pass1_val_tta_lj.sbatch
set -euo pipefail

if [[ -z "${_PASS1_VAL_IN_CONTAINER:-}" ]]; then
  _HOST_PROJECT="${HOME}/luka/code/XPlainVerse-ACMChallenge"
  _DEFAULT_REPO_LC="$(echo "${GITHUB_REPOSITORY:-TheLukaDragar/XPlainVerse-ACMChallenge}" | tr '[:upper:]' '[:lower:]')"
  _IMAGE="${LJ_APPTAINER_IMAGE:-docker://ghcr.io/${_DEFAULT_REPO_LC}-lj:latest}"
  _SIF="${HOME}/containers/xplainverse-acmchallenge.sif"
  _BIND="${HOME}:${HOME},/primoz:/primoz"
  [[ -d "${_HOST_PROJECT}" ]] && _BIND="${_BIND},${_HOST_PROJECT}:/workspace/XPlainVerse-ACMChallenge"
  _INNER='export _PASS1_VAL_IN_CONTAINER=1; exec bash scripts/run_pass1_val_tta_lj.sh'
  if [[ -f "${_SIF}" ]]; then
    exec apptainer exec --nv -B "${_BIND}" --pwd /workspace/XPlainVerse-ACMChallenge "${_SIF}" bash -c "${_INNER}"
  else
    exec apptainer exec --nv -B "${_BIND}" --pwd /workspace/XPlainVerse-ACMChallenge "${_IMAGE}" bash -c "${_INNER}"
  fi
fi

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-${_SCRIPT_ROOT}}"
EXP_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier"
MANIFEST_DIR="${MANIFEST_DIR:-${EXP_DIR}/manifests}"
ENS_CKPT="${ENS_CKPT:-/home/jakob/luka/runs/pass1_ensemble/bombek_so400m_dinov2_20260528-225201/best_ckpt/ckpt.pt}"
VAL_MANIFEST="${VAL_MANIFEST:-${MANIFEST_DIR}/manifest_val.parquet}"
VAL_MANIFEST_TTA="${VAL_MANIFEST_TTA:-${MANIFEST_DIR}/manifest_val_tta.parquet}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

_TS="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/pass1_val_tta/${_TS}}"
mkdir -p "${OUT_DIR}"

echo "=== Pass-1 full val TTA (lj) ==="
echo "  ckpt:          ${ENS_CKPT}"
echo "  val_manifest:  ${VAL_MANIFEST}"
echo "  out_dir:       ${OUT_DIR}"
echo

if [[ ! -f "${VAL_MANIFEST_TTA}" ]]; then
  echo "=== build manifest_val_tta (110k orig + 110k flip) ==="
  python3 "${EXP_DIR}/build_manifest_tta.py" \
    --input "${VAL_MANIFEST}" \
    --output "${VAL_MANIFEST_TTA}"
else
  echo "=== reusing ${VAL_MANIFEST_TTA} ==="
fi

echo "=== ensemble inference (220k forwards) ==="
python3 "${EXP_DIR}/eval_ensemble_tta.py" \
  --ckpt "${ENS_CKPT}" \
  --manifest "${VAL_MANIFEST_TTA}" \
  --labels-manifest "${VAL_MANIFEST}" \
  --out "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"

echo
echo "=== Done ==="
echo "  predictions: ${OUT_DIR}/predictions.parquet"
echo "  metrics:     ${OUT_DIR}/metrics.json  (thr_best_f1 on full 110k val)"
