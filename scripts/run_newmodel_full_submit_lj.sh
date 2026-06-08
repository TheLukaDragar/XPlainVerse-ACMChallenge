#!/usr/bin/env bash
# Full new-model submission: from-scratch+val Pass-1 detector @ p_fake_mean 0.375,
# FULL pass-2 complex over all 200k test images (conditioned on the new verdicts),
# full compressor, CodaBench zip. No flip-patch / no reuse of old explanations.
#
# Unlike run_calibrated_resubmit_lj.sh (which patched only flipped rows), this
# regenerates every complex explanation against the new detector's labels — cleaner,
# avoids id-merge bugs.
#
# Usage (GPU node or via lj_ghcr_image_exec / sbatch):
#   LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=08:00:00 bash scripts/run_newmodel_full_submit_lj.sh
#
# Env overrides:
#   OUT_DIR, START_STEP (1-5), NEW_TTA, NEW_SCORE_COL, NEW_THRESHOLD,
#   PASS2_ADAPTERS, COMPRESSOR_ADAPTERS, SHARD_COUNT

set -euo pipefail

if [[ -z "${_NEWMODEL_IN_CONTAINER:-}" ]]; then
  export _NEWMODEL_IN_CONTAINER=1
  _INNER="export _NEWMODEL_IN_CONTAINER=1 HOME=/home/jakob"
  for _v in OUT_DIR START_STEP NEW_TTA NEW_SCORE_COL NEW_THRESHOLD PASS2_ADAPTERS COMPRESSOR_ADAPTERS SHARD_COUNT; do
    if [[ -n "${!_v:-}" ]]; then _INNER+=" ${_v}=$(printf '%q' "${!_v}")"; fi
  done
  _INNER+="; exec bash scripts/run_newmodel_full_submit_lj.sh"
  exec ./scripts/lj_ghcr_image_exec.sh bash -c "${_INNER}"
fi

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-${_SCRIPT_ROOT}}"
EVAL_DIR="${CODE_ROOT}/evaluation"
MANIFEST_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier/manifests"

NEW_TTA="${NEW_TTA:-/home/jakob/luka/runs/pass1_test_tta_newmodel/predictions.parquet}"
FULL_TEST_MANIFEST="${FULL_TEST_MANIFEST:-${MANIFEST_DIR}/manifest_test.parquet}"
NEW_SCORE_COL="${NEW_SCORE_COL:-p_fake_mean}"
NEW_THRESHOLD="${NEW_THRESHOLD:-0.375}"
PASS2_ADAPTERS="${PASS2_ADAPTERS:-/home/jakob/luka/runs/vlm_v2_grpo/job_48855/checkpoint-26400}"
COMPRESSOR_ADAPTERS="${COMPRESSOR_ADAPTERS:-/home/jakob/luka/runs/compressor_vl/checkpoint-10000}"
SHARD_COUNT="${SHARD_COUNT:-4}"
START_STEP="${START_STEP:-1}"

_TS="$(date -u +%Y%m%d-%H%M%S)"
ROOT_OUT="${OUT_DIR:-/home/jakob/luka/runs/submission_newmodel_full_${_TS}}"
OUT_DIR="${ROOT_OUT}"
mkdir -p "${OUT_DIR}"

export PYTHONNOUSERSITE=1
export PATH="/usr/local/bin:/usr/bin:/bin"
unset CONDA_PREFIX CONDA_DEFAULT_ENV 2>/dev/null || true
for _cuda_lib in cu121 cu12 cu13; do
  _nv="/usr/local/lib/python3.10/dist-packages/nvidia/${_cuda_lib}/lib"
  [[ -d "${_nv}" ]] && { export LD_LIBRARY_PATH="${_nv}:${LD_LIBRARY_PATH:-}"; break; }
done
export TORCH_COMPILE_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0

PASS1_PRED="${OUT_DIR}/pass1_test_predictions.parquet"
PASS2_OUT="${OUT_DIR}/pass2_full"
COMPRESSOR_OUT="${OUT_DIR}/compressor_full"
COMPLEX_OUT="${PASS2_OUT}/complex_explanations.jsonl"
SUBMISSION_JSONL="${COMPRESSOR_OUT}/submission.jsonl"
SUBMISSION_ZIP="${OUT_DIR}/submission.zip"

echo "=== New-model FULL submission pipeline ==="
echo "  out_dir:     ${OUT_DIR}"
echo "  start_step:  ${START_STEP}"
echo "  new_tta:     ${NEW_TTA}"
echo "  operating:   ${NEW_SCORE_COL} @ ${NEW_THRESHOLD}"
echo "  pass2_adapt: ${PASS2_ADAPTERS}"
echo

if [[ "${START_STEP}" -le 1 ]]; then
  echo "=== [1/5] Recut Pass-1 (new model) ==="
  python3 "${EVAL_DIR}/prepare_recalibrated_pass1.py" \
    --test-tta "${NEW_TTA}" \
    --test-manifest "${FULL_TEST_MANIFEST}" \
    --out-dir "${OUT_DIR}" \
    --new-score-col "${NEW_SCORE_COL}" \
    --new-threshold "${NEW_THRESHOLD}"
fi

if [[ "${START_STEP}" -le 2 ]]; then
  mkdir -p "${PASS2_OUT}"
  echo "=== [2/5] FULL Pass-2 complex over 200k (${SHARD_COUNT} GPU) ==="
  (
    export PASS1_PRED ADAPTERS="${PASS2_ADAPTERS}" OUT_DIR="${PASS2_OUT}"
    export TEST_MANIFEST="${FULL_TEST_MANIFEST}"
    export THRESHOLD="${NEW_THRESHOLD}" PASS1_SCORE_COL="${NEW_SCORE_COL}"
    export _PASS2_TEST_IN_CONTAINER=1 SHARD_COUNT

    for SHARD in $(seq 0 $((SHARD_COUNT - 1))); do
      CUDA_VISIBLE_DEVICES="${SHARD}" SHARD_ID="${SHARD}" SHARD_COUNT="${SHARD_COUNT}" \
        bash "${CODE_ROOT}/scripts/run_pass2_test_complex_lj.sh" &
    done
    wait

    MERGE_ONLY=1 SHARD_ID=0 SHARD_COUNT="${SHARD_COUNT}" \
      bash "${CODE_ROOT}/scripts/run_pass2_test_complex_lj.sh"
  )
fi
echo "  complex: $(wc -l < "${COMPLEX_OUT}") rows"

if [[ "${START_STEP}" -le 3 ]]; then
  mkdir -p "${COMPRESSOR_OUT}"
  echo "=== [3/5] FULL compressor on all fakes (${SHARD_COUNT} GPU) ==="
  (
    export OUT_DIR="${COMPRESSOR_OUT}" COMPLEX_IN="${COMPLEX_OUT}" ADAPTERS="${COMPRESSOR_ADAPTERS}"
    export _COMPRESSOR_IN_CONTAINER=1 SHARD_COUNT

    for SHARD in $(seq 0 $((SHARD_COUNT - 1))); do
      CUDA_VISIBLE_DEVICES="${SHARD}" SHARD_ID="${SHARD}" SHARD_COUNT="${SHARD_COUNT}" \
        bash "${CODE_ROOT}/scripts/run_compressor_test_lj.sh" &
    done
    wait

    MERGE_ONLY=1 SHARD_ID=0 SHARD_COUNT="${SHARD_COUNT}" \
      bash "${CODE_ROOT}/scripts/run_compressor_test_lj.sh"
  )
fi
echo "  submission_jsonl: $(wc -l < "${SUBMISSION_JSONL}") rows"

if [[ "${START_STEP}" -le 4 ]]; then
  echo "=== [4/5] CodaBench zip ==="
  python3 "${CODE_ROOT}/scripts/build_codabench_submission.py" \
    --input "${SUBMISSION_JSONL}" \
    --output "${SUBMISSION_ZIP}" \
    --manifest "${FULL_TEST_MANIFEST}"
fi

if [[ "${START_STEP}" -le 5 ]]; then
  echo "=== [5/5] Summary ==="
  python3 - <<PY
import json
from pathlib import Path
import pandas as pd

out = Path("${OUT_DIR}")
pass1 = pd.read_parquet(out / "pass1_test_predictions.parquet")
sub_lines = sum(1 for _ in (Path("${SUBMISSION_JSONL}")).open() if _.strip())
summary = {
    "new_tta": "${NEW_TTA}",
    "score_col": "${NEW_SCORE_COL}",
    "threshold": ${NEW_THRESHOLD},
    "n_test": int(len(pass1)),
    "pred_fake_rate": float(pass1["pred_label"].mean()),
    "submission_rows": sub_lines,
    "submission_zip": "${SUBMISSION_ZIP}",
}
(out / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
fi

echo
echo "=== Done ==="
echo "  submission.jsonl : ${SUBMISSION_JSONL}"
echo "  submission.zip   : ${SUBMISSION_ZIP}"
echo "  Upload submission.zip to CodaBench."
