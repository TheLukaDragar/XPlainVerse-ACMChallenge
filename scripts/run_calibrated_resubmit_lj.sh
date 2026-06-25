#!/usr/bin/env bash
# Recalibrated submission: p_fake_mean @ 0.11 + flip-patch Pass-2/compressor + CodaBench zip.
#
# Only re-infers explanations for images whose Pass-1 verdict flipped vs deployed
# (p_fake_orig @ 0.084). Pass-1 labels for all 200k are recut from existing TTA scores.
#
# Usage (GPU node or via lj_ghcr_image_exec / sbatch):
#   bash scripts/run_calibrated_resubmit_lj.sh
#
# Env overrides:
#   OUT_DIR, START_STEP (1-6), OLD_COMPLEX, OLD_COMPRESSOR_INFER, ADAPTERS, COMPRESSOR_ADAPTERS
#   RUN_COMPRESSOR_INFER=0  skip compressor GPU step when reusing an existing patch

set -euo pipefail

if [[ -z "${_CALIBRATED_IN_CONTAINER:-}" ]]; then
  export _CALIBRATED_IN_CONTAINER=1
  _INNER="export _CALIBRATED_IN_CONTAINER=1 HOME=/home/jakob"
  for _v in OUT_DIR START_STEP RUN_COMPRESSOR_INFER OLD_COMPLEX OLD_COMPRESSOR_INFER \
            TEST_TTA TEST_MANIFEST NEW_SCORE_COL NEW_THRESHOLD BASE_COMPLEX \
            PASS2_ADAPTERS COMPRESSOR_ADAPTERS SHARD_COUNT; do
    if [[ -n "${!_v:-}" ]]; then _INNER+=" ${_v}=$(printf '%q' "${!_v}")"; fi
  done
  _INNER+="; exec bash scripts/run_calibrated_resubmit_lj.sh"
  exec ./scripts/lj_ghcr_image_exec.sh bash -c "${_INNER}"
fi

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-${_SCRIPT_ROOT}}"
EVAL_DIR="${CODE_ROOT}/evaluation"
MANIFEST_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier/manifests"
PROMPT_FILE="${CODE_ROOT}/dataset/prompt_v2.txt"

TEST_TTA="${TEST_TTA:-/home/jakob/luka/runs/pass1_test_tta/20260605-170149/predictions.parquet}"
TEST_MANIFEST="${TEST_MANIFEST:-${MANIFEST_DIR}/manifest_test.parquet}"
OLD_COMPLEX="${OLD_COMPLEX:-/home/jakob/luka/runs/pass2_test_complex/20260606-195337/complex_explanations.jsonl}"
OLD_COMPRESSOR_INFER="${OLD_COMPRESSOR_INFER:-/home/jakob/luka/runs/compressor_test/20260607-080830/compressor_infer.jsonl}"
PASS2_ADAPTERS="${PASS2_ADAPTERS:-/home/jakob/luka/runs/vlm_v2_grpo/job_48855/checkpoint-26400}"
COMPRESSOR_ADAPTERS="${COMPRESSOR_ADAPTERS:-/home/jakob/luka/runs/compressor_vl/checkpoint-10000}"

NEW_SCORE_COL="${NEW_SCORE_COL:-p_fake_mean}"
NEW_THRESHOLD="${NEW_THRESHOLD:-0.11}"
SHARD_COUNT="${SHARD_COUNT:-4}"

_TS="$(date -u +%Y%m%d-%H%M%S)"
ROOT_OUT="${OUT_DIR:-/home/jakob/luka/runs/submission_calibrated_mean011_${_TS}}"
OUT_DIR="${ROOT_OUT}"
FULL_TEST_MANIFEST="${TEST_MANIFEST}"
START_STEP="${START_STEP:-1}"
mkdir -p "${OUT_DIR}"

export PYTHONNOUSERSITE=1
export PATH="/usr/local/bin:/usr/bin:/bin"
unset CONDA_PREFIX CONDA_DEFAULT_ENV 2>/dev/null || true
for _cuda_lib in cu121 cu12 cu13; do
  _nv="/usr/local/lib/python3.10/dist-packages/nvidia/${_cuda_lib}/lib"
  [[ -d "${_nv}" ]] && { export LD_LIBRARY_PATH="${_nv}:${LD_LIBRARY_PATH:-}"; break; }
done
export TORCH_COMPILE_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0

echo "=== Calibrated resubmit pipeline ==="
echo "  out_dir:     ${OUT_DIR}"
echo "  start_step:  ${START_STEP}"
echo "  operating:   ${NEW_SCORE_COL} @ ${NEW_THRESHOLD}"
echo "  pass2_adapt: ${PASS2_ADAPTERS}"
echo

PASS2_OUT="${OUT_DIR}/pass2_flip"
COMPRESSOR_OUT="${OUT_DIR}/compressor_flip"
PASS1_PRED="${OUT_DIR}/pass1_test_predictions.parquet"
FLIP_MANIFEST="${OUT_DIR}/manifest_test_flips.parquet"
COMPLEX_MERGED="${OUT_DIR}/complex_explanations_merged.jsonl"
COMPRESSOR_MERGED="${OUT_DIR}/compressor_infer_merged.jsonl"
SUBMISSION_JSONL="${OUT_DIR}/submission.jsonl"
SUBMISSION_ZIP="${OUT_DIR}/submission.zip"

# Cross-model safe: flips are computed vs the labels the base explanations were
# conditioned on (BASE_COMPLEX, defaults to OLD_COMPLEX). Required when TEST_TTA
# comes from a different Pass-1 model than the one that produced OLD_COMPLEX.
BASE_COMPLEX="${BASE_COMPLEX:-${OLD_COMPLEX}}"
if [[ "${START_STEP}" -le 1 ]]; then
  echo "=== [1/6] Recut Pass-1 + flip manifest (flips vs base explanations' labels) ==="
  python3 "${EVAL_DIR}/prepare_recalibrated_pass1.py" \
    --test-tta "${TEST_TTA}" \
    --test-manifest "${FULL_TEST_MANIFEST}" \
    --out-dir "${OUT_DIR}" \
    --new-score-col "${NEW_SCORE_COL}" \
    --new-threshold "${NEW_THRESHOLD}" \
    --base-complex "${BASE_COMPLEX}"
fi

N_FLIPS="$(python3 -c "import json; print(json.load(open('${OUT_DIR}/recalibration_summary.json'))['n_flips'])")"
echo "  flips: ${N_FLIPS}"

if [[ "${START_STEP}" -le 2 ]]; then
  mkdir -p "${PASS2_OUT}"
  echo "=== [2/6] Pass-2 complex on ${N_FLIPS} flipped images (${SHARD_COUNT} GPU) ==="
  (
    export PASS1_PRED ADAPTERS="${PASS2_ADAPTERS}" OUT_DIR="${PASS2_OUT}"
    export TEST_MANIFEST="${FLIP_MANIFEST}"
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

COMPLEX_PATCH="${PASS2_OUT}/complex_explanations.jsonl"
echo "  complex_patch: $(wc -l < "${COMPLEX_PATCH}") rows"

if [[ "${START_STEP}" -le 3 ]]; then
echo "=== [3/6] Merge complex explanations ==="
python3 "${EVAL_DIR}/merge_jsonl_by_id.py" \
  --base "${OLD_COMPLEX}" \
  --patch "${COMPLEX_PATCH}" \
  --output "${COMPLEX_MERGED}" \
  --sort-by-id
fi

RUN_COMPRESSOR_INFER="${RUN_COMPRESSOR_INFER:-}"
if [[ -z "${RUN_COMPRESSOR_INFER}" ]]; then
  [[ "${START_STEP}" -le 2 ]] && RUN_COMPRESSOR_INFER=1 || RUN_COMPRESSOR_INFER=0
fi

if [[ "${RUN_COMPRESSOR_INFER}" == 1 ]]; then
  echo "=== [4/6] Compressor on flipped fakes only ==="
  mkdir -p "${COMPRESSOR_OUT}"
  (
    export OUT_DIR="${COMPRESSOR_OUT}" COMPLEX_IN="${COMPLEX_PATCH}" ADAPTERS="${COMPRESSOR_ADAPTERS}"
    export _COMPRESSOR_IN_CONTAINER=1 SHARD_COUNT

    for SHARD in $(seq 0 $((SHARD_COUNT - 1))); do
      CUDA_VISIBLE_DEVICES="${SHARD}" SHARD_ID="${SHARD}" SHARD_COUNT="${SHARD_COUNT}" \
        bash "${CODE_ROOT}/scripts/run_compressor_test_lj.sh" &
    done
    wait

    MERGE_ONLY=1 SHARD_ID=0 SHARD_COUNT="${SHARD_COUNT}" \
      bash "${CODE_ROOT}/scripts/run_compressor_test_lj.sh"
  )
else
  echo "=== [4/6] Compressor infer skipped (RUN_COMPRESSOR_INFER=0) ==="
fi

# Prefer canonical compressor_flip/; fall back to legacy pass2_flip/compressor_flip/ from buggy runs.
if [[ -n "${COMPRESSOR_PATCH_INFER:-}" ]]; then
  :
elif [[ -f "${COMPRESSOR_OUT}/compressor_infer.jsonl" ]]; then
  COMPRESSOR_PATCH_INFER="${COMPRESSOR_OUT}/compressor_infer.jsonl"
else
  COMPRESSOR_PATCH_INFER="${PASS2_OUT}/compressor_flip/compressor_infer.jsonl"
fi
if [[ "${START_STEP}" -le 5 && "${SKIP_COMPRESSOR_MERGE:-0}" != 1 ]]; then
  echo "=== [4b/6] Merge compressor infer (base + flip patch) ==="
  python3 "${EVAL_DIR}/merge_compressor_infer.py" \
    --base-infer "${OLD_COMPRESSOR_INFER}" \
    --base-complex "${OLD_COMPLEX}" \
    --patch-infer "${COMPRESSOR_PATCH_INFER}" \
    --output "${COMPRESSOR_MERGED}"
fi

if [[ "${START_STEP}" -le 5 ]]; then
echo "=== [5/6] Build full submission JSONL ==="
python3 "${EVAL_DIR}/build_simple_explanations.py" \
  --complex "${COMPLEX_MERGED}" \
  --compressor-infer "${COMPRESSOR_MERGED}" \
  --output "${SUBMISSION_JSONL}" \
  --errors-json "${OUT_DIR}/simple_errors.json"
fi

if [[ "${START_STEP}" -le 6 ]]; then
echo "=== [6/6] CodaBench zip ==="
python3 "${CODE_ROOT}/scripts/build_codabench_submission.py" \
  --input "${SUBMISSION_JSONL}" \
  --output "${SUBMISSION_ZIP}" \
  --manifest "${FULL_TEST_MANIFEST}"
fi

python3 - <<PY
import json
from pathlib import Path
import pandas as pd

out = Path("${OUT_DIR}")
summary = json.loads((out / "recalibration_summary.json").read_text())
pass1 = pd.read_parquet(out / "pass1_test_predictions.parquet")
sub_lines = sum(1 for _ in (out / "submission.jsonl").open() if _.strip())
summary["submission_rows"] = sub_lines
summary["submission_zip"] = str(out / "submission.zip")
summary["pass1_pred_fake_rate"] = float(pass1["pred_label"].mean())
(out / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo
echo "=== Done ==="
echo "  submission.jsonl : ${SUBMISSION_JSONL}"
echo "  submission.zip     : ${SUBMISSION_ZIP}"
echo "  Upload submission.zip to CodaBench."
