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
#   OUT_DIR, OLD_COMPLEX, OLD_COMPRESSOR_INFER, ADAPTERS, COMPRESSOR_ADAPTERS

set -euo pipefail

if [[ -z "${_CALIBRATED_IN_CONTAINER:-}" ]]; then
  export _CALIBRATED_IN_CONTAINER=1
  exec ./scripts/lj_ghcr_image_exec.sh bash -c 'export _CALIBRATED_IN_CONTAINER=1 HOME=/home/jakob; exec bash scripts/run_calibrated_resubmit_lj.sh'
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
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/submission_calibrated_mean011_${_TS}}"
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
echo "  operating:   ${NEW_SCORE_COL} @ ${NEW_THRESHOLD}"
echo "  pass2_adapt: ${PASS2_ADAPTERS}"
echo

echo "=== [1/6] Recut Pass-1 + flip manifest ==="
python3 "${EVAL_DIR}/prepare_recalibrated_pass1.py" \
  --test-tta "${TEST_TTA}" \
  --test-manifest "${TEST_MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --new-score-col "${NEW_SCORE_COL}" \
  --new-threshold "${NEW_THRESHOLD}"

PASS1_PRED="${OUT_DIR}/pass1_test_predictions.parquet"
FLIP_MANIFEST="${OUT_DIR}/manifest_test_flips.parquet"
N_FLIPS="$(python3 -c "import json; print(len(json.load(open('${OUT_DIR}/recalibration_summary.json'))['flip_sample_ids']))")"
echo "  flips: ${N_FLIPS}"

PASS2_OUT="${OUT_DIR}/pass2_flip"
mkdir -p "${PASS2_OUT}"

echo "=== [2/6] Pass-2 complex on ${N_FLIPS} flipped images (${SHARD_COUNT} GPU) ==="
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

COMPLEX_PATCH="${PASS2_OUT}/complex_explanations.jsonl"
echo "  complex_patch: $(wc -l < "${COMPLEX_PATCH}") rows"

echo "=== [3/6] Merge complex explanations ==="
COMPLEX_MERGED="${OUT_DIR}/complex_explanations_merged.jsonl"
python3 "${EVAL_DIR}/merge_jsonl_by_id.py" \
  --base "${OLD_COMPLEX}" \
  --patch "${COMPLEX_PATCH}" \
  --output "${COMPLEX_MERGED}" \
  --sort-by-id

echo "=== [4/6] Compressor on flipped fakes only ==="
COMPRESSOR_OUT="${OUT_DIR}/compressor_flip"
mkdir -p "${COMPRESSOR_OUT}"
export OUT_DIR="${COMPRESSOR_OUT}" COMPLEX_IN="${COMPLEX_PATCH}" ADAPTERS="${COMPRESSOR_ADAPTERS}"
export _COMPRESSOR_IN_CONTAINER=1

for SHARD in $(seq 0 $((SHARD_COUNT - 1))); do
  CUDA_VISIBLE_DEVICES="${SHARD}" SHARD_ID="${SHARD}" SHARD_COUNT="${SHARD_COUNT}" \
    bash "${CODE_ROOT}/scripts/run_compressor_test_lj.sh" &
done
wait

MERGE_ONLY=1 SHARD_ID=0 SHARD_COUNT="${SHARD_COUNT}" \
  bash "${CODE_ROOT}/scripts/run_compressor_test_lj.sh"

# run_compressor merge writes submission but only from patch complex — grab infer only.
COMPRESSOR_PATCH_INFER="${COMPRESSOR_OUT}/compressor_infer.jsonl"
COMPRESSOR_MERGED="${OUT_DIR}/compressor_infer_merged.jsonl"
python3 "${EVAL_DIR}/merge_jsonl_by_id.py" \
  --base "${OLD_COMPRESSOR_INFER}" \
  --patch "${COMPRESSOR_PATCH_INFER}" \
  --output "${COMPRESSOR_MERGED}"

echo "=== [5/6] Build full submission JSONL ==="
SUBMISSION_JSONL="${OUT_DIR}/submission.jsonl"
python3 "${EVAL_DIR}/build_simple_explanations.py" \
  --complex "${COMPLEX_MERGED}" \
  --compressor-infer "${COMPRESSOR_MERGED}" \
  --output "${SUBMISSION_JSONL}" \
  --errors-json "${OUT_DIR}/simple_errors.json"

echo "=== [6/6] CodaBench zip ==="
SUBMISSION_ZIP="${OUT_DIR}/submission.zip"
python3 "${CODE_ROOT}/scripts/build_codabench_submission.py" \
  --input "${SUBMISSION_JSONL}" \
  --output "${SUBMISSION_ZIP}" \
  --manifest "${TEST_MANIFEST}"

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
