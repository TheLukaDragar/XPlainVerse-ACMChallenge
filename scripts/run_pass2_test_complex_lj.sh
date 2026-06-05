#!/usr/bin/env bash
# Pass-2 VLM on full test split — complex explanations only (simple later).
#
# Pass-1 verdicts: precomputed TTA ensemble predictions.parquet (p_fake_mean).
# Model: GRPO checkpoint-26400 by default.
#
# Sharding (4 GPUs on one node):
#   SHARD_ID=0 SHARD_COUNT=4 CUDA_VISIBLE_DEVICES=0 bash scripts/run_pass2_test_complex_lj.sh
#
# From Slurm login:
#   sbatch scripts/sbatch_pass2_test_complex_lj.sbatch
#
# Env:
#   PASS1_PRED      Pass-1 wide predictions.parquet
#   TEST_MANIFEST   manifest_test.parquet
#   ADAPTERS        GRPO/SFT LoRA checkpoint
#   SHARD_ID        0..SHARD_COUNT-1 (default 0)
#   SHARD_COUNT     parallel shards (default 1; sbatch uses 4)
#   MERGE_ONLY      1 = only merge shards + build complex JSONL (skip infer)
set -euo pipefail

if [[ -z "${_PASS2_TEST_IN_CONTAINER:-}" ]]; then
  _HOST_PROJECT="${HOME}/luka/code/XPlainVerse-ACMChallenge"
  _DEFAULT_REPO_LC="$(echo "${GITHUB_REPOSITORY:-TheLukaDragar/XPlainVerse-ACMChallenge}" | tr '[:upper:]' '[:lower:]')"
  _IMAGE="${LJ_APPTAINER_IMAGE:-docker://ghcr.io/${_DEFAULT_REPO_LC}-lj:latest}"
  _SIF="${HOME}/containers/xplainverse-acmchallenge.sif"
  _BIND="${HOME}:${HOME},/primoz:/primoz"
  [[ -d "${_HOST_PROJECT}" ]] && _BIND="${_BIND},${_HOST_PROJECT}:/workspace/XPlainVerse-ACMChallenge"
  export _PASS2_TEST_IN_CONTAINER=1
  _INNER='export _PASS2_TEST_IN_CONTAINER=1; exec bash scripts/run_pass2_test_complex_lj.sh'
  if [[ -f "${_SIF}" ]]; then
    exec apptainer exec --nv -B "${_BIND}" --pwd /workspace/XPlainVerse-ACMChallenge "${_SIF}" bash -c "${_INNER}"
  else
    exec apptainer exec --nv -B "${_BIND}" --pwd /workspace/XPlainVerse-ACMChallenge "${_IMAGE}" bash -c "${_INNER}"
  fi
fi

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CODE_ROOT:-}" ]]; then :; elif [[ -d "${_SCRIPT_ROOT}/evaluation" ]]; then CODE_ROOT="${_SCRIPT_ROOT}";
elif [[ -d /workspace/XPlainVerse-ACMChallenge/evaluation ]]; then CODE_ROOT="/workspace/XPlainVerse-ACMChallenge";
else CODE_ROOT="${HOME}/luka/code/XPlainVerse-ACMChallenge"; fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
for _cuda_lib in cu121 cu12 cu13; do
  _nv="/usr/local/lib/python3.10/dist-packages/nvidia/${_cuda_lib}/lib"
  [[ -d "${_nv}" ]] && { export LD_LIBRARY_PATH="${_nv}:${LD_LIBRARY_PATH:-}"; break; }
done
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
_SHIM="${CODE_ROOT}/scripts/lj_swift_compat"
[[ -f "${_SHIM}/sitecustomize.py" ]] && export PYTHONPATH="${_SHIM}:${PYTHONPATH:-}"

EVAL_DIR="${CODE_ROOT}/evaluation"
MANIFEST_DIR="${MANIFEST_DIR:-${CODE_ROOT}/research/experiments/02_pass1_classifier/manifests}"
PROMPT_FILE="${PROMPT_FILE:-${CODE_ROOT}/dataset/prompt.txt}"
TEST_MANIFEST="${TEST_MANIFEST:-${MANIFEST_DIR}/manifest_test.parquet}"
PASS1_PRED="${PASS1_PRED:-/home/jakob/luka/runs/pass1_test_tta/20260605-170149/predictions.parquet}"
ADAPTERS="${ADAPTERS:-/home/jakob/luka/runs/vlm_v2_grpo/job_48855/checkpoint-26400}"
THRESHOLD="${THRESHOLD:-0.129}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SHARD_ID="${SHARD_ID:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
MERGE_ONLY="${MERGE_ONLY:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

_TS="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/pass2_test_complex/${_TS}}"
mkdir -p "${OUT_DIR}"

CONDITIONED="${OUT_DIR}/pass2_infer_shard${SHARD_ID}.jsonl"
INFER_SHARD="${OUT_DIR}/infer_shard${SHARD_ID}.jsonl"
INFER_MERGED="${OUT_DIR}/infer.jsonl"
COMPLEX_OUT="${OUT_DIR}/complex_explanations.jsonl"

echo "=== Pass-2 test complex (lj) ==="
echo "  code_root:     ${CODE_ROOT}"
echo "  adapters:      ${ADAPTERS}"
echo "  pass1_pred:    ${PASS1_PRED}"
echo "  test_manifest: ${TEST_MANIFEST}"
echo "  shard:         ${SHARD_ID}/${SHARD_COUNT}  GPU=${CUDA_VISIBLE_DEVICES}"
echo "  out_dir:       ${OUT_DIR}"
echo

if [[ "${MERGE_ONLY}" != "1" ]]; then
  echo "=== [1/3] build conditioned infer shard ==="
  if [[ -s "${CONDITIONED}" ]]; then
    echo "  [resume] ${CONDITIONED}"
  else
    python3 "${EVAL_DIR}/build_pass2_infer_test.py" \
      --test-manifest "${TEST_MANIFEST}" \
      --ensemble-pred "${PASS1_PRED}" \
      --prompt-file "${PROMPT_FILE}" \
      --threshold "${THRESHOLD}" \
      --shard-id "${SHARD_ID}" \
      --shard-count "${SHARD_COUNT}" \
      --out "${CONDITIONED}" \
      --verdicts-json "${OUT_DIR}/pass1_verdicts_shard${SHARD_ID}.json"
  fi

  echo "=== [2/3] VLM infer (${ADAPTERS##*/}) ==="
  _N_COND="$(wc -l < "${CONDITIONED}")"
  _N_INFER="$( [[ -f "${INFER_SHARD}" ]] && wc -l < "${INFER_SHARD}" || echo 0 )"
  if [[ "${_N_INFER}" -ge "${_N_COND}" && "${_N_INFER}" -gt 0 ]]; then
    echo "  [resume] ${INFER_SHARD} (${_N_INFER} rows)"
  else
    python3 /opt/ms-swift/swift/cli/infer.py \
      --model Qwen/Qwen3-VL-8B-Instruct --model_type qwen3_vl --use_hf true \
      --adapters "${ADAPTERS}" \
      --val_dataset "${CONDITIONED}" \
      --infer_backend transformers \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --result_path "${INFER_SHARD}"
  fi
  echo "  shard ${SHARD_ID} done: ${INFER_SHARD}"
fi

if [[ "${SHARD_COUNT}" -gt 1 && "${SHARD_ID}" -eq 0 && "${MERGE_ONLY}" == "1" ]] || \
   [[ "${SHARD_COUNT}" -eq 1 && "${MERGE_ONLY}" != "1" ]]; then
  echo "=== [3/3] merge shards + build complex explanations ==="
  _SHARD_PATHS=()
  for ((i=0; i<SHARD_COUNT; i++)); do
    _SHARD_PATHS+=("${OUT_DIR}/infer_shard${i}.jsonl")
  done
  if [[ "${SHARD_COUNT}" -gt 1 ]]; then
    python3 "${EVAL_DIR}/merge_infer_shards.py" \
      --shards "${_SHARD_PATHS[@]}" \
      --output "${INFER_MERGED}"
  else
    cp "${INFER_SHARD}" "${INFER_MERGED}"
  fi
  python3 "${EVAL_DIR}/build_complex_explanations.py" \
    --infer "${INFER_MERGED}" \
    --pass1-pred "${PASS1_PRED}" \
    --output "${COMPLEX_OUT}" \
    --errors-json "${OUT_DIR}/complex_errors.json"
  echo
  echo "=== Done ==="
  echo "  infer:   ${INFER_MERGED}"
  echo "  complex: ${COMPLEX_OUT}"
fi
