#!/usr/bin/env bash
# Compressor on test fake complex explanations -> simple; reals copy complex.
#
# Run on GPU node (SSH):
#   bash scripts/run_compressor_test_lj.sh
#
# 4-GPU parallel (default):
#   SHARD_COUNT=4 bash scripts/run_compressor_test_lj.sh
set -euo pipefail

if [[ -z "${_COMPRESSOR_IN_CONTAINER:-}" && -x "${HOME}/xplainverse_exec.sh" ]]; then
  export _COMPRESSOR_IN_CONTAINER=1
  exec "${HOME}/xplainverse_exec.sh" bash "$0" "$@"
fi

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CODE_ROOT:-}" ]]; then :; elif [[ -d "${_SCRIPT_ROOT}/evaluation" ]]; then CODE_ROOT="${_SCRIPT_ROOT}";
elif [[ -d /workspace/XPlainVerse-ACMChallenge/evaluation ]]; then CODE_ROOT="/workspace/XPlainVerse-ACMChallenge";
else CODE_ROOT="${HOME}/luka/code/XPlainVerse-ACMChallenge"; fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PATH="/usr/local/bin:/usr/bin:/bin"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PYTHON_EXE CONDA_EXE LD_PRELOAD 2>/dev/null || true
export CC=/usr/bin/cc CXX=/usr/bin/c++ GCC=/usr/bin/gcc TRITON_DISABLE_LINE_INFO=1

for _cuda_lib in cu121 cu12 cu13; do
  _nv="/usr/local/lib/python3.10/dist-packages/nvidia/${_cuda_lib}/lib"
  [[ -d "${_nv}" ]] && { export LD_LIBRARY_PATH="${_nv}:${LD_LIBRARY_PATH:-}"; break; }
done
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

EVAL_DIR="${CODE_ROOT}/evaluation"
COMPLEX_IN="${COMPLEX_IN:-/home/jakob/luka/runs/pass2_test_complex/20260606-195337/complex_explanations.jsonl}"
ADAPTERS="${ADAPTERS:-/home/jakob/luka/runs/compressor_vl/checkpoint-10000}"
PROMPT_FILE="${PROMPT_FILE:-${CODE_ROOT}/dataset/prompt_v2.txt}"
SHARD_ID="${SHARD_ID:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
MERGE_ONLY="${MERGE_ONLY:-0}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

_TS="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/compressor_test/${_TS}}"
mkdir -p "${OUT_DIR}"

CONDITIONED="${OUT_DIR}/compressor_infer_shard${SHARD_ID}.jsonl"
INFER_SHARD="${OUT_DIR}/infer_shard${SHARD_ID}.jsonl"
INFER_MERGED="${OUT_DIR}/compressor_infer.jsonl"
SUBMISSION_OUT="${OUT_DIR}/submission.jsonl"

echo "=== Compressor test (lj) ==="
echo "  complex:    ${COMPLEX_IN}"
echo "  adapters:   ${ADAPTERS}"
echo "  prompt:     ${PROMPT_FILE}"
echo "  shard:      ${SHARD_ID}/${SHARD_COUNT} GPU=${CUDA_VISIBLE_DEVICES}"
echo "  out_dir:    ${OUT_DIR}"
echo

if [[ "${MERGE_ONLY}" != "1" ]]; then
  echo "=== [1/3] build compressor infer shard ==="
  if [[ -s "${CONDITIONED}" ]]; then
    echo "  [resume] ${CONDITIONED}"
  else
    python3 "${EVAL_DIR}/build_compressor_infer_test.py" \
      --complex "${COMPLEX_IN}" \
      --prompt-file "${PROMPT_FILE}" \
      --shard-id "${SHARD_ID}" \
      --shard-count "${SHARD_COUNT}" \
      --out "${CONDITIONED}"
  fi

  echo "=== [2/3] compressor infer ==="
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
      --max_batch_size "${MAX_BATCH_SIZE}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --result_path "${INFER_SHARD}"
  fi
fi

if [[ "${SHARD_COUNT}" -gt 1 && "${SHARD_ID}" -eq 0 && "${MERGE_ONLY}" == "1" ]] || \
   [[ "${SHARD_COUNT}" -eq 1 && "${MERGE_ONLY}" != "1" ]]; then
  echo "=== [3/3] merge + build submission ==="
  _SHARD_PATHS=()
  for ((i=0; i<SHARD_COUNT; i++)); do
    _SHARD_PATHS+=("${OUT_DIR}/infer_shard${i}.jsonl")
  done
  if [[ "${SHARD_COUNT}" -gt 1 ]]; then
    : > "${INFER_MERGED}"
    for ((i=0; i<SHARD_COUNT; i++)); do
      cat "${OUT_DIR}/infer_shard${i}.jsonl" >> "${INFER_MERGED}"
    done
    echo "concatenated ${SHARD_COUNT} compressor shards -> ${INFER_MERGED}"
  else
    cp "${INFER_SHARD}" "${INFER_MERGED}"
  fi
  python3 "${EVAL_DIR}/build_simple_explanations.py" \
    --complex "${COMPLEX_IN}" \
    --compressor-infer "${INFER_MERGED}" \
    --output "${SUBMISSION_OUT}" \
    --errors-json "${OUT_DIR}/simple_errors.json"
  echo
  echo "=== Done ==="
  echo "  compressor_infer: ${INFER_MERGED}"
  echo "  submission:       ${SUBMISSION_OUT}"
fi
