#!/usr/bin/env bash
# Retry Qwen stages for samples with null fields in _stage_cache.jsonl.
# Then patch per_sample from cache and re-merge final_scores.json (full 200k metrics).
#
# Usage (login node):
#   OUT_DIR=~/luka/runs/smoke_upstream_eval/results_sub7_full_200k_qwen_vllm \
#   ./scripts/run_full_test_qwen_eval_lj.sh   # normal entry dispatches container
#
# Or directly:
#   OUT_DIR=... ./scripts/retry_failed_qwen_lj.sh
set -euo pipefail

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-${_SCRIPT_ROOT}}"

OUT_DIR="${OUT_DIR:-${HOME}/luka/runs/smoke_upstream_eval/results_sub7_full_200k_qwen_vllm}"
UPSTREAM_EVAL="${UPSTREAM_EVAL:-${HOME}/luka/code/XPlainVerse-upstream-eval}"
EVAL_DIR="${UPSTREAM_EVAL}/evaluation"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-4B}"
QWEN_BATCH="${QWEN_BATCH:-16}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
EXTRACTION_MAX_TOKENS="${EXTRACTION_MAX_TOKENS:-2048}"
COVERAGE_MAX_TOKENS="${COVERAGE_MAX_TOKENS:-2048}"
MAX_RETRY_ATTEMPTS="${MAX_RETRY_ATTEMPTS:-3}"
LOG="${LOG:-${OUT_DIR}/qwen_retry.log}"
STATUS="${STATUS:-${OUT_DIR}/qwen_retry_status.txt}"

SHARD_DIR="${OUT_DIR}/shards"
SCORED_GT="${UPSTREAM_EVAL}/evaluation/ground_truth/test/complex_ground_truth_entity_facts.jsonl"

if [[ -z "${_QWEN_RETRY_IN_CONTAINER:-}" && ! -d /.singularity.d && ! -f /.dockerenv ]]; then
  export _QWEN_RETRY_IN_CONTAINER=1
  export LJ_APPTAINER_IMAGE="${LJ_APPTAINER_IMAGE:-docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest}"
  _INNER="export _QWEN_RETRY_IN_CONTAINER=1 HOME=${HOME} LJ_APPTAINER_IMAGE=$(printf '%q' "${LJ_APPTAINER_IMAGE}")"
  for _v in OUT_DIR UPSTREAM_EVAL QWEN_MODEL QWEN_BATCH LOG STATUS LJ_GPU_GRES LJ_GPU_TIME; do
    [[ -n "${!_v:-}" ]] && _INNER+=" ${_v}=$(printf '%q' "${!_v}")"
  done
  _INNER+="; exec bash scripts/retry_failed_qwen_lj.sh"
  exec env LJ_APPTAINER_IMAGE="${LJ_APPTAINER_IMAGE}" \
    LJ_GPU_GRES="${LJ_GPU_GRES:-gpu:1}" LJ_GPU_TIME="${LJ_GPU_TIME:-02:00:00}" \
    "${CODE_ROOT}/scripts/lj_ghcr_image_exec.sh" bash -c "${_INNER}"
fi

export PYTHONNOUSERSITE=1
for _cuda_lib in cu13 cu121 cu12; do
  _nv="/usr/local/lib/python3.10/dist-packages/nvidia/${_cuda_lib}/lib"
  [[ -d "${_nv}" ]] && { export LD_LIBRARY_PATH="${_nv}:${LD_LIBRARY_PATH:-}"; break; }
done
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

mkdir -p "${OUT_DIR}"
write_status() { printf '%s %s\n' "$(date -Is)" "$1" | tee -a "${STATUS}"; }

FAIL_JSON="${OUT_DIR}/qwen_failures.json"
python3 "${CODE_ROOT}/scripts/find_qwen_failures.py" --out-dir "${OUT_DIR}" --output "${FAIL_JSON}"
COUNT=$(python3 -c "import json; print(json.load(open('${FAIL_JSON}'))['count'])")
SHARDS=$(python3 -c "import json; print(' '.join(map(str,json.load(open('${FAIL_JSON}'))['shards_affected'])))")

write_status "START retry failed=${COUNT} shards=${SHARDS:-none}"

if [[ "${COUNT}" -eq 0 ]]; then
  write_status "no failures; merging only"
  python3 "${CODE_ROOT}/scripts/merge_qwen_shards.py" \
    --shard-dirs $(seq -f "${OUT_DIR}/shard_%g" 0 3) \
    --output "${OUT_DIR}/final_scores.json" \
    --per-sample-output "${OUT_DIR}/per_sample_scores.jsonl"
  write_status "DONE exit=0"
  exit 0
fi

python3 "${CODE_ROOT}/scripts/patch_evaluate_val_lj.py" \
  --source "${EVAL_DIR}/evaluate_val.py" \
  --dest "${EVAL_DIR}/evaluate_val_lj.py"
cp "${CODE_ROOT}/evaluation/utils/llm_helpers.py" "${EVAL_DIR}/utils/llm_helpers.py"

_vllm_ready() {
  python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${1}/v1/models', timeout=5)"
}

_run_shard_retry() {
  local shard="$1"
  local port="${VLLM_BASE_PORT}"  # always 8000 — one GPU, sequential shards
  local shard_out="${OUT_DIR}/shard_${shard}"

  if command -v fuser >/dev/null 2>&1; then
    fuser -k -TERM "${port}/tcp" 2>/dev/null || true
    sleep 3
    fuser -k -KILL "${port}/tcp" 2>/dev/null || true
  fi
  sleep 10

  write_status "shard ${shard}: starting vLLM port=${port}"
  (
    export CUDA_VISIBLE_DEVICES=0
    vllm serve "${QWEN_MODEL}" \
      --host 127.0.0.1 --port "${port}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --language-model-only \
      --reasoning-parser qwen3 \
      --default-chat-template-kwargs '{"enable_thinking": false}' \
      >> "${LOG}" 2>&1
  ) &
  local vllm_pid=$!

  local ready=0
  for _ in $(seq 1 120); do
    if _vllm_ready "${port}" 2>/dev/null; then ready=1; break; fi
    kill -0 "${vllm_pid}" 2>/dev/null || break
    sleep 5
  done
  [[ "${ready}" -eq 1 ]] || { write_status "shard ${shard}: vLLM failed"; kill "${vllm_pid}" 2>/dev/null || true; return 1; }

  write_status "shard ${shard}: Qwen retry evaluate_val.py"
  (
    export CUDA_VISIBLE_DEVICES=0
    cd "${EVAL_DIR}"
    python3 evaluate_val_lj.py \
      --submission "${SHARD_DIR}/shard_${shard}.zip" \
      --ground-truth "${SHARD_DIR}/shard_${shard}_reference.jsonl" \
      --gt-entity-facts "${SCORED_GT}" \
      --no-update-gt-entity-facts \
      --output-dir "${shard_out}" \
      --backend vllm \
      --base-url "http://127.0.0.1:${port}/v1" \
      --model-name "${QWEN_MODEL}" \
      --qwen-batch-size "${QWEN_BATCH}" \
      --extraction-max-tokens "${EXTRACTION_MAX_TOKENS}" \
      --coverage-max-tokens "${COVERAGE_MAX_TOKENS}" \
      --skip-bert-sle \
      --no-preload-models \
      >> "${LOG}" 2>&1
  ) || { kill "${vllm_pid}" 2>/dev/null || true; return 1; }

  kill "${vllm_pid}" 2>/dev/null || true
  wait "${vllm_pid}" 2>/dev/null || true
  if command -v fuser >/dev/null 2>&1; then
    fuser -k -TERM "${port}/tcp" 2>/dev/null || true
    sleep 3
    fuser -k -KILL "${port}/tcp" 2>/dev/null || true
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    for _ in $(seq 1 120); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
      [[ -z "${used}" || "${used}" -le 2048 ]] && break
      sleep 2
    done
  else
    sleep 15
  fi
  write_status "shard ${shard}: retry DONE"
}

for SHARD in ${SHARDS}; do
  attempt=1
  while [[ "${attempt}" -le "${MAX_RETRY_ATTEMPTS}" ]]; do
    write_status "shard ${SHARD}: attempt ${attempt}/${MAX_RETRY_ATTEMPTS}"
    _run_shard_retry "${SHARD}" || exit 1
    FAIL_JSON="${OUT_DIR}/qwen_failures.json"
    python3 "${CODE_ROOT}/scripts/find_qwen_failures.py" --out-dir "${OUT_DIR}" --output "${FAIL_JSON}"
    REMAIN=$(python3 -c "import json; d=json.load(open('${FAIL_JSON}')); print(sum(1 for s in d['samples'] if s['shard']==${SHARD}))")
    write_status "shard ${SHARD}: ${REMAIN} failures remaining"
    [[ "${REMAIN}" -eq 0 ]] && break
    attempt=$((attempt + 1))
  done
done

write_status "patching per_sample from cache"
python3 "${CODE_ROOT}/scripts/patch_per_sample_from_cache.py" --out-dir "${OUT_DIR}"

write_status "re-merge full 200k scores"
python3 "${CODE_ROOT}/scripts/merge_qwen_shards.py" \
  --shard-dirs $(seq -f "${OUT_DIR}/shard_%g" 0 3) \
  --output "${OUT_DIR}/final_scores.json" \
  --per-sample-output "${OUT_DIR}/per_sample_scores.jsonl" \
  | tee -a "${LOG}"

python3 "${CODE_ROOT}/scripts/find_qwen_failures.py" --out-dir "${OUT_DIR}" --output "${FAIL_JSON}"
COUNT_AFTER=$(python3 -c "import json; print(json.load(open('${FAIL_JSON}'))['count'])")
write_status "failures after retry: ${COUNT_AFTER}"
write_status "DONE exit=0"
