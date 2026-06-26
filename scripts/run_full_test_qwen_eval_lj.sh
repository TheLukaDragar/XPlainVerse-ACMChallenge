#!/usr/bin/env bash
# 10k scored test rows — Qwen entity/facts via vLLM batch API, 4× GPU shards.
#
# Uses GHCR eval image (vLLM 0.23 + /v1/chat/completions/batch).
# Patches upstream llm_helpers with fork batch client before eval.
#
# From Slurm login:
#   LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=24:00:00 \
#     LJ_APPTAINER_IMAGE=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_full_test_qwen_eval_lj.sh
set -euo pipefail

if [[ -z "${_QWEN_EVAL_IN_CONTAINER:-}" ]]; then
  export _QWEN_EVAL_IN_CONTAINER=1
  export LJ_APPTAINER_IMAGE="${LJ_APPTAINER_IMAGE:-docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest}"
  _INNER="export _QWEN_EVAL_IN_CONTAINER=1 HOME=${HOME} LJ_APPTAINER_IMAGE=$(printf '%q' "${LJ_APPTAINER_IMAGE}")"
  for _v in OUT_DIR SUBMISSION UPSTREAM_EVAL QWEN_MODEL NUM_GPUS QWEN_BATCH BERT_BATCH SLE_BATCH; do
    if [[ -n "${!_v:-}" ]]; then
      _INNER+=" ${_v}=$(printf '%q' "${!_v}")"
    fi
  done
  _INNER+="; exec bash scripts/run_full_test_qwen_eval_lj.sh"
  exec env LJ_APPTAINER_IMAGE="${LJ_APPTAINER_IMAGE}" ./scripts/lj_ghcr_image_exec.sh bash -c "${_INNER}"
fi

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-${_SCRIPT_ROOT}}"
UPSTREAM_EVAL="${UPSTREAM_EVAL:-${HOME}/luka/code/XPlainVerse-upstream-eval}"
EVAL_DIR="${UPSTREAM_EVAL}/evaluation"
EVAL_PY="${EVAL_DIR}/evaluate_val.py"

OUT_DIR="${OUT_DIR:-${HOME}/luka/runs/smoke_upstream_eval/results_sub7_qwen_10k_vllm_batch}"
LOG="${LOG:-${HOME}/luka/runs/smoke_upstream_eval/qwen_vllm_batch.log}"
STATUS="${STATUS:-${HOME}/luka/runs/smoke_upstream_eval/qwen_vllm_batch_status.txt}"
SUBMISSION="${SUBMISSION:-${HOME}/luka/runs/smoke_upstream_eval/sub7_full_stem.zip}"
GT_REF="${GT_REF:-${UPSTREAM_EVAL}/evaluation/ground_truth/test/reference.jsonl}"
GT_CACHE="${GT_CACHE:-${UPSTREAM_EVAL}/evaluation/ground_truth/test/complex_ground_truth_entity_facts.jsonl}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-4B}"
NUM_GPUS="${NUM_GPUS:-4}"
QWEN_BATCH="${QWEN_BATCH:-16}"
BERT_BATCH="${BERT_BATCH:-128}"
SLE_BATCH="${SLE_BATCH:-128}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"

FILTER_DIR="${OUT_DIR}/filtered"
SHARD_DIR="${OUT_DIR}/shards"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
for _cuda_lib in cu13 cu121 cu12; do
  _nv="/usr/local/lib/python3.10/dist-packages/nvidia/${_cuda_lib}/lib"
  [[ -d "${_nv}" ]] && { export LD_LIBRARY_PATH="${_nv}:${LD_LIBRARY_PATH:-}"; break; }
done
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

mkdir -p "${OUT_DIR}"

write_status() {
  printf '%s %s\n' "$(date -Is)" "$1" | tee -a "${STATUS}"
}

_vllm_ready() {
  local base_url="$1"
  python3 - <<PY
import urllib.request
urllib.request.urlopen("${base_url}/models", timeout=5)
PY
}

cleanup_vllm() {
  for _pid in "${VLLM_PIDS[@]:-}"; do
    if [[ -n "${_pid}" ]] && kill -0 "${_pid}" 2>/dev/null; then
      kill "${_pid}" 2>/dev/null || true
      wait "${_pid}" 2>/dev/null || true
    fi
  done
}

on_exit() {
  local code=$?
  cleanup_vllm
  if [[ ${code} -eq 0 ]]; then
    write_status "DONE exit=0"
    if [[ -f "${OUT_DIR}/final_scores.json" ]]; then
      python3 - <<PY >>"${STATUS}"
import json
from pathlib import Path
d = json.loads(Path("${OUT_DIR}/final_scores.json").read_text())
for k in (
    "samples_merged", "complex_entity_f1", "complex_facts_f1", "complex_overall_score",
    "grounding_score", "explanation_score",
):
    print(f"  {k}: {d.get(k)}")
PY
    fi
  else
    write_status "FAILED exit=${code} (see ${LOG})"
  fi
}
trap on_exit EXIT

if [[ -f "${OUT_DIR}/final_scores.json" ]]; then
  write_status "SKIP: ${OUT_DIR}/final_scores.json exists"
  exit 0
fi

write_status "START host=$(hostname) gpus=${NUM_GPUS} batch=${QWEN_BATCH} out=${OUT_DIR}"

{
  echo "=== sub7 Qwen eval (vLLM batch API, ${NUM_GPUS} GPU) ==="
  echo "host: $(hostname)"
  echo "started: $(date -Is)"
  echo "eval_py: ${EVAL_PY}"
  python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.device_count())"
} | tee "${LOG}"

if ! command -v vllm >/dev/null 2>&1; then
  write_status "ERROR: vllm not on PATH — use GHCR eval image"
  exit 1
fi

if [[ ! -f "${EVAL_PY}" ]]; then
  write_status "ERROR: missing ${EVAL_PY}"
  exit 1
fi

write_status "patching upstream llm_helpers with vLLM batch client"
cp "${CODE_ROOT}/evaluation/utils/llm_helpers.py" "${EVAL_DIR}/utils/llm_helpers.py"

write_status "filtering to score_explanations=true rows"
python3 "${CODE_ROOT}/scripts/filter_scored_test_subset.py" \
  --reference "${GT_REF}" \
  --gt-entity-facts "${GT_CACHE}" \
  --output-dir "${FILTER_DIR}" 2>&1 | tee -a "${LOG}"

SCORED_REF="${FILTER_DIR}/reference_scored.jsonl"
SCORED_GT="${FILTER_DIR}/gt_entity_facts_scored.jsonl"

write_status "splitting submission into ${NUM_GPUS} shards"
python3 "${CODE_ROOT}/scripts/split_submission_shards.py" \
  --submission "${SUBMISSION}" \
  --reference "${SCORED_REF}" \
  --output-dir "${SHARD_DIR}" \
  --num-shards "${NUM_GPUS}" 2>&1 | tee -a "${LOG}"

VLLM_PIDS=()
EVAL_PIDS=()

for SHARD in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((VLLM_BASE_PORT + SHARD))
  SHARD_OUT="${OUT_DIR}/shard_${SHARD}"
  mkdir -p "${SHARD_OUT}"
  (
    export CUDA_VISIBLE_DEVICES="${SHARD}"
    vllm serve "${QWEN_MODEL}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --language-model-only \
      >> "${LOG}" 2>&1
  ) &
  VLLM_PIDS+=($!)
  write_status "shard ${SHARD}: vLLM pid=${VLLM_PIDS[-1]} port=${PORT} gpu=${SHARD}"
done

for SHARD in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((VLLM_BASE_PORT + SHARD))
  BASE_URL="http://127.0.0.1:${PORT}/v1"
  write_status "shard ${SHARD}: waiting for ${BASE_URL}"
  _ready=0
  for _i in $(seq 1 120); do
    if _vllm_ready "${BASE_URL}" 2>/dev/null; then
      _ready=1
      break
    fi
    if ! kill -0 "${VLLM_PIDS[$SHARD]}" 2>/dev/null; then
      write_status "shard ${SHARD}: vLLM died during startup"
      exit 1
    fi
    sleep 5
  done
  if [[ "${_ready}" -ne 1 ]]; then
    write_status "shard ${SHARD}: vLLM not ready after 600s"
    exit 1
  fi
done
write_status "all ${NUM_GPUS} vLLM servers ready"

for SHARD in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((VLLM_BASE_PORT + SHARD))
  BASE_URL="http://127.0.0.1:${PORT}/v1"
  SHARD_OUT="${OUT_DIR}/shard_${SHARD}"

  (
    export CUDA_VISIBLE_DEVICES="${SHARD}"
    write_status "shard ${SHARD}: evaluate_val.py"
    cd "${EVAL_DIR}"
    python3 "${EVAL_PY}" \
      --submission "${SHARD_DIR}/shard_${SHARD}.zip" \
      --ground-truth "${SHARD_DIR}/shard_${SHARD}_reference.jsonl" \
      --gt-entity-facts "${SCORED_GT}" \
      --no-update-gt-entity-facts \
      --output-dir "${SHARD_OUT}" \
      --backend vllm \
      --base-url "${BASE_URL}" \
      --model-name "${QWEN_MODEL}" \
      --qwen-batch-size "${QWEN_BATCH}" \
      --bertscore-batch-size "${BERT_BATCH}" \
      --sle-batch-size "${SLE_BATCH}" \
      --no-preload-models \
      >> "${LOG}" 2>&1
  ) &
  EVAL_PIDS+=($!)
  write_status "shard ${SHARD}: eval pid=${EVAL_PIDS[-1]}"
done

FAIL=0
for SHARD in $(seq 0 $((NUM_GPUS - 1))); do
  if ! wait "${EVAL_PIDS[$SHARD]}"; then
    write_status "shard ${SHARD}: eval FAILED"
    FAIL=1
  else
    write_status "shard ${SHARD}: eval DONE"
  fi
done
cleanup_vllm
[[ ${FAIL} -eq 0 ]] || exit 1

write_status "merging shard outputs"
python3 "${CODE_ROOT}/scripts/merge_qwen_shards.py" \
  --shard-dirs $(seq -f "${OUT_DIR}/shard_%g" 0 $((NUM_GPUS - 1))) \
  --output "${OUT_DIR}/final_scores.json" \
  --per-sample-output "${OUT_DIR}/per_sample_scores.jsonl" \
  2>&1 | tee -a "${LOG}"

write_status "wrote ${OUT_DIR}/final_scores.json"
