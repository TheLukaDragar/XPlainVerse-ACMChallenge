#!/usr/bin/env bash
# 10k scored test rows — Qwen entity/facts via vLLM batch API, 4× GPU shards.
#
# Two-phase to avoid BERT OOM (vLLM must be stopped before DeBERTa loads):
#   Phase 1: vLLM + Qwen extract/coverage → _stage_cache.jsonl per shard
#   Phase 2: BERT + SLE only (--skip-qwen), merge final_scores.json
#
# From Slurm login:
#   LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=24:00:00 \
#     LJ_APPTAINER_IMAGE=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_full_test_qwen_eval_lj.sh
#
# Phase 2 only (if phase 1 cache exists):
#   PHASE=qwen_bert ./scripts/lj_ghcr_image_exec.sh bash scripts/run_full_test_qwen_eval_lj.sh
#   # or PHASE=bert for BERT/SLE only
set -euo pipefail

if [[ -z "${_QWEN_EVAL_IN_CONTAINER:-}" && ! -d /.singularity.d && ! -f /.dockerenv ]]; then
  export _QWEN_EVAL_IN_CONTAINER=1
  export LJ_APPTAINER_IMAGE="${LJ_APPTAINER_IMAGE:-docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest}"
  _INNER="export _QWEN_EVAL_IN_CONTAINER=1 HOME=${HOME} LJ_APPTAINER_IMAGE=$(printf '%q' "${LJ_APPTAINER_IMAGE}")"
  for _v in OUT_DIR SUBMISSION UPSTREAM_EVAL QWEN_MODEL NUM_GPUS QWEN_BATCH BERT_BATCH SLE_BATCH PHASE FILTER_SCORED_ONLY LOG STATUS; do
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
UPSTREAM_EVAL_PY="${EVAL_DIR}/evaluate_val.py"
EVAL_PY="${EVAL_DIR}/evaluate_val_lj.py"

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
# Official evaluate_val.py defaults (orhnizr README / --extraction-max-tokens).
EXTRACTION_MAX_TOKENS="${EXTRACTION_MAX_TOKENS:-1024}"
COVERAGE_MAX_TOKENS="${COVERAGE_MAX_TOKENS:-1024}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"
PHASE="${PHASE:-all}"  # all | qwen | bert
# 1 = only score_explanations=true rows (~10k local protocol); 0 = full reference (200k).
FILTER_SCORED_ONLY="${FILTER_SCORED_ONLY:-1}"

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

VLLM_PIDS=()

_kill_pid_tree() {
  local pid="$1"
  [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null && return 0
  local child
  for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
    _kill_pid_tree "${child}"
  done
  kill -TERM "${pid}" 2>/dev/null || true
}

_kill_vllm_ports() {
  for SHARD in $(seq 0 $((NUM_GPUS - 1))); do
    local PORT=$((VLLM_BASE_PORT + SHARD))
    if command -v fuser >/dev/null 2>&1; then
      fuser -k -TERM "${PORT}/tcp" 2>/dev/null || true
    fi
  done
  sleep 2
  for SHARD in $(seq 0 $((NUM_GPUS - 1))); do
    local PORT=$((VLLM_BASE_PORT + SHARD))
    if command -v fuser >/dev/null 2>&1; then
      fuser -k -KILL "${PORT}/tcp" 2>/dev/null || true
    fi
  done
}

_wait_gpus_free() {
  local max_wait="${1:-120}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    sleep 10
    return 0
  fi
  write_status "waiting up to ${max_wait}s for GPU memory release"
  local i
  for i in $(seq 1 "${max_wait}"); do
    local ok=1
    for SHARD in $(seq 0 $((NUM_GPUS - 1))); do
      local used
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${SHARD}" 2>/dev/null | tr -d ' ')
      if [[ -n "${used}" && "${used}" -gt 2048 ]]; then
        ok=0
        break
      fi
    done
    if [[ "${ok}" -eq 1 ]]; then
      write_status "GPUs free (<2GiB used per device)"
      return 0
    fi
    sleep 1
  done
  write_status "WARN: GPU memory still high after ${max_wait}s — continuing anyway"
}

cleanup_vllm() {
  write_status "stopping vLLM servers"
  for _pid in "${VLLM_PIDS[@]:-}"; do
    _kill_pid_tree "${_pid}"
    wait "${_pid}" 2>/dev/null || true
  done
  _kill_vllm_ports
  VLLM_PIDS=()
  _wait_gpus_free 120
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
    "samples_merged", "complex_entity_f1", "complex_facts_f1", "complex_evidence_f1",
    "complex_overall_score", "grounding_score", "explanation_score",
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

write_status "START host=$(hostname) gpus=${NUM_GPUS} batch=${QWEN_BATCH} phase=${PHASE} out=${OUT_DIR}"

{
  echo "=== sub7 Qwen eval (vLLM batch, ${NUM_GPUS} GPU, phase=${PHASE}) ==="
  echo "host: $(hostname)"
  echo "started: $(date -Is)"
  python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.device_count())"
} | tee $([[ "${PHASE}" == "bert" ]] && echo -a || true) "${LOG}"

if [[ "${PHASE}" == "bert" || "${PHASE}" == "all" ]] && ! command -v vllm >/dev/null 2>&1 && [[ "${PHASE}" != "bert" ]]; then
  write_status "ERROR: vllm not on PATH — use GHCR eval image"
  exit 1
fi

if [[ ! -f "${UPSTREAM_EVAL_PY}" ]]; then
  write_status "ERROR: missing ${UPSTREAM_EVAL_PY}"
  exit 1
fi

write_status "patching evaluate_val.py -> evaluate_val_lj.py (stage cache + two-phase)"
python3 "${CODE_ROOT}/scripts/patch_evaluate_val_lj.py" \
  --source "${UPSTREAM_EVAL_PY}" \
  --dest "${EVAL_PY}"

write_status "patching upstream llm_helpers with vLLM batch client"
cp "${CODE_ROOT}/evaluation/utils/llm_helpers.py" "${EVAL_DIR}/utils/llm_helpers.py"

if [[ "${FILTER_SCORED_ONLY}" == "1" ]]; then
  if [[ ! -f "${FILTER_DIR}/reference_scored.jsonl" ]]; then
    write_status "filtering to score_explanations=true rows"
    python3 "${CODE_ROOT}/scripts/filter_scored_test_subset.py" \
      --reference "${GT_REF}" \
      --gt-entity-facts "${GT_CACHE}" \
      --output-dir "${FILTER_DIR}" 2>&1 | tee -a "${LOG}"
  fi
  SCORED_REF="${FILTER_DIR}/reference_scored.jsonl"
  SCORED_GT="${FILTER_DIR}/gt_entity_facts_scored.jsonl"
else
  write_status "using full reference (${GT_REF}) — all rows, no score_explanations filter"
  SCORED_REF="${GT_REF}"
  SCORED_GT="${GT_CACHE}"
fi

if [[ ! -f "${SHARD_DIR}/shard_0.zip" ]]; then
  write_status "splitting submission into ${NUM_GPUS} shards"
  python3 "${CODE_ROOT}/scripts/split_submission_shards.py" \
    --submission "${SUBMISSION}" \
    --reference "${SCORED_REF}" \
    --output-dir "${SHARD_DIR}" \
    --num-shards "${NUM_GPUS}" 2>&1 | tee -a "${LOG}"
fi

_run_eval_shard() {
  local shard="$1"
  shift
  local extra_args=("$@")
  local shard_out="${OUT_DIR}/shard_${shard}"
  mkdir -p "${shard_out}"
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    cd "${EVAL_DIR}"
    python3 "${EVAL_PY}" \
      --submission "${SHARD_DIR}/shard_${shard}.zip" \
      --ground-truth "${SHARD_DIR}/shard_${shard}_reference.jsonl" \
      --gt-entity-facts "${SCORED_GT}" \
      --no-update-gt-entity-facts \
      --output-dir "${shard_out}" \
      --backend vllm \
      --base-url "http://127.0.0.1:$((VLLM_BASE_PORT + shard))/v1" \
      --model-name "${QWEN_MODEL}" \
      --qwen-batch-size "${QWEN_BATCH}" \
      --extraction-max-tokens "${EXTRACTION_MAX_TOKENS}" \
      --coverage-max-tokens "${COVERAGE_MAX_TOKENS}" \
      --bertscore-batch-size "${BERT_BATCH}" \
      --sle-batch-size "${SLE_BATCH}" \
      --no-preload-models \
      "${extra_args[@]}"
  ) >> "${LOG}" 2>&1
}

# --- Phase 1: Qwen (vLLM must be running) ---
if [[ "${PHASE}" == "all" || "${PHASE}" == "qwen" ]]; then
  write_status "PHASE 1: Qwen extract + coverage"
  # Non-thinking: official default is --enable-thinking off (not passed below).
  # Qwen3.5 + vLLM also needs enable_thinking=false on serve + per-request (llm_helpers).
  for SHARD in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((VLLM_BASE_PORT + SHARD))
    (
      export CUDA_VISIBLE_DEVICES="${SHARD}"
    vllm serve "${QWEN_MODEL}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --language-model-only \
      --reasoning-parser qwen3 \
      --default-chat-template-kwargs '{"enable_thinking": false}' \
      >> "${LOG}" 2>&1
    ) &
    VLLM_PIDS+=($!)
    write_status "shard ${SHARD}: vLLM pid=${VLLM_PIDS[-1]} port=${PORT} gpu=${SHARD}"
  done

  for SHARD in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((VLLM_BASE_PORT + SHARD))
    write_status "shard ${SHARD}: waiting for http://127.0.0.1:${PORT}/v1"
    _ready=0
    for _i in $(seq 1 120); do
      if _vllm_ready "http://127.0.0.1:${PORT}/v1" 2>/dev/null; then
        _ready=1
        break
      fi
      if ! kill -0 "${VLLM_PIDS[$SHARD]}" 2>/dev/null; then
        write_status "shard ${SHARD}: vLLM died during startup"
        exit 1
      fi
      sleep 5
    done
    [[ "${_ready}" -eq 1 ]] || { write_status "shard ${SHARD}: vLLM not ready"; exit 1; }
  done
  write_status "all ${NUM_GPUS} vLLM servers ready"

  EVAL_PIDS=()
  for SHARD in $(seq 0 $((NUM_GPUS - 1))); do
    (
      write_status "shard ${SHARD}: phase1 evaluate_val.py"
      _run_eval_shard "${SHARD}" --skip-bert-sle
    ) &
    EVAL_PIDS+=($!)
    write_status "shard ${SHARD}: phase1 pid=${EVAL_PIDS[-1]}"
  done

  FAIL=0
  for SHARD in $(seq 0 $((NUM_GPUS - 1))); do
    if ! wait "${EVAL_PIDS[$SHARD]}"; then
      write_status "shard ${SHARD}: phase1 FAILED"
      FAIL=1
    else
      write_status "shard ${SHARD}: phase1 DONE"
    fi
  done
  cleanup_vllm
  [[ ${FAIL} -eq 0 ]] || exit 1

  write_status "PHASE 1 complete; computing intermediate Qwen scores"
  python3 "${CODE_ROOT}/scripts/compute_qwen_intermediate_scores.py" \
    --shard-dirs $(seq -f "${OUT_DIR}/shard_%g" 0 $((NUM_GPUS - 1))) \
    --output "${OUT_DIR}/intermediate_qwen_scores.json" \
    2>&1 | tee -a "${LOG}"
fi

# --- Phase 2: BERT + SLE (no vLLM) ---
if [[ "${PHASE}" == "all" || "${PHASE}" == "bert" ]]; then
  if [[ "${PHASE}" == "bert" ]]; then
    _kill_vllm_ports
    _wait_gpus_free 60
  fi
  write_status "PHASE 2: BERT + SLE (vLLM stopped)"
  EVAL_PIDS=()
  for SHARD in $(seq 0 $((NUM_GPUS - 1))); do
    (
      write_status "shard ${SHARD}: phase2 evaluate_val.py"
      _run_eval_shard "${SHARD}" --skip-qwen
    ) &
    EVAL_PIDS+=($!)
    write_status "shard ${SHARD}: phase2 pid=${EVAL_PIDS[-1]}"
  done

  FAIL=0
  for SHARD in $(seq 0 $((NUM_GPUS - 1))); do
    if ! wait "${EVAL_PIDS[$SHARD]}"; then
      write_status "shard ${SHARD}: phase2 FAILED"
      FAIL=1
    else
      write_status "shard ${SHARD}: phase2 DONE"
    fi
  done
  [[ ${FAIL} -eq 0 ]] || exit 1
fi

write_status "merging shard outputs"
python3 "${CODE_ROOT}/scripts/merge_qwen_shards.py" \
  --shard-dirs $(seq -f "${OUT_DIR}/shard_%g" 0 $((NUM_GPUS - 1))) \
  --output "${OUT_DIR}/final_scores.json" \
  --per-sample-output "${OUT_DIR}/per_sample_scores.jsonl" \
  2>&1 | tee -a "${LOG}"

write_status "wrote ${OUT_DIR}/final_scores.json"
