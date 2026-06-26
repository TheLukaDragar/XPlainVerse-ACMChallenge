#!/usr/bin/env bash
# Full test-set Qwen grounding eval on Lj.
#
# MUST run inside the GHCR eval Docker image (vLLM 0.23 + torch cu130).
# See docker/README.md and upstream README "vLLM Judge Backend".
#
#   DO:  lj_ghcr_image_exec.sh  →  ghcr.io/.../xplainverse-acmchallenge:latest
#   DON'T: conda, miniconda, ~/xplainverse_exec.sh (old SIF, no vLLM), --backend transformers
#
# From Slurm login:
#   LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=72:00:00 \
#     LJ_APPTAINER_IMAGE=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_full_test_qwen_eval_lj.sh
set -euo pipefail

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "${_SCRIPT_ROOT}/evaluation" ]]; then
  CODE_ROOT="${CODE_ROOT:-${_SCRIPT_ROOT}}"
elif [[ -d /workspace/XPlainVerse-ACMChallenge/evaluation ]]; then
  CODE_ROOT="${CODE_ROOT:-/workspace/XPlainVerse-ACMChallenge}"
else
  CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
fi

UPSTREAM_EVAL="${UPSTREAM_EVAL:-${HOME}/luka/code/XPlainVerse-upstream-eval}"
if [[ -f "${UPSTREAM_EVAL}/evaluation/evaluate_val.py" ]]; then
  EVAL_PY="${EVAL_PY:-${UPSTREAM_EVAL}/evaluation/evaluate_val.py}"
  EVAL_DIR="$(dirname "${EVAL_PY}")"
else
  EVAL_PY="${EVAL_PY:-${CODE_ROOT}/evaluation/evaluate_val.py}"
  EVAL_DIR="${CODE_ROOT}/evaluation"
fi

OUT_DIR="${OUT_DIR:-${HOME}/luka/runs/smoke_upstream_eval/results_sub7_full_qwen_vllm}"
LOG="${LOG:-${HOME}/luka/runs/smoke_upstream_eval/qwen_vllm_eval.log}"
STATUS="${STATUS:-${HOME}/luka/runs/smoke_upstream_eval/qwen_vllm_eval_status.txt}"
SUBMISSION="${SUBMISSION:-${HOME}/luka/runs/smoke_upstream_eval/sub7_full_stem.zip}"
GT_REF="${GT_REF:-${UPSTREAM_EVAL}/evaluation/ground_truth/test/reference.jsonl}"
GT_CACHE="${GT_CACHE:-${UPSTREAM_EVAL}/evaluation/ground_truth/test/complex_ground_truth_entity_facts.jsonl}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-4B}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
QWEN_BATCH="${QWEN_BATCH:-16}"
BERT_BATCH="${BERT_BATCH:-128}"
SLE_BATCH="${SLE_BATCH:-128}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

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

cleanup_vllm() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    write_status "stopping vLLM pid=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
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
    "complex_bert_f1", "complex_entity_f1", "complex_evidence_f1", "complex_overall_score",
    "simple_overall_score", "grounding_score", "explanation_score", "detection_macro_f1", "final_score",
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

write_status "START host=$(hostname) image=GHCR backend=vllm model=${QWEN_MODEL} out=${OUT_DIR}"

{
  echo "=== full test Qwen eval (vLLM in Docker image) ==="
  echo "host: $(hostname)"
  echo "started: $(date -Is)"
  echo "eval_py: ${EVAL_PY}"
  echo "submission: ${SUBMISSION}"
  echo "ground_truth: ${GT_REF}"
  python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.device_count())"
} | tee "${LOG}"

if ! command -v vllm >/dev/null 2>&1; then
  write_status "ERROR: vllm not on PATH — you are NOT in the GHCR eval image."
  write_status "  Use: LJ_APPTAINER_IMAGE=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest ./scripts/lj_ghcr_image_exec.sh bash $0"
  write_status "  Do NOT use conda or ~/xplainverse_exec.sh"
  exit 1
fi

write_status "starting vLLM serve port=${VLLM_PORT} (in-container, see docker/Dockerfile)"
vllm serve "${QWEN_MODEL}" \
  --host 127.0.0.1 \
  --port "${VLLM_PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --language-model-only \
  >> "${LOG}" 2>&1 &
VLLM_PID=$!
write_status "vLLM pid=${VLLM_PID}"

_vllm_ready() {
  python3 - <<PY
import urllib.request
urllib.request.urlopen("${VLLM_BASE_URL}/models", timeout=5)
PY
}

write_status "waiting for ${VLLM_BASE_URL}"
for _i in $(seq 1 120); do
  if _vllm_ready 2>/dev/null; then
    write_status "vLLM ready (${_i} checks)"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    write_status "vLLM died during startup"
    tail -40 "${LOG}" >> "${STATUS}" || true
    exit 1
  fi
  sleep 5
done
if ! _vllm_ready 2>/dev/null; then
  write_status "vLLM not ready after 600s"
  exit 1
fi

write_status "running evaluate_val.py --backend vllm (full test)"
export PYTHONPATH="${EVAL_DIR}:${PYTHONPATH:-}"
python3 "${EVAL_PY}" \
  --submission "${SUBMISSION}" \
  --ground-truth "${GT_REF}" \
  --gt-entity-facts "${GT_CACHE}" \
  --no-update-gt-entity-facts \
  --output-dir "${OUT_DIR}" \
  --backend vllm \
  --base-url "${VLLM_BASE_URL}" \
  --model-name "${QWEN_MODEL}" \
  --qwen-batch-size "${QWEN_BATCH}" \
  --bertscore-batch-size "${BERT_BATCH}" \
  --sle-batch-size "${SLE_BATCH}" \
  --no-preload-models \
  2>&1 | tee -a "${LOG}"

write_status "wrote ${OUT_DIR}/final_scores.json"
