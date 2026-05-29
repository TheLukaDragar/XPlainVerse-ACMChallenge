#!/usr/bin/env bash
# Two-stage end-to-end eval on lj with the OFFICIAL scorer (Qwen entity/facts +
# BERTScore + SLE), on ~N balanced val samples.
#
#   Pass-1 verdict  : our SigLIP2+DINOv2 ensemble (VERDICT_SOURCE=ensemble)
#                     or ground truth (VERDICT_SOURCE=gt, quick upper bound)
#   Pass-2 explain  : Qwen3-VL v2 SFT adapter, conditioned on the verdict via the
#                     v2 HYPOTHETICAL_{FAKE,REAL} prompts (matches training)
#   Score           : evaluation/evaluate_val.py with Qwen3.5-4B
#
# Runs inside the single lj container. From the Slurm login node:
#   LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=08:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/eval_pipeline_v2_lj.sh
#
# Key env:
#   ADAPTERS        Pass-2 adapter dir (default: latest vlm_v2 checkpoint-1655)
#   VERDICT_SOURCE  ensemble | gt        (default ensemble)
#   N               number of val samples (default 1000, balanced real/fake)
#   ENS_CKPT        ensemble ckpt.pt
#   THRESHOLD       p_fake decision threshold for ensemble (default 0.5)
#   SIMPLE_MODE     first_sentence | match_verdict | copy (default first_sentence)
#   OUT_DIR         results dir
set -euo pipefail

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
# ms-swift main imports FSDP2 on torch 2.4.1 — keep the shim on PYTHONPATH.
_SHIM="${CODE_ROOT}/scripts/lj_swift_compat"
[[ -f "${_SHIM}/sitecustomize.py" ]] && export PYTHONPATH="${_SHIM}:${PYTHONPATH:-}"

EXP_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier"
EVAL_DIR_SRC="${CODE_ROOT}/evaluation"
VAL_INFER="${VAL_INFER:-${CODE_ROOT}/dataset/val_vlm_infer_v2.jsonl}"
GT="${GT:-${CODE_ROOT}/evaluation/data/val_ground_truth.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-${CODE_ROOT}/dataset/prompt_v2.txt}"

ADAPTERS="${ADAPTERS:-/home/jakob/luka/runs/vlm_v2/run-20260529-104845/checkpoint-1655}"
VERDICT_SOURCE="${VERDICT_SOURCE:-ensemble}"
N="${N:-1000}"
ENS_CKPT="${ENS_CKPT:-/home/jakob/luka/runs/pass1_ensemble/bombek_so400m_dinov2_20260528-225201/best_ckpt/ckpt.pt}"
THRESHOLD="${THRESHOLD:-0.5}"
SIMPLE_MODE="${SIMPLE_MODE:-first_sentence}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-4B}"
QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

_TS="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/eval_pipeline_v2/${VERDICT_SOURCE}_n${N}_${_TS}}"
mkdir -p "${OUT_DIR}"

CONDITIONED="${OUT_DIR}/pass2_infer.jsonl"
INFER_JSONL="${OUT_DIR}/infer.jsonl"
SUBMISSION="${OUT_DIR}/submission.jsonl"
EVAL_OUT="${OUT_DIR}/eval_results"
PASS1_MANIFEST="${OUT_DIR}/pass1_manifest.parquet"
PASS1_PRED="${OUT_DIR}/pass1_pred"

echo "=== Two-stage v2 eval (lj) ==="
echo "  code_root:      ${CODE_ROOT}"
echo "  adapters:       ${ADAPTERS}"
echo "  verdict_source: ${VERDICT_SOURCE}  (N=${N}, threshold=${THRESHOLD})"
echo "  ensemble ckpt:  ${ENS_CKPT}"
echo "  simple_mode:    ${SIMPLE_MODE}"
echo "  scorer:         ${QWEN_MODEL}"
echo "  out_dir:        ${OUT_DIR}"
echo

# --- Stage 1: verdicts -------------------------------------------------------
ENS_PRED_ARGS=()
if [[ "${VERDICT_SOURCE}" == "ensemble" ]]; then
  echo "=== $(date -u +%H:%M:%S) [1/5] Pass-1 ensemble on ${N} samples ==="
  python3 "${EVAL_DIR_SRC}/build_pass2_infer.py" manifest \
    --val-infer "${VAL_INFER}" --gt "${GT}" --n "${N}" --out "${PASS1_MANIFEST}"
  ( cd "${EXP_DIR}" && python3 eval_ensemble.py \
      --ckpt "${ENS_CKPT}" --manifest "${PASS1_MANIFEST}" \
      --out "${PASS1_PRED}" --slice 0 --batch-size "${QWEN_BATCH_SIZE}" )
  ENS_PRED_ARGS=(--verdict-source ensemble --ensemble-pred "${PASS1_PRED}/predictions.parquet" --threshold "${THRESHOLD}")
else
  echo "=== $(date -u +%H:%M:%S) [1/5] Pass-1 verdict = ground truth ==="
  ENS_PRED_ARGS=(--verdict-source gt)
fi

# --- Stage 2a: build conditioned infer set -----------------------------------
echo "=== $(date -u +%H:%M:%S) [2/5] build conditioned Pass-2 infer set ==="
python3 "${EVAL_DIR_SRC}/build_pass2_infer.py" infer \
  --val-infer "${VAL_INFER}" --gt "${GT}" --n "${N}" \
  --prompt-file "${PROMPT_FILE}" "${ENS_PRED_ARGS[@]}" \
  --out "${CONDITIONED}" --verdicts-json "${OUT_DIR}/pass1_verdicts.json"

# --- Stage 2b: VLM infer (conditioned) ---------------------------------------
echo "=== $(date -u +%H:%M:%S) [3/5] Qwen3-VL infer (${ADAPTERS##*/}) ==="
python3 /opt/ms-swift/swift/cli/infer.py \
  --model Qwen/Qwen3-VL-8B-Instruct --model_type qwen3_vl --use_hf true \
  --adapters "${ADAPTERS}" \
  --val_dataset "${CONDITIONED}" \
  --infer_backend transformers \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --result_path "${INFER_JSONL}"

# --- Stage 3: build submission -----------------------------------------------
echo "=== $(date -u +%H:%M:%S) [4/5] build submission (simple_mode=${SIMPLE_MODE}) ==="
( cd "${EVAL_DIR_SRC}" && python3 build_submission.py \
    --infer "${INFER_JSONL}" --output "${SUBMISSION}" \
    --simple-mode "${SIMPLE_MODE}" --errors-json "${OUT_DIR}/submission_errors.json" ) || true

# --- Stage 4: official scorer ------------------------------------------------
echo "=== $(date -u +%H:%M:%S) [5/5] official eval (${QWEN_MODEL}) ==="
( cd "${EVAL_DIR_SRC}" && python3 evaluate_val.py \
    --submission "${SUBMISSION}" --ground-truth "${GT}" \
    --output-dir "${EVAL_OUT}" --backend transformers \
    --model-name "${QWEN_MODEL}" --qwen-batch-size "${QWEN_BATCH_SIZE}" \
    --device-map "cuda:0" )

echo
echo "=== DONE $(date -u +%H:%M:%S) ==="
if [[ -f "${EVAL_OUT}/final_scores.json" ]]; then
  python3 -c "import json; s=json.load(open('${EVAL_OUT}/final_scores.json')); \
print('\n'.join(f'  {k}: {s.get(k)}' for k in ['samples_completed','complex_overall_score','complex_entity_f1','complex_facts_f1','complex_evidence_f1','complex_bert_f1','simple_overall_score','simple_bert_f1','simple_sle_score']))"
fi
echo "results: ${OUT_DIR}"
