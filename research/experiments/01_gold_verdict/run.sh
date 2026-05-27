#!/usr/bin/env bash
# Run gold-verdict-conditioning experiment on ckpt-2400.
#
# Sequence (single GPU):
#   For each variant in {baseline, conditioned, structured}:
#     1. swift infer  → infer.jsonl
#     2. build_submission.py → submission.jsonl
#     3. evaluate_val.py → final_scores.json
#
# Each variant uses 200 rows = 100 real + 100 fake (built by make_subsets.py).
# ckpt-2400 was the best ROUGE-L checkpoint from runs/vlm_full SFT.

set -euo pipefail

WORKSPACE_ROOT=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge
CODE_ROOT="${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge"
EXP_DIR="${CODE_ROOT}/research/experiments/01_gold_verdict"
# Default: oldest surviving checkpoint (2400 was pruned by save_total_limit=5).
ADAPTERS="${ADAPTERS:-${WORKSPACE_ROOT}/runs/vlm_full/v1-20260524-214014/checkpoint-3600}"
if [[ ! -d "${ADAPTERS}" ]]; then
  echo "error: adapters checkpoint not found: ${ADAPTERS}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export TORCH_COMPILE_DISABLE=1
export IMAGE_MAX_TOKEN_NUM=1024
export MAX_PIXELS=1003520

if [[ -n "${VARIANTS:-}" ]]; then
  read -ra VARIANTS <<< "${VARIANTS}"
else
  VARIANTS=(baseline conditioned structured)
fi

mkdir -p "${EXP_DIR}/results"

echo "=== Gold-verdict experiment ==="
echo "GPU:       ${CUDA_VISIBLE_DEVICES}"
echo "ADAPTERS:  ${ADAPTERS}"
echo "variants:  ${VARIANTS[*]}"
echo

for v in "${VARIANTS[@]}"; do
  VAL_JSONL="${EXP_DIR}/subsets/${v}.jsonl"
  OUT_DIR="${EXP_DIR}/results/${v}"
  INFER_JSONL="${OUT_DIR}/infer.jsonl"
  SUBMISSION_JSONL="${OUT_DIR}/submission.jsonl"
  EVAL_DIR="${OUT_DIR}/eval_results"

  if [[ -f "${EVAL_DIR}/final_scores.json" ]]; then
    echo "[skip] ${v} already has final_scores.json"
    continue
  fi

  mkdir -p "${OUT_DIR}"

  if [[ -f "${INFER_JSONL}" ]] && [[ "$(wc -l < "${INFER_JSONL}")" -ge 200 ]]; then
    echo "[skip] ${v} infer already done -> ${INFER_JSONL}"
  else
    echo "=== $(date -u +%H:%M:%S) [${v}] infer 200 rows ==="
    /usr/bin/python3 /opt/ms-swift/swift/cli/infer.py \
      --model Qwen/Qwen3-VL-8B-Instruct \
      --model_type qwen3_vl \
      --use_hf true \
      --adapters "${ADAPTERS}" \
      --val_dataset "${VAL_JSONL}" \
      --infer_backend transformers \
      --max_new_tokens 512 \
      --result_path "${INFER_JSONL}" \
      > "${OUT_DIR}/infer.stdout.txt" 2> "${OUT_DIR}/infer.stderr.txt"
    echo "[done] infer rc=$? -> ${INFER_JSONL}"
  fi

  if [[ -f "${SUBMISSION_JSONL}" ]]; then
    echo "[skip] ${v} submission already done -> ${SUBMISSION_JSONL}"
  else
    echo "=== $(date -u +%H:%M:%S) [${v}] build submission ==="
    ( cd "${CODE_ROOT}/evaluation" && \
      /usr/bin/python3 build_submission.py \
        --infer "${INFER_JSONL}" \
        --output "${SUBMISSION_JSONL}" \
        --simple-mode match_verdict \
        --errors-json "${OUT_DIR}/submission_errors.json" \
    ) > "${OUT_DIR}/submission.stdout.txt" 2>&1
    echo "[done] submission -> ${SUBMISSION_JSONL}"
  fi

  echo "=== $(date -u +%H:%M:%S) [${v}] official eval ==="
  ( cd "${CODE_ROOT}/evaluation" && \
    /usr/bin/python3 evaluate_val.py \
      --submission "${SUBMISSION_JSONL}" \
      --ground-truth "${CODE_ROOT}/evaluation/data/val_ground_truth.jsonl" \
      --output-dir "${EVAL_DIR}" \
      --backend transformers \
      --model-name Qwen/Qwen3.5-4B \
      --qwen-batch-size 4 \
      --device-map cuda:0 \
  ) > "${OUT_DIR}/eval.stdout.txt" 2> "${OUT_DIR}/eval.stderr.txt"
  echo "[done] eval rc=$? -> ${EVAL_DIR}/final_scores.json"
  echo
done

echo "=== ALL VARIANTS DONE $(date -u +%H:%M:%S) ==="
echo
for v in "${VARIANTS[@]}"; do
  echo "--- ${v} ---"
  if [[ -f "${EXP_DIR}/results/${v}/eval_results/final_scores.json" ]]; then
    /usr/bin/python3 -c "import json; s=json.load(open('${EXP_DIR}/results/${v}/eval_results/final_scores.json')); print('\n'.join(f'  {k}: {s.get(k)}' for k in ['samples_completed','detection_f1','complex_overall_score','complex_entity_f1','complex_evidence_f1','complex_bert_f1','simple_overall_score','explanation_score','simple_bert_f1','simple_sle_score']))"
  else
    echo "  MISSING"
  fi
done
