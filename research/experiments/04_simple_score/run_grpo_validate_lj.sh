#!/usr/bin/env bash
# Validate a GRPO compressor checkpoint on the fake-200 holdout:
#   compressor infer (GT complex input) -> simple -> official BERT+SLE score.
# Compares against the SFT baseline (probe/cmp_baseline) and GT simple.
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
EXP="${CODE_ROOT}/research/experiments/04_simple_score"
PROBE="${EXP}/probe"
COMPLEX_IN="${PROBE}/fake200_complex.jsonl"
REF="${PROBE}/fake200_ref.jsonl"
ADAPTERS="${ADAPTERS:-/home/jakob/luka/runs/compressor_grpo/20260613-201713/checkpoint-4800}"
PROMPT_FILE="${PROMPT_FILE:-${CODE_ROOT}/dataset/prompt_v2.txt}"
TAG="${TAG:-grpo4800}"

export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

out="${PROBE}/cmp_${TAG}"
mkdir -p "${out}"
cd "${CODE_ROOT}/evaluation"

echo "=== [${TAG}] build infer ==="
python3 build_compressor_infer_test.py \
  --complex "${COMPLEX_IN}" --prompt-file "${PROMPT_FILE}" \
  --shard-id 0 --shard-count 1 --out "${out}/infer_in.jsonl"

echo "=== [${TAG}] swift infer (adapters=${ADAPTERS}) ==="
python3 /opt/ms-swift/swift/cli/infer.py \
  --model Qwen/Qwen3-VL-8B-Instruct --model_type qwen3_vl --use_hf true \
  --adapters "${ADAPTERS}" \
  --val_dataset "${out}/infer_in.jsonl" \
  --infer_backend transformers \
  --max_batch_size 32 --max_new_tokens 128 \
  --result_path "${out}/infer_out.jsonl"

echo "=== [${TAG}] build submission ==="
python3 build_simple_explanations.py \
  --complex "${REF}" \
  --compressor-infer "${out}/infer_out.jsonl" \
  --output "${out}/submission.jsonl" \
  --errors-json "${out}/errors.json"

echo "=== [${TAG}] score ==="
python3 evaluate_simple_explanations.py \
  --submission "${out}/submission.jsonl" \
  --ground-truth "${REF}" \
  --output "${PROBE}/report_cmp_${TAG}.json" \
  --no-progress

echo
echo "=== aggregate (baseline vs ${TAG}) ==="
python3 "${EXP}/aggregate.py" "${PROBE}/report_cmp_baseline.json" "${PROBE}/report_cmp_${TAG}.json"

echo
echo "=== sample outputs (GT | SFT-baseline | ${TAG}) ==="
python3 - "$REF" "${PROBE}/cmp_baseline/submission.jsonl" "${out}/submission.jsonl" <<'PY'
import json, sys
ref, base, grpo = sys.argv[1:4]
def load(p):
    d={}
    for l in open(p):
        r=json.loads(l); d[r["sample_id"]]=r
    return d
R=load(ref); B=load(base); G=load(grpo)
ids=[i for i in G if i in B and i in R][:10]
for i in ids:
    print("-"*80)
    print("GT  :", R[i].get("simple_explanation","")[:160])
    print("SFT :", B[i].get("simple_explanation","")[:160])
    print("GRPO:", G[i].get("simple_explanation","")[:160])
PY
