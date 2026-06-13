#!/usr/bin/env bash
# Compressor prompt A/B/C on a 200-fake holdout (GT complex as input) -> score simple.
# Runs inside the eval container (ms-swift). Dispatch with lj GPU srun.
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
EXP="${CODE_ROOT}/research/experiments/04_simple_score"
PROBE="${EXP}/probe"
COMPLEX_IN="${PROBE}/fake200_complex.jsonl"
REF="${PROBE}/fake200_ref.jsonl"
ADAPTERS="${ADAPTERS:-/home/jakob/luka/runs/compressor_vl/checkpoint-10000}"

export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

declare -A PROMPTS=(
  [baseline]="${CODE_ROOT}/dataset/prompt_v2.txt"
  [ultrasimple]="${EXP}/prompt_ultrasimple.txt"
  [tworeasons]="${EXP}/prompt_tworeasons.txt"
)

cd "${CODE_ROOT}/evaluation"
for name in baseline ultrasimple tworeasons; do
  pf="${PROMPTS[$name]}"
  out="${PROBE}/cmp_${name}"
  mkdir -p "${out}"
  echo "=== [${name}] build infer ==="
  python3 build_compressor_infer_test.py \
    --complex "${COMPLEX_IN}" --prompt-file "${pf}" \
    --shard-id 0 --shard-count 1 --out "${out}/infer_in.jsonl"

  echo "=== [${name}] swift infer ==="
  python3 /opt/ms-swift/swift/cli/infer.py \
    --model Qwen/Qwen3-VL-8B-Instruct --model_type qwen3_vl --use_hf true \
    --adapters "${ADAPTERS}" \
    --val_dataset "${out}/infer_in.jsonl" \
    --infer_backend transformers \
    --max_batch_size 32 --max_new_tokens 128 \
    --result_path "${out}/infer_out.jsonl"

  echo "=== [${name}] build submission ==="
  python3 build_simple_explanations.py \
    --complex "${REF}" \
    --compressor-infer "${out}/infer_out.jsonl" \
    --output "${out}/submission.jsonl" \
    --errors-json "${out}/errors.json"

  echo "=== [${name}] score ==="
  python3 evaluate_simple_explanations.py \
    --submission "${out}/submission.jsonl" \
    --ground-truth "${REF}" \
    --output "${PROBE}/report_cmp_${name}.json" \
    --no-progress
done

echo
echo "=== aggregate ==="
python3 "${EXP}/aggregate.py" "${PROBE}"/report_cmp_*.json
