#!/usr/bin/env bash
# Score probe submissions with the official simple evaluator (BERT + SLE) in the eval container.
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
EXP="${CODE_ROOT}/research/experiments/04_simple_score"
PROBE="${EXP}/probe"
REF="${PROBE}/ref_subset.jsonl"

# Node has internet (HF downloads run here); allow fetching the BERTScore model.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

cd "${CODE_ROOT}/evaluation"
for name in oracle real_firstsent real_w35 real_w25; do
  echo "=== scoring ${name} ==="
  python3 evaluate_simple_explanations.py \
    --submission "${PROBE}/submission_${name}.jsonl" \
    --ground-truth "${REF}" \
    --output "${PROBE}/report_${name}.json" \
    --no-progress
done

echo
echo "=== aggregate ==="
python3 "${EXP}/aggregate.py" "${PROBE}"/report_*.json
