#!/usr/bin/env bash
# Build v2 ms-swift training JSONL on a CPU compute node.
#
# Submit from login:
#   sbatch scripts/sbatch_build_train_v2.sbatch
#
# Override knobs at submit time:
#   sbatch --export=ALL,HYPOTHETICAL_RATIO=0.6,FAKE_REAL_RATIO=3 \
#          scripts/sbatch_build_train_v2.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f /shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_build_v2_<JOBID>.out

#SBATCH --job-name=xpv-build-v2
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_build_v2_%j.out
#SBATCH --error=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_build_v2_%j.err

set -euo pipefail

WORKSPACE_ROOT=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge
CODE_ROOT="${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge"
CONTAINER=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest

HYPOTHETICAL_RATIO="${HYPOTHETICAL_RATIO:-0.50}"
FAKE_REAL_RATIO="${FAKE_REAL_RATIO:-4.0}"
FILTER_MIN_SENTENCES="${FILTER_MIN_SENTENCES:-3}"
FILTER_MIN_WORDS="${FILTER_MIN_WORDS:-80}"
FILTER_MIN_CONNECTIVES="${FILTER_MIN_CONNECTIVES:-0}"
PROMPT_FILE="${PROMPT_FILE:-${CODE_ROOT}/dataset/prompt_v2.txt}"

mkdir -p "${WORKSPACE_ROOT}/runs"
mkdir -p "${CODE_ROOT}/dataset/build_logs"

echo "=== sbatch_build_train_v2 ==="
echo "job:              ${SLURM_JOB_ID}"
echo "node:             ${SLURM_NODELIST}"
echo "container:        ${CONTAINER}"
echo "prompt:           ${PROMPT_FILE}"
echo "hypothetical:     ${HYPOTHETICAL_RATIO}"
echo "fake_real_ratio:  ${FAKE_REAL_RATIO}"
echo "filter:           sentences>=${FILTER_MIN_SENTENCES} "\
  "words>=${FILTER_MIN_WORDS} connectives>=${FILTER_MIN_CONNECTIVES}"
echo "started:          $(date -Is)"
echo

srun \
  --container-image "${CONTAINER}" \
  --container-mount-home \
  --container-mounts "${WORKSPACE_ROOT}:${WORKSPACE_ROOT}" \
  --container-workdir "${CODE_ROOT}" \
  bash -c "
    set -euo pipefail
    /usr/bin/python3 ${CODE_ROOT}/dataset/build_swift_jsonl_v2.py \
      --splits train val \
      --prompt-file ${PROMPT_FILE} \
      --hypothetical-ratio ${HYPOTHETICAL_RATIO} \
      --fake-real-ratio ${FAKE_REAL_RATIO} \
      --filter-min-sentences ${FILTER_MIN_SENTENCES} \
      --filter-min-words ${FILTER_MIN_WORDS} \
      --filter-min-connectives ${FILTER_MIN_CONNECTIVES}
  "

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
ls -la ${CODE_ROOT}/dataset/train_vlm_v2.jsonl ${CODE_ROOT}/dataset/val_vlm_v2.jsonl 2>&1 || true
echo
echo "Row counts:"
wc -l ${CODE_ROOT}/dataset/train_vlm_v2.jsonl ${CODE_ROOT}/dataset/val_vlm_v2.jsonl 2>&1 || true
