#!/bin/bash
# Gold-verdict experiment — structured prompt variant only.
#
# Submit from login:
#   sbatch scripts/sbatch_gold_verdict_structured.sbatch
#
# Override checkpoint (must exist on disk):
#   sbatch --export=ALL,ADAPTERS=/path/to/checkpoint-3600 \
#          scripts/sbatch_gold_verdict_structured.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f /shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_gold_verdict_structured_<JOBID>.out
#
#SBATCH --job-name=xpv-gold-struct
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_gold_verdict_structured_%j.out
#SBATCH --error=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_gold_verdict_structured_%j.err

set -euo pipefail

WORKSPACE_ROOT=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge
CODE_ROOT="${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge"
CONTAINER=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest
EXP_DIR="${CODE_ROOT}/research/experiments/01_gold_verdict"
ADAPTERS="${ADAPTERS:-${WORKSPACE_ROOT}/runs/vlm_full/v1-20260524-214014/checkpoint-3600}"

mkdir -p "${WORKSPACE_ROOT}/runs"

echo "=== sbatch_gold_verdict_structured ==="
echo "job:        ${SLURM_JOB_ID}"
echo "node:       ${SLURM_NODELIST}"
echo "container:  ${CONTAINER}"
echo "adapters:   ${ADAPTERS}"
echo "variant:    structured"
echo "started:    $(date -Is)"
echo

if [[ ! -d "${ADAPTERS}" ]]; then
  echo "error: checkpoint missing: ${ADAPTERS}" >&2
  exit 1
fi

srun \
  --container-image "${CONTAINER}" \
  --container-mount-home \
  --container-mounts "${WORKSPACE_ROOT}:${WORKSPACE_ROOT}" \
  --container-workdir "${CODE_ROOT}" \
  bash -c "
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=0
    export LD_LIBRARY_PATH=\"/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:\${LD_LIBRARY_PATH:-}\"
    export TORCH_COMPILE_DISABLE=1
    export IMAGE_MAX_TOKEN_NUM=1024
    export MAX_PIXELS=1003520
    export ADAPTERS=${ADAPTERS}
    export VARIANTS=structured
    exec ${EXP_DIR}/run.sh
  " | tee -a "${EXP_DIR}/run_structured_slurm_${SLURM_JOB_ID}.log"

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
if [[ -f "${EXP_DIR}/results/structured/eval_results/final_scores.json" ]]; then
  echo "=== structured final_scores.json ==="
  python3 -m json.tool "${EXP_DIR}/results/structured/eval_results/final_scores.json"
else
  echo "WARNING: structured eval_results/final_scores.json not found — check logs"
  exit 1
fi
