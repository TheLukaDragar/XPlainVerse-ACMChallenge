#!/bin/bash
# Quick val eval for a Pass-2 adapter (GRPO or SFT) — official scorer, N=500 balanced.
#
# Submit:
#   sbatch scripts/sbatch_eval_grpo_quick_lj.sbatch
#   sbatch --export=ALL,ADAPTERS=/path/to/checkpoint-26400,EVAL_TAG=grpo26400 \
#          scripts/sbatch_eval_grpo_quick_lj.sbatch
#
#SBATCH --job-name=xpv-eval-quick
#SBATCH --partition=elixir-interno
#SBATCH --nodelist=elixir-lj-gpu-01.elixir.ul.si
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/home/jakob/luka/runs/slurm_eval_quick_%j.out
#SBATCH --error=/home/jakob/luka/runs/slurm_eval_quick_%j.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
DEFAULT_REPO_LC="$(echo "${GITHUB_REPOSITORY:-TheLukaDragar/XPlainVerse-ACMChallenge}" | tr '[:upper:]' '[:lower:]')"
CONTAINER="${LJ_APPTAINER_IMAGE:-docker://ghcr.io/${DEFAULT_REPO_LC}-lj:latest}"
APPTAINER_BIND="${LJ_APPTAINER_BIND:-${HOME}:${HOME},/primoz:/primoz}"

ADAPTERS="${ADAPTERS:-/home/jakob/luka/runs/vlm_v2_grpo/job_48855/checkpoint-26400}"
EVAL_TAG="${EVAL_TAG:-grpo}"
N="${N:-500}"
VERDICT_SOURCE="${VERDICT_SOURCE:-ensemble}"

echo "=== sbatch_eval_grpo_quick_lj ==="
echo "job:       ${SLURM_JOB_ID}"
echo "adapters:  ${ADAPTERS}"
echo "N:         ${N}  verdict: ${VERDICT_SOURCE}"
echo "started:   $(date -Is)"

INNER="set -euo pipefail
export PYTHONNOUSERSITE=1
export ADAPTERS=$(printf '%q' "${ADAPTERS}")
export N=${N}
export VERDICT_SOURCE=$(printf '%q' "${VERDICT_SOURCE}")
export CUDA_VISIBLE_DEVICES=0
export QWEN_BATCH_SIZE=16
cd $(printf '%q' "${CODE_ROOT}")
chmod +x scripts/eval_pipeline_v2_lj.sh
exec bash scripts/eval_pipeline_v2_lj.sh
"

srun --gres=gpu:1 --cpus-per-task=32 --mem=128G \
  bash -c "exec apptainer exec --nv -B $(printf '%q' "${APPTAINER_BIND}") --env PYTHONNOUSERSITE=1 $(printf '%q' "${CONTAINER}") bash -c $(printf '%q' "${INNER}")"

echo "=== eval job ${SLURM_JOB_ID} done $(date -Is) ==="
