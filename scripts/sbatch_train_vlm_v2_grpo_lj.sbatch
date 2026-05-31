#!/bin/bash
# Pass-2 GRPO (complex BERTScore reward) on Ljubljana (4 × A100 80GB).
#
# Continues from v2 SFT LoRA ckpt-1655. Saves every 100 steps; re-submit with
# ADAPTERS=<latest checkpoint> to resume if the 48h wall hits before 1 epoch.
#
# Submit from the Slurm login node:
#   sbatch scripts/sbatch_train_vlm_v2_grpo_lj.sbatch
#
# Resume from a mid-run checkpoint:
#   sbatch --export=ALL,ADAPTERS=/home/jakob/luka/runs/vlm_v2_grpo/run-YYYYMMDD-HHMMSS/checkpoint-5000 \
#          scripts/sbatch_train_vlm_v2_grpo_lj.sbatch
#
# Effective batch = per_device 1 × 4 GPUs × grad_accum 2 = 8 prompts/step.
# At ~15 s/step, 48 h ≈ 11k steps (~88k samples); full 363k rows ≈ 45k steps.
#
#SBATCH --job-name=xpv-grpo-v2-lj
#SBATCH --partition=elixir-interno
#SBATCH --nodelist=elixir-lj-gpu-01.elixir.ul.si
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jakob/luka/runs/slurm_grpo_v2_lj_%j.out
#SBATCH --error=/home/jakob/luka/runs/slurm_grpo_v2_lj_%j.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/luka/runs}"
DEFAULT_REPO_LC="$(echo "${GITHUB_REPOSITORY:-TheLukaDragar/XPlainVerse-ACMChallenge}" | tr '[:upper:]' '[:lower:]')"
CONTAINER="${LJ_APPTAINER_IMAGE:-docker://ghcr.io/${DEFAULT_REPO_LC}-lj:latest}"
APPTAINER_BIND="${LJ_APPTAINER_BIND:-${HOME}:${HOME},/primoz:/primoz}"

mkdir -p "${RUNS_ROOT}"

if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
  WANDB_API_KEY="$(python3 - "${HOME}/.netrc" <<'PY'
import netrc, sys
try:
    print(netrc.netrc(sys.argv[1]).authenticators("api.wandb.ai")[2])
except (FileNotFoundError, TypeError):
    pass
PY
)"
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"

_AUTH_ENV=""
if [[ -n "${APPTAINER_DOCKER_USERNAME:-}" ]]; then
  _AUTH_ENV+="export APPTAINER_DOCKER_USERNAME=$(printf '%q' "${APPTAINER_DOCKER_USERNAME}"); "
fi
if [[ -n "${APPTAINER_DOCKER_PASSWORD:-}" ]]; then
  _AUTH_ENV+="export APPTAINER_DOCKER_PASSWORD=$(printf '%q' "${APPTAINER_DOCKER_PASSWORD}"); "
fi

echo "=== sbatch_train_vlm_v2_grpo_lj ==="
echo "job:        ${SLURM_JOB_ID}"
echo "node:       ${SLURM_JOB_NODELIST:-?}"
echo "container:  ${CONTAINER}"
echo "code_root:  ${CODE_ROOT}"
echo "adapters:   ${ADAPTERS:-/home/jakob/luka/runs/vlm_v2/run-20260529-104845/checkpoint-1655}"
echo "wandb:      $([[ -n "${WANDB_API_KEY}" ]] && echo "key loaded (${#WANDB_API_KEY} chars)" || echo "not set")"
echo "started:    $(date -Is)"
echo

INNER="${_AUTH_ENV}
set -euo pipefail
export PYTHONNOUSERSITE=1
export WANDB_API_KEY=$(printf '%q' "${WANDB_API_KEY}")
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ADAPTERS=${ADAPTERS:-/home/jakob/luka/runs/vlm_v2/run-20260529-104845/checkpoint-1655}
export PER_DEVICE_BS=${PER_DEVICE_BS:-1}
export GRAD_ACCUM=${GRAD_ACCUM:-2}
export NUM_EPOCHS=${NUM_EPOCHS:-1}
export NUM_GENERATIONS=${NUM_GENERATIONS:-8}
export USE_VLLM=${USE_VLLM:-true}
export VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.4}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
export SAVE_STEPS=${SAVE_STEPS:-100}
export LOG_STEPS=${LOG_STEPS:-1}
export REPORT_TO=${REPORT_TO:-wandb}
export ADD_VERSION=false
export OUTPUT_DIR=${OUTPUT_DIR:-${RUNS_ROOT}/vlm_v2_grpo/job_${SLURM_JOB_ID}}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-vlm_v2_grpo_job_${SLURM_JOB_ID}}
cd $(printf '%q' "${CODE_ROOT}")
chmod +x scripts/train_vlm_v2_grpo_lj.sh
exec bash scripts/train_vlm_v2_grpo_lj.sh
"

srun --gres=gpu:4 --cpus-per-task=64 --mem=256G \
  bash -c "exec apptainer exec --nv -B $(printf '%q' "${APPTAINER_BIND}") --env PYTHONNOUSERSITE=1 $(printf '%q' "${CONTAINER}") bash -c $(printf '%q' "${INNER}")"

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
