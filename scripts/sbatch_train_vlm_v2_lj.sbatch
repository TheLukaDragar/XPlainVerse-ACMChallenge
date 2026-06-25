#!/bin/bash
# Stage-2 Qwen3-VL v2 SFT on Ljubljana (elixir-lj-gpu-01, 4 × A100 80GB).
#
# Batch job so training survives logout. Runs inside the single lj training
# container (GHCR -lj image, docker/Dockerfile.lj) via Apptainer — the same
# container used for Pass-1, keeping the lj stack aligned. The devel-base image
# ships nvcc (deepspeed) + flash_attn; train_vlm_v2_lj.sh adds the FSDP2 shim.
#
# Submit from the Slurm login node:
#   sbatch scripts/sbatch_train_vlm_v2_lj.sbatch
#
# Override knobs at submit time:
#   sbatch --export=ALL,NUM_EPOCHS=2,PER_DEVICE_BS=2,GRAD_ACCUM=4 \
#          scripts/sbatch_train_vlm_v2_lj.sbatch
#
# Effective batch 32 = per_device 2 × 4 GPUs × grad_accum 4.
#
#SBATCH --job-name=xpv-vlm-v2-lj
#SBATCH --partition=elixir-interno
#SBATCH --nodelist=elixir-lj-gpu-01.elixir.ul.si
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jakob/luka/runs/slurm_vlm_v2_lj_%j.out
#SBATCH --error=/home/jakob/luka/runs/slurm_vlm_v2_lj_%j.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/luka/runs}"
DEFAULT_REPO_LC="$(echo "${GITHUB_REPOSITORY:-TheLukaDragar/XPlainVerse-ACMChallenge}" | tr '[:upper:]' '[:lower:]')"
CONTAINER="${LJ_APPTAINER_IMAGE:-docker://ghcr.io/${DEFAULT_REPO_LC}-lj:latest}"
APPTAINER_BIND="${LJ_APPTAINER_BIND:-${HOME}:${HOME},/primoz:/primoz}"

mkdir -p "${RUNS_ROOT}"

# Load W&B key from login-node ~/.netrc so the container can sync.
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

# Forward GHCR creds (only needed if the -lj package is private).
_AUTH_ENV=""
if [[ -n "${APPTAINER_DOCKER_USERNAME:-}" ]]; then
  _AUTH_ENV+="export APPTAINER_DOCKER_USERNAME=$(printf '%q' "${APPTAINER_DOCKER_USERNAME}"); "
fi
if [[ -n "${APPTAINER_DOCKER_PASSWORD:-}" ]]; then
  _AUTH_ENV+="export APPTAINER_DOCKER_PASSWORD=$(printf '%q' "${APPTAINER_DOCKER_PASSWORD}"); "
fi

echo "=== sbatch_train_vlm_v2_lj ==="
echo "job:        ${SLURM_JOB_ID}"
echo "node:       ${SLURM_JOB_NODELIST:-?}"
echo "container:  ${CONTAINER}"
echo "code_root:  ${CODE_ROOT}"
echo "wandb:      $([[ -n "${WANDB_API_KEY}" ]] && echo "key loaded (${#WANDB_API_KEY} chars)" || echo "not set (REPORT_TO=tensorboard to disable)")"
echo "started:    $(date -Is)"
echo

INNER="${_AUTH_ENV}
set -euo pipefail
export PYTHONNOUSERSITE=1
export WANDB_API_KEY=$(printf '%q' "${WANDB_API_KEY}")
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export PER_DEVICE_BS=${PER_DEVICE_BS:-2}
export GRAD_ACCUM=${GRAD_ACCUM:-4}
export NUM_EPOCHS=${NUM_EPOCHS:-1}
export OUTPUT_DIR=${OUTPUT_DIR:-${RUNS_ROOT}/vlm_v2/job_${SLURM_JOB_ID}}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-vlm_v2_lj_job_${SLURM_JOB_ID}}
export REPORT_TO=${REPORT_TO:-wandb}
export ADD_VERSION=false
cd $(printf '%q' "${CODE_ROOT}")
chmod +x scripts/train_vlm_v2_lj.sh
exec bash scripts/train_vlm_v2_lj.sh
"

srun --gres=gpu:4 --cpus-per-task=64 --mem=256G \
  bash -c "exec apptainer exec --nv -B $(printf '%q' "${APPTAINER_BIND}") --env PYTHONNOUSERSITE=1 $(printf '%q' "${CONTAINER}") bash -c $(printf '%q' "${INNER}")"

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
