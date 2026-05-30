#!/bin/bash
# Resume Qwen3-VL v2 SFT on axa from a prior checkpoint (default: job_91935 ckpt-8000).
#
# Submit:
#   sbatch scripts/sbatch_train_vlm_v2_axa4_resume.sbatch
#
# Override checkpoint:
#   RESUME_CKPT=/path/to/checkpoint-8000 sbatch scripts/sbatch_train_vlm_v2_axa4_resume.sbatch
#
#SBATCH --job-name=xpv-vlm-v2-axa4
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --nodelist=axa
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=850G
#SBATCH --time=7-00:00:00
#SBATCH --output=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_vlm_v2_axa4_%j.out
#SBATCH --error=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_vlm_v2_axa4_%j.err

set -euo pipefail

WORKSPACE_ROOT=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge
CODE_ROOT=${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge
CONTAINER=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest
RESUME_CKPT="${RESUME_CKPT:-${WORKSPACE_ROOT}/runs/vlm_v2_axa4/job_91935/checkpoint-8000}"

if [[ ! -d "${RESUME_CKPT}" ]]; then
  echo "error: resume checkpoint not found: ${RESUME_CKPT}" >&2
  exit 1
fi

mkdir -p "${WORKSPACE_ROOT}/runs"

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
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "error: WANDB_API_KEY not set and no api.wandb.ai entry in ${HOME}/.netrc" >&2
  exit 1
fi
export WANDB_API_KEY

echo "=== sbatch_train_vlm_v2_axa4_resume ==="
echo "job:        ${SLURM_JOB_ID}"
echo "nodes:      ${SLURM_JOB_NODELIST}"
echo "resume:     ${RESUME_CKPT}"
echo "container:  ${CONTAINER}"
echo "started:    $(date -Is)"
echo

srun \
  --container-image "${CONTAINER}" \
  --container-mount-home \
  --container-mounts "${WORKSPACE_ROOT}:${WORKSPACE_ROOT}" \
  --container-workdir "${CODE_ROOT}" \
  --export=ALL,WANDB_API_KEY,SLURM_JOB_ID,SLURM_JOB_USER,SLURM_GPUS_ON_NODE \
  --label \
  bash -lc "
    set -euo pipefail
    export WORKSPACE_ROOT=${WORKSPACE_ROOT}
    export CODE_ROOT=${CODE_ROOT}
    export FRIDA_USER_HOME=${HOME}
    export NPROC_PER_NODE=4
    export PER_DEVICE_BS=2
    export GRAD_ACCUM=4
    export OUTPUT_DIR=${WORKSPACE_ROOT}/runs/vlm_v2_axa4
    export WANDB_RUN_NAME=vlm_v2_axa4_1ep_resume
    export ADD_VERSION=false
    export RESUME_FROM_CHECKPOINT=${RESUME_CKPT}
    chmod +x ${CODE_ROOT}/scripts/train_vlm_v2_frida.sh
    exec ${CODE_ROOT}/scripts/train_vlm_v2_frida.sh
  "

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
