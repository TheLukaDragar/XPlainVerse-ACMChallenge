#!/bin/bash
# Qwen3-VL v2 SFT on Frida — aga node (4 × A100 40GB).
#
# Submit:
#   sbatch scripts/sbatch_train_vlm_v2_aga4.sbatch
#
# Effective batch 32: per_device=2 × 4 GPUs × grad_accum=4
#
#SBATCH --job-name=xpv-vlm-v2-aga4
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --nodelist=aga
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=400G
#SBATCH --time=7-00:00:00
#SBATCH --output=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_vlm_v2_aga4_%j.out
#SBATCH --error=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_vlm_v2_aga4_%j.err

set -euo pipefail

WORKSPACE_ROOT=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge
CODE_ROOT=${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge
CONTAINER=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest

mkdir -p "${WORKSPACE_ROOT}/runs"

# Resolve on login node (scontrol not available inside Pyxis).
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"

# Load W&B key from login-node ~/.netrc for the root-run container.
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

echo "=== sbatch_train_vlm_v2_aga4 ==="
echo "job:        ${SLURM_JOB_ID}"
echo "nodes:      ${SLURM_JOB_NODELIST}"
echo "master:     ${MASTER_ADDR}"
echo "ntasks:     ${SLURM_NTASKS:-?}"
echo "container:  ${CONTAINER}"
echo "workspace:  ${WORKSPACE_ROOT}"
echo "wandb:      key loaded (${#WANDB_API_KEY} chars)"
echo "started:    $(date -Is)"
echo

srun \
  --container-image "${CONTAINER}" \
  --container-mount-home \
  --container-mounts "${WORKSPACE_ROOT}:${WORKSPACE_ROOT}" \
  --container-workdir "${CODE_ROOT}" \
  --export=ALL,MASTER_ADDR="${MASTER_ADDR}",WANDB_API_KEY,SLURM_JOB_ID,SLURM_JOB_USER,SLURM_GPUS_ON_NODE \
  --label \
  bash -lc "
    set -euo pipefail
    export WORKSPACE_ROOT=${WORKSPACE_ROOT}
    export CODE_ROOT=${CODE_ROOT}
    export FRIDA_USER_HOME=${HOME}
    export NPROC_PER_NODE=4
    export PER_DEVICE_BS=2
    export GRAD_ACCUM=4
    export OUTPUT_DIR=${WORKSPACE_ROOT}/runs/vlm_v2_aga4
    export WANDB_RUN_NAME=vlm_v2_aga4_1ep
    export ADD_VERSION=false
    chmod +x ${CODE_ROOT}/scripts/train_vlm_v2_frida.sh
    exec ${CODE_ROOT}/scripts/train_vlm_v2_frida.sh
  "

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
