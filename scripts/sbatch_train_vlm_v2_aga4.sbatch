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

echo "=== sbatch_train_vlm_v2_aga4 ==="
echo "job:        ${SLURM_JOB_ID}"
echo "nodes:      ${SLURM_JOB_NODELIST}"
echo "ntasks:     ${SLURM_NTASKS:-?}"
echo "container:  ${CONTAINER}"
echo "workspace:  ${WORKSPACE_ROOT}"
echo "started:    $(date -Is)"
echo

srun \
  --container-image "${CONTAINER}" \
  --container-mount-home \
  --container-mounts "${WORKSPACE_ROOT}:${WORKSPACE_ROOT}" \
  --container-workdir "${CODE_ROOT}" \
  --label \
  bash -lc "
    set -euo pipefail
    export WORKSPACE_ROOT=${WORKSPACE_ROOT}
    export CODE_ROOT=${CODE_ROOT}
    export NPROC_PER_NODE=4
    export PER_DEVICE_BS=2
    export GRAD_ACCUM=4
    export OUTPUT_DIR=${WORKSPACE_ROOT}/runs/vlm_v2_aga4
    export WANDB_RUN_NAME=vlm_v2_aga4_1ep
    chmod +x ${CODE_ROOT}/scripts/train_vlm_v2_frida.sh
    exec ${CODE_ROOT}/scripts/train_vlm_v2_frida.sh
  "

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
