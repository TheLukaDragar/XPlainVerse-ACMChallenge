#!/bin/bash
# Compressor GRPO on Frida — 4×A100 on axa (fastest free node; 40GB).
#
# Submit:
#   sbatch scripts/sbatch_train_compressor_grpo_4gpu.sbatch
#
#SBATCH --job-name=xpv-compressor-grpo
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --nodelist=axa
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=850G
# 48h fits before the Jun 15 23:00 maintenance reservation so the job starts now.
# Checkpoints every 200 steps; resume after maintenance for the rest.
#SBATCH --time=48:00:00
#SBATCH --output=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_compressor_grpo_%j.out
#SBATCH --error=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_compressor_grpo_%j.err

set -euo pipefail

WORKSPACE_ROOT=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge
CODE_ROOT=${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge
CONTAINER=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest
SFT_CKPT=${WORKSPACE_ROOT}/runs/compressor_vl_sft/job_93030/checkpoint-10000

if [[ ! -d "${SFT_CKPT}" ]]; then
  echo "error: SFT checkpoint missing: ${SFT_CKPT}" >&2
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
  echo "error: WANDB_API_KEY not set" >&2
  exit 1
fi
export WANDB_API_KEY

echo "=== sbatch_train_compressor_grpo_4gpu ==="
echo "job:        ${SLURM_JOB_ID}"
echo "nodes:      ${SLURM_JOB_NODELIST}"
echo "sft ckpt:   ${SFT_CKPT}"
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
    export SFT_ADAPTERS=${SFT_CKPT}
    export OUTPUT_DIR=${WORKSPACE_ROOT}/runs/compressor_grpo
    export WANDB_RUN_NAME=compressor_grpo_frida_4gpu
    chmod +x ${CODE_ROOT}/scripts/train_compressor_grpo_frida.sh
    exec ${CODE_ROOT}/scripts/train_compressor_grpo_frida.sh
  "

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
