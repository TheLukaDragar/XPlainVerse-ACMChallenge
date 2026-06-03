#!/bin/bash
# Compressor SFT (complex -> simple) on Frida — 4×GPU, no node pin (fastest start).
#
# Trains Qwen3.5-4B LoRA on dataset/train_compressor.jsonl (320k fake rows).
# Excludes aga (known NCCL/GPU issues May 2026).
#
# Submit:
#   sbatch scripts/sbatch_train_compressor_4gpu.sbatch
#
# Smoke:
#   TRAIN_SLICE=64 VAL_SLICE=32 MAX_STEPS=2 sbatch --export=ALL,TRAIN_SLICE,VAL_SLICE,MAX_STEPS ...
#
#SBATCH --job-name=xpv-compressor
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --exclude=aga
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_compressor_%j.out
#SBATCH --error=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_compressor_%j.err

set -euo pipefail

WORKSPACE_ROOT=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge
CODE_ROOT=${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge
CONTAINER=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest

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

echo "=== sbatch_train_compressor_4gpu ==="
echo "job:        ${SLURM_JOB_ID}"
echo "nodes:      ${SLURM_JOB_NODELIST}"
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
    export PER_DEVICE_BS=8
    export GRAD_ACCUM=1
    export OUTPUT_DIR=${WORKSPACE_ROOT}/runs/compressor_sft
    export WANDB_RUN_NAME=compressor_sft_4gpu_1ep
    export ADD_VERSION=false
    chmod +x ${CODE_ROOT}/scripts/train_compressor_frida.sh
    exec ${CODE_ROOT}/scripts/train_compressor_frida.sh
  "

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
