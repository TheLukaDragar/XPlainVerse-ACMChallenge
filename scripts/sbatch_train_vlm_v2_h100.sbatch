#!/bin/bash
# Qwen3-VL v2 SFT on Frida, defaulting to one full H100 node (8 × H100 80GB).
#
# Submit the real run:
#   sbatch scripts/sbatch_train_vlm_v2_h100.sbatch
#
# Smoke test on any single A100/A100_80GB:
#   sbatch --job-name=xpv-v2-smoke --gres=gpu:A100:1 --cpus-per-task=16 \
#     --mem=80G --time=00:45:00 \
#     --export=ALL,NPROC_PER_NODE=1,CUDA_VISIBLE_DEVICES=0,PER_DEVICE_BS=1,GRAD_ACCUM=1,\
#TRAIN_SLICE=32,VAL_SLICE=16,MAX_STEPS=2,EVAL_STEPS=999999,SAVE_STEPS=999999,\
#PREDICT_WITH_GENERATE=false,REPORT_TO=tensorboard,OUTPUT_DIR=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/vlm_v2_smoke \
#     scripts/sbatch_train_vlm_v2_h100.sbatch
#
# Multi-node-capable note:
#   Override --nodes=N --ntasks=N and set/request an appropriate GPU type/count.
#   train_vlm_v2_frida.sh derives NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT,
#   and NPROC_PER_NODE for ms-swift/torchrun from the Slurm allocation.
#
#SBATCH --job-name=xpv-vlm-v2-h100
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1500G
#SBATCH --time=7-00:00:00
#SBATCH --output=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_vlm_v2_h100_%j.out
#SBATCH --error=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_vlm_v2_h100_%j.err

set -euo pipefail

WORKSPACE_ROOT=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge
CODE_ROOT=${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge
CONTAINER=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest

mkdir -p "${WORKSPACE_ROOT}/runs"

echo "=== sbatch_train_vlm_v2_h100 ==="
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
    chmod +x ${CODE_ROOT}/scripts/train_vlm_v2_frida.sh
    exec ${CODE_ROOT}/scripts/train_vlm_v2_frida.sh
  "

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
