#!/bin/bash
# Full VLM LoRA SFT on XPlainVerse (450k rows, 1× A100 80GB, partition frida).
#
# Submit from the login node:
#   sbatch scripts/sbatch_train_vlm_full.sbatch
#
# Override GPU type (e.g. H100, B300) at submit time:
#   sbatch --gres=gpu:H100:1 scripts/sbatch_train_vlm_full.sbatch
#
# 4-GPU run (sets NPROC_PER_NODE=4, DeepSpeed ZeRO-2):
#   sbatch --gres=gpu:A100_80GB:4 --cpus-per-task=64 --mem=256G \
#          --export=ALL,NPROC_PER_NODE=4,CUDA_VISIBLE_DEVICES=0,1,2,3 \
#          scripts/sbatch_train_vlm_full.sbatch
#
# Disable W&B (use tensorboard instead):
#   sbatch --export=ALL,REPORT_TO=tensorboard scripts/sbatch_train_vlm_full.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f /shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_vlm_full_<JOBID>.out
#   sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode
#
#SBATCH --job-name=xpv-vlm-full
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_vlm_full_%j.out
#SBATCH --error=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_vlm_full_%j.err

set -euo pipefail

WORKSPACE_ROOT=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge
CODE_ROOT=${WORKSPACE_ROOT}/code/XPlainVerse-ACMChallenge
CONTAINER=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:latest

mkdir -p "${WORKSPACE_ROOT}/runs"

echo "=== sbatch_train_vlm_full ==="
echo "job:        ${SLURM_JOB_ID}"
echo "node:       ${SLURM_NODELIST}"
echo "container:  ${CONTAINER}"
echo "workspace:  ${WORKSPACE_ROOT}"
echo "started:    $(date -Is)"
echo

# Pyxis mounts:
#   --container-mount-home   → $HOME visible inside (for HF cache, W&B creds, Cursor CLI)
#   --container-mounts       → full workspace tree (data/ + code/ + runs/)
#   $SCRATCH                 → auto-mounted by Pyxis (node-local fast scratch)
#
# train_vlm_full.sh reads all tunable knobs from env vars — override via
# --export=ALL,VAR=val at sbatch submit time.
srun \
  --container-image "${CONTAINER}" \
  --container-mount-home \
  --container-mounts "${WORKSPACE_ROOT}:${WORKSPACE_ROOT}" \
  --container-workdir "${CODE_ROOT}" \
  bash -c "
    set -euo pipefail
    export WORKSPACE_ROOT=${WORKSPACE_ROOT}
    export CODE_ROOT=${CODE_ROOT}
    chmod +x ${CODE_ROOT}/scripts/train_vlm_full.sh
    exec ${CODE_ROOT}/scripts/train_vlm_full.sh
  "

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
