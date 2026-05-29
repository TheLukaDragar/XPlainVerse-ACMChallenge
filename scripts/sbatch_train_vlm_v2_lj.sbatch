#!/bin/bash
# Stage-2 Qwen3-VL v2 SFT on Ljubljana (elixir-lj-gpu-01, 4 × A100 80GB).
#
# Batch job so training survives logout. Runs inside the VLM training SIF via
# ~/xplainverse_exec.sh (torch 2.11 + flash_attn), the same container v1 used —
# NOT the GHCR -lj Pass-1 image (which lacks a usable deepspeed/nvcc + FSDP2).
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
mkdir -p "${RUNS_ROOT}"

# Bind the GPU-node NVMe dataset into the SIF (xplainverse_exec.sh does not).
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:-/primoz:/primoz}"
export SINGULARITY_BINDPATH="${SINGULARITY_BINDPATH:-${APPTAINER_BINDPATH}}"

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

# v2 training knobs (override via --export at submit time).
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export REPORT_TO="${REPORT_TO:-wandb}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/vlm_v2/job_${SLURM_JOB_ID}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-vlm_v2_lj_job_${SLURM_JOB_ID}}"

echo "=== sbatch_train_vlm_v2_lj ==="
echo "job:        ${SLURM_JOB_ID}"
echo "node:       ${SLURM_JOB_NODELIST:-?}"
echo "code_root:  ${CODE_ROOT}"
echo "container:  ${XPLAINVERSE_SIF:-${HOME}/containers/xplainverse-acmchallenge.sif}"
echo "gpus:       ${NPROC_PER_NODE}  per_device=${PER_DEVICE_BS} grad_accum=${GRAD_ACCUM}"
echo "output:     ${OUTPUT_DIR}"
echo "wandb:      $([[ -n "${WANDB_API_KEY}" ]] && echo "key loaded (${#WANDB_API_KEY} chars)" || echo "not set (REPORT_TO=tensorboard to disable)")"
echo "started:    $(date -Is)"
echo

cd "${CODE_ROOT}"
srun --gres=gpu:4 --cpus-per-task=64 --mem=256G \
  ~/xplainverse_exec.sh env \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    PER_DEVICE_BS="${PER_DEVICE_BS}" \
    GRAD_ACCUM="${GRAD_ACCUM}" \
    NUM_EPOCHS="${NUM_EPOCHS}" \
    REPORT_TO="${REPORT_TO}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
    WANDB_API_KEY="${WANDB_API_KEY}" \
    ADD_VERSION=false \
    bash scripts/train_vlm_v2_lj.sh

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
