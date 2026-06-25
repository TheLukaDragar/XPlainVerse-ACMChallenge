#!/bin/bash
# Stage-2 Qwen3-VL v2 SFT with an UNFROZEN vision tower — Ljubljana (4 × A100 80GB).
#
# Motivation: with freeze_vit=true (sbatch_train_vlm_v2_lj.sbatch) the model learns
# the right *style* but hallucinates / mis-grounds the evidence objects, tanking
# entity+facts F1 (70% of the complex score). Training the ViT + aligner lets the
# model attend to the actual evidence regions.
#
# Differences vs the frozen run:
#   FREEZE_VIT=false, FREEZE_ALIGNER=false  -> vision tower is trained
#   VIT_LR=1e-5, ALIGNER_LR=1e-5            -> small LR so pretrained features
#                                              are refined, not destroyed
#   VIT_GRAD_CKPT=true                      -> checkpoint the ViT to save memory
#   PER_DEVICE_BS=1, GRAD_ACCUM=8           -> eff batch 32, fits the extra
#                                              ViT activations/optimizer state
#   LEARNING_RATE=1e-4                      -> slightly gentler LLM-LoRA LR
#
# Same single lj container (GHCR -lj) via Apptainer; deepspeed ZeRO-2 shards the
# now-larger optimizer state across the 4 GPUs.
#
# Submit from the Slurm login node:
#   sbatch scripts/sbatch_train_vlm_v2_unfrozen_lj.sbatch
#
# Optional: raise visual resolution for finer artifacts (more memory):
#   sbatch --export=ALL,MAX_PIXELS=1605632,PER_DEVICE_BS=1,GRAD_ACCUM=8 \
#          scripts/sbatch_train_vlm_v2_unfrozen_lj.sbatch
#
#SBATCH --job-name=xpv-vlm-v2-unfrozen-lj
#SBATCH --partition=elixir-interno
#SBATCH --nodelist=elixir-lj-gpu-01.elixir.ul.si
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jakob/luka/runs/slurm_vlm_v2_unfrozen_lj_%j.out
#SBATCH --error=/home/jakob/luka/runs/slurm_vlm_v2_unfrozen_lj_%j.err

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

_AUTH_ENV=""
if [[ -n "${APPTAINER_DOCKER_USERNAME:-}" ]]; then
  _AUTH_ENV+="export APPTAINER_DOCKER_USERNAME=$(printf '%q' "${APPTAINER_DOCKER_USERNAME}"); "
fi
if [[ -n "${APPTAINER_DOCKER_PASSWORD:-}" ]]; then
  _AUTH_ENV+="export APPTAINER_DOCKER_PASSWORD=$(printf '%q' "${APPTAINER_DOCKER_PASSWORD}"); "
fi

# Unfreeze knobs (override at submit time with --export=ALL,...).
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
VIT_LR="${VIT_LR:-1e-5}"
ALIGNER_LR="${ALIGNER_LR:-1e-5}"
REPORT_TO="${REPORT_TO:-wandb}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/vlm_v2_unfrozen/job_${SLURM_JOB_ID}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-vlm_v2_unfrozen_lj_job_${SLURM_JOB_ID}}"

echo "=== sbatch_train_vlm_v2_unfrozen_lj ==="
echo "job:        ${SLURM_JOB_ID}"
echo "node:       ${SLURM_JOB_NODELIST:-?}"
echo "container:  ${CONTAINER}"
echo "code_root:  ${CODE_ROOT}"
echo "unfreeze:   vit+aligner  vit_lr=${VIT_LR} aligner_lr=${ALIGNER_LR}  llm_lr=${LEARNING_RATE}"
echo "batch:      per_device=${PER_DEVICE_BS} grad_accum=${GRAD_ACCUM} gpus=${NPROC_PER_NODE}"
echo "output:     ${OUTPUT_DIR}"
echo "wandb:      $([[ -n "${WANDB_API_KEY}" ]] && echo "key loaded (${#WANDB_API_KEY} chars)" || echo "not set")"
echo "started:    $(date -Is)"
echo

INNER="${_AUTH_ENV}
set -euo pipefail
export PYTHONNOUSERSITE=1
export WANDB_API_KEY=$(printf '%q' "${WANDB_API_KEY}")
export NPROC_PER_NODE=${NPROC_PER_NODE}
export PER_DEVICE_BS=${PER_DEVICE_BS}
export GRAD_ACCUM=${GRAD_ACCUM}
export NUM_EPOCHS=${NUM_EPOCHS}
export LEARNING_RATE=${LEARNING_RATE}
export FREEZE_VIT=false
export FREEZE_ALIGNER=false
export FREEZE_LLM=false
export VIT_GRAD_CKPT=true
export VIT_LR=${VIT_LR}
export ALIGNER_LR=${ALIGNER_LR}
export REPORT_TO=${REPORT_TO}
export OUTPUT_DIR=${OUTPUT_DIR}
export WANDB_RUN_NAME=${WANDB_RUN_NAME}
export ADD_VERSION=false
${MAX_PIXELS:+export MAX_PIXELS=${MAX_PIXELS}}
cd $(printf '%q' "${CODE_ROOT}")
chmod +x scripts/train_vlm_v2_lj.sh
exec bash scripts/train_vlm_v2_lj.sh
"

srun --gres=gpu:4 --cpus-per-task=64 --mem=256G \
  bash -c "exec apptainer exec --nv -B $(printf '%q' "${APPTAINER_BIND}") --env PYTHONNOUSERSITE=1 $(printf '%q' "${CONTAINER}") bash -c $(printf '%q' "${INNER}")"

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
