#!/bin/bash
#SBATCH --job-name=pass1-val-tta
#SBATCH --partition=elixir-interno
#SBATCH --nodelist=elixir-lj-gpu-01.elixir.ul.si
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/jakob/luka/runs/pass1_val_tta/slurm-%j.out
#SBATCH --error=/home/jakob/luka/runs/pass1_val_tta/slurm-%j.err

set -euo pipefail

PROJECT_DIR="${HOME}/luka/code/XPlainVerse-ACMChallenge"
DEFAULT_REPO_LC="$(echo "${GITHUB_REPOSITORY:-TheLukaDragar/XPlainVerse-ACMChallenge}" | tr '[:upper:]' '[:lower:]')"
LJ_APPTAINER_IMAGE="${LJ_APPTAINER_IMAGE:-docker://ghcr.io/${DEFAULT_REPO_LC}-lj:latest}"
APPTAINER_BIND="${LJ_APPTAINER_BIND:-${HOME}:${HOME},/primoz:/primoz},${PROJECT_DIR}:/workspace/XPlainVerse-ACMChallenge"

mkdir -p "${HOME}/luka/runs/pass1_val_tta"
cd "${PROJECT_DIR}"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export _PASS1_VAL_IN_CONTAINER=1

apptainer exec --nv \
  -B "${APPTAINER_BIND}" \
  --env PYTHONNOUSERSITE=1 \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env _PASS1_VAL_IN_CONTAINER=1 \
  --pwd /workspace/XPlainVerse-ACMChallenge \
  "${LJ_APPTAINER_IMAGE}" \
  bash scripts/run_pass1_val_tta_lj.sh
