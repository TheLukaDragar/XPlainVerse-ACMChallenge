#!/bin/bash
# Build stage-2 (v2) ms-swift JSONL on Ljubljana inside the lj training container.
#
# Pure-stdlib build, but reads ~560k manifest rows + explanation JSON files, so it
# runs on the GPU node compute allocation (not the login node). Uses the GHCR -lj
# image via Apptainer — the same aligned lj container as training.
#
# Submit from the Slurm login node:
#   sbatch scripts/sbatch_build_train_v2_lj.sbatch
#
# Override knobs:
#   sbatch --export=ALL,HYPOTHETICAL_RATIO=0.6,FAKE_REAL_RATIO=3 \
#          scripts/sbatch_build_train_v2_lj.sbatch
#
#SBATCH --job-name=xpv-build-v2-lj
#SBATCH --partition=elixir-interno
#SBATCH --nodelist=elixir-lj-gpu-01.elixir.ul.si
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/jakob/luka/runs/slurm_build_v2_lj_%j.out
#SBATCH --error=/home/jakob/luka/runs/slurm_build_v2_lj_%j.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/luka/runs}"
DEFAULT_REPO_LC="$(echo "${GITHUB_REPOSITORY:-TheLukaDragar/XPlainVerse-ACMChallenge}" | tr '[:upper:]' '[:lower:]')"
CONTAINER="${LJ_APPTAINER_IMAGE:-docker://ghcr.io/${DEFAULT_REPO_LC}-lj:latest}"
APPTAINER_BIND="${LJ_APPTAINER_BIND:-${HOME}:${HOME},/primoz:/primoz}"

mkdir -p "${RUNS_ROOT}"

_AUTH_ENV=""
if [[ -n "${APPTAINER_DOCKER_USERNAME:-}" ]]; then
  _AUTH_ENV+="export APPTAINER_DOCKER_USERNAME=$(printf '%q' "${APPTAINER_DOCKER_USERNAME}"); "
fi
if [[ -n "${APPTAINER_DOCKER_PASSWORD:-}" ]]; then
  _AUTH_ENV+="export APPTAINER_DOCKER_PASSWORD=$(printf '%q' "${APPTAINER_DOCKER_PASSWORD}"); "
fi

echo "=== sbatch_build_train_v2_lj ==="
echo "job:        ${SLURM_JOB_ID}"
echo "node:       ${SLURM_JOB_NODELIST:-?}"
echo "container:  ${CONTAINER}"
echo "code_root:  ${CODE_ROOT}"
echo "started:    $(date -Is)"
echo

INNER="${_AUTH_ENV}
set -euo pipefail
export PYTHONNOUSERSITE=1
export HYPOTHETICAL_RATIO=${HYPOTHETICAL_RATIO:-0.50}
export FAKE_REAL_RATIO=${FAKE_REAL_RATIO:-4.0}
export FILTER_MIN_SENTENCES=${FILTER_MIN_SENTENCES:-3}
export FILTER_MIN_WORDS=${FILTER_MIN_WORDS:-80}
export FILTER_MIN_CONNECTIVES=${FILTER_MIN_CONNECTIVES:-0}
cd $(printf '%q' "${CODE_ROOT}")
chmod +x scripts/build_train_v2_lj.sh
exec bash scripts/build_train_v2_lj.sh
"

srun --cpus-per-task=16 --mem=32G \
  bash -c "exec apptainer exec -B $(printf '%q' "${APPTAINER_BIND}") --env PYTHONNOUSERSITE=1 $(printf '%q' "${CONTAINER}") bash -c $(printf '%q' "${INNER}")"

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
