#!/usr/bin/env bash
# CPU node: unpack XPlainVerse *.tar.part-* (see extract_xplainverse_explanations.sh).
# Submit from login:  sbatch scripts/sbatch_extract_xplainverse.sbatch
# FRIDA: ~/.cursor/skills/frida-slurm-orchestration/SKILL.md — edit #SBATCH --account if required.

#SBATCH --job-name=xplainverse-unpack
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
##SBATCH --account=YOUR_ACCOUNT   # uncomment/set if Slurm requires it
#SBATCH --output=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_unpack_%j.out
#SBATCH --error=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_unpack_%j.err

set -euo pipefail

# Slurm copies this script to spool — do not use BASH_SOURCE for repo path.
# Submit from the repo root:  cd .../XPlainVerse-ACMChallenge && sbatch scripts/sbatch_extract_xplainverse.sbatch
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/code/XPlainVerse-ACMChallenge}}"
export XPLAINVERSE_DATA_ROOT="${XPLAINVERSE_DATA_ROOT:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/data/XPlainVerse}"
export UNPACK_TRAIN_IMAGES="${UNPACK_TRAIN_IMAGES:-1}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "extract_script=${REPO_ROOT}/scripts/extract_xplainverse_explanations.sh"
test -f "${REPO_ROOT}/scripts/extract_xplainverse_explanations.sh" || {
  echo "error: missing ${REPO_ROOT}/scripts/extract_xplainverse_explanations.sh — cd repo root then sbatch, or set REPO_ROOT" >&2
  exit 1
}
echo "XPLAINVERSE_DATA_ROOT=${XPLAINVERSE_DATA_ROOT} UNPACK_TRAIN_IMAGES=${UNPACK_TRAIN_IMAGES}"

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL bash "${REPO_ROOT}/scripts/extract_xplainverse_explanations.sh"
else
  bash "${REPO_ROOT}/scripts/extract_xplainverse_explanations.sh"
fi
echo "unpack finished exit=$?"
