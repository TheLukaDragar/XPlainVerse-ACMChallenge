#!/usr/bin/env bash
# CPU job: download + unpack XPlainVerse test images (HF multipart tar).
#
# Submit from repo root:
#   sbatch scripts/sbatch_download_test_xplainverse.sbatch
#
#SBATCH --job-name=xpv-test-dl
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_test_download_%j.out
#SBATCH --error=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_test_download_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/code/XPlainVerse-ACMChallenge}}"
export XPLAINVERSE_DATA_ROOT="${XPLAINVERSE_DATA_ROOT:-/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/data/XPlainVerse}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "XPLAINVERSE_DATA_ROOT=${XPLAINVERSE_DATA_ROOT}"
echo "job=${SLURM_JOB_ID} node=${SLURM_NODELIST} started=$(date -Is)"

test -f "${REPO_ROOT}/scripts/download_extract_xplainverse_test.sh"

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL bash "${REPO_ROOT}/scripts/download_extract_xplainverse_test.sh"
else
  bash "${REPO_ROOT}/scripts/download_extract_xplainverse_test.sh"
fi

echo "finished exit=$? at $(date -Is)"
