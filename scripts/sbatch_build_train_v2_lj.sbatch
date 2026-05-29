#!/bin/bash
# Build stage-2 (v2) ms-swift JSONL on Ljubljana inside the SIF.
#
# Pure-stdlib build, but reads ~560k manifest rows + explanation JSON files,
# so it runs on the GPU node compute allocation (not the login node). Uses
# ~/xplainverse_exec.sh so /workspace + /primoz are bound consistently.
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
mkdir -p "${RUNS_ROOT}"

# Bind the GPU-node NVMe dataset into the SIF (xplainverse_exec.sh does not).
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:-/primoz:/primoz}"
export SINGULARITY_BINDPATH="${SINGULARITY_BINDPATH:-${APPTAINER_BINDPATH}}"

export HYPOTHETICAL_RATIO="${HYPOTHETICAL_RATIO:-0.50}"
export FAKE_REAL_RATIO="${FAKE_REAL_RATIO:-4.0}"
export FILTER_MIN_SENTENCES="${FILTER_MIN_SENTENCES:-3}"
export FILTER_MIN_WORDS="${FILTER_MIN_WORDS:-80}"
export FILTER_MIN_CONNECTIVES="${FILTER_MIN_CONNECTIVES:-0}"

echo "=== sbatch_build_train_v2_lj ==="
echo "job:        ${SLURM_JOB_ID}"
echo "node:       ${SLURM_JOB_NODELIST:-?}"
echo "code_root:  ${CODE_ROOT}"
echo "started:    $(date -Is)"
echo

cd "${CODE_ROOT}"
srun --cpus-per-task=16 --mem=32G \
  ~/xplainverse_exec.sh env \
    HYPOTHETICAL_RATIO="${HYPOTHETICAL_RATIO}" \
    FAKE_REAL_RATIO="${FAKE_REAL_RATIO}" \
    FILTER_MIN_SENTENCES="${FILTER_MIN_SENTENCES}" \
    FILTER_MIN_WORDS="${FILTER_MIN_WORDS}" \
    FILTER_MIN_CONNECTIVES="${FILTER_MIN_CONNECTIVES}" \
    bash scripts/build_train_v2_lj.sh

echo
echo "=== job ${SLURM_JOB_ID} finished at $(date -Is) ==="
