#!/usr/bin/env bash
# Hold a B300 allocation on partition frida (batch-only — no interactive srun on frida).
# After submit:  JOBID=$(sbatch --parsable scripts/sbatch_frida_b300_hold.sh)
# Attach shell:   srun --overlap --jobid="${JOBID}" --pty bash
# Done:            scancel "${JOBID}"
#
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:B300:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=b300-hold-test
#SBATCH --output=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_b300_hold_%j.out
#SBATCH --error=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_b300_hold_%j.err

set -euo pipefail
echo "holder start host=$(hostname) job=${SLURM_JOB_ID:-} $(date -Is)"
nvidia-smi -L || true
sleep infinity
