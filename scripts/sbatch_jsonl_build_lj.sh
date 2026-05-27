#!/usr/bin/env bash
# Submit a Slurm job on elixir-lj-gpu-01 to build full ms-swift JSONL inside Apptainer.
#
# Usage (from repo root on Slurm login):
#   ./scripts/sbatch_jsonl_build_lj.sh
#   DATA_ROOT=/path/to/XPlainVerse ./scripts/sbatch_jsonl_build_lj.sh
#
# Logs: logs/hf_fetch/jsonl_build_full_<jobid>.{out,err}
# Outputs: dataset/*.jsonl (gitignored; large)

set -euo pipefail

REPO="${REPO:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
DATA_ROOT="${DATA_ROOT:-${HOME}/luka/data/XPlainVerse}"
PARTITION="${LJ_PARTITION:-elixir-interno}"
GPU_NODE="${LJ_GPU_NODE:-elixir-lj-gpu-01.elixir.ul.si}"
GPU_TIME="${LJ_GPU_TIME:-12:00:00}"
GPU_MEM="${LJ_GPU_MEM:-128G}"
GPU_CPUS="${LJ_GPU_CPUS:-16}"
GPU_GRES="${LJ_GPU_GRES:-gpu:4}"
XEXEC="${HOME}/xplainverse_exec.sh"

cd "${REPO}"
mkdir -p logs/hf_fetch

JOBFILE="$(mktemp "${TMPDIR:-/tmp}/xpv_jsonl_full_XXXXXX.sbatch")"
trap 'rm -f "${JOBFILE}"' EXIT

{
  echo "#!/bin/bash"
  echo "#SBATCH -p ${PARTITION}"
  echo "#SBATCH -w ${GPU_NODE}"
  echo "#SBATCH --gres=${GPU_GRES}"
  echo "#SBATCH --mem=${GPU_MEM}"
  echo "#SBATCH --cpus-per-task=${GPU_CPUS}"
  echo "#SBATCH --time=${GPU_TIME}"
  echo "#SBATCH -J xpv-jsonl-full"
  echo "#SBATCH -D ${REPO}"
  echo "#SBATCH -o logs/hf_fetch/jsonl_build_full_%j.out"
  echo "#SBATCH -e logs/hf_fetch/jsonl_build_full_%j.err"
  echo "set -euo pipefail"
  echo "exec \"${XEXEC}\" python3 dataset/build_swift_jsonl.py \\"
  echo "  --data-root \"${DATA_ROOT}\" --output-dir dataset"
} > "${JOBFILE}"

chmod +x "${JOBFILE}"
sbatch "${JOBFILE}"
