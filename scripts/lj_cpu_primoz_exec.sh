#!/usr/bin/env bash
# CPU-only command on elixir-lj-gpu-01 (for /primoz NVMe) — no GPU allocation.
#
# Use this for manifest builds, unzip, file walks, etc. while GPUs train.
#
# Usage:
#   ./scripts/lj_cpu_primoz_exec.sh python3 research/experiments/02_pass1_classifier/build_manifest_external.py --only dfbench

set -euo pipefail

PARTITION="${LJ_PARTITION:-elixir-interno}"
GPU_NODE="${LJ_GPU_NODE:-elixir-lj-gpu-01.elixir.ul.si}"
PROJECT_DIR="${LJ_PROJECT_DIR:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
CPU_MEM="${LJ_CPU_MEM:-32G}"
CPU_CPUS="${LJ_CPU_CPUS:-8}"
CPU_TIME="${LJ_CPU_TIME:-02:00:00}"

export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:-/primoz:/primoz,${HOME}:${HOME}}"
export SINGULARITY_BINDPATH="${SINGULARITY_BINDPATH:-${APPTAINER_BINDPATH}}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command...>" >&2
  exit 1
fi

if hostname 2>/dev/null | grep -q 'elixir-lj-gpu'; then
  cd "${PROJECT_DIR}"
  exec "$@"
fi

if ! command -v srun >/dev/null 2>&1; then
  echo "error: srun unavailable" >&2
  exit 1
fi

REMOTE_CMD=""
for arg in "$@"; do
  REMOTE_CMD+="$(printf '%q ' "${arg}")"
done

echo "CPU dispatch -> ${GPU_NODE} (${CPU_CPUS} cpus, ${CPU_MEM}, ${CPU_TIME})" >&2

exec srun -p "${PARTITION}" -w "${GPU_NODE}" -n1 \
  --cpus-per-task="${CPU_CPUS}" --mem="${CPU_MEM}" --time="${CPU_TIME}" \
  bash -lc "export APPTAINER_BINDPATH=$(printf '%q' "${APPTAINER_BINDPATH}") SINGULARITY_BINDPATH=$(printf '%q' "${SINGULARITY_BINDPATH}") && cd $(printf '%q' "${PROJECT_DIR}") && ${REMOTE_CMD}"
