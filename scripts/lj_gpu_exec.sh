#!/usr/bin/env bash
# Run a command on elixir-lj-gpu-01 inside the XPlainVerse Apptainer container.
#
# - Already on elixir-lj-gpu-* → delegates to ~/xplainverse_exec.sh
# - On slurm login / other host → srun to GPU node, then Apptainer
#
# Usage:
#   ./scripts/lj_gpu_exec.sh python3 -c 'import torch; print(torch.cuda.device_count())'
#   ./scripts/lj_gpu_exec.sh bash scripts/train_vlm_full_lj.sh
#
# Override Slurm allocation (quick checks):
#   LJ_GPU_TIME=00:30:00 LJ_GPU_COUNT=1 ./scripts/lj_gpu_exec.sh nvidia-smi

set -euo pipefail

PARTITION="${LJ_PARTITION:-elixir-interno}"
GPU_NODE="${LJ_GPU_NODE:-elixir-lj-gpu-01.elixir.ul.si}"
PROJECT_DIR="${LJ_PROJECT_DIR:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
GPU_GRES="${LJ_GPU_GRES:-gpu:4}"
GPU_MEM="${LJ_GPU_MEM:-64G}"
GPU_CPUS="${LJ_GPU_CPUS:-16}"
GPU_TIME="${LJ_GPU_TIME:-12:00:00}"

# Local NVMe dataset on elixir-lj-gpu-01 (not visible on login node).
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:-/primoz:/primoz}"
export SINGULARITY_BINDPATH="${SINGULARITY_BINDPATH:-${APPTAINER_BINDPATH}}"

_lj_xplainverse_exec() {
  ~/xplainverse_exec.sh "$@"
}

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command...>" >&2
  echo "  wraps ~/xplainverse_exec.sh on ${GPU_NODE}" >&2
  exit 1
fi

if hostname 2>/dev/null | grep -q 'elixir-lj-gpu'; then
  exec _lj_xplainverse_exec "$@"
fi

if ! command -v srun >/dev/null 2>&1; then
  echo "error: not on GPU node and srun unavailable. SSH to ${GPU_NODE} or use cursor_gpu_lj.sh." >&2
  exit 1
fi

# Build a safely quoted remote command.
REMOTE_CMD=""
for arg in "$@"; do
  REMOTE_CMD+="$(printf '%q ' "${arg}")"
done

echo "Dispatching to ${GPU_NODE} (partition ${PARTITION}, ${GPU_GRES}, ${GPU_TIME})..." >&2

exec srun -p "${PARTITION}" -w "${GPU_NODE}" -n1 \
  --gres="${GPU_GRES}" --mem="${GPU_MEM}" --cpus-per-task="${GPU_CPUS}" --time="${GPU_TIME}" \
  bash -lc "export APPTAINER_BINDPATH=$(printf '%q' "${APPTAINER_BINDPATH}") SINGULARITY_BINDPATH=$(printf '%q' "${SINGULARITY_BINDPATH}") && cd $(printf '%q' "${PROJECT_DIR}") && ~/xplainverse_exec.sh ${REMOTE_CMD}"
