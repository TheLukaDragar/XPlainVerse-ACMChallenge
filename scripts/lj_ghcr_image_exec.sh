#!/usr/bin/env bash
# Run a command on elixir-lj-gpu-01 inside the CI-built GHCR training image (Dockerfile.lj).
# Uses Apptainer `docker://` (OCI) instead of ~/xplainverse_exec.sh + local .sif.
#
# Default image: ghcr.io/<lowercase repo>-lj:latest (see .github/workflows/container-lj.yml).
# This is a separate GHCR package from the CUDA 13 / vLLM eval image (same repo name without -lj).
#
# Usage (Slurm login, repo root):
#   chmod +x scripts/lj_ghcr_image_exec.sh
#   LJ_GPU_TIME=00:45:00 ./scripts/lj_ghcr_image_exec.sh python3 -c 'import torch, flash_attn; print(torch.__version__, flash_attn.__version__)'
#
# Private GHCR (required unless the package is public):
#   export APPTAINER_DOCKER_USERNAME=TheLukaDragar
#   export APPTAINER_DOCKER_PASSWORD=<classic PAT with read:packages>
#
# Override tag from a green Actions run, e.g. commit 6224dd3 on the shared package (pre -lj split):
#   LJ_APPTAINER_IMAGE=docker://ghcr.io/thelukadragar/xplainverse-acmchallenge:sha-6224dd3-slurm

set -euo pipefail

PARTITION="${LJ_PARTITION:-elixir-interno}"
GPU_NODE="${LJ_GPU_NODE:-elixir-lj-gpu-01.elixir.ul.si}"
PROJECT_DIR="${LJ_PROJECT_DIR:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
GPU_GRES="${LJ_GPU_GRES:-gpu:4}"
GPU_MEM="${LJ_GPU_MEM:-64G}"
GPU_CPUS="${LJ_GPU_CPUS:-16}"
GPU_TIME="${LJ_GPU_TIME:-12:00:00}"

DEFAULT_REPO_LC="$(echo "${GITHUB_REPOSITORY:-TheLukaDragar/XPlainVerse-ACMChallenge}" | tr '[:upper:]' '[:lower:]')"
LJ_APPTAINER_IMAGE="${LJ_APPTAINER_IMAGE:-docker://ghcr.io/${DEFAULT_REPO_LC}-lj:latest}"
APPTAINER_BIND="${LJ_APPTAINER_BIND:-${HOME}:${HOME},/primoz:/primoz}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command...>" >&2
  echo "  default LJ_APPTAINER_IMAGE=${LJ_APPTAINER_IMAGE}" >&2
  exit 1
fi

run_inner() {
  cd "${PROJECT_DIR}"
  export PYTHONNOUSERSITE=1
  exec apptainer exec --nv \
    -B "${APPTAINER_BIND}" \
    --env PYTHONNOUSERSITE=1 \
    "${LJ_APPTAINER_IMAGE}" \
    "$@"
}

if hostname 2>/dev/null | grep -q 'elixir-lj-gpu'; then
  run_inner "$@"
fi

if ! command -v srun >/dev/null 2>&1; then
  echo "error: srun unavailable." >&2
  exit 1
fi

echo "Dispatching to ${GPU_NODE} (image=${LJ_APPTAINER_IMAGE})..." >&2
if [[ -z "${APPTAINER_DOCKER_USERNAME:-}" || -z "${APPTAINER_DOCKER_PASSWORD:-}" ]]; then
  echo "note: APPTAINER_DOCKER_USERNAME/PASSWORD unset — pull fails if GHCR package is private (401 → manifest unknown)." >&2
fi

# Forward GHCR creds to the GPU node (srun does not inherit login env by default).
_AUTH_EXPORT=""
if [[ -n "${APPTAINER_DOCKER_USERNAME:-}" ]]; then
  _AUTH_EXPORT+="export APPTAINER_DOCKER_USERNAME=$(printf '%q' "${APPTAINER_DOCKER_USERNAME}"); "
fi
if [[ -n "${APPTAINER_DOCKER_PASSWORD:-}" ]]; then
  _AUTH_EXPORT+="export APPTAINER_DOCKER_PASSWORD=$(printf '%q' "${APPTAINER_DOCKER_PASSWORD}"); "
fi

INNER="${_AUTH_EXPORT}export PYTHONNOUSERSITE=1; cd $(printf '%q' "${PROJECT_DIR}") && exec apptainer exec --nv --env PYTHONNOUSERSITE=1 -B $(printf '%q' "${APPTAINER_BIND}") $(printf '%q' "${LJ_APPTAINER_IMAGE}")"
for _a in "$@"; do
  INNER+=" $(printf '%q' "${_a}")"
done

exec srun -p "${PARTITION}" -w "${GPU_NODE}" -n1 \
  --gres="${GPU_GRES}" --mem="${GPU_MEM}" --cpus-per-task="${GPU_CPUS}" --time="${GPU_TIME}" \
  bash -c "${INNER}"
