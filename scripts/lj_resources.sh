#!/usr/bin/env bash
# Lj node CPU / thread defaults for ms-swift training and dataset builds.
# Source from train_vlm_full_lj.sh (do not execute directly).
#
# Uses Slurm allocation when present (srun/sbatch), else nproc on elixir-lj-gpu-01.

lj_detect_cpus() {
  if [[ -n "${LJ_CPUS_OVERRIDE:-}" ]]; then
    echo "${LJ_CPUS_OVERRIDE}"
    return
  fi
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    echo "${SLURM_CPUS_PER_TASK}"
    return
  fi
  if [[ -n "${SLURM_JOB_CPUS_PER_NODE:-}" ]]; then
    echo "${SLURM_JOB_CPUS_PER_NODE}"
    return
  fi
  nproc 2>/dev/null || echo 32
}

# Sets DATASET_NUM_PROC, DATALOADER_NUM_WORKERS, and BLAS/thread env.
# Args: NPROC_PER_NODE (GPU processes sharing the CPU allocation).
lj_apply_cpu_defaults() {
  local nproc_gpu="${1:-1}"
  local cpus total reserve per map_workers

  cpus="$(lj_detect_cpus)"
  LJ_CPUS_TOTAL="${cpus}"

  # HuggingFace map / packing preprocess — use most of the Slurm CPU slice.
  map_workers=$(( cpus - 8 ))
  if [[ "${map_workers}" -lt 4 ]]; then
    map_workers=4
  fi
  if [[ "${map_workers}" -gt 64 ]]; then
    map_workers=64
  fi
  export DATASET_NUM_PROC="${DATASET_NUM_PROC:-${map_workers}}"

  # Per-rank DataLoader workers (total ≈ nproc_gpu × this).
  reserve=$(( nproc_gpu + 4 ))
  per=$(( (cpus - reserve) / nproc_gpu ))
  if [[ "${per}" -lt 2 ]]; then
    per=2
  fi
  if [[ "${per}" -gt 16 ]]; then
    per=16
  fi
  export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-${per}}"

  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
}
