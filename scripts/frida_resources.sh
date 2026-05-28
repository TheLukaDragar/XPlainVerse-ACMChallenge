#!/usr/bin/env bash
# Frida Slurm CPU / worker defaults for ms-swift distributed training.
# Source this file from a training script; do not execute it directly.

frida_detect_cpus() {
  if [[ -n "${FRIDA_CPUS_OVERRIDE:-}" ]]; then
    echo "${FRIDA_CPUS_OVERRIDE}"
    return
  fi
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    echo "${SLURM_CPUS_PER_TASK}"
    return
  fi
  if [[ -n "${SLURM_JOB_CPUS_PER_NODE:-}" ]]; then
    echo "${SLURM_JOB_CPUS_PER_NODE%%(*}"
    return
  fi
  nproc 2>/dev/null || echo 32
}

frida_cuda_visible_devices() {
  local nproc_gpu="${1:-1}"
  local out=""
  local i
  for ((i = 0; i < nproc_gpu; i++)); do
    if [[ -n "${out}" ]]; then
      out+=","
    fi
    out+="${i}"
  done
  echo "${out}"
}

# Sets DATASET_NUM_PROC, DATALOADER_NUM_WORKERS, and BLAS/thread env.
# Args: NPROC_PER_NODE (GPU ranks sharing the Slurm CPU allocation).
frida_apply_cpu_defaults() {
  local nproc_gpu="${1:-1}"
  local cpus reserve per map_workers

  cpus="$(frida_detect_cpus)"
  FRIDA_CPUS_TOTAL="${cpus}"

  # HuggingFace map / metadata preprocessing. Cap to avoid hammering shared FS.
  map_workers=$((cpus - 16))
  if [[ "${map_workers}" -lt 4 ]]; then
    map_workers=4
  fi
  if [[ "${map_workers}" -gt 64 ]]; then
    map_workers=64
  fi
  export DATASET_NUM_PROC="${DATASET_NUM_PROC:-${map_workers}}"

  # Per-rank DataLoader workers. Keep total workers bounded on 8-GPU jobs.
  reserve=$((nproc_gpu + 8))
  per=$(((cpus - reserve) / nproc_gpu))
  if [[ "${per}" -lt 2 ]]; then
    per=2
  fi
  if [[ "${per}" -gt 12 ]]; then
    per=12
  fi
  export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-${per}}"

  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
}
