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

# Resolve torchrun MASTER_ADDR on Frida (scontrol is often absent inside Pyxis).
# Args: NNODES (optional, default 1).
frida_resolve_master_addr() {
  local nnodes="${1:-${NNODES:-1}}"

  if [[ -n "${MASTER_ADDR:-}" ]]; then
    echo "${MASTER_ADDR}"
    return
  fi

  # Single-node multi-GPU: loopback is reliable inside Pyxis containers.
  # Short hostnames like "aga" often fail NCCL bind with:
  #   "Call to bind failed: Cannot assign requested address"
  if [[ "${nnodes}" -eq 1 ]]; then
    echo "127.0.0.1"
    return
  fi

  if [[ -n "${SLURM_STEP_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol show hostnames "${SLURM_STEP_NODELIST}" | head -1
    return
  fi
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1
    return
  fi
  hostname -s 2>/dev/null || hostname
}

# NCCL defaults for single-node PCIe GPU boxes (aga/ana/axa).
frida_export_nccl_env() {
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
}

# Load W&B key for Pyxis containers that run as root with HOME=/root.
frida_load_wandb_key() {
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    return 0
  fi

  local netrc=""
  for netrc in \
    "${FRIDA_NETRC:-}" \
    "${HOME}/.netrc" \
    "/shared/home/${USER:-}/.netrc"; do
    [[ -n "${netrc}" && -f "${netrc}" ]] || continue
    WANDB_API_KEY="$(python3 - "${netrc}" <<'PY'
import netrc, sys
path = sys.argv[1]
try:
    print(netrc.netrc(path).authenticators("api.wandb.ai")[2])
except (FileNotFoundError, TypeError):
    pass
PY
)" || true
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      export WANDB_API_KEY
      export NETRC="${netrc}"
      return 0
    fi
  done
  return 1
}

# Single-process HF cache warm before torchrun (avoids DDP barrier during snapshot_download).
frida_warm_hf_model() {
  local model_id="${1:-Qwen/Qwen3-VL-8B-Instruct}"
  echo "prefetch: ensuring ${model_id} is in HF cache (${HF_HOME:-default})..." >&2
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("${model_id}")
print("prefetch: ok")
PY
}
