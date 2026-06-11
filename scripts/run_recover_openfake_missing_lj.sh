#!/usr/bin/env bash
# Recover missing OpenFake JPEGs from parquet (PNG/WebP/JPEG -> JPEG via PIL).
# CPU-only on elixir-lj-gpu-01. After this finishes:
#   ./scripts/finish_external_manifest_lj.sh
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
LOG="${HOME}/luka/runs/prepare_external/recover_openfake_missing.log"
mkdir -p "$(dirname "$LOG")"
LJ_CPU_TIME="${LJ_CPU_TIME:-72:00:00}" LJ_CPU_CPUS="${LJ_CPU_CPUS:-16}" LJ_CPU_MEM="${LJ_CPU_MEM:-64G}" \
  "${CODE_ROOT}/scripts/lj_cpu_primoz_exec.sh" \
  python3 "${CODE_ROOT}/scripts/recover_openfake_missing_primoz.py" --max-group 14 \
  2>&1 | tee "$LOG"

echo "=== recovery done — rebuilding manifest ==="
LJ_CPU_TIME="${LJ_CPU_TIME:-02:00:00}" \
  "${CODE_ROOT}/scripts/finish_external_manifest_lj.sh" 2>&1 | tee -a "$LOG"

echo "=== ready to train ==="
echo "LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=120:00:00 ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_ensemble_external_all_lj.sh"
