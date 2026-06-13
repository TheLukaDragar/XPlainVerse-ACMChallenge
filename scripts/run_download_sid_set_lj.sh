#!/usr/bin/env bash
# Dispatch SID_Set download to elixir-lj-gpu-01 (/primoz NVMe).
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
LOG_ROOT="${LOG_ROOT:-${HOME}/luka/runs/download_sid_set}"
mkdir -p "${LOG_ROOT}"

echo "Starting SID_Set download (~140 GB) on /primoz..."
LJ_CPU_TIME="${LJ_CPU_TIME:-24:00:00}" \
LJ_CPU_CPUS="${LJ_CPU_CPUS:-8}" \
LJ_CPU_MEM="${LJ_CPU_MEM:-32G}" \
  "${CODE_ROOT}/scripts/lj_cpu_primoz_exec.sh" \
  bash "${CODE_ROOT}/scripts/download_sid_set_primoz.sh" \
  2>&1 | tee "${LOG_ROOT}/download.log"

echo "Log: ${LOG_ROOT}/download.log"
echo "Data: /primoz/luka/external/SID_Set/"
echo "Primoz log: /primoz/luka/external/_logs/SID_Set.log"
