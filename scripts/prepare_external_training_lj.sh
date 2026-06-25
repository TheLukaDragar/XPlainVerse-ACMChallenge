#!/usr/bin/env bash
# Prepare external datasets + manifest_all_v1 for Pass-1 combined training.
# CPU-only on elixir-lj-gpu-01 (/primoz + $HOME) — does not use GPUs.
#
# Steps:
#   1. Extract DFBench ZIPs -> /primoz/luka/external/DFBench/DFBench/
#   2. Extract OpenFake parquet 0-14 -> /primoz/luka/external/OpenFake_jpeg/
#   3. Build manifest_all_v1.parquet (XP pooled + OpenFake + DFBench)
#
# Usage:
#   ./scripts/prepare_external_training_lj.sh
#   SKIP_EXTRACT=1 ./scripts/prepare_external_training_lj.sh   # manifests only

set -euo pipefail

if [[ -d "${HOME}/luka/code/XPlainVerse-ACMChallenge/research" ]]; then
  CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
else
  CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

LOG_ROOT="${LOG_ROOT:-${HOME}/luka/runs/prepare_external}"
mkdir -p "${LOG_ROOT}"

_run() {
  local name="$1"
  shift
  echo "=== ${name} ==="
  LJ_CPU_TIME="${LJ_CPU_TIME:-48:00:00}" LJ_CPU_CPUS="${LJ_CPU_CPUS:-16}" LJ_CPU_MEM="${LJ_CPU_MEM:-64G}" \
    "${CODE_ROOT}/scripts/lj_cpu_primoz_exec.sh" "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
}

if [[ "${SKIP_EXTRACT:-}" != "1" ]]; then
  _run extract_dfbench python3 "${CODE_ROOT}/scripts/extract_dfbench_primoz.py"
  _run extract_openfake python3 "${CODE_ROOT}/scripts/extract_openfake_jpeg_primoz.py" --max-group 14
fi

_run build_manifests python3 "${CODE_ROOT}/research/experiments/02_pass1_classifier/build_manifest_external.py" \
  --only all_v1 --require-exists

MANIFEST="${CODE_ROOT}/research/experiments/02_pass1_classifier/manifests/external/manifest_all_v1.parquet"
echo
echo "=== Ready ==="
echo "manifest: ${MANIFEST}"
python3 - <<PY
import pandas as pd
df = pd.read_parquet("${MANIFEST}")
print(f"rows={len(df)}")
print(df.label.value_counts())
print(df.source.value_counts())
PY
