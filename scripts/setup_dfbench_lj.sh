#!/usr/bin/env bash
# Extract DFBench images on /primoz and build manifest_dfbench_train.parquet.
#
# Usage:
#   ./scripts/setup_dfbench_lj.sh              # full extract (skip if already done)
#   ./scripts/setup_dfbench_lj.sh --repair     # re-extract LIVE + ali_flux_schnell only
#   MANIFEST_ONLY=1 ./scripts/setup_dfbench_lj.sh
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
LOG_ROOT="${LOG_ROOT:-${HOME}/luka/runs/setup_dfbench}"
mkdir -p "${LOG_ROOT}"

REPAIR=0
if [[ "${1:-}" == "--repair" ]]; then
  REPAIR=1
fi

_run() {
  local name="$1"
  shift
  echo "=== ${name} ==="
  LJ_CPU_TIME="${LJ_CPU_TIME:-04:00:00}" LJ_CPU_CPUS="${LJ_CPU_CPUS:-8}" LJ_CPU_MEM="${LJ_CPU_MEM:-32G}" \
    "${CODE_ROOT}/scripts/lj_cpu_primoz_exec.sh" "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
}

if [[ "${MANIFEST_ONLY:-}" != "1" ]]; then
  if [[ "${REPAIR}" == "1" ]]; then
    _run extract_repair python3 "${CODE_ROOT}/scripts/extract_dfbench_primoz.py" \
      --only LIVE ali_flux_schnell
  else
    _run extract_all python3 "${CODE_ROOT}/scripts/extract_dfbench_primoz.py"
  fi
fi

_run build_manifest python3 "${CODE_ROOT}/research/experiments/02_pass1_classifier/build_manifest_external.py" \
  --only dfbench --require-exists

MANIFEST="${CODE_ROOT}/research/experiments/02_pass1_classifier/manifests/external/manifest_dfbench_train.parquet"
echo
echo "=== DFBench ready ==="
echo "images: /primoz/luka/external/DFBench/DFBench/"
echo "manifest: ${MANIFEST}"
python3 - <<PY
import pandas as pd
df = pd.read_parquet("${MANIFEST}")
print(f"rows={len(df)}")
print("labels:", df.label.value_counts().to_dict())
print("generators:", df.generator.value_counts().to_dict())
PY
