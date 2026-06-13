#!/usr/bin/env bash
# Audit SID_Set, extract JPEGs on /primoz, build manifest_sid_set_trainval.parquet.
#
# Usage:
#   ./scripts/setup_sid_set_lj.sh
#   MANIFEST_ONLY=1 ./scripts/setup_sid_set_lj.sh   # after extract done
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
LOG_ROOT="${LOG_ROOT:-${HOME}/luka/runs/setup_sid_set}"
mkdir -p "${LOG_ROOT}"

_run() {
  local name="$1"
  shift
  echo "=== ${name} ==="
  LJ_CPU_TIME="${LJ_CPU_TIME:-24:00:00}" LJ_CPU_CPUS="${LJ_CPU_CPUS:-16}" LJ_CPU_MEM="${LJ_CPU_MEM:-64G}" \
    "${CODE_ROOT}/scripts/lj_cpu_primoz_exec.sh" "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
}

if [[ "${SKIP_AUDIT:-}" != "1" ]]; then
  _run audit_sid_set python3 "${CODE_ROOT}/scripts/audit_sid_set_primoz.py"
fi

if [[ "${MANIFEST_ONLY:-}" != "1" ]]; then
  _run extract_sid_set python3 "${CODE_ROOT}/scripts/extract_sid_set_jpeg_primoz.py" \
    --splits train validation --workers "${SID_WORKERS:-8}"
fi

_run build_manifest python3 "${CODE_ROOT}/research/experiments/02_pass1_classifier/build_manifest_external.py" \
  --only sid_set --require-exists

MANIFEST="${CODE_ROOT}/research/experiments/02_pass1_classifier/manifests/external/manifest_sid_set_trainval.parquet"
echo
echo "=== SID_Set ready ==="
echo "images: /primoz/luka/external/SID_Set_jpeg/"
echo "manifest: ${MANIFEST}"
python3 - <<PY
import pandas as pd
df = pd.read_parquet("${MANIFEST}")
print(f"rows={len(df)}")
print("labels:", df.label.value_counts().to_dict())
print("generators:", df.generator.value_counts().to_dict())
print("splits:", df.split.value_counts().to_dict() if "split" in df.columns else "n/a")
PY
