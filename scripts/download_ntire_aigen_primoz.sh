#!/usr/bin/env bash
# Download + extract NTIRE 2026 Robust AI-Gen Detection train set to /primoz.
# Run on elixir-lj-gpu-01 or via ./scripts/run_download_ntire_aigen_lj.sh from login node.
#
# HF: deepfakesMSU/NTIRE-RobustAIGenDetection-train
#   ~114 GB, 6 zip shards, ~277k images (0=real, 1=generated), in-the-wild degradations.
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
BASE="${NTIRE_AIGEN_ROOT:-/primoz/luka/external/NTIRE_AIGen}"
LOG_DIR="/primoz/luka/external/_logs"
mkdir -p "${LOG_DIR}" "${BASE}"

export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=0
export NTIRE_AIGEN_ROOT="${BASE}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG_DIR}/NTIRE_AIGen.log"; }

log "NTIRE AIGen download -> ${BASE}"
df -h /primoz | tee -a "${LOG_DIR}/NTIRE_AIGen.log"

python3 "${CODE_ROOT}/scripts/download_ntire_aigen_recover.py" 2>&1 | tee -a "${LOG_DIR}/NTIRE_AIGen.log"

# Extract shards (skip if already extracted: shard_i/labels.csv present).
if [[ "${SKIP_EXTRACT:-0}" != "1" ]]; then
  for zip in "${BASE}"/shard_*.zip; do
    [[ -f "${zip}" ]] || continue
    name="$(basename "${zip}" .zip)"          # shard_0
    idx="${name#shard_}"                        # 0
    dest="${BASE}/shard_${idx}"
    if [[ -f "${dest}/labels.csv" ]]; then
      log "extract skip ${name} (labels.csv exists)"
      continue
    fi
    log "unzip ${zip} -> ${BASE}"
    unzip -q -o "${zip}" -d "${BASE}" 2>&1 | tail -3 | tee -a "${LOG_DIR}/NTIRE_AIGen.log" || {
      log "WARN unzip failed for ${zip}"
      continue
    }
    # Some archives may unpack as images/+labels.csv at root or under shard_i/.
    if [[ ! -f "${dest}/labels.csv" && -f "${BASE}/labels.csv" ]]; then
      mkdir -p "${dest}"
      mv "${BASE}/labels.csv" "${dest}/labels.csv" 2>/dev/null || true
      [[ -d "${BASE}/images" ]] && mv "${BASE}/images" "${dest}/images" 2>/dev/null || true
    fi
  done
fi

log "NTIRE AIGen done"
du -sh "${BASE}" | tee -a "${LOG_DIR}/NTIRE_AIGen.log"
for d in "${BASE}"/shard_*/; do
  [[ -d "${d}" ]] || continue
  n=$(find "${d}/images" -maxdepth 1 -type f 2>/dev/null | wc -l)
  log "  $(basename "${d}"): ${n} images"
done
