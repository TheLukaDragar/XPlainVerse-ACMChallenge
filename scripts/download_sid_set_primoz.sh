#!/usr/bin/env bash
# Download SID_Set (Open Images v7 reals + synthetic + tampered) to /primoz.
# Run on elixir-lj-gpu-01 or via ./scripts/run_download_sid_set_lj.sh from login node.
#
# HF: saberzl/SID_Set (~140 GB, train+validation only; test set not on HF)
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
BASE="/primoz/luka/external/SID_Set"
OLD="/home/jakob/luka/data/external/SID_Set"
LOG_DIR="/primoz/luka/external/_logs"
mkdir -p "${LOG_DIR}" "${BASE}"

export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=0

log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG_DIR}/SID_Set.log"; }

if [[ -d "${OLD}" ]] && [[ "$(ls -A "${OLD}" 2>/dev/null | wc -l)" -gt 0 ]]; then
  log "rsync partial ${OLD} -> ${BASE}"
  rsync -a --info=progress2 "${OLD}/" "${BASE}/" >> "${LOG_DIR}/rsync-SID_Set.log" 2>&1 || true
fi

log "SID_Set reliable download -> ${BASE}"
df -h /primoz | tee -a "${LOG_DIR}/SID_Set.log"

python3 "${CODE_ROOT}/scripts/download_sid_set_recover.py" 2>&1 | tee -a "${LOG_DIR}/SID_Set.log"

log "SID_Set done"
du -sh "${BASE}" | tee -a "${LOG_DIR}/SID_Set.log"
ls "${BASE}/data" 2>/dev/null | wc -l | xargs -I{} log "parquet shards: {}"
