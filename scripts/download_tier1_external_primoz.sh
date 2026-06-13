#!/usr/bin/env bash
# Download Tier-1 external deepfake datasets to /primoz/luka/external/ (GPU node NVMe).
# Run ON elixir-lj-gpu-01 (not login node — /primoz is not mounted on slurm).
#
# Usage:
#   bash scripts/download_tier1_external_primoz.sh          # all four
#   bash scripts/download_tier1_external_primoz.sh DFBench GenImage
#
set -euo pipefail

BASE="/primoz/luka/external"
OLD="/home/jakob/luka/data/external"
HF="/home/jakob/miniconda3/bin/hf"
LOG_DIR="${BASE}/_logs"
mkdir -p "${LOG_DIR}" "${BASE}"/{DFBench,OpenFake,GenImage,DRCT-2M,SID_Set}

log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG_DIR}/master.log"; }

migrate_partial() {
  local name="$1"
  if [[ -d "${OLD}/${name}" ]] && [[ "$(ls -A "${OLD}/${name}" 2>/dev/null | wc -l)" -gt 0 ]]; then
    log "rsync partial ${OLD}/${name} -> ${BASE}/${name}"
    rsync -a --info=progress2 "${OLD}/${name}/" "${BASE}/${name}/" \
      >> "${LOG_DIR}/rsync-${name}.log" 2>&1 || true
  fi
}

download_dfbench() {
  log "DFBench -> ${BASE}/DFBench"
  migrate_partial DFBench
  "${HF}" download IntMeGroup/DFBench --repo-type dataset \
    --local-dir "${BASE}/DFBench" \
    >> "${LOG_DIR}/DFBench.log" 2>&1
  log "DFBench done"
}

download_genimage() {
  log "GenImage -> ${BASE}/GenImage"
  migrate_partial GenImage
  "${HF}" download ENSTA-U2IS/GenImage --repo-type dataset \
    --local-dir "${BASE}/GenImage" \
    >> "${LOG_DIR}/GenImage.log" 2>&1
  log "GenImage done"
}

download_openfake() {
  log "OpenFake -> ${BASE}/OpenFake (~3.4TB — check df -h /primoz first)"
  migrate_partial OpenFake
  # snapshot_download handles resume; 8 workers
  python3 - <<'PY' >> "${LOG_DIR}/OpenFake.log" 2>&1
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ComplexDataLab/OpenFake",
    repo_type="dataset",
    local_dir="/primoz/luka/external/OpenFake",
    max_workers=8,
    resume_download=True,
)
print("OpenFake snapshot_download finished")
PY
  log "OpenFake done"
}

download_sid_set() {
  log "SID_Set -> ${BASE}/SID_Set (~140 GB)"
  migrate_partial SID_Set
  bash "$(dirname "${BASH_SOURCE[0]}")/download_sid_set_primoz.sh"
  log "SID_Set done"
}

download_drct2m() {
  log "DRCT-2M -> ${BASE}/DRCT-2M"
  migrate_partial DRCT-2M
  python3 -m pip install --user -q modelscope 2>>"${LOG_DIR}/DRCT-2M.log" || true
  python3 - <<'PY' >> "${LOG_DIR}/DRCT-2M.log" 2>&1
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download("BokingChen/DRCT-2M", local_dir="/primoz/luka/external/DRCT-2M")
print("DRCT-2M download finished")
PY
  log "DRCT-2M done"
}

df -h /primoz | tee -a "${LOG_DIR}/master.log"

TARGETS=("${@:-DFBench OpenFake GenImage DRCT-2M}")
for t in "${TARGETS[@]}"; do
  case "$t" in
    DFBench)    download_dfbench ;;
    OpenFake)   download_openfake ;;
    GenImage)   download_genimage ;;
    SID_Set)    download_sid_set ;;
    DRCT-2M)    download_drct2m ;;
    *) log "unknown target: $t" ;;
  esac
done

log "all requested downloads finished"
du -sh "${BASE}"/* | tee -a "${LOG_DIR}/master.log"
