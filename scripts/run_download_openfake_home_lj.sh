#!/usr/bin/env bash
# Resume OpenFake download on home disk (~3.4 TB). Reliable per-file fallback.
set -euo pipefail

OF_DIR="/home/jakob/luka/data/external/OpenFake"
LOG="${OF_DIR}/download_recover.log"
mkdir -p "${OF_DIR}"

export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "OpenFake download -> ${OF_DIR}"
echo "log: ${LOG}"
cd "${OF_DIR}"
exec python3 download_openfake_recover.py 2>&1 | tee -a "${LOG}"
