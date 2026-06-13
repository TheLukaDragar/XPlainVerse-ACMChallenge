#!/usr/bin/env bash
# Periodic download status (safe — read-only, does not touch other tmux sessions).
# Usage:
#   ./scripts/monitor_downloads_lj.sh          # print once
#   INTERVAL=300 ./scripts/monitor_downloads_lj.sh --loop
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
INTERVAL="${INTERVAL:-300}"
LOG="${LOG:-${HOME}/luka/runs/download_monitor.log}"
mkdir -p "$(dirname "${LOG}")"

status_once() {
  {
    echo "=== $(date -Iseconds) ==="

    if [[ -d /home/jakob/luka/data/external/OpenFake/core ]]; then
      OF_BYTES=$(find /home/jakob/luka/data/external/OpenFake/core -name '*.parquet' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
      OF_SHARDS=$(find /home/jakob/luka/data/external/OpenFake/core -name '*.parquet' 2>/dev/null | wc -l)
      OF_GB=$(python3 -c "print(f'{$OF_BYTES/1e9:.1f}')")
      echo "OpenFake: ${OF_GB} GB core parquet, ${OF_SHARDS} shards (~50% of ~3.4 TB target)"
      if pgrep -af 'download_openfake_recover.py' >/dev/null 2>&1; then
        echo "OpenFake: RUNNING"
      else
        echo "OpenFake: STOPPED (no download_openfake_recover.py)"
      fi
      tail -1 /home/jakob/luka/data/external/OpenFake/download_recover.log 2>/dev/null || true
    fi

    if pgrep -af 'download_sid_set_recover.py' >/dev/null 2>&1; then
      echo "SID_Set dispatch: RUNNING (recover script on gpu node)"
    elif pgrep -af 'run_download_sid_set_lj.sh' >/dev/null 2>&1; then
      echo "SID_Set dispatch: RUNNING (wrapper)"
    else
      echo "SID_Set dispatch: STOPPED"
    fi
    tail -3 "${HOME}/luka/runs/download_sid_set/download.log" 2>/dev/null || true

    echo "--- primoz SID_Set (quick) ---"
    LJ_CPU_TIME=00:05:00 "${CODE_ROOT}/scripts/lj_cpu_primoz_exec.sh" bash -lc \
      'du -sh /primoz/luka/external/SID_Set 2>/dev/null; ls /primoz/luka/external/SID_Set/data/*.parquet 2>/dev/null | wc -l | xargs -I{} echo "SID_Set shards: {}/283"; tail -1 /primoz/luka/external/_logs/SID_Set.log 2>/dev/null' \
      2>/dev/null || echo "primoz check skipped"
    echo
  } | tee -a "${LOG}"
}

if [[ "${1:-}" == "--loop" ]]; then
  while true; do
    status_once
    sleep "${INTERVAL}"
  done
else
  status_once
fi
