#!/usr/bin/env bash
# Start Cursor "My Machines" worker on Elixir (typically slurm login node).
# Worker stays connected to cursor.com/agents; GPU work is dispatched via
# scripts/lj_gpu_exec.sh (see .cursor/rules/xplainverse-lj-node.mdc).
#
# Prerequisites:
#   agent login
#
# Usage:
#   ./scripts/run_cursor_worker_lj.sh              # detached tmux (default)
#   ./scripts/run_cursor_worker_lj.sh --attach     # start + attach
#   ./scripts/run_cursor_worker_lj.sh --kill-only  # stop worker + tmux
#   ./scripts/run_cursor_worker_lj.sh --foreground # no tmux
#
# Reattach:
#   tmux attach -t cursor-worker-lj
#
# Then open Agents panel → environment: elixir-lj-gpu

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${TMUX_SESSION:-cursor-worker-lj}"
WORKER_NAME="${CURSOR_WORKER_NAME:-elixir-lj-gpu}"
LOG_DIR="${REPO_ROOT}/scripts/agent_tasks/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/cursor_worker_lj_${TIMESTAMP}.log"

export PATH="${HOME}/.local/bin:${PATH}"

MODE="${1:-}"

if ! command -v agent >/dev/null 2>&1; then
  echo "error: 'agent' not found. Install: curl -fsSL https://cursor.com/install | bash" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1 && [[ "${MODE}" != "--foreground" ]]; then
  echo "error: 'tmux' not found. Install tmux or use --foreground" >&2
  exit 1
fi

if ! agent status 2>&1 | grep -q 'Logged in'; then
  echo "error: not logged in. Run: agent login" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

# Runner script avoids tmux/bash -lc quoting bugs (flags were parsed as positional args).
RUNNER="${LOG_DIR}/.cursor_worker_lj_runner.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REPO_ROOT}'
export PATH="${HOME}/.local/bin:\${PATH}"
echo "=== Cursor worker (lj) ==="
echo "started: \$(date -Is)"
echo "host:    \$(hostname)"
echo "worker:  ${WORKER_NAME}"
echo "repo:    ${REPO_ROOT}"
echo "log:     ${LOG_FILE}"
echo
echo "Agents panel → pick environment: ${WORKER_NAME}"
echo "GPU/container commands → scripts/lj_gpu_exec.sh (see .cursor/rules/xplainverse-lj-node.mdc)"
echo
exec agent worker start --name '${WORKER_NAME}' --worker-dir '${REPO_ROOT}' --verbose
EOF
chmod +x "${RUNNER}"

TMUX_INNER="bash '${RUNNER}' 2>&1 | tee '${LOG_FILE}'; echo; echo 'worker exited — press Enter'; read -r"

kill_previous_worker() {
  local killed_any=false

  while IFS= read -r sess; do
    [[ -z "${sess}" ]] && continue
    echo "Killing tmux session: ${sess}"
    tmux kill-session -t "${sess}" 2>/dev/null || true
    killed_any=true
  done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -E '^cursor-worker-lj' || true)

  local pids
  pids="$(pgrep -f "agent worker start.*--name ${WORKER_NAME}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "Killing previous worker PIDs: ${pids}"
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    sleep 1
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
    killed_any=true
  fi

  if [[ "${killed_any}" == false ]]; then
    echo "No previous lj worker found."
  fi
}

run_foreground() {
  echo "Starting worker (foreground) → ${LOG_FILE}"
  bash "${RUNNER}" 2>&1 | tee "${LOG_FILE}"
}

start_tmux() {
  local attach="${1:-false}"

  kill_previous_worker

  echo "Starting Cursor worker in tmux: ${SESSION}"
  echo "  name:    ${WORKER_NAME}"
  echo "  attach:  tmux attach -t ${SESSION}"
  echo "  detach:  Ctrl-b d"
  echo "  kill:    ./scripts/run_cursor_worker_lj.sh --kill-only"
  echo "  panel:   https://cursor.com/agents (env: ${WORKER_NAME})"
  echo "  log:     ${LOG_FILE}"

  if [[ "${attach}" == "true" ]]; then
    tmux new-session -s "${SESSION}" "${TMUX_INNER}"
  else
    tmux new-session -d -s "${SESSION}" "${TMUX_INNER}"
    sleep 1
    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
      echo "error: tmux session died immediately. Check log:" >&2
      echo "  ${LOG_FILE}" >&2
      tail -20 "${LOG_FILE}" 2>/dev/null >&2 || true
      exit 1
    fi
    echo "Detached — worker keeps running if you disconnect."
  fi
}

case "${MODE}" in
  --foreground)
    kill_previous_worker
    run_foreground
    ;;
  --attach)
    start_tmux true
    ;;
  --kill-only)
    kill_previous_worker
    ;;
  --help|-h)
    sed -n '2,20p' "$0" | sed 's/^# \?//'
    ;;
  ""|--background)
    start_tmux false
    ;;
  *)
    echo "error: unknown option: ${MODE}" >&2
    echo "usage: $0 [--attach | --foreground | --kill-only | --help]" >&2
    exit 1
    ;;
esac
