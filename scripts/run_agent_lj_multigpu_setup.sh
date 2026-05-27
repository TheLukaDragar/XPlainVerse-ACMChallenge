#!/usr/bin/env bash
# Launch Cursor Agent CLI to autonomously set up lj multi-GPU training.
# Runs inside tmux so SSH/tunnel drops do not kill the agent.
#
# Prerequisites:
#   agent login   (or export CURSOR_API_KEY=...)
#
# Usage:
#   ./scripts/run_agent_lj_multigpu_setup.sh              # detached tmux (default)
#   ./scripts/run_agent_lj_multigpu_setup.sh --attach   # attach to running/fresh session
#   ./scripts/run_agent_lj_multigpu_setup.sh --foreground # no tmux (blocks this terminal)
#   ./scripts/run_agent_lj_multigpu_setup.sh --kill-only  # stop previous runs only
#
# Reattach after disconnect:
#   tmux attach -t cursor-lj-multigpu-setup
#
# Logs: scripts/agent_tasks/logs/lj_multigpu_<timestamp>.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_FILE="${REPO_ROOT}/scripts/agent_tasks/lj_multigpu_training_setup.md"
LOG_DIR="${REPO_ROOT}/scripts/agent_tasks/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/lj_multigpu_${TIMESTAMP}.log"
SESSION="${TMUX_SESSION:-cursor-lj-multigpu-setup}"

export PATH="${HOME}/.local/bin:${PATH}"

MODE="${1:-}"

if ! command -v agent >/dev/null 2>&1; then
  echo "error: 'agent' not found. Install: curl -fsSL https://cursor.com/install | bash" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1 && [[ "${MODE}" != "--foreground" ]]; then
  echo "error: 'tmux' not found. Install tmux or run with --foreground" >&2
  exit 1
fi

if ! agent status 2>&1 | grep -q 'Logged in'; then
  echo "error: not logged in. Run: agent login" >&2
  exit 1
fi

if [[ ! -f "${TASK_FILE}" ]]; then
  echo "error: task file missing: ${TASK_FILE}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

PROMPT="Autonomous setup task. Read and execute every step in ${TASK_FILE} (plan + success criteria). Workspace: ${REPO_ROOT}. Use ~/xplainverse_exec.sh for container/GPU checks on lj. Report a structured completion summary when done."

# Shell command run inside tmux (tee log + exit marker for status checks).
INNER_CMD=$(cat <<EOF
set -euo pipefail
cd '${REPO_ROOT}'
export PATH="${HOME}/.local/bin:\${PATH}"
echo "=== Cursor agent lj setup ==="
echo "started: \$(date -Is)"
echo "log: ${LOG_FILE}"
echo "session: ${SESSION}"
echo
agent \\
  --print \\
  --output-format stream-json \\
  --stream-partial-output \\
  --force \\
  --trust \\
  --approve-mcps \\
  --workspace '${REPO_ROOT}' \\
  --model composer-2.5 \\
  '${PROMPT}'
echo
echo "=== agent finished: \$(date -Is) exit=\$? ==="
EOF
)

kill_previous_agents() {
  local killed_any=false

  # Kill our tmux session(s) for this task.
  while IFS= read -r sess; do
    [[ -z "${sess}" ]] && continue
    echo "Killing tmux session: ${sess}"
    tmux kill-session -t "${sess}" 2>/dev/null || true
    killed_any=true
  done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -E '^cursor-lj-multigpu' || true)

  # Kill stray agent processes for this lj setup task (nohup / old runs).
  local pids
  pids="$(pgrep -f 'agent.*lj_multigpu_training_setup\.md' 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "Killing previous agent PIDs: ${pids}"
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    sleep 1
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
    killed_any=true
  fi

  if [[ "${killed_any}" == false ]]; then
    echo "No previous lj agent runs found."
  fi
}

run_foreground() {
  echo "Starting agent (foreground, no tmux) → ${LOG_FILE}"
  bash -lc "${INNER_CMD}" 2>&1 | tee "${LOG_FILE}"
}

start_tmux() {
  local attach="${1:-false}"

  kill_previous_agents

  echo "Starting agent in tmux session: ${SESSION}"
  echo "  log:     ${LOG_FILE}"
  echo "  attach:  tmux attach -t ${SESSION}"
  echo "  detach:  Ctrl-b d"
  echo "  kill:    tmux kill-session -t ${SESSION}"

  if [[ "${attach}" == "true" ]]; then
    tmux new-session -s "${SESSION}" "bash -lc $(printf '%q' "${INNER_CMD}") 2>&1 | tee '${LOG_FILE}'"
  else
    tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${INNER_CMD}") 2>&1 | tee '${LOG_FILE}'"
    echo "Detached — agent keeps running if you disconnect."
  fi
}

case "${MODE}" in
  --foreground)
    kill_previous_agents
    run_foreground
    ;;
  --attach)
    start_tmux true
    ;;
  --kill-only)
    kill_previous_agents
    ;;
  --help|-h)
    sed -n '2,18p' "$0" | sed 's/^# \?//'
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
