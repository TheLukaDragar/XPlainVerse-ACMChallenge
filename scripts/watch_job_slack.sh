#!/usr/bin/env bash
# Poll a Slurm job until RUNNING, then notify via Slack (best-effort) and exit.
#
# Usage:
#   JOB_ID=91476 SLACK_USER=U0B4EHTH4RM ./scripts/watch_job_slack.sh
#
# Run in tmux on the login node; polls every POLL_SECS (default 300).

set -euo pipefail

JOB_ID="${JOB_ID:?set JOB_ID}"
SLACK_USER="${SLACK_USER:-U0B4EHTH4RM}"
POLL_SECS="${POLL_SECS:-300}"
LOG="${LOG:-${HOME}/job_watch_${JOB_ID}.log}"

mkdir -p "$(dirname "${LOG}")"

notify() {
  local msg="$1"
  echo "$(date -Is) ${msg}" | tee -a "${LOG}"

  export PATH="${HOME}/.local/bin:${PATH}"
  if command -v agent >/dev/null 2>&1 && [[ -n "${CURSOR_API_KEY:-}" ]]; then
    agent -p --force "Send a Slack direct message to user ${SLACK_USER}. Message text (verbatim):

${msg}" >> "${LOG}" 2>&1 || true
  fi
}

notify "watch_job_slack: monitoring job ${JOB_ID} (poll every ${POLL_SECS}s)"

while true; do
  state="$(sacct -j "${JOB_ID}" -o State -n -X 2>/dev/null | head -1 | tr -d ' ' || true)"
  node="$(sacct -j "${JOB_ID}" -o NodeList -n -X 2>/dev/null | head -1 | tr -d ' ' || true)"

  case "${state}" in
    RUNNING)
      notify "*xpv-vlm-v2-a100 job ${JOB_ID} is RUNNING*
Node: ${node:-ana}
Logs: \`/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/slurm_vlm_v2_a100_${JOB_ID}.out\`
Monitor: \`tail -f ...\` or W&B run \`vlm_v2_a100_1ep\`"
      exit 0
      ;;
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY)
      notify "watch_job_slack: job ${JOB_ID} ended with state=${state} (no RUNNING notification sent)"
      exit 1
      ;;
    "")
      if ! squeue -j "${JOB_ID}" -h 2>/dev/null | grep -q .; then
        notify "watch_job_slack: job ${JOB_ID} disappeared from queue (state unknown)"
        exit 1
      fi
      ;;
  esac

  sleep "${POLL_SECS}"
done
