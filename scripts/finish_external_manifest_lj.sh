#!/usr/bin/env bash
# Rebuild manifest_all_v1 after extraction completes (require all files on disk).
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
LJ_CPU_TIME="${LJ_CPU_TIME:-02:00:00}" \
  "${CODE_ROOT}/scripts/lj_cpu_primoz_exec.sh" \
  python3 "${CODE_ROOT}/research/experiments/02_pass1_classifier/build_manifest_external.py" \
  --only all_v1 --require-exists
