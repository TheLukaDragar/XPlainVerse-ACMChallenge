#!/usr/bin/env bash
# Pass-1 test TTA for external warmstart best ckpt (200k test, orig+flip).
#
#   LJ_GPU_GRES=gpu:1 LJ_GPU_TIME=08:00:00 ./scripts/run_pass1_test_tta_warmstart_lj.sh
set -euo pipefail

export ENS_CKPT="${ENS_CKPT:-/home/jakob/luka/runs/pass1_ensemble/external_all_warmstart_20260612-105218/best_ckpt/ckpt.pt}"
export OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/pass1_test_tta_warmstart}"
export THRESHOLD="${THRESHOLD:-0.47}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

exec "$(dirname "$0")/run_pass1_test_tta_lj.sh"
