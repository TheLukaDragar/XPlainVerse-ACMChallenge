#!/usr/bin/env bash
# Pass-1 test TTA on 4 GPUs (100k forwards each, merge at end).
#
#   LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=03:00:00 ./scripts/run_pass1_test_tta_4gpu_lj.sh
#
# Env: ENS_CKPT, OUT_DIR, THRESHOLD, BATCH_SIZE, SHARD_COUNT (default 4)
set -euo pipefail

if [[ -z "${_PASS1_TEST_4GPU_IN_CONTAINER:-}" ]]; then
  export _PASS1_TEST_4GPU_IN_CONTAINER=1
  _INNER="export _PASS1_TEST_4GPU_IN_CONTAINER=1 HOME=/home/jakob"
  for _v in ENS_CKPT OUT_DIR THRESHOLD BATCH_SIZE NUM_WORKERS TEST_IMAGES_DIR MANIFEST_DIR SHARD_COUNT; do
    if [[ -n "${!_v:-}" ]]; then
      _INNER+=" ${_v}=$(printf '%q' "${!_v}")"
    fi
  done
  _INNER+="; exec bash scripts/run_pass1_test_tta_4gpu_lj.sh"
  exec ./scripts/lj_ghcr_image_exec.sh bash -c "${_INNER}"
fi

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-${_SCRIPT_ROOT}}"
EXP_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier"
MANIFEST_DIR="${MANIFEST_DIR:-${EXP_DIR}/manifests}"
ENS_CKPT="${ENS_CKPT:-/home/jakob/luka/runs/pass1_ensemble/external_all_warmstart_20260612-105218/best_ckpt/ckpt.pt}"
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/pass1_test_tta_warmstart_4gpu}"
THRESHOLD="${THRESHOLD:-0.47}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SHARD_COUNT="${SHARD_COUNT:-4}"
MANIFEST_TTA="${MANIFEST_DIR}/manifest_test_tta.parquet"

mkdir -p "${OUT_DIR}"
export PYTHONNOUSERSITE=1

echo "=== Pass-1 test TTA ${SHARD_COUNT}-GPU ==="
echo "  ckpt:    ${ENS_CKPT}"
echo "  out:     ${OUT_DIR}"
echo "  shards:  ${SHARD_COUNT}"
echo

if [[ ! -f "${MANIFEST_TTA}" ]]; then
  python3 "${EXP_DIR}/build_test_manifest.py" \
    --test-images-dir "${TEST_IMAGES_DIR:-/primoz/luka/XPlainVerse/data/XPlainVerse/test/images}" \
    --out-dir "${MANIFEST_DIR}" --tta
fi

_PIDS=()
for SHARD in $(seq 0 $((SHARD_COUNT - 1))); do
  CUDA_VISIBLE_DEVICES="${SHARD}" \
    python3 "${EXP_DIR}/eval_ensemble_test.py" \
      --ckpt "${ENS_CKPT}" \
      --manifest "${MANIFEST_TTA}" \
      --out "${OUT_DIR}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --device cuda:0 \
      --shard-id "${SHARD}" \
      --shard-count "${SHARD_COUNT}" \
      --threshold "${THRESHOLD}" &
  _PIDS+=($!)
  echo "  launched shard ${SHARD} on GPU ${SHARD} (pid $!)"
done

echo "waiting for ${#_PIDS[@]} shards..."
for _pid in "${_PIDS[@]}"; do
  wait "${_pid}"
done

echo "=== merging shards ==="
python3 "${EXP_DIR}/eval_ensemble_test.py" \
  --out "${OUT_DIR}" \
  --merge-only \
  --threshold "${THRESHOLD}"

echo "=== Done ==="
echo "  wide : ${OUT_DIR}/predictions.parquet"
echo "  long : ${OUT_DIR}/predictions_long.parquet"
