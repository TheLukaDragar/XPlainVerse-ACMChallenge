#!/usr/bin/env bash
# Launch 4 compressor shards on GPU node (no Slurm). Run via SSH:
#   bash scripts/run_compressor_test_4gpu.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-/home/jakob/luka/runs/compressor_test/$(date -u +%Y%m%d-%H%M%S)}"
export OUT_DIR SHARD_COUNT=4 COMPLEX_IN ADAPTERS PROMPT_FILE

mkdir -p "${OUT_DIR}"
echo "=== compressor 4gpu ==="
echo "  out_dir: ${OUT_DIR}"

for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${SHARD}" SHARD_ID="${SHARD}" SHARD_COUNT=4 \
    bash "${SCRIPT_DIR}/run_compressor_test_lj.sh" &
done
wait

MERGE_ONLY=1 SHARD_ID=0 SHARD_COUNT=4 bash "${SCRIPT_DIR}/run_compressor_test_lj.sh"
echo "=== compressor 4gpu done: ${OUT_DIR} ==="
