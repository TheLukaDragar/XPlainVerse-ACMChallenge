#!/usr/bin/env bash
# DINOv3-MAC inference with flip TTA on a manifest (4-GPU sharded).
#
#   LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=06:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_dinov3_test_tta_lj.sh
set -euo pipefail

export HOME="${HOME:-/home/jakob}"
export HF_HOME="${HF_HOME:-/home/jakob/.cache/huggingface}"
CODE_ROOT="${CODE_ROOT:-${HOME}/luka/code/XPlainVerse-ACMChallenge}"
EXP_DIR="${CODE_ROOT}/research/experiments/02_pass1_classifier"
MANIFEST_DIR="${MANIFEST_DIR:-${EXP_DIR}/manifests}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONNOUSERSITE=1

CKPT="${CKPT:-/home/jakob/luka/runs/pass1_dinov3_mac/dinov3_mac_20260616-205827/best_ckpt/ckpt.pt}"
MANIFEST="${MANIFEST:-${MANIFEST_DIR}/manifest_test.parquet}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-12}"
_TS="$(date -u +%Y%m%d-%H%M%S)"
OUT="${OUT:-/home/jakob/luka/runs/pass1_dinov3_test_tta/${_TS}}"

echo "=== DINOv3-MAC TTA inference ==="
echo "  ckpt     : ${CKPT}"
echo "  manifest : ${MANIFEST}"
echo "  out      : ${OUT}  img=${IMAGE_SIZE} bs=${BATCH_SIZE} gpus=${NPROC_PER_NODE}"

cd "${EXP_DIR}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
python3 -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  eval_dinov3_test.py \
  --manifest "${MANIFEST}" --ckpt "${CKPT}" --out "${OUT}" \
  --image-size "${IMAGE_SIZE}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}"

echo "=== Done ==="
echo "preds: ${OUT}/predictions.parquet"
