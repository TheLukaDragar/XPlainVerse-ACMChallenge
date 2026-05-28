#!/usr/bin/env bash
# SigLIP2-giant LoRA — tuned recipe (web / Bombek / HF deepfake best practices).
#
# vs Run A (lora-r16-bs32): lower LR, cosine+warmup, r32, eff batch 128, 4 epochs.
#
# From login node:
#   LJ_GPU_GRES=gpu:2 LJ_GPU_TIME=06:00:00 \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_siglip_lora_tuned_lj.sh
#
# After training, 10k val eval:
#   CKPT=~/luka/runs/pass1_siglip2_giant/<run>/best_ckpt/ckpt.pt \
#     OUT_DIR=~/luka/runs/pass1_siglip2_giant/<run>/eval_10k \
#     ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_eval_10k.sh

set -euo pipefail

_CODE="${HOME}/luka/code/XPlainVerse-ACMChallenge"
if [[ ! -d "${_CODE}/research" ]] && [[ -d /workspace/XPlainVerse-ACMChallenge ]]; then
  _CODE="/workspace/XPlainVerse-ACMChallenge"
fi

export CODE_ROOT="${CODE_ROOT:-${_CODE}}"
export BACKBONE="${BACKBONE:-baseline_models/pass1/siglip2-giant}"
export LORA=1
export LORA_R="${LORA_R:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
export LR_HEAD="${LR_HEAD:-2e-4}"
export LR_BACKBONE="${LR_BACKBONE:-5e-5}"
export LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export EPOCHS="${EPOCHS:-4}"
# r32 LoRA + SigLIP2-giant @ 384px OOMs at bs 64/GPU on A100 80GB (~77 GiB used).
# bs 32/GPU (eff 64 on 2 GPU) matches Run A memory; use 4 GPU for eff 128 if needed.
export BATCH_SIZE="${BATCH_SIZE:-32}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VAL_SLICE="${VAL_SLICE:-10000}"

_RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
export OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/luka/runs/pass1_siglip2_giant/lora-r32-cosine-4ep-${_RUN_TS}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-pass1_siglip_lora_r32_cosine_4ep_${_RUN_TS}}"

exec bash "${CODE_ROOT}/scripts/run_pass1_lj.sh"
