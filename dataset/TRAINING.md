# XPlainVerse VLM training — flag rationale

This document explains every important `swift sft` choice for Qwen3-VL-8B on our
`train_vlm.jsonl`. Use the shell scripts in `scripts/` rather than copying
commands by hand.

## Quick start

```bash
cd /shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/code/XPlainVerse-ACMChallenge

# 1. Sanity (~100 steps, 500 train rows, ~15–30 min on 1× A100)
chmod +x scripts/train_vlm_sanity.sh
./scripts/train_vlm_sanity.sh

# 2. Full 260k run (1× A100, ~8125 steps/epoch, many hours)
chmod +x scripts/train_vlm_full.sh
./scripts/train_vlm_full.sh

# 3. Full run on 4× GPU (matches baseline effective batch 64, ~4063 steps)
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_vlm_full.sh
```

## Model & data

| Flag | Value | Why |
|------|-------|-----|
| `--model` | `Qwen/Qwen3-VL-8B-Instruct` | Fresh instruct base; we train our prompt format from scratch (not the challenge baseline merge). |
| `--model_type` | `qwen3_vl` | ms-swift won't always infer VL template from HF id alone. |
| `--use_hf` | `true` | Download from HuggingFace (ModelScope mirror also works with `false`). |
| `--dataset` | `dataset/train_vlm.jsonl` | 260k balanced rows: user prompt + GT complex + `Verdict:` |
| `--val_dataset` | `dataset/val_vlm.jsonl#2000` | Small val slice for `eval_loss` during training. Full 110k infer uses `val_vlm_infer.jsonl` separately. |

## LoRA / what gets trained

| Flag | Value | Why |
|------|-------|-----|
| `--tuner_type` | `lora` | ms-swift 4.x flag (baseline README uses deprecated `--train_type`). |
| `--lora_rank` / `--lora_alpha` | `16` / `32` | Same as XPlainVerse baseline. |
| `--target_modules` | `all-linear` | Standard ms-swift VL recipe. |
| `--freeze_vit` | `true` | ViT already strong; saves ~40% VRAM. Baseline does the same. |
| `--freeze_aligner` | `true` | Projector frozen; LoRA only on LLM linear layers. |
| `--freeze_llm` | `false` | LoRA adapters **are** added to the LLM (default). |

## Memory / speed

| Flag | Value | Why |
|------|-------|-----|
| `--torch_dtype` | `bfloat16` | A100 native; matches baseline. |
| `--attn_impl` | `sdpa` | **No `flash_attn` installed** on this machine. Do not use `--padding_free` without flash-attn. |
| `--gradient_checkpointing` | `true` | Trades compute for VRAM; required at batch 1–2 on 8B VL. |
| `--max_length` | `2048` | Prompt ~170 tok + image ~1024 + answer ~320 ≈ 1514; 2048 is safe headroom. |
| `IMAGE_MAX_TOKEN_NUM` | `1024` | Qwen3-VL default env var (set in scripts). |
| `MAX_PIXELS` | `1003520` | Matches ms-swift Qwen3-VL recommendation. |

## Batch size & schedule

| Setting | 1× A100 sanity | 1× A100 full | 4× A100 full |
|---------|----------------|--------------|--------------|
| `per_device_train_batch_size` | 1 | 1 | 2 |
| `gradient_accumulation_steps` | 8 | 32 | 8 |
| `NPROC_PER_NODE` | 1 | 1 | 4 |
| **Effective batch** | 8 | 32 | 64 |
| Steps / epoch @ 260k | — | ~8125 | ~4063 |
| `--max_steps` / epochs | `max_steps=100` | `num_train_epochs=1` | `num_train_epochs=1` |
| `--learning_rate` | `2e-4` | `2e-4` | `2e-4` |

Baseline used eff batch **64** on 4× GPU for 4063 steps. Our 1-GPU recipe uses
eff batch **32** (same epoch, 2× more steps, longer wall time but fits one card).

## What we deliberately omit

| Flag | Reason |
|------|--------|
| `--padding_free` / `--packing` | Require `flash_attn`; not installed. |
| `--loss_scale ignore_empty_think` | Qwen3-VL-**Instruct** is non-thinking; our targets have no `` blocks. |
| `--enable_thinking` | Inference-only flag; irrelevant for SFT. |
| `--deepspeed zero2` | Optional for multi-GPU; add if OOM on 4× run. |

## After training

```bash
# Merge LoRA (optional, faster infer)
swift export \
  --adapters runs/vlm_sanity/vx-*/checkpoint-* \
  --merge_lora true \
  --output_dir runs/vlm_sanity_merged

# Infer on val (transformers backend — vLLM weak on Qwen3-VL)
CUDA_VISIBLE_DEVICES=0 swift infer \
  --adapters runs/vlm_sanity/vx-*/checkpoint-* \
  --model_type qwen3_vl \
  --val_dataset dataset/val_vlm_infer.jsonl#20 \
  --infer_backend pt \
  --max_new_tokens 512 \
  --temperature 0 \
  --result_path runs/vlm_sanity_infer.jsonl
```

Post-process: strip thinking → parse `Verdict:` → run compressor →
`evaluate_val.py`. (Scripts for that pipeline are next.)

## Environment variables (always export)

```bash
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_MAX_TOKEN_NUM=1024
```

These are set automatically inside `scripts/train_vlm_*.sh`.
