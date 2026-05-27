# Pass-1 binary classifier — fine-tuned VFM

**Status:** STAGED, NOT YET RUN. Wait for `01_gold_verdict` validation results before launching.

## Purpose

Train a binary `{real, fake}` classifier on XPlainVerse-train (450k images) using a modern Vision Foundation Model + linear head + LoRA. This becomes Pass-1 of the two-stage pipeline. The Pass-2 VLM gets the verdict from this classifier and only generates the explanation.

## Why this design

See `research/02_literature_review.md` and `research/04_strategy.md` for full citations. Short version:

- **Backbone: DINOv3-Large** (300M params, ungated, 96.5% on GenImage in Simplicity Prevails 2026)
- **Linear head** first as baseline (typical 90%+ in published work)
- **LoRA r=16 alpha=32** on backbone if linear isn't enough
- **Loss:** binary cross-entropy. **Optimizer:** AdamW. **Schedule:** cosine, 1 epoch.

## Data flow

```
train_vlm.jsonl  ─┬─→ manifest_train.parquet   (450k rows: image_path, label_int)
                  └─→ manifest_train_balanced.parquet  (260k = 130k real + 130k fake)
val_vlm_infer.jsonl + val_ground_truth.jsonl
                  ─→ manifest_val.parquet      (110k rows)
```

We start with the 1:1 balanced subset (260k) for the first run because the unbalanced 320k:130k split was the root cause of VLM mode collapse.

## Files

| File | Purpose |
|------|---------|
| `build_manifest.py` | Convert train_vlm.jsonl + val_ground_truth.jsonl into image-path/label parquets |
| `train_pass1.py` | DINOv3-Large + linear head training loop (HF Transformers + PEFT) |
| `eval_pass1.py` | Run trained classifier on val, output per-sample probs + AUC/AP/F1 |
| `run.sh` | Launcher that chains build → train → eval |

## Decision rule (DO NOT LAUNCH UNTIL VALIDATION RESULT)

After `01_gold_verdict` completes, check:

- If `conditioned` complex_overall ≥ baseline + 0.03 → **launch Pass-1 training** (claim confirmed)
- If `structured` complex_overall ≥ baseline + 0.05 → **launch Pass-1 + adopt structured prompt** (best path)
- If both ≤ baseline + 0.01 → **do NOT launch Pass-1**. The verdict is not the bottleneck; pivot to fake-only SFT continuation or prompt engineering.

## Resource estimate

- 1× A100 80GB
- DINOv3-Large at 224×224, batch 64, ~1 step/sec
- 260k images / 64 batch = 4063 steps/epoch
- ~70 minutes/epoch
- 2 epochs + linear-head warmup + LoRA fine-tune ≈ 4-6 hours

## Expected outputs

`runs/pass1_dinov3/v1-{date}/`
- `best_ckpt/` — model weights
- `val_predictions.parquet` — per-sample p(fake)
- `metrics.json` — AUC, AP, calibrated threshold, accuracy at default and calibrated thresholds, per-class accuracy
- `tensorboard/` — training curves
