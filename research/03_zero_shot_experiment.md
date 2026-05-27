# Zero-shot AIGC detector experiment — XPlainVerse val

**Date:** 26 May 2026, 22:00 UTC  
**Goal:** Test whether pretrained AIGC binary detectors can replace the VLM as a Pass-1 verdict source without further training.  
**Output dir:** `runs/zero_shot_aigc/`

## TL;DR

**Zero-shot transfer fails on XPlainVerse. None of the 6 tested detectors beat our VLM's 0.72 verdict accuracy.** Domain shift from their training distribution (GenImage / OpenFake / synthetic web data) to XPlainVerse is severe.

## Setup

- **Source:** `Lunahera/simplicityprevails` (HF) — official released 7-baseline collection from Simplicity Prevails (arXiv 2602.01738, 2026)
- **Eval data:** 1000 images from XPlainVerse val = **500 real + 500 fake** (deterministic sample from `data/XPlainVerse/val/images/{real,fake}/` first-N-sorted)
- **Hardware:** A100 80GB (separate from the running VLM SFT)
- **Wall time:** ~10 min for 6/7 baselines. DINOv3-Linear (7B model) failed because `facebook/dinov3-vit7b16-pretrain-lvd1689m` is gated on HF (HTTP 403)
- **Total disk:** 38 GB checkpoints

## Results — at default threshold 0.5

| Model | AUC | AP | Acc | Real acc | Fake acc |
|-------|-----|----|-----|----------|----------|
| SigLIP-Linear | 0.615 | 0.601 | 0.558 | 0.852 | 0.264 |
| **PE-CLIP-Linear** | **0.726** | **0.725** | 0.546 | 0.984 | 0.108 |
| DINOv2-Linear | 0.586 | 0.567 | 0.522 | 0.890 | 0.154 |
| MetaCLIP-Linear | 0.558 | 0.566 | 0.529 | 0.922 | 0.136 |
| SigLIP2-Linear (giant) | 0.688 | 0.678 | 0.530 | 0.980 | 0.080 |
| MetaCLIP2-Linear (giant) | 0.635 | 0.647 | 0.555 | 0.942 | 0.168 |
| 6-model ensemble (avg prob) | 0.666 | 0.660 | 0.530 | 0.980 | 0.070 |
| **VLM Qwen3-VL ckpt-3200** | — | — | **0.720** | **0.916** | **0.552** |

## Results — at oracle-calibrated threshold (best F1 on this val sample)

This is the **upper bound** of what each model could achieve with perfect threshold calibration. Note: the threshold is fit on the same val data we evaluate on, so this slightly overestimates true performance.

| Model | Best threshold | Acc | Real acc | Fake acc |
|-------|---------------:|----:|---------:|---------:|
| SigLIP-Linear | 0.0425 | 0.599 | 0.57 | 0.63 |
| **PE-CLIP-Linear** | **0.0103** | **0.679** | 0.71 | 0.65 |
| DINOv2-Linear | 0.0543 | 0.573 | 0.72 | 0.43 |
| MetaCLIP-Linear | 0.0393 | 0.556 | 0.71 | 0.40 |
| SigLIP2-Linear | 0.0032 | 0.640 | 0.56 | 0.72 |
| MetaCLIP2-Linear | 0.0072 | 0.609 | 0.64 | 0.58 |
| Ensemble | 0.0632 | 0.621 | 0.60 | 0.65 |

**Best calibrated zero-shot model = PE-CLIP-Linear at 0.679 accuracy. Still 4.1 points below the VLM's 0.720.**

## Probability distribution diagnostics

Why default threshold 0.5 is broken: all 6 models output very low fake-probabilities for *both* classes. Probabilities are crammed near 0.

| Model | Mean p(fake) on real samples | Mean p(fake) on fake samples | Separation |
|-------|------------------------------:|------------------------------:|-----------:|
| SigLIP-Linear | 0.178 | 0.289 | 0.111 |
| PE-CLIP-Linear | 0.038 | 0.152 | 0.114 |
| DINOv2-Linear | 0.128 | 0.190 | 0.062 |
| MetaCLIP-Linear | 0.103 | 0.161 | 0.058 |
| SigLIP2-Linear | 0.036 | 0.114 | 0.078 |
| MetaCLIP2-Linear | 0.077 | 0.186 | 0.109 |

The signal is there (separation > 0) but tiny. Their training real distribution matches ours; their training fake distribution does not. XPlainVerse fakes look "more real than typical fakes" to these detectors.

## Why this is consistent with the literature

The Simplicity Prevails paper (arXiv 2602.01738, our source for these checkpoints) reports:
- DINOv3-Linear gets **96.5% on GenImage** (in-distribution)
- DINOv3-Linear gets **94.0% on in-the-wild** (different generators)
- The drop from in-distribution to in-the-wild is ~2.5 percentage points for *the strongest* backbone

Our zero-shot test is essentially asking the same question on yet another out-of-distribution dataset (XPlainVerse uses different generators and post-processing than GenImage / OpenFake). The drops we observe (96% → ~60% calibrated) are larger than their in-the-wild drops, which suggests XPlainVerse fakes are **further from their training distribution than typical "in-the-wild"**.

## Conclusion

1. **Don't use any of these zero-shot.** Even the best one is 4 points below the VLM and would degrade the pipeline.
2. **The features still work** — AUC 0.6–0.73 means the backbones partially separate real from fake. The classification heads are wrong because they were fit on a different distribution.
3. **Fine-tuning is required.** A binary classifier head trained on 450k labeled XPlainVerse-train images should reach the published 90%+ accuracy ceiling, based on:
   - Simplicity Prevails 2026 (96.5% on GenImage)
   - Bombek1 99.10% on OpenFake
   - NTIRE 2026 winners 0.9974 ROC-AUC

## Files for reference

- `runs/zero_shot_aigc/SUMMARY.md` — this experiment write-up with full numbers
- `runs/zero_shot_aigc/results/{model}.json` — per-sample predictions
- `runs/zero_shot_aigc/summary.json` — aggregate metrics
- `runs/zero_shot_aigc/run_all.sh` + `analyze.py` — re-runnable pipeline
- `runs/zero_shot_aigc/simplicityprevails/` — cloned repo + LFS weights (38GB)
