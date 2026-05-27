# Strategy — recommended pipeline

Two-stage decoupled pipeline with separate Pass-1 verdict (fine-tuned VFM classifier) and Pass-2 conditional explanation (existing VLM, prompt-modified).

## Architecture

```
                 image
                   │
                   ▼
        ┌──────────────────────┐
        │ Pass 1: AIGC verdict │   DINOv3-Large (or SigLIP2-SO400M)
        │ frozen VFM + LoRA    │   + linear binary head, fine-tuned
        │ + linear head        │   on 450k XPlainVerse-train
        └──────────┬───────────┘
                   │ verdict ∈ {real, fake}
                   ▼
        ┌──────────────────────┐
        │ Pass 2: explanation  │   Qwen3-VL-8B (our SFT'd model)
        │ VLM, conditional     │   + prompt template that hard-codes
        │ on Pass-1 verdict    │   the verdict from Pass 1
        └──────────┬───────────┘
                   │
                   ▼
   {complex_explanation, simple_explanation}
```

## Why this design — claim by claim

### Claim 1: Decoupling Pass-1 from Pass-2 fixes mode collapse

**Direct support:**
- M2F2-Det (CVPR 2025): removing detection-aware features from joint VLM costs **−10.12% accuracy**. Conversely, providing them improves both judgment F1 (+4.51%) and explanation CIDEr/ROUGE-L.
- Sun et al. 2021 (arXiv 2110.02056): predict-then-explain is more data-efficient and avoids exposure bias compared to joint or explain-then-predict.
- Marasovic et al. 2022 (arXiv 2205.03401): *"LLMs tend to generate nonfactual explanations when making wrong predictions"* — wrong verdict ⇒ wrong explanation in joint generation.

**What we don't have direct support for:**
- The specific claim "Pass-2 conditioning fixes mode collapse regardless of where the verdict comes from." This is a corollary of the three above but no paper tested exactly that. **Recommended cheap experiment** before committing to this path: feed gold verdicts into our existing VLM via prompt and measure entity/facts F1 lift on val. If we see ≥3 points of complex_overall improvement, the claim holds and the bigger investment in training a Pass-1 classifier is justified.

### Claim 2: A fine-tuned VFM classifier will reach 90%+ accuracy

**Direct support (in-distribution numbers):**
- Simplicity Prevails 2026: DINOv3-Linear on GenImage = 96.5%
- DINOv3-Forensics 2026: +10 absolute points over specialized SOTA on 4 benchmarks
- Bombek1 SigLIP2+DINOv2 ensemble on OpenFake: 99.10% accuracy, 0.9997 AUC
- NTIRE 2026 winners: 0.9974 clean ROC-AUC

**What we have evidence against (zero-shot ≠ fine-tuned):**
- Our zero-shot experiment shows 0.679 best calibrated accuracy. The published 96%+ is for *trained* heads. We must train.

### Claim 3: Stop training the VLM at the current ROUGE-L plateau

**Our own data:**
- ROUGE-L peaked at step 2400 (34.73), declined by step 3200 (34.34)
- 13% through 1 epoch, ETA 12 more days for full epoch
- Train loss still decreasing — model is overfitting to template style, not improving downstream metric

**Decision:** Stop SFT, switch resources to Pass-1 training. Use ckpt-2400 as Pass-2 backbone.

### Claim 4: DINOv3-Large is the right Pass-1 backbone

**Pros:**
- Best zero-shot AUC in our backbone family (DINOv3-Linear is reported to beat DINOv2 by +11% / +30%, but we couldn't test DINOv3 directly because the 7B variant is gated)
- Self-supervised on 1.689B images (LVD-1689M) — strongest "natural image" prior
- Linear probe matches specialized SOTA on AIGC (Simplicity Prevails 2026)
- Ungated for Large (300M) and Huge+ (1B) variants

**Alternatives considered:**
- **SigLIP2-SO400M** (Bombek1 recipe): also good, slightly worse on GenImage (94.5% vs 96.5%). Used by NTIRE winners but in giant variant. **Use as ensemble partner.**
- **DINOv3-7B**: would be best but gated on HF. Skip.
- **PE-CLIP**: best in our zero-shot test, but linear head only. Could include in ensemble.
- **CLIP / DINOv2**: outdated, beaten by both DINOv3 and SigLIP2 in 2025-2026 papers.

### Claim 5: Two-stage > joint training

**Direct support:**
- Multimodal-CoT 2023: two-stage (rationale → answer) beats GPT-3.5 by **+16% on ScienceQA** (75.17 → 91.68%)
- VIGIL CVPR 2026: explicit "these capabilities cannot be acquired in a single training pass" — uses 3 progressive stages
- Two-Stage Reasoning-Infused Learning 2025: +8.7 percentage points (z=6.88, p<.001) over single-stage SFT on emotion classification
- Nature Digital Medicine 2025 two-stage prompting: −63% incorrect-knowledge errors

**Counter-evidence we acknowledge:**
- GenCLS++ (arXiv 2504.19898, 2025) shows CoT prompts hurt pure classification. **Resolution:** we don't add CoT to Pass-1 — it's a single linear classifier. CoT only happens implicitly inside Pass-2 explanation.

## Concrete plan with effort estimates

| # | Step | Effort | Expected lift |
|---|------|--------|--------------:|
| 1 | Stop VLM SFT at ckpt-2400, free GPU 1 | 0 | — |
| 2 | Cheap validation: feed gold verdicts to current VLM via prompt template, measure complex_overall on 200-sample val | 1 hour | This validates the decoupling claim *before* we invest in Pass-1 training |
| 3 | Fine-tune DINOv3-Large + linear head + LoRA on 450k XPlainVerse-train (binary cross-entropy, ~1 epoch) | 1–2 days, 1 GPU | Pass-1 verdict acc 0.55 → 0.90+ on fakes |
| 4 | (optional) Add SigLIP2-SO400M as ensemble partner | +1 day | +1–2 acc points based on Bombek1 recipe |
| 5 | Modify Pass-2 prompt: hard-code Pass-1 verdict, ask for 5 specific regions on fake / 5 anchoring evidence on real | 0.5 day | entity/facts F1 0.27 → 0.40+ (if M2F2-Det's +10% holds) |
| 6 | End-to-end eval on 1k val sample | 0.5 day | Final complex_overall target ≥ 0.5 |

**Total: ~4–5 days for a first complete pipeline. 21 days remain to deadline (15 Jun 2026).**

## What we explicitly drop

- ❌ Continuing the current VLM SFT past ckpt-2400 (ROUGE-L plateau, overfitting to template style)
- ❌ Zero-shot use of any simplicityprevails baseline (proven 4 points below VLM)
- ❌ Joint single-pass classification + explanation (wrong-verdict-poisons-explanation problem)
- ❌ Explain-then-predict (Multimodal-CoT order; less data-efficient per Sun et al. 2021)
- ❌ Adding CoT prompts to Pass-1 (GenCLS++ shows it hurts pure classification)

## Risks

1. **Domain shift in Pass-1 training:** XPlainVerse-train and val both come from the same distribution, so this should be tractable. But we should verify train/val match by inspecting a few generators.
2. **Pass-2 still has its own template bias:** even with correct verdict, the VLM may still produce a 1-region explanation. Mitigate via prompt engineering and possibly a second SFT pass on fake-only data with the conditional prompt.
3. **Compute budget:** training Pass-1 on 450k images at 224×224 = ~1 hour per epoch on A100. Should fit easily.
