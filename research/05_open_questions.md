# Open questions — claims I cannot fully back with citations

Honest list of things I asserted in conversation but where the literature evidence is **indirect**. Flagged so we don't over-commit before validating.

## 1. "Pass-2 conditioning fixes mode collapse regardless of where the verdict comes from"

**What I said:** giving the VLM a hard-coded verdict like "this image IS fake, list 5 specific artifacts" will fix mode collapse even if the verdict came from a separate classifier (vs. from gold labels or its own internal reasoning).

**What we have:**
- M2F2-Det shows decoupled detection features → +10.12% acc, +4.51% F1, better explanations. ✅ Strong support for the architectural pattern.
- Marasovic 2022 shows joint generation produces unfaithful explanations on wrong predictions. ✅ Support for *why* decoupling helps.
- Sun et al. 2021 PtE: more data-efficient than EtP. ✅ Support for *order*.

**What we don't have:**
- A direct ablation in any paper showing: "given fixed VLM, conditioning on a *correct verdict from any source* (vs. joint generation) yields equivalent improvement." This is the specific claim. The closest is M2F2-Det but their detection features are integrated via a cross-attention bridge, not just a prompt-level token.
- No paper explicitly compares "verdict via prompt token" vs "verdict via cross-attention" vs "verdict via fine-tuning" as the conditioning mechanism.

**Recommended validation (1-day experiment):**
- Take 200 val rows, prepend "FORENSIC ANALYSIS — this image has been determined to be {gold_label}." to the prompt.
- Run inference with current ckpt-2400, measure complex_overall and per-class entity/facts F1.
- If gold-conditional > unconditional by ≥3 points complex_overall: claim holds, build Pass-1 classifier.
- If <1 point lift: claim is wrong, the issue is template bias not verdict bias. Pivot to fake-only SFT instead.

## 2. "VLM verdict is the bottleneck (vs. explanation quality)"

**What I implied:** mode collapse is the dominant problem; if we fix verdicts, complex_overall will jump.

**What we have:**
- 78 real-called-fake + 483 fake-called-real errors out of 2000 = 28% verdict error. These contribute 0 entity F1.
- If verdicts were 100% correct, the upper bound is current_F1_when_correct × 1.0. Our current entity_f1 is 0.323 on the official 32-sample run, but that includes correctly-classified samples too.

**What we don't have:**
- A clean breakdown: entity_f1 conditional on verdict_correct vs verdict_wrong on the *official* eval. We computed it on training-time predictions but not on the official scoring pipeline.

**Recommended validation:** Run the official `evaluate_val.py` on a 200-sample slice of ckpt-2400 predictions, separated by verdict-correct vs verdict-wrong. If the gap is, say, 0.55 vs 0.10, our priority is verdict (28% of samples × 0.45 lift = 12.6 points lift potential). If the gap is 0.40 vs 0.30, priority is explanation depth.

## 3. "Fine-tuned DINOv3 will reach 90%+ on XPlainVerse"

**What I said:** based on Simplicity Prevails 96.5% on GenImage and 94.0% in-the-wild.

**What we have:**
- These numbers are real but for *those* datasets. XPlainVerse may be harder.
- Our zero-shot diagnostic showed lower probability separation (~0.1) than reported in the paper (>0.5 typical for trained linear head on in-distribution).

**What we don't have:**
- Any paper testing on XPlainVerse specifically (it's a new ACM-MM 2026 challenge dataset).

**Why I still believe it:**
- 450k labeled training images is a *lot*. Even a difficult distribution becomes tractable with that scale.
- The zero-shot experiment shows the *features* contain the signal (AUC 0.73 with PE-CLIP). The head just doesn't transfer. A trained head on 450k samples will fix that.
- Bombek1's recipe on OpenFake hits 99.10% — XPlainVerse is in the same family.

**Risk if wrong:** if Pass-1 only reaches, say, 80% accuracy, that's still better than the VLM's 72%. So this is bounded downside even in the worst case.

## 4. "Stop training the VLM at the current ROUGE-L plateau"

**What I said:** ROUGE-L peaked at step 2400, more SFT will overfit style.

**What we have:**
- Direct evidence in our own runs: ROUGE-L 34.73 (step 2400) → 34.58 (step 2800) → 34.34 (step 3200). Token accuracy still rising (overfitting indicator).

**What we don't have:**
- Whether ROUGE-L is a faithful proxy for the official entity/facts F1. They may decouple — the model could be improving at fact-level detail while ROUGE-L looks flat.

**Recommended validation:** run official evaluate_val.py on a 200-sample sample at ckpt-3200 and ckpt-2400. If 3200 > 2400 on complex_overall, ROUGE-L plateau is misleading and we should keep training.

## 5. "DINOv3 > SigLIP2 for our task"

**What I said:** DINOv3-Linear 96.5% > SigLIP2-Linear 94.5% on GenImage in Simplicity Prevails.

**What we have:**
- Their numbers: DINOv3 better on GenImage (96.5 vs 94.5), comparable on in-the-wild (94.0 vs 88.0).
- NTIRE 2026 winners actually used SigLIP2-giant variants, not DINOv3.
- Bombek1 ensemble uses both.

**What this means:**
- Difference is small. Try both. Ensemble if compute allows.
- "DINOv3 is best" is a 2-point claim, not a settled-issue claim.

## 6. "VLM has been struggling specifically because of the bimodal training distribution"

**What I said:** real targets are 43 words / fake targets 113 words; this asymmetry caused the mode collapse.

**What we have:**
- Direct measurement on 50k train rows: real median 43 words, fake median 113 words.
- Strong correlation between this asymmetry and "predict real → produce short text → mode collapse."

**What we don't have:**
- A controlled ablation. The asymmetry exists, the mode collapse exists, but causation isn't proven. Could also be class imbalance (2.5:1 fake:real) or harder underlying task on fakes.

**Why I still believe it:**
- The 65% prevalence of "this picture looks real because…" openers in predictions is a strong indicator the model picked up the simple template as default.
- Easy fix to test in Pass 2 — split prompts by class.

---

## What this list means for the next 21 days

Before we sink 4–5 days into building the Pass-1 classifier, we should run **two cheap validation experiments** (1 day total):

1. **Gold-verdict prompt-conditioning experiment** — directly tests claim #1 above
2. **Per-verdict-class entity F1 breakdown on official eval** — directly tests claim #2 above

If both confirm the hypothesis, build Pass-1 with high confidence. If they don't, we save 4 days and pivot.

This list should be revisited after each major experiment. Move resolved questions into `02_literature_review.md` (with our own numbers added) or close them out.
