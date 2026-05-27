# Literature review — citations with concrete numbers

Each entry: paper title, authors/venue/year, **what they actually showed**, why it matters for our pipeline.

---

## 1. AIGC binary detection — choice of backbone

### Simplicity Prevails: Generalizable AIGI Detection in Visual Foundation Models
- **arXiv:** 2602.01738 (2026)
- **Repo:** https://huggingface.co/Lunahera/simplicityprevails (we cloned it; see `runs/zero_shot_aigc/`)
- **Claim:** A *frozen* modern Vision Foundation Model + a single linear layer beats specialized AIGC detectors.

**Numbers (linear probe accuracies):**

| Backbone | GenImage avg | In-the-wild avg | Notes |
|----------|--------------|-----------------|-------|
| **DINOv3-Linear** | **96.5%** | **94.0%** | Best modern VFM in their tests |
| DINOv2-Linear | 85.2% (calc'd) | ~64% | Older — DINOv3 is +11% on GenImage, +30.4% on in-the-wild |
| MetaCLIP-2 | 89.5% (calc'd) | — | +12.6% over MetaCLIP |
| SigLIP2-Linear | 94.5% | 88% | |
| Specialized OMAT (prior SOTA) | 94.6% | 90.9% | Beaten by frozen DINOv3 |

**Why we cite this:** establishes that for in-distribution data, a frozen modern VFM + linear head ≈ SOTA. Justifies our plan to fine-tune one of these on XPlainVerse-train rather than build a specialized architecture.

### DINOv3 Beats Specialized Detectors: A Simple Foundation Model Baseline for Image Forensics
- **arXiv:** 2604.16083 (2026)
- **Claim:** DINOv3 + LoRA + lightweight conv decoder + boundary-aware loss > specialized SOTA on 4 benchmarks
- **Numbers:** Up to **+10 absolute points** over previous SOTA. Even smallest DINOv3 variant beats prior specialized methods on data-scarce MVSS.
- **Why we cite this:** justifies LoRA on DINOv3 as the recipe for our Pass-1 classifier.

### Brought a Gun to a Knife Fight: Modern VFM Baselines on In-the-Wild AI Image Detection
- **arXiv:** 2509.12995 (2025)
- **Claim:** Linear probe on modern VFM beats forensic detectors by **>20% accuracy** on in-the-wild data
- **Numbers (in-the-wild):** DINOv3 96.3 / 93.7 / 90.5 / 92.4 across 4 datasets. CLIP (2021) 84.3 / 70.5 / 55.6. Specialized NPR 88.8 / 97.8 / 3.18 / 57.2 (catastrophic on cross-domain).
- **Why we cite this:** confirms that older detectors don't generalize but modern VFM frozen features do.

### Layer Transition Discrepancy (LTD)
- **arXiv:** 2603.10598 (CVPR 2026)
- **Repo:** https://github.com/yywencs/LTD
- **Claim:** Frozen CLIP-ViT intermediate-layer features + dynamic layer selection
- **Numbers:** UFD 96.90% / DRCT-2M 99.54% / GenImage 91.62% — beat ForgeLens (95.56), FatFormer (95.98)
- **Why we cite this:** alternative if simple linear probe doesn't suffice. Probably overkill for in-distribution.

### Bombek1 / royhuang199712 SigLIP2 + DINOv2 ensemble
- **HF:** Bombek1/ai-image-detector-siglip-dinov2 (2025)
- **Architecture:** SigLIP2-SO400M + DINOv2-Large, both with LoRA r=32, fused via MLP head
- **Numbers (OpenFake validation):** AUC 0.9997, accuracy 99.10%, cross-dataset 97.15%. Per-generator: DALL-E 3 100%, Midjourney V6 96.33%
- **Why we cite this:** existing public model showing the ensemble recipe works at scale on the right kind of data.

### NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild
- **arXiv:** 2604.11487 (CVPR 2026 workshop)
- **Numbers:** Top team (MICV) 0.9974 clean ROC-AUC / 0.9723 robust. #2 Ant International 0.9971 / 0.9711. Both used SigLIP2-giant variants.
- **Why we cite this:** sets the upper bound for what's achievable on this kind of binary task. Justifies using SigLIP2 variants when we want maximal robustness to image transforms.

### CO-SPY: Combining Semantic and Pixel Features (CVPR 2025)
- **arXiv:** 2503.18286
- **Repo:** https://github.com/Megum1/Co-Spy
- **Numbers:** Average AP 96.02%, accuracy 87.06% on diffusion in-the-wild (CO-SPYBench, includes FLUX). +11–34% over baselines.
- **Why we cite this:** combines semantic (CLIP) + pixel (NPR) features. A fallback if pure DINOv3 underperforms.

---

## 2. Decoupled detection-then-explanation in forensic VLMs

### Rethinking Vision-Language Model in Face Forensics: M2F2-Det (CVPR 2025)
- **arXiv:** 2503.20188
- **Repo:** https://github.com/CHELSEA234/M2F2_Det
- **Architecture:** CLIP image encoder + Forgery Prompt Learning + frequency-token H_F + Bridge Adapter → LLM for explanation. Two-stage training (detection backbone, then LLM alignment).

**Critical ablation numbers (Fig. 5):**
- Removing forgery features H_F: **−10.12% accuracy**
- Beats fine-tuned LLaVA by **+4.51% F1** on judgment
- Beats DDVQA-BLIP by **+7.74% accuracy**
- Best CIDEr and ROUGE-L for explanation quality

**Why we cite this:** **closest published architecture to ours.** Direct empirical evidence that decoupling detection from explanation improves both detection accuracy and explanation quality. The 10.12% drop when forgery features are removed = direct evidence for our Pass-1 classifier idea.

### IFDL-VLM
- **Repo:** https://github.com/sha0fengGuo/IFDL-VLM
- **Architecture:** Stage 1 detection/localization (SIDA-style) → Stage 2 LLaVA-based explanation
- **Why we cite this:** another two-stage forensic VLM, public code.

### VIGIL: Part-Grounded Structured Reasoning for Generalizable Deepfake Detection (CVPR 2026)
- **arXiv:** 2603.21526
- **Architecture:** 3 progressive stages — SFT for structured format → rejection sampling on hard cases → GRPO with part-aware rewards
- **Numbers:** Stage ablations show +0.9–1.0% per reward component, final 93.3% accuracy
- **Why we cite this:** "plan-then-examine" pipeline. Explicit statement: *"these capabilities cannot be acquired in a single training pass."*

---

## 3. Predict-then-explain ordering (theory)

### Are Training Resources Insufficient? Predict First Then Explain!
- **arXiv:** 2110.02056 (Sun et al. 2021)
- **Claim:** Predict-then-explain (PtE) is more data-efficient and training-efficient than explain-then-predict (EtP)
- **Result:** PtE always more training-efficient than EtP; better when explanation data is scarce; free from exposure bias
- **Why we cite this:** **directly justifies the order of our two-step approach** (verdict first, explanation second). The opposite order (Multimodal-CoT style) requires more explanation supervision.

### Multimodal-CoT (Zhang et al. 2023)
- **arXiv:** 2302.00923
- **Repo:** https://github.com/amazon-science/mm-cot
- **Architecture:** Two-stage — rationale generation → answer inference (this is EtP, opposite of what we want)
- **Numbers:** Beats GPT-3.5 by **+16% (75.17 → 91.68%)** on ScienceQA with <1B model
- **Why we cite this:** establishes that **decoupled two-stage** (regardless of order) helps multimodal reasoning. The +16% delta is the canonical "decoupling helps" number.

### The Unreliability of Explanations in Few-shot Prompting
- **arXiv:** 2205.03401 (Marasovic et al. 2022)
- **Claim:** *"LLMs tend to generate nonfactual explanations when making wrong predictions."*
- **Why we cite this:** direct evidence that **wrong verdict ⇒ unfaithful explanation** in joint generation. Argues for separating the verdict source from the explanation generation.

### ExPred: Explain and Predict, and then Predict Again
- **arXiv:** 2101.04109 (Zhang et al. 2021)
- **Numbers:** +7–47% over end-to-end baselines on Movie Reviews, FEVER, MultiRC by adding task-specific supervision in the explanation stage
- **Why we cite this:** alternative to PtE — supervised explanations work when you have rationale data. Less applicable to us since we use rationales (the GT explanations) only as training targets, not pipeline supervision.

### Two-Stage Reasoning-Infused Learning
- **arXiv:** 2507.00214 (2025)
- **Claim:** Training a Llama-3.2-1B model to output reasoning + label vs label only: **+8.7 absolute percentage points** accuracy on dair-ai/emotion (z=6.88, p<.001)
- **Why we cite this:** modern evidence that adding rationale generation during SFT improves classification. Indirect support for our Pass-2 conditional explanation prompt.

### Two-stage prompting framework with predefined verification (Nature Digital Medicine 2025)
- **DOI:** 10.1038/s41746-025-02146-4
- **Architecture:** Initial Diagnosis → Verification → Final Diagnosis
- **Numbers (MedQA + NEJM):** **+5.2% accuracy, −16.0% uncertainty, +23.3% consistency, −63% incorrect-knowledge errors. +4% over vanilla CoT.**
- **Why we cite this:** "verify-then-decide" loop. Same intuition as our two-stage but in medical domain. The 63% reduction in *factual* errors is most relevant to our entity/facts F1.

---

## 4. Counter-evidence / caveats

### GenCLS++: Generative Classification in LLMs (arXiv 2504.19898, 2025)
- **Claim:** *"Classification tasks often achieve optimal results without explicit reasoning prompts."*
- **Numbers:** +3.46% naive SFT improvement; **CoT prompts hurt** classification compared to direct generation
- **Caveat for our case:** their finding applies to *pure classification* (verdict only). Our task has classification + grounded explanation. Doesn't undermine our Pass-2 conditional explanation, but we should NOT add CoT-style "let me think step by step" before the verdict.

### EACL 2026 — Reasoning Trade-offs at Low FPR
- **PDF:** aclanthology.org/2026.eacl-long.190.pdf
- **Claim:** *"reasoning improves accuracy but systematically degrades recall at low-FPR operating points"*
- **Numbers:** Token-based scoring beats self-verbalized scoring; ensembling Think-On + Think-Off recovers both metrics
- **Caveat:** if we ever optimize for high-precision fake detection (e.g., to avoid false positives on real images), explicit reasoning can hurt. Probably not relevant since challenge eval is balanced.

### Stop Reasoning! When MLLMs with CoT Reasoning Meet Adversarial Images
- **arXiv:** 2402.14899
- **Claim:** CoT reasoning gives only marginal robustness; "stop-reasoning attack" can dissolve the gain
- **Caveat:** not relevant for our adversarial setting (we don't expect adversarial inputs in val).

---

## 5. Foundation models (backbone candidates with size/source)

| Backbone | Params | HF id | Year | Notes |
|----------|--------|-------|------|-------|
| DINOv3-Large | 300M | `facebook/dinov3-vitl16-pretrain-lvd1689m` | Aug 2025 | Released under commercial license, ungated |
| DINOv3-Huge+ | 1B | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | Aug 2025 | Ungated, larger |
| DINOv3-7B | 7B | `facebook/dinov3-vit7b16-pretrain-lvd1689m` | Aug 2025 | **Gated** — needs HF access acceptance |
| SigLIP2-SO400M-384 | 400M | `google/siglip2-so400m-patch14-384` | Feb 2025 | Bombek1 ensemble uses this |
| SigLIP2-giant-opt-384 | 1.1B | `google/siglip2-giant-opt-patch16-384` | Feb 2025 | NTIRE 2026 winners use this |
| SigLIP-Large-384 | 400M | `google/siglip-large-patch16-384` | 2023 | Older |
| MetaCLIP-2-giant | 1B | `facebook/metaclip-2-worldwide-giant` | Jul 2025 | |
| PE-Core-L14-336 | ~330M | facebook/PE-Core (uses vendored code) | Apr 2025 | Best zero-shot AUC on our val (0.726) |
| DINOv2-Large | 300M | `facebook/dinov2-large` | 2023 | Beaten by DINOv3 by 11–30 points |

---

## 6. Numbers we cite for the eventual paper

| Claim | Source | Number |
|-------|--------|--------|
| "Decoupled detection improves explanation quality" | M2F2-Det CVPR 2025 | −10.12% acc when removing forgery features |
| "Predict-then-explain is more data-efficient" | Sun et al. 2021 | qualitative + experiments |
| "Two-stage decoupled multimodal reasoning helps" | Multimodal-CoT 2023 | +16% over GPT-3.5 on ScienceQA |
| "Verdict errors propagate to explanations" | Marasovic et al. 2022 | qualitative |
| "Modern VFMs beat specialized AIGC detectors" | Simplicity Prevails 2026 | DINOv3-Linear 96.5% vs OMAT 94.6% on GenImage |
| "Forgery-aware features matter" | VLAForge CVPR 2026 | +18% AUROC from forgery-aware mask alone |
| "Verification stages reduce factual errors" | Nature Digital Medicine 2025 | −63% incorrect-knowledge errors |
| "DINOv3 > DINOv2 generation gap" | Simplicity Prevails 2026 | +11% on GenImage, +30.4% on in-the-wild |
