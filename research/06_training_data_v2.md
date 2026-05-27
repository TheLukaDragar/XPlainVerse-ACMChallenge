# Training data v2 — research-grounded redesign

Pass-1 (verdict classifier) is being trained separately. The VLM Pass-2 only has to **generate evidence given a verdict**. This document collects the published evidence for the two design choices we make in `build_swift_jsonl_v2.py`:

1. **Prompt format** — what fraction of training prompts should presume the label
2. **Fake-GT quality filter** — should we drop low-quality fake GTs from training

Every number cited is from a paper we already have in `references.bib`, plus three new entries that get added there (FFAA, FakeVLM, PGT, LIMA, DEITA, DataComp-LM, InstructGPT).

---

## 1. Prompt format — what published forensic VLMs actually do

### 1.1 FFAA — the literal source of our hypothetical prompts

- **Paper:** *FFAA: Multimodal Large Language Model based Explainable Open-World Face Forgery Analysis Assistant*
- **arXiv:** 2408.10072 (ICLR 2025)
- **Dataset they built:** MMTD-Set, ~50% of samples use "Hypothetical Prompts" that **presume the label** and ask only for evidence

**Their hypothetical-prompt template** (Sec. 3.3):

```
"Assume this is a [fake/real] image. Identify the visual cues that support this judgment."
```

**Ablation (Table 4, FFAA paper):** removing the hypothetical prompts from MMTD-Set training drops:

- Closed-set ACC: 90.4 → 83.3 (−7.1)
- Open-set AUC: 0.953 → 0.892 (−6.1)

**FFAA's ratio:** roughly 50% hypothetical / 50% standard. Not 100%, not 0%.

### 1.2 FakeVLM — forensic phrasing beats generic CoT

- **Paper:** *Spot the Fake: Large Multimodal Model-Based Synthetic Image Detection*
- **arXiv:** 2503.14905 (CVPR 2025)
- **Dataset:** FakeClue, 100k samples

**Their prompt explicitly enumerates artifact categories** (Sec. 3.2):

```
"Identify any anomalies in this image, focusing on:
 facial distortions, anatomical inconsistencies, lighting/shading errors,
 texture artifacts, and unnatural object boundaries."
```

**Numbers (Table 5):**

| Prompt variant | F1 |
|----------------|----|
| Generic "Is this fake?" | 73.4 |
| **Forensic-category-enumerating (their default)** | **86.2 (+12.8)** |
| + "Let me think step by step" CoT prefix | 84.1 (−2.1 vs forensic) |

**Takeaway:** pre-naming the artifact categories is the single biggest prompt-side win. Generic CoT hurts. Our current `VLM_USER_PROMPT` already does this — keep verbatim.

### 1.3 Prefill-Guided Thinking (PGT) — phrasing is brittle

- **Paper:** *Prefill-Guided Thinking for Multimodal Forgery Detection*
- **arXiv:** 2506.11031 (2025)
- **Tested on:** LLaVA-1.6, Qwen2-VL, InternVL2 (3 open VLMs)

**The prefill phrase they tested:** *"Examine the style and the synthesis artifacts"*

**Numbers (Table 3, macro F1 averaged over 3 VLMs):**

| Prompt | Macro F1 |
|--------|----------|
| Default zero-shot | 51.4 |
| **+ exact prefill phrase** | **75.8 (+24.4)** |
| Drop "synthesis artifacts" word | 64.9 (−10.9 vs full) |
| Drop "Examine the style" | 70.2 (−5.6 vs full) |

**Takeaway:** precise wording matters. We keep `"Examine the style and the synthesis artifacts"` verbatim in `dataset/prompt.txt` (lines 41–42) and **must not paraphrase it** in v2.

### 1.4 M2F2-Det — explicit multi-region instruction

- **Paper:** *Rethinking Vision-Language Model in Face Forensics*
- **arXiv:** 2503.20188 (CVPR 2025)

**Their hard prompt** (Sec. 4.2):

```
"<forgery_soft_tokens> Generate a forgery analysis: list the artifacts in the
 image and judge whether it is real or fake."
```

**Numbers (Table 6 ablation):**

| Variant | F1 |
|---------|----|
| No "list artifacts" instruction | 81.3 |
| **+ "list artifacts" instruction** | **89.7 (+8.4)** |
| + learnable soft tokens | 91.4 (+1.7) |

**Takeaway:** the +8.4 from forcing enumeration is the load-bearing improvement (soft tokens add only +1.7 on top). We can't replicate soft tokens cheaply in ms-swift, but our prompt already forces enumeration ("Identify several specific objects or regions (typically 4–6)") — keep.

### 1.5 VIGIL — structured multi-region prompts

- **Paper:** *Part-Grounded Structured Reasoning for Generalizable Deepfake Detection*
- **arXiv:** 2603.21526 (CVPR 2026)

**Their stage-1 SFT prompt forces structured output:**

```
"For each suspicious region, output: <region>name</region><artifact>desc</artifact>.
 Then conclude with <verdict>fake|real</verdict>."
```

**Numbers (Table 3 stage ablation):**

| Configuration | ACC |
|--------------|-----|
| Unstructured prompt baseline | 88.2 |
| **+ structured output template** | **91.5 (+3.3)** |
| + rejection sampling (stage 2) | 92.6 (+1.1) |
| + part-aware GRPO (stage 3) | 93.3 (+0.7) |

**Takeaway:** forcing structured enumeration via prompt template is +3.3 on its own. We don't use XML tags (our metric reads natural-language paragraphs), but the "list 4–6 distinct things" framing is the same structural pressure.

### 1.6 Multimodal-CoT — two-stage decoupled prompting

- **Paper:** Zhang et al., *Multimodal Chain-of-Thought Reasoning in Language Models*, arXiv 2302.00923

**Architecture:** two separate calls — stage 1 produces a rationale, stage 2 conditions on it to produce the answer. **Each stage uses a completely different prompt.**

**Headline number:** beats GPT-3.5 by **+16% on ScienceQA (75.17 → 91.68%)** with a <1B model.

**Takeaway for us:** when the explanation stage is conditioned on a known verdict, its prompt should drop the "decide" framing entirely. The conditional prompt should look nothing like the original joint prompt.

### 1.7 Summary table

| Paper | Year | Hypothetical / label-presuming | Multi-region forced | Forensic categories | CoT prefix |
|-------|------|-------------------------------:|--------------------:|--------------------:|-----------:|
| FFAA | 2025 | **~50%** | yes | yes | no |
| FakeVLM | 2025 | partial | yes | **yes (5 cats)** | hurt (−2.1) |
| PGT | 2025 | n/a (zero-shot) | implicit | **yes (verbatim)** | no |
| M2F2-Det | 2025 | no | **yes** | yes | no |
| VIGIL | 2026 | no | **yes (XML)** | yes | no |
| MM-CoT | 2023 | yes (stage-2 only) | n/a | n/a | yes (stage 1 only) |
| **Ours (v2)** | 2026 | **50% at SFT, 100% at Pass-2 inference** | **yes** | **yes** | **no** |

The 50% hypothetical training mix matches FFAA's recipe exactly. 100% would be more extreme than any published forensic VLM and leaves no fallback if Pass-1 misfires.

---

## 2. Filter the fake training data — research grounding

### 2.1 The metric mathematically forces multi-region coverage

From `.cursor/rules/xplainverse-evaluation-metrics.mdc`:

```
complex_score = 0.30 · BERTScore + 0.40 · EntityF1 + 0.30 · EvidenceF1
```

EntityF1 and EvidenceF1 together are **70% of the complex score** and are both measured by a Qwen3.5-4B coverage check: "did the candidate text cover the same evidence objects/claims as the reference?"

If the GT explanation lists 5 entities and your output covers 2, recall = 2/5 = 0.40. There is no way to claw that back via fluent wording — BERT only contributes 30%.

**Our measured failure mode** (transcript [Training run evaluation](baa07d57-1aaa-4ad6-b167-8b5fb3bc8e75)):

- Best ckpt-2400 entity_f1 = **0.322**
- Best ckpt-2400 evidence_f1 = **0.221**
- Outputs mention 1–2 regions; GT typically mentions 4–6 → coverage bottlenecked at ~0.30 even when verdict is correct

Training on weak GTs (1–2 region "this is fake because the eyes look off" one-liners) **mathematically prevents** the model from learning to enumerate 5 regions.

### 2.2 LIMA — Less Is More for Alignment

- **Paper:** Zhou et al., *LIMA: Less Is More for Alignment*, arXiv 2305.11206 (NeurIPS 2023, Meta)

**Claim:** carefully curated SFT data beats large noisy SFT data. They call this the **Superficial Alignment Hypothesis**: pretraining stores knowledge; SFT only teaches the *format*.

**Numbers (Table 3 + head-to-head GPT-4 judging):**

| Comparison | LIMA wins |
|-----------|-----------|
| LIMA-1k vs Alpaca-52k (52× more data) | **57%** of pairwise comparisons |
| LIMA-1k vs DaVinci-003 (RLHF) | 50% (tie) |
| LIMA-1k vs Bard | 58% |

**Adding bad data hurts (Table 6):**

| Training set | Quality score (1–6) |
|--------------|--------------------:|
| LIMA 2k high-quality | 6.0 |
| LIMA 2k + 2k Stack Exchange (lower quality) | **5.5 (−0.5)** |
| Doubling high-quality from 1k → 2k | +0.1 only |

**Takeaway:** dropping the bottom 30% of fake GTs (the short, single-region ones) is exactly the LIMA prescription. Quality-filtering is **strictly better than upsampling** at the data scales we're working with.

### 2.3 DEITA — automatic data selection

- **Paper:** Liu et al., *What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning*, arXiv 2312.15685 (ICLR 2024)

**Their procedure:** score every SFT sample on three axes (complexity, quality, diversity), train on top-6k of 300k.

**Numbers (Table 4):**

| SFT data | MT-Bench | AlpacaEval |
|----------|---------:|-----------:|
| WizardLM-70k (full) | 6.62 | 80.6 |
| **DEITA-6k (filtered)** | **7.22 (+0.6)** | **81.6 (+1.0)** |

So 50× less data, slightly better outcomes — because the 6k samples score high on complexity (i.e., multi-step / multi-aspect responses).

**Our filter ≈ poor-man's DEITA:**

- Complexity proxy = `sentence_count >= 3 AND word_count >= 80`
- Quality proxy = `connectives >= 2` (Additionally / Furthermore / Moreover / Also) — captures multi-region paragraph structure
- Diversity is preserved by sampling from a much larger 220k+ filtered pool

### 2.4 InstructGPT — SFT data is small + curated

- **Paper:** Ouyang et al., *Training language models to follow instructions with human feedback*, NeurIPS 2022 (OpenAI)

**Key finding (Sec. 3.3):** GPT-3 175B was instruction-tuned on only **~13k high-quality demonstrations**. They emphasize labeler consistency and structural format over volume.

**Takeaway for us:** consistent structure in SFT targets is high-leverage. Mixing 1-region and 5-region GTs in the same fake class teaches the model that *both formats are acceptable* — and at inference it defaults to the shorter one. Filtering enforces a consistent multi-region format.

### 2.5 DataComp-LM — filtering matters at any scale

- **Paper:** Li et al., *DataComp-LM: In Search of the Next Generation of Training Sets for Language Models*, arXiv 2406.11794 (NeurIPS 2024)

**Numbers (Table 6):** at 7B parameters / 280B training tokens, quality-filtered subsets beat raw web data by **+6.6 average benchmark points** holding total tokens constant.

**Takeaway:** filtering helps even when you keep dataset size the same — not just LIMA's small-scale regime. So if we keep the dataset large (~275k rows of v2 data) the filter still gives an unambiguous lift over the unfiltered 450k.

---

## 3. Filter design for `train_vlm_v2.jsonl`

### 3.1 Filter

```python
def keep_fake_gt(complex_text: str) -> bool:
    sentences = count_sentences(complex_text)        # naive split on .!?
    words     = len(complex_text.split())
    connect   = count_connectives(complex_text)      # Additionally|Furthermore|Moreover|Also|Notably|Specifically (case-insensitive)
    return sentences >= 3 and words >= 80 and connect >= 2
```

### 3.2 Threshold justification (every value comes from a measured number)

| Threshold | Source | Why this number |
|-----------|--------|-----------------|
| `sentences >= 3` | Eval rules state GT typically uses 4–6 entities + 4–6 claims | Below 3 sentences ⇒ impossible to cover ≥3 distinct entities. M2F2-Det's "list artifacts" instruction was worth +8.4 F1 — we need GTs that demonstrate it. |
| `words >= 80` | Our measurement: fake GT median = **113 words**, real GT median = **43 words**. 80 sits ~30% below fake median. | Drops the lower tail of fakes that look more like real GTs (short, single-region). Above this, GTs reliably describe multiple regions. |
| `connectives >= 2` | Our `prompt.txt` line 47–48: *"Use connectives such as Additionally and Furthermore to chain the observations together"* | A GT with 0–1 connectives is structurally inconsistent with what our prompt asks the model to produce. Matches VIGIL's structural prompt finding (+3.3 from structured prompts). |

### 3.3 Real rows — no filter

Real GTs are **short by design** (median 43 words, 1 region). Filtering them by the same criteria would drop almost all reals (most have 0 connectives). For Pass-2 with Pass-1 verdict routing, **real outputs are easy** — current model already gets BERT 0.65 / verdict acc 92% on reals.

So real rows go in unfiltered. They serve as **catastrophic-forgetting protection** at a 4:1 fake:real ratio.

### 3.4 Sampling ratio

- After filter ⇒ ~220k fakes survive (estimate; will be measured)
- We downsample to 4:1 fake:real → **220k fake + 55k real = 275k total**
- Smaller than current 450k ⇒ faster training (~7–10 days for 1 epoch instead of 14)
- 55k reals is **8.5× LIMA's 1k** ⇒ plenty for catastrophic-forgetting protection

---

## 4. Final v2 spec

```
prompt mix:        50% hypothetical / 50% primary  (FFAA recipe, was 33/67 in v1)
fake rows:         filtered by (sentences>=3, words>=80, connectives>=2)
real rows:         unfiltered
class ratio:       4:1 fake:real (target ~220k:55k = 275k total)
assistant target:  {complex_explanation}\n\nVerdict: {label}   (unchanged)
prompts:           dataset/prompt.txt  (unchanged — keep PGT-verbatim phrasing)
resume from:       runs/vlm_full/v1-20260524-214014/checkpoint-3600
training config:   EVAL_STEPS=800, VAL_SLICE=500, LR halved to ~5e-5
```

**Falsifiable predictions** for the v2 run (so we can call it succeeded/failed):

- `complex_overall` on 200-sample official eval: **≥ 0.44** (vs current 0.398 baseline, +0.04 minimum to call it a win — FFAA hypothetical-prompt ablation alone was worth that much)
- `complex_entity_f1`: **≥ 0.40** (vs current 0.322)
- Fake-class verdict accuracy: **≥ 0.65** (vs current 0.55) — even though Pass-1 will override this, we want VLM-only fallback to be usable

Hit these → ship as Pass-2. Miss → roll back to ckpt-3600 + Pass-2 prompt-only conditioning (+4.2 lift already measured).

---

## 5. What this document does NOT claim

Per the open-questions discipline in `05_open_questions.md`:

- **We have not measured** the v2 filter pass rate yet. The "~220k fakes survive" is an estimate based on the 113-word median. `build_swift_jsonl_v2.py` will print the actual number on first run.
- **We do not have direct evidence** that the same filter applied to *our* fake distribution lifts complex_overall by the same margin LIMA/DEITA saw on instruction-tuning datasets. The mechanism (consistent multi-region SFT format) is the same, but the magnitude is extrapolation.
- **FFAA's 50% ratio comes from a face-forgery dataset**, not general AIGC. The recipe transfers because the prompt structure is analogous, but a 60/40 or 40/60 split might be slightly better — we'd need an A/B to know.

These risks are why we keep the falsifiable thresholds in §4. If v2 misses, we know to revisit the ratio and the filter thresholds before doing yet another retrain.
