# GRPO reward plan — XPlainVerse VLM stage

This document captures **why** GRPO is worth doing for our pipeline, **how much gain to expect**, and a **concrete reward design** aligned with the official evaluator.

Related:
- Scoring rules: `.cursor/rules/xplainverse-evaluation-metrics.mdc`
- Official evaluator: `evaluation/evaluate_val.py`
- Train JSONL already carries `sample_id` and `label` (required for ORM kwargs)

---

## 1. Problem statement

**SFT optimizes token likelihood / rouge.** The leaderboard optimizes:

```
complex_overall = 0.30 · BERT + 0.40 · EntityF1 + 0.30 · EvidenceF1
simple_overall  = 0.70 · BERT + 0.30 · SLE_norm
```

Entity + evidence = **70% of complex score**. Both require Qwen3.5-4B to extract evidence objects and atomic claims, then check bidirectional coverage.

**Observed SFT gap (100-step sanity smoke, 8 samples):**
- Verdict match: ~6–7/8 (often correct)
- Entity/facts recall on fake rows: ~0.35–0.65 when verdict correct (2/5 GT objects typical)
- Wrong-verdict rows: ~0.15–0.25 overall

SFT teaches *style* and *average length*; it does **not** directly optimize “mention every GT evidence object with a specific claim.” GRPO only helps if the **reward measures that**.

---

## 2. What the literature says (realistic expectations)

| Setting | Typical gain over SFT | Judge requirement |
|---------|----------------------|-------------------|
| Math/code GRPO (DeepSeek-R1) | Large on target domain | **Exact** verifiable checks (boxed answer, unit tests) |
| General reasoning GRPO ([Scalpel vs Hammer](https://arxiv.org/pdf/2507.10616)) | Modest in-domain, small OOD drift | Rule-based, well-aligned |
| Small-model math GRPO (misaligned format reward) | **Negative** | Bad proxy → reward hacking |
| VLM domain GRPO ([S-GRPO](https://arxiv.org/html/2604.16557)) | ~1–3% on domain metrics | Task metric aligned |
| **XPlainVerse (this project)** | **+0.05–0.10 absolute on `complex_overall`** | Must track entity + claim coverage |

**Bottom line:** A good judge is **critical** for us. A bad judge (BERT-only, format-only, noisy self-extract) often gives **zero or negative** gain. GRPO **amplifies** existing capability; it does not invent vision from scratch.

---

## 3. Design principles

1. **Align online reward with official metric weights** — same 0.3 / 0.4 / 0.3 structure where possible.
2. **Pre-cache everything on the GT side** — extractions from train complex text are fixed; never re-extract GT during training.
3. **Minimize Qwen calls per completion** — full eval is 4 calls (extract×2 + coverage×2); online GRPO should use **1–2 calls**.
4. **Local for cheap signals** — verdict parse, format, BERTScore, length gates run on train GPU / CPU.
5. **Remote async for slow signals** — Qwen coverage via OpenAI-compatible API on the fast machine (`AsyncORM`).
6. **Full `evaluate_val.py` offline only** — calibration every N checkpoints, not every step.
7. **Anti-hacking guards** — penalize generic fluff, wrong length class, missing `Verdict:`, hallucination patterns.

---

## 4. Reward stack (recommended)

### Tier A — Local, free (every completion)

| Signal | Weight in composite | Implementation |
|--------|---------------------|----------------|
| **Verdict match** | 0.10 | Parse `Verdict: real\|fake`, compare to dataset `label` |
| **Format compliance** | 0.05 | Must end with `\n\nVerdict: {label}`; no `<reasoning>` tags |
| **Length class** | 0.05 | Fake: target 80–180 words; Real: 25–70 words (soft penalty outside band) |
| **Anti-generic penalty** | −0.05 max | Regex hit list: `AI-generated`, `artifacts are visible`, `looks synthetic` without named object |

These are **gates**, not the main signal — they stop GRPO from collapsing to one-liners or wrong-class templates.

### Tier B — Local BERT (batched on train node)

| Signal | Weight | Notes |
|--------|--------|-------|
| **Complex BERT F1** | 0.15 | Same DeBERTa model as evaluator; compare completion (minus verdict line) to cached GT complex text |

Reuse `evaluation/utils/llm_helpers.get_bert_scorer()`. Batch across the GRPO group for efficiency.

### Tier C — Remote Qwen (async API on reward machine)

| Signal | Weight | Qwen calls | Notes |
|--------|--------|------------|-------|
| **Entity recall (GT→pred)** | 0.25 | 1 coverage | Cached GT extraction JSON + candidate complex text |
| **Evidence recall (GT→pred)** | 0.20 | *(same call)* | From same coverage response: `claim_coverage` |
| **Precision (pred→GT)** | 0.15 | 1 coverage *(optional)* | Extract pred once per unique completion; coverage vs GT text |

**Default online (2 Qwen calls):**
1. Coverage: `cached_gt_json` + candidate → entity_coverage, claim_coverage (**recall**)
2. Coverage: `pred_extraction_json` + gt_complex_text → entity_coverage, claim_coverage (**precision**)

**Cheaper ablation (1 Qwen call):** skip pred extraction + precision; use recall-only proxy:

```
entity_proxy = entity_coverage_gt_to_pred
evidence_proxy  = claim_coverage_gt_to_pred
entity_f1 ≈ entity_proxy   (assume precision ≈ 1 early in training)
evidence_f1  ≈ evidence_proxy
```

Start with 1-call mode for micro-runs; enable 2-call mode once throughput is benchmarked.

### Composite reward formula

Mirror official complex score:

```python
def complex_reward(r):
    return (
        0.30 * r.bert_f1
        + 0.40 * harmonic_mean(r.entity_recall, r.entity_precision)
        + 0.30 * harmonic_mean(r.evidence_recall, r.evidence_precision)
    )
```

Add Tier A gates **before** mixing (multiply or subtract):

```python
reward = complex_reward(r)
reward *= (0.5 + 0.5 * verdict_match)          # 0 if wrong verdict, full if correct
reward += format_bonus + length_bonus - generic_penalty
reward = clip(reward, 0.0, 1.0)
```

**Optional:** expose sub-scores to W&B for debugging (`reward/entity_recall`, etc.).

---

## 5. What NOT to use as primary reward

| Bad reward | Why it fails |
|------------|--------------|
| Rouge / cross-entropy proxy | Already optimized by SFT; ignores entity recall |
| BERT only | Fluent generic paragraphs; entity F1 flat |
| Verdict + format only | +0.01–0.03 on real metric |
| Full 4-call eval sequential | Too slow; ~minutes per GRPO step |
| Self-extract GT each step | Non-stationary, expensive, drifty |
| Pred-side extract without precision | Model learns to spam object names without claims |

---

## 6. Architecture

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Train machine (GRPO)       │         │  Reward machine (fast API)    │
│  Qwen3-VL + vLLM rollouts   │  HTTP   │  vLLM serve Qwen3.5-4B        │
│  ms-swift AsyncORM plugin   │ ──────► │  OpenAI /v1/chat/completions  │
│  BERT + verdict (local)     │  async  │  extraction + coverage only │
└─────────────────────────────┘         └──────────────────────────────┘
         │                                           ▲
         │  reads                                    │  precomputed once
         ▼                                           │
  cache/train_gt_extractions.jsonl ──────────────────┘
  (sample_id → diagnostic_entities, evidence_claims)
```

**Env vars (reward plugin):**
```bash
export XPLAINVERSE_REWARD_BASE_URL=http://REWARD_HOST:8000/v1
export XPLAINVERSE_REWARD_MODEL=Qwen/Qwen3.5-4B
export XPLAINVERSE_GT_CACHE=evaluation/cache/train_gt_extractions.jsonl
export XPLAINVERSE_REWARD_MODE=recall_only   # or full_f1
```

---

## 7. Pre-computation (do before GRPO)

### 7.1 `evaluation/precompute_gt_extractions.py`

For every **train** row (450k):
- Input: `sample_id`, GT complex text (from manifest / train JSONL labels)
- Output: JSONL `{sample_id, diagnostic_entities, evidence_claims}`
- Run once with same extraction prompt as `evaluation/prompts/semantic_extraction_prompt.txt`
- Backend: local vLLM or remote API (batch overnight)

Val cache (110k) optional — useful for offline calibration, not required for train GRPO.

### 7.2 `evaluation/cache/gt_complex_text.jsonl`

Map `sample_id → complex_explanation` for BERT and precision coverage. Can be derived from train manifest in the same script.

---

## 8. Implementation checklist

| # | File | Purpose |
|---|------|---------|
| 1 | `evaluation/precompute_gt_extractions.py` | One-time GT entity/claim cache |
| 2 | `external_plugins/xplainverse_rewards.py` | `AsyncORM` + local ORMs registered in `orms` |
| 3 | `scripts/benchmark_reward_api.sh` | Measure latency @ batch 8/16/32 on reward machine |
| 4 | `scripts/train_vlm_grpo.sh` | GRPO from SFT LoRA checkpoint |
| 5 | `scripts/eval_checkpoint.sh` | infer → submission → `evaluate_val.py` subset | **done** |
| 6 | `evaluation/build_submission.py` | Parse `Verdict:` from infer JSONL | **done** |
| 7 | `scripts/serve_reward_judge.sh` | vLLM OpenAI server for Qwen3.5-4B judge | **done** |

### ms-swift registration (sketch)

```python
# external_plugins/xplainverse_rewards.py
from swift.rewards import ORM, AsyncORM, orms

class VerdictORM(ORM): ...
class FormatORM(ORM): ...
class LengthORM(ORM): ...
class BertComplexORM(ORM): ...
class XPlainVerseCoverageAsyncORM(AsyncORM): ...

orms['xplainverse_verdict'] = VerdictORM
orms['xplainverse_format'] = FormatORM
orms['xplainverse_length'] = LengthORM
orms['xplainverse_bert'] = BertComplexORM
orms['xplainverse_coverage'] = XPlainVerseCoverageAsyncORM
```

Train command (later):
```bash
swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --adapters runs/vlm_full/.../checkpoint-best \
  --dataset dataset/train_vlm.jsonl \
  --external_plugins external_plugins/xplainverse_rewards.py \
  --reward_funcs xplainverse_verdict xplainverse_format xplainverse_bert xplainverse_coverage \
  --use_vllm true \
  --vllm_mode server \
  ...
```

Dataset kwargs available in ORM: `sample_id`, `label` (already in JSONL).

---

## 9. Phased rollout

### Phase 0 — SFT baseline (current)
- Finish `train_vlm_full.sh`
- Run `eval_checkpoint.sh` on 500–2000 val rows → record `complex_overall`, entity/facts F1

### Phase 1 — Reward infra (no GRPO yet)
- Precompute train GT extractions
- Benchmark reward API: target **< 2s per completion** @ 1 coverage call, batch 16
- Unit-test ORM on 20 fixed completions vs `evaluate_val.py` scores (correlation > 0.85)

### Phase 2 — Micro GRPO (50 steps, 500–2k train subset)
- 1-call recall-only reward
- `num_generations=4`, small LR (`5e-6`), KL coef default
- Compare micro-run checkpoint vs SFT on same 500 val rows
- **Success criterion:** entity F1 +0.03–0.05 absolute without verdict collapse

### Phase 3 — Full GRPO
- 450k train (or stratified 100k if too slow)
- Enable 2-call full F1 if API keeps up
- Offline full eval every 200–500 steps

### Phase 4 — Compressor (separate)
- SFT compressor first (fake complex → simple)
- Optional small GRPO with local rewards only:
  - Simple BERT vs GT simple (0.7)
  - SLE norm (0.3)
  - Hard cap: max 60 words, no technical jargon list

---

## 10. Calibration protocol

After each GRPO checkpoint:

1. Infer 1000 val images → `build_submission.py`
2. Run `evaluate_val.py` (full metric)
3. Log scatter: `online_reward` vs `complex_overall` per sample
4. Track **rank correlation** — if < 0.7, fix reward before continuing

| Metric | SFT baseline (fill after full SFT) | GRPO target |
|--------|-----------------------------------|-------------|
| `complex_overall` | TBD | +0.05–0.10 |
| `complex_entity_f1` | TBD | +0.10–0.15 |
| `complex_evidence_f1` | TBD | +0.08–0.12 |
| `complex_bert_f1` | TBD | +0.02–0.05 |
| Verdict accuracy | TBD | maintain ±1% |

---

## 11. Throughput budget

Rough GRPO step (group size G=4, batch B=1 prompt):
- 4 completions × (1–2 Qwen coverage + optional pred extract)
- At 2s/call, 8 calls ≈ 16s reward + rollout time

If too slow:
- Reduce `num_generations` to 2 for debugging
- Use `recall_only` mode
- Increase API batch concurrency (`asyncio.Semaphore(32)`)
- Subsample train to 50–100k stratified by label

---

## 12. Default recommendation (TL;DR)

**Use this reward mix for VLM GRPO:**

| Component | Share of signal | Where |
|-----------|-----------------|-------|
| GT→pred entity + facts coverage (Qwen) | **~50%** | Remote async |
| Complex BERT vs GT | **~15%** | Local |
| Verdict + format + length | **~20%** | Local |
| Pred→GT precision (optional) | **~15%** | Remote async |

**Do first:** precompute GT cache → benchmark API → micro GRPO 50 steps → full eval to validate +0.03 entity F1.

**Do not:** run full `evaluate_val.py` every GRPO step.

---

## 13. Open decisions

| Decision | Recommendation | Alternative |
|----------|----------------|-------------|
| 1 vs 2 Qwen calls | Start 1-call recall-only | 2-call full F1 when API proven |
| Train on full 450k vs 100k | 100k stratified first | Full 450k if throughput OK |
| GRPO after 1 epoch SFT vs best checkpoint | Best val rouge / offline entity checkpoint | Latest epoch |
| Include simple in VLM GRPO | **No** — compressor is stage 2 | Joint training (complex) |

---

*Last updated: 2026-05-24. Revise after first SFT baseline eval numbers are recorded in §10.*
