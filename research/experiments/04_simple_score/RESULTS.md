# Simple-score exploration — results

Simple overall = `0.7·BERT_f1 + 0.3·SLE_norm`, `SLE_norm = (clip(SLE,-1,4)+1)/5`.
Leaderboard simple overall (sub 795187) = **0.677**.

## A. Structural ceiling (oracle text, 1000 fake + 1000 real val)

Pred = GT text (fake→GT simple, real→GT complex), so BERT≈1.0. Isolates SLE + real shorten tradeoff.

| policy | class | BERT | SLE_norm | simple | weighted |
|--------|-------|------|----------|--------|----------|
| oracle | fake | 1.000 | 0.501 | 0.850 | |
| oracle | real | 1.000 | 0.475 | 0.843 | **0.847** |
| real_firstsent | real | 0.860 | 0.635 | 0.792 | 0.824 |
| real_w35 | real | 0.926 | 0.554 | 0.814 | 0.834 |
| real_w25 | real | 0.845 | 0.687 | 0.798 | 0.826 |

**Takeaways**
- **Structural ceiling ≈ 0.847** with perfect text. Actual 0.677 → **~0.17 headroom is all text quality**, not pipeline structure.
- **Shortening real simple HURTS**: BERT loss (70%) > SLE gain (30%). Keep reals = copy complex.
- Even perfect fake simple caps SLE_norm at ~0.50 → GT simple is not very "simple". A metric-aware
  compressor can exceed this by trading wording for simplicity while holding BERT.

## B. Compressor prompt A/B/C (real model outputs, 200 fakes, GT complex input)

`compressor_vl/checkpoint-10000`, greedy, max_new_tokens=128.

| prompt | BERT | SLE_raw | SLE_norm | simple |
|--------|------|---------|----------|--------|
| baseline (prompt_v2) | 0.7378 | 1.711 | 0.542 | 0.6791 |
| tworeasons (~45w, 2 obj) | 0.7352 | 1.829 | 0.566 | 0.6844 |
| **ultrasimple (~20w, child)** | **0.7418** | **1.913** | **0.583** | **0.6940** |

**Takeaways**
- Baseline compressor on oracle complex = **0.679** ≈ leaderboard 0.677 → compressor output is the binding constraint.
- **`ultrasimple` prompt = +0.015 simple on fakes, no retrain** — gains on BOTH SLE (+0.04) and BERT (+0.004).
  Shorter, plainer outputs match the short GT simple better (higher F1) and read simpler.
- Prompt swap is off-SFT-distribution but the model still follows it. Zero detection risk.

## Recommendations

1. **Now (free):** swap compressor inference prompt to `ultrasimple` in the submission pipeline.
   Expected ≈ +0.008 simple overall (fakes are 60k/110k) → ≈ +0.002 challenge total.
2. **Real images:** keep copy-complex (shortening hurts). Real simple quality rides on complex quality.
3. **Big lever (later):** retrain compressor with **GRPO**, reward = `0.7·BERT(vs GT simple) + 0.3·SLE`.
   SFT only mimics GT (caps at the 0.85 oracle ceiling); GRPO optimizes the metric directly and can push
   SLE above the GT's own ~0.50 while holding BERT. SFT v2 with cleaner/shorter targets is a smaller step.

## Repro

```bash
python3 research/experiments/04_simple_score/build_probe.py --n-per-class 1000
LJ_GPU_GRES=gpu:1 ./scripts/lj_gpu_exec.sh bash research/experiments/04_simple_score/run_probe_lj.sh
# compressor A/B/C (GPU node, eval container):
srun ... ~/xplainverse_exec.sh bash research/experiments/04_simple_score/run_compressor_promptab_lj.sh
```
