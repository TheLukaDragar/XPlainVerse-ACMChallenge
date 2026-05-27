# Problem statement — what the VLM is currently doing wrong

All numbers are from our own runs on `runs/vlm_full/v1-20260524-214014`.

## Training state (snapshot 26 May 2026, 22:00 UTC)

| Metric | Step 1 | Step 3700 (current) |
|--------|--------|---------------------|
| Train loss | 2.677 | 0.871 |
| Token accuracy | 49.3% | 73.0% |
| Progress | 0.004% | **13.2% of 1 epoch** |
| Wall time | — | 1d 22h 59m |
| ETA full epoch | — | +12d 21h |
| GPU memory | — | 19 GB / 80 GB |

## Eval ROUGE-L on 2000 val rows (training-time eval, every 400 steps)

| Step | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 |
|------|---------|---------|---------|--------|
| 400 | 32.32 | 11.54 | 27.51 | 9.76 |
| 800 | 38.92 | 16.25 | 33.77 | 14.20 |
| 1200 | 36.54 | 14.63 | 31.43 | 12.47 |
| 1600 | 39.33 | 16.79 | 34.26 | 14.90 |
| 2000 | 39.42 | 16.74 | 34.30 | 14.75 |
| **2400 (best)** | **40.07** | **17.16** | **34.73** | **14.94** |
| 2800 | 39.79 | 16.73 | 34.58 | 14.75 |
| 3200 | 39.20 | 16.63 | 34.34 | 14.76 |

**ROUGE-L plateaued at step 2400.** Training loss is still decreasing, but downstream metric is flat. This suggests further SFT will mostly memorize style rather than improve coverage.

## Verdict accuracy on 2000 val (ckpt-3200, training-time predict.jsonl)

| Class | Count | Predicted correctly | Rate |
|-------|-------|---------------------|------|
| Real | 922 | 844 | **91.5%** |
| Fake | 1078 | 595 | **55.2%** |
| **Overall** | **2000** | **1439** | **72.0%** |

**Asymmetry: 78 real-called-fake errors vs 483 fake-called-real errors.** Fakes contribute 86% of all errors despite being 54% of the data.

## Style template collapse (our own pred analysis)

| Pattern in prediction text | Count out of 2000 | Rate |
|----------------------------|-------------------|------|
| `"This picture/looks real because…"` opener | 1299 | **65%** |
| Technical fake-style phrases (`"visual inconsistencies"`, `"digital artifacts"`, `"suggest manipulation"`) | 527 | 26% |
| Other | 174 | 9% |

The model has learned two output modes — short simple-real (43-word median in training targets) and long technical-fake (113-word median in training targets) — and **defaults to the easier real mode** when the image is ambiguous.

## Official challenge eval (small sample, ckpt-2400)

Run on 32 val rows via `scripts/eval_checkpoint_one_gpu.sh`:

| Metric | Score |
|--------|-------|
| `complex_overall_score` | **0.391** |
| `complex_entity_f1` | 0.323 |
| `complex_facts_f1` | 0.221 |
| `complex_bert_f1` | 0.649 |
| `simple_overall_score` | 0.496 |
| `simple_bert_f1` | 0.591 |
| `simple_sle_score` | 0.363 |

For reference, the sanity ckpt-100 (8 samples) scored 0.256 complex_overall — so we more than doubled entity/facts F1 over the sanity baseline.

## Training data asymmetry (root cause of mode collapse)

Sampled from first 50k rows of `dataset/train_vlm.jsonl`:

| Class | Count | Mean words | Median | p10 | p90 | Style |
|-------|-------|------------|--------|-----|-----|-------|
| Real targets | 14366 | **43** | 43 | 36 | 51 | Simple "this picture looks real because…" |
| Fake targets | 35635 | **124** | 113 | 83 | 196 | Technical "the image contains…" |

**Fake targets are 2.6× longer than real targets and use a different register.** This bimodal distribution + class imbalance (also 2.5:1 fake:real) creates the conditions for mode collapse during SFT.

## Failure mode summary

1. **Verdict bias toward "real"** when the image doesn't have obvious artifacts (483 fake-called-real errors)
2. **Single-region reasoning** — even when verdict is right, the explanation typically covers 1–2 evidence regions vs the 4–6 regions in GT, capping entity_f1 around 0.32
3. **Coupled failure** — wrong verdict ⇒ wrong style ⇒ explanation argues the opposite case ⇒ entity/facts F1 → 0 on those samples
