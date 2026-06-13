# XPlainVerse — Submission Log & Plan

Single source of truth for what we submitted, leaderboard scores, and reusable assets.
(Leaderboard numbers come from Slack agent DMs; pipeline pred-fake from `pipeline_summary.json`.)

## Leaderboard results (CodaBench, FRI team = theluka)

Scorer: `abhijeet1317/xdd-scorer:2026-v6` (Overall = detection + explanation).

| # | Operating point | Pred fake % | Detection F1 | Notes |
|---|---|---:|---:|---|
| 1 | `p_fake_orig @ 0.084` (deployed Bombek-arch ensemble, train-only) | 39.1% | **0.690** | first full pipeline; FRI ~#9 |
| 2 | `p_fake_mean @ 0.11` (flip-patch, 9,148 regen) | 35.9% | **~0.698** | calibration didn't transfer |
| 3 | **warmstart `p_fake_mean @ 0.49`** (2M external, flip-patch 42,325 regen) | 41.2% | **0.848966** | submission 795187, 2026-06-13 16:08 — **+0.16 jump** |

### Full board snapshot — 2026-06-13 ~16:12 CEST (Final Results)

| # | Team | Overall | Det F1 | Det Acc | Complex BERT | Simple Overall | Explanation |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | HIT VIRLAB (hit_xiaolun) | 0.81083 | 0.920087 | 0.92088 | 0.698534 | 0.704611 | 0.701573 |
| 2 | xz (xuzhu) | 0.807453 | 0.881308 | 0.88253 | 0.69702 | 0.770177 | 0.733598 |
| 3 | XJTU GenAI (xinyiyin) | 0.794198 | 0.877939 | 0.87916 | 0.692885 | 0.728029 | 0.710457 |
| 4 | ZKRJ (gongj) | 0.78303 | 0.904072 | 0.904185 | 0.688374 | 0.635603 | 0.661988 |
| 5 | MSUteam (artfil) | 0.774954 | 0.883572 | 0.883885 | 0.694615 | 0.638058 | 0.666336 |
| **6** | **FRI (theluka)** | **0.771605** | **0.848966** | **0.84918** | **0.711251** | **0.677236** | **0.694243** |

**Big result:** the 2M-external warmstart detector lifted Detection F1 **0.690 → 0.849 (+0.159)**,
moving FRI from ~#9 to **#6 overall**. Confirms the old diagnosis: the gap was a real
generalization/data problem, not architecture or threshold. More/diverse data was the lever.

**Our complex BERT (0.711) is now the BEST on the board** (next: artfil 0.6946, hit 0.6985).
Detection is still the only remaining gap: leaders are 0.88–0.92 vs our 0.849 (~0.03–0.07 behind).
Simple Overall (0.677) is mid-pack — xz 0.770, xinyiyin 0.728 are ahead → a simple-explanation
upgrade is the cheapest remaining explanation lever.

Diagnosis (notes 2026-06-07/08, now partly resolved): val macro F1 ~0.86 but old leaderboard
~0.69 came from val→test **score shift** + real generalization gap. The external-data retrain
closed most of it.

## Models (holdout macro F1 = 1k unseen; leaky for pooled models)

| Model | Holdout macro F1 | Trained on val? | Test signal |
|---|---:|---|---|
| Trainval resume (XP-only) | **0.972** | yes (pooled) | none |
| **External warmstart** (2M ext + val) | **0.970** | yes (pooled) | none |
| Trainval / external scratch | 0.944–0.948 | yes | none |
| Deployed Bombek-arch (train-only) | 0.852 | no | **0.690 leaderboard** |
| Retrain macro-F1 / giant | 0.834–0.860 (full val) | no | not submitted |

Warmstart calibration (holdout TTA): best `p_fake_mean @ 0.49` → holdout macro F1 0.972,
test pred-fake 41.2%. See `~/luka/runs/pass1_calibration/warmstart_holdout_*.json`.

## Reusable assets (avoid recompute)

| Asset | Path |
|---|---|
| Base complex (200k, deployed verdicts) | `~/luka/runs/pass2_test_complex/20260606-195337/complex_explanations.jsonl` |
| Base compressor infer (78,257 fakes) | `~/luka/runs/compressor_test/20260607-080830/compressor_infer.jsonl` |
| Calibrated submission.jsonl (200k complex+simple) | `~/luka/runs/submission_calibrated_mean011_20260608-105939/submission.jsonl` |
| Warmstart test TTA preds | `~/luka/runs/pass1_test_tta_warmstart/predictions.parquet` |
| Pass-2 GRPO adapters | `~/luka/runs/vlm_v2_grpo/job_48855/checkpoint-26400` |
| Compressor adapters | `~/luka/runs/compressor_vl/checkpoint-10000` |

## Submission builds on disk

| Dir | Detector | Pred fake % | Zip | Status |
|---|---|---:|---|---|
| `submission_calibrated_mean011_20260608-105939` | mean @ 0.11 | 35.9% | yes | submitted (#2) |
| `submission_newmodel_full` | newmodel mean @ 0.375 | 39.5% | yes | not uploaded |
| `submission_warmstart_flippatch_049` | **warmstart mean @ 0.49** | 41.2% | yes | **READY — consistent explanations, upload this** |
| `submission_warmstart_detonly_049` | warmstart mean @ 0.49 | 41.2% | zip removed | DO NOT USE — stale explanations on 40k flipped rows |

### `submission_warmstart_flippatch_049` (correct warmstart build)
Pass-1 (warmstart mean @ 0.49) → Pass-2 complex on 42,325 flipped rows → compressor
simple on the 23,269 new fakes → merged with 157,675 unchanged explanations → zip.
Validated: 200k rows each file, 0 real simple≠complex, 0 fake simple==complex, 0 empty.
Built via `run_calibrated_resubmit_lj.sh` with cross-model flip detection (`--base-complex`).

## Pass-2 fast strategy

Detection and explanation are scored **independently** on the leaderboard.
- **Detection-only resubmit** (`scripts/build_warmstart_detection_submission.py`): override
  labels with a new Pass-1 op-point, reuse existing explanations unchanged. CPU-only, instant.
  Used for `submission_warmstart_detonly_049`. ~40k rows (20%) have label/explanation mismatch.
- **Flip-patch** (`scripts/run_calibrated_resubmit_lj.sh`): regenerate explanations only for
  flipped verdicts. Warmstart @ 0.49 vs base = ~42k flips (21%) → ~1.5h vs ~6h full.
  TODO: `prepare_recalibrated_pass1.py` only detects flips within one model's TTA file; needs
  a cross-model mode (old labels from base complex jsonl) for warmstart.

## Branch map

| Branch | Purpose | Status |
|---|---|---|
| `external-manifest-mix` | 2M external data, warmstart train, test/holdout TTA, calibration, det-only submit | **active** |
| `pass1-trainval-resume-e2-4` | XP-only resume → 0.972 holdout | done |
| `pass1-test-tta` | original test TTA (broken wrapper) | superseded |
| `pass1-calibrate-macrof1` | val/test threshold sweep | merged into work |
| `pass2-test-complex` | Pass-2 complex (200k) | shipped |
| `grpo-complex-reward` | GRPO complex reward (ckpt-26400) | shipped |
| `codabench-evaluator` / `codabench-id-filename` | scorer + zip builder | on main |
| `pass1-retrain-macrof1` / `trainval-scratch` / `ensemble-giant` / `timm-fullft` | retrain experiments (flat ~0.86) | dead ends |

## Next (after #6 with Det F1 0.849)

Phase ends **18 June 2026 01:59 CEST**. Overall = detection + explanation; we lead complex
BERT, trail on detection (~0.03–0.07) and simple.

1. **Detection (biggest lever, 0.849 → ~0.88+):** leaders at 0.88–0.92. Options:
   - threshold re-tune for the warmstart model on the leaderboard signal (we now have a real
     test read at 0.49 → 41.2% fake; try a small sweep around it, 1 slot);
   - finish/keep training warmstart (was still improving each epoch) or ensemble warmstart +
     trainval-resume scores (decorrelated data mixes).
2. **Simple explanation (cheap explanation gain):** ours 0.677 vs xz 0.770 / xinyiyin 0.728.
   Compressor prompt/length tuning → likely +0.02–0.05 explanation with no detection risk.
3. **Complex is already #1 (0.711)** — leave it; don't spend slots there.
4. Reserve slots for clear hypotheses, not threshold noise.
