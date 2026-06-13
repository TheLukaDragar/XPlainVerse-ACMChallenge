# XPlainVerse — Submission Log & Plan

Single source of truth for what we submitted, leaderboard scores, and reusable assets.
(Leaderboard numbers come from Slack agent DMs; pipeline pred-fake from `pipeline_summary.json`.)

## Leaderboard results (CodaBench, FRI team)

| # | Operating point | Pred fake % | det macro F1 | Notes |
|---|---|---:|---:|---|
| 1 | `p_fake_orig @ 0.084` (deployed Bombek-arch ensemble, train-only) | 39.1% | **0.690** | first full pipeline; FRI ~#9 |
| 2 | `p_fake_mean @ 0.11` (flip-patch, 9,148 regen) | 35.9% | **~0.698** | calibration didn't transfer; slot ~wasted |

Explanation side was strong: complex BERT 0.690, simple 0.666, explanation_score 0.678
(≈ #3 if ranked on explanations alone). **Detection is the bottleneck.**

Diagnosis (notes 2026-06-07/08): val macro F1 ~0.86 but leaderboard ~0.69. Cause =
val→test **score shift** (`p_fake_orig` mean 0.36 val vs 0.24 test) + real generalization
gap, NOT architecture. More/better data > threshold tweaks.

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

## Next

1. Upload `submission_warmstart_flippatch_049/submission.zip` → read whether the
   2M-external warmstart detector beats 0.690 on leaderboard (only honest test signal).
   Explanations are consistent (flip-patched), so this also reflects explanation score fairly.
2. If flat → stop on this detector; explanations are already competitive (~0.68).
3. Reserve remaining CodaBench slots for a clear detection hypothesis, not threshold tweaks.
