# Pass-1 ensemble — strategy (test submission)

## Model

- **Checkpoint:** `pass1_ensemble/bombek_so400m_dinov2_20260528-225201/best_ckpt/ckpt.pt`
- SigLIP2-SO400M + DINOv2-Large ensemble (Bombek recipe)

## Threshold calibration (full 110k val)

**Never use a threshold from a small balanced slice (e.g. n=500).** Always calibrate on **all 110k val** with the same score column you deploy on test.

| Run | Path | Notes |
|-----|------|-------|
| Val TTA (220k forwards) | `/home/jakob/luka/runs/pass1_val_tta/20260605-215505/` | orig + flip stored; metrics below |

### F1-optimal thresholds (full 110k val, Jun 2026)

| Score column | thr_best_f1 | AUC | Acc@thr | Fake F1 | Pred fake % |
|--------------|-------------|-----|---------|---------|-------------|
| **`p_fake_orig`** | **0.0839** | 0.936 | 85.8% | 0.871 | 55.0% |
| `p_fake_flip` | 0.0804 | 0.936 | 85.8% | 0.871 | 55.6% |
| `p_fake_mean` (orig+flip)/2 | 0.0820 | 0.939 | 86.3% | 0.876 | 55.9% |

**Deployed on test (current):** `p_fake_orig` @ **0.0839**.

Old wrong default `0.129` came from n=500 val in an eval pipeline — do not use.

## Horizontal flip TTA

We ran **orig + flip** on test (400k forwards) and on val (220k forwards). Analysis on test:

- Pearson(orig, flip) ≈ **0.98**; median |Δp| ≈ **0.005**
- Verdict disagreement @ 0.129 ≈ **5.4%** (mostly near-threshold images)

**Decision for submission:** **do not use flip scores for labels now.**

- Keep flip columns in parquet for analysis / future experiments.
- **Pass-2 conditioning + detection label:** `p_fake_orig` only.
- **Threshold:** 0.0839 from full val on `p_fake_orig`.

Possible later: `p_fake_mean`, majority vote, or flip only for borderline band — not for v1 submit.

## Test predictions

| Artifact | Path |
|----------|------|
| Raw TTA infer | `/home/jakob/luka/runs/pass1_test_tta/20260605-170149/predictions.parquet` |
| **Final (orig @ 0.0839)** | `/home/jakob/luka/runs/pass1_test/final/predictions.parquet` |

Recut without re-inference:

```bash
python3 research/experiments/02_pass1_classifier/recut_pass1_predictions.py \
  --input /home/jakob/luka/runs/pass1_test_tta/20260605-170149/predictions.parquet \
  --output /home/jakob/luka/runs/pass1_test/final/predictions.parquet \
  --score-col p_fake_orig \
  --threshold 0.0838903859257698
```

Or calibrate threshold from val in the same script (`--calibrate-from`).

## Pass-2 pipeline env

```bash
export PASS1_PRED=/home/jakob/luka/runs/pass1_test/final/predictions.parquet
export THRESHOLD=0.0838903859257698
export PASS1_SCORE_COL=p_fake_orig   # or p_fake column after recut
```
