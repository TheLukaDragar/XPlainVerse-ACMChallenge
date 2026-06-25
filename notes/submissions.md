# XPlainVerse — Submission Log & Plan

Single source of truth for what we submitted, leaderboard scores, and reusable assets.
(Leaderboard numbers come from Slack agent DMs; pipeline pred-fake from `pipeline_summary.json`.)

## Final status — competition closed 18 June 2026 01:59 CEST

**Public leaderboard entry:** submission **801131** (`sub7.zip`, uploaded 2026-06-17 13:45) — **Overall 0.790824** (best FRI score).

Built on disk but **not uploaded** (CodaBench daily limit 1/1; quota resets UTC midnight = 02:00 CEST, **1 min after** phase end):

| Build | Path | Pred fake % | Notes |
|---|---|---:|---|
| DINOv3 e3 alone @ 0.337 | `~/luka/runs/submission_dinov3e3_0337_dino/submission.zip` | 43.53% | 4,514 flip-patch vs last DINO @0.338 |
| DINOv3 fused @ 0.360 flip-patch | `~/luka/runs/submission_dinov3fused_036_flippatch/submission.zip` | 42.75% | uploaded earlier as sub7 candidate |

CodaBench UI states daily limit resets at “midnight server time” but counter uses **UTC calendar dates** while times display in **CEST** — final resubmit blocked without organizer exception. Email sent with e3 zip attached.

## Leaderboard results (CodaBench, FRI team = theluka)

Scorer: `abhijeet1317/xdd-scorer:2026-v6` (Overall = detection + explanation).

| # | Operating point | Pred fake % | Overall | Notes |
|---|---|---:|---:|---|
| 1 | `p_fake_orig @ 0.084` (deployed Bombek-arch ensemble, train-only) | 39.1% | **0.690** | first full pipeline; FRI ~#9 |
| 2 | `p_fake_mean @ 0.11` (flip-patch, 9,148 regen) | 35.9% | **~0.698** | calibration didn't transfer |
| 3 | **warmstart `p_fake_mean @ 0.49`** (2M external, flip-patch 42,325 regen) | 41.2% | **0.848966** | 795187, 2026-06-13 — **+0.16 Det F1 jump** |
| 4 | **GRPO simple resubmit** (det+complex unchanged, GRPO compressor ckpt-4800) | 41.2% | **0.784693** | 796442, 2026-06-14 — **#6 → #4**; Simple 0.730 |
| 5 | **DINOv3 fused + flip-patch** (logit-fuse @0.360; 15,101 flips regen) | 42.8% | holdout **0.9808** macro-F1 | sub7.zip **801131** → **0.790824 overall** |
| 6 | DINOv3 e3 alone @ 0.337 (flip-patch 4,514) | 43.53% | — | zip on disk only; deadline blocked |

**Flip-patch (not label-only):** 15,101 verdicts flip vs the #4/all_v3 build (9,250 fake→real, 5,851 real→fake). Explanations regenerated for ALL flips via `run_calibrated_resubmit_lj.sh` (Pass-2 complex ckpt-26400 + GRPO compressor ckpt-4800), so the 5,851 newly-fake images get fake-style complex + short GRPO simple instead of stale "looks real" text. Verified: 0 fake rows with simple==complex, 0 real rows with simple!=complex, 0 empty. The earlier label-only zip `submission_dinov3fused_036/` is superseded.

## DINOv3-MAC ensemble (2026-06-17) — biggest detection lever since external data

Reproduced the NTIRE 2026 winner recipe (DINO-MAC, github.com/qcf-568/DINOMAC):
DINOv3-Large + LoRA r32/α64 on attn.qkv + MAC head (AVG+CLS+4 REG over last 4 layers) +
deep supervision. Trained on **all_v4** (3.24M: XPlainVerse + OpenFake + DFBench + SID_Set +
**NTIRE 277k in-the-wild**), **512px** (matches native test res exactly), 2 epochs, focal.

| Detector (1k holdout, TTA) | macro-F1 |
|---|---:|
| ensemble v3 (prev best) | 0.9708 |
| ensemble warmstart | 0.9719 |
| **DINOv3-MAC alone** | **0.9788** |
| **logit-fuse DINOv3 0.6 + v3 0.4** | **0.9808** (+0.009) |

DINOv3↔ensemble error corr **0.95** vs ensemble-internal **0.984** → genuinely decorrelated,
which is why fusion pays off. Fused test @ 0.360 → 42.8% fake (next to the proven 41.2% point).
Artifacts: ckpt `pass1_dinov3_mac/dinov3_mac_20260616-205827/best_ckpt/ckpt.pt`;
test preds `pass1_dinov3_test_tta/20260617-091636/{predictions,fused_test}.parquet`.
Rebuild submission: `scripts/build_fused_detection_submission.py`.

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

### Full board snapshot — 2026-06-14 ~11:41 CEST (after GRPO simple resubmit)

| # | Team | Overall | Det F1 | Det Acc | Complex BERT | Simple Overall | Explanation |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | HIT VIRLAB (hit_xiaolun) | 0.81083 | 0.920087 | 0.92088 | 0.698534 | 0.704611 | 0.701573 |
| 2 | xz (xuzhu) | 0.807453 | 0.881308 | 0.88253 | 0.69702 | 0.770177 | 0.733598 |
| 3 | XJTU GenAI (xinyiyin) | 0.794198 | 0.877939 | 0.87916 | 0.692885 | 0.728029 | 0.710457 |
| **4** | **FRI (theluka)** | **0.784693** | **0.848966** | **0.84918** | **0.711251** | **0.729591** | **0.720421** |

**GRPO simple win (submission 796442):** regenerated only the `simple` field with the GRPO
compressor (ckpt-4800, reward = 0.7·BERT + 0.3·SLE). Detection + complex byte-identical to 795187.
Result: **Simple Overall 0.677236 → 0.729591 (+0.0524)** → **Overall 0.771605 → 0.784693 (+0.0131)**,
**#6 → #4**. Matches the holdout prediction (fake simple 0.679 → 0.80; blended ≈ +0.05).
Note: GRPO collapsed to a short "looks like a sticker/doll, does not look real" template — legitimate
since simple is scored only by BERT+SLE (no LLM), confirmed on the official Evaluation page.

**Standing now (#4):** complex BERT 0.711 still **#1 on board**; simple 0.730 now ~#3 (xz 0.770, XJTU 0.728).
**Detection 0.849 is the ONLY remaining lever** — to pass #3 XJTU (0.794198) need +0.0095 overall ≈
**+0.019 Det F1** (0.849 → 0.868), or some simple headroom toward xz's 0.770. #2/#1 need detection
0.88–0.92 (we're 0.849). → **Phase B: detection retrain on OpenFake(full)+SID_Set+DFBench.**
Phase ended **2026-06-18 01:59 CEST**; used **7/11** submission slots.

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
| `cursor/dinov3-alone-e3-a544` | DINOv3 e3 test TTA + flip-patch submit script | **final challenge push** |
| `cursor/dinov3-ensemble-a544` | DINOv3-MAC + fused detection | merged work |
| `pass1-trainval-resume-e2-4` | XP-only resume → 0.972 holdout | done |
| `pass1-test-tta` | original test TTA (broken wrapper) | superseded |
| `pass1-calibrate-macrof1` | val/test threshold sweep | merged into work |
| `pass2-test-complex` | Pass-2 complex (200k) | shipped |
| `grpo-complex-reward` | GRPO complex reward (ckpt-26400) | shipped |
| `codabench-evaluator` / `codabench-id-filename` | scorer + zip builder | on main |
| `pass1-retrain-macrof1` / `trainval-scratch` / `ensemble-giant` / `timm-fullft` | retrain experiments (flat ~0.86) | dead ends |

## Experiment log — VLM-as-detector (NEGATIVE, 2026-06-13)

Ran the Pass-2 VLM with the neutral "decide" prompt on all 1000 holdout images
(`~/luka/runs/holdout_vlm_detect/`), parsed its `Verdict:` line, compared to Pass-1
(warmstart @ 0.49) and GT:

| Detector | Acc | Macro F1 | fake rate |
|---|---:|---:|---:|
| Pass-1 warmstart | 0.972 | 0.972 | 0.527 |
| VLM neutral-decide | 0.772 | 0.772 | 0.413 |

- Disagree 232/1000; on disagreements **Pass-1 right 216, VLM right 16**.
- Best non-hurting ensemble = avg-prob w_p1=0.8 → macroF1 0.9738 (+0.002, noise).
- **Conclusion: VLM is a much weaker classifier; drop the VLM-detector/ensemble idea.**
  The `54dda78` win (VLM caught a composite Pass-1 missed) is 1 of only 16 — not representative.
- Detection lever remains: targeted data retrain (graphic/doc fakes + low-quality reals +
  adversarial/overlay augmentation), per the holdout error analysis.

## Post-challenge archive (2026-06-18)

All submission zips and checkpoints live under `~/luka/runs/` (not in git). Key git branches:

| Branch | Contents |
|---|---|
| `cursor/dinov3-alone-e3-a544` | e3 TTA + flip-patch orchestration (`run_dinov3alone_e3_submit_lj.sh`) |
| `cursor/dinov3-ensemble-a544` | DINOv3-MAC model + fused detection builder |
| `cursor/external-manifest-mix-a544` | 2M external data + warmstart pipeline |
| `cursor/grpo-complex-reward-a544` | GRPO Pass-2 + compressor |

Paper deadline **30 June 2026** (on-site ACM-MM presentation required for prizes).
