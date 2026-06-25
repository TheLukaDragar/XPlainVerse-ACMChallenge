# XPlainVerse ACM Challenge — Full Archive (June 2026)

**Purpose:** Single recovery document if disks, nodes, or accounts are wiped.  
**Team:** FRI (`theluka` on CodaBench)  
**Repo:** `git@github.com:TheLukaDragar/XPlainVerse-ACMChallenge.git`  
**Challenge closed:** 18 June 2026 01:59 CEST  

**Final public submission:** CodaBench **801131** (`sub7.zip`) — **Overall 0.790824**  
**Best on-disk build (not uploaded):** `~/luka/runs/submission_dinov3e3_0337_dino/submission.zip`

---

## 1. What we built (pipeline overview)

Two-task challenge: **detection** (real/fake) + **explanations** (complex + simple).

```
Pass-1 classifier (ensemble / DINOv3-MAC)  →  label per test image
Pass-2 VLM (GRPO complex, Qwen3-VL-8B)     →  complex_explanation
Compressor (GRPO simple, Qwen3.5-4B)     →  simple_explanation (fake only)
merge + zip                                →  CodaBench submission
```

**Flip-patch strategy:** When detection threshold/model changes, only regenerate explanations for rows whose **label flipped** vs a base submission (Pass-2 + compressor on flip manifest only). Scripts: `evaluation/prepare_recalibrated_pass1.py`, `scripts/run_calibrated_resubmit_lj.sh`.

**Scoring (public):** Overall = (detection_macro_F1 + explanation_score) / 2. We reached **#4 overall**, **#1 complex BERT (0.711)** on the final board.

---

## 2. Git branches (code map)

| Branch | What it contains |
|--------|------------------|
| `main` | Baseline challenge repo, evaluators, dataset builders |
| `cursor/external-manifest-mix-a544` | 2M external data mixing, warmstart Pass-1 training |
| `cursor/grpo-complex-reward-a544` | GRPO Pass-2 complex (checkpoint-26400) |
| `cursor/dinov3-ensemble-a544` | DINOv3-MAC model + logit fusion + fused submission builder |
| `cursor/dinov3-alone-e3-a544` | Final e3 TTA + flip-patch orchestration |
| `cursor/codabench-evaluator-a544` | Local CodaBench scorer |
| `cursor/pass1-*` | Pass-1 experiments (calibration, TTA, retrain) |

**Docs in repo:**

| File | Content |
|------|---------|
| `notes/submissions.md` | Leaderboard log, submission history, scores |
| `notes/ARCHIVE.md` | This file — paths, checkpoints, data |
| `.cursor/rules/xplainverse-*.mdc` | Challenge context, metrics, lj cluster rules |
| `dataset/README.md` | VLM/compressor JSONL, prompts, GRPO plan |
| `scripts/LJ_TRAINING.md` | Lj GPU node, Apptainer, JSONL build |

---

## 3. Hosts & mount paths

### Ljubljana cluster (`elixir-lj-gpu-01`)

| Role | Path |
|------|------|
| Code (host) | `~/luka/code/XPlainVerse-ACMChallenge` |
| Code (in container) | `/workspace/XPlainVerse-ACMChallenge` |
| Runs / checkpoints | `~/luka/runs/` |
| XPlainVerse images (local copy) | `~/luka/data/XPlainVerse/` (**367 GB**) |
| External datasets | `~/luka/data/external/` |
| GPU data on `/primoz` | `/primoz/luka/XPlainVerse/data/XPlainVerse` (bind-mounted in Apptainer) |
| Apptainer SIF | `~/containers/xplainverse-acmchallenge.sif` |
| Container entry | `~/xplainverse_exec.sh` |
| Slurm dispatch from login | `./scripts/lj_gpu_exec.sh` |
| DINOv3 TTA (needs timm) | `./scripts/lj_ghcr_image_exec.sh` (GHCR image with flash-attn) |

### Shared workspace (reference path from rules)

`/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/data/XPlainVerse/` — may exist on other machines; **primary copy used for training: `~/luka/data/XPlainVerse`**.

---

## 4. Datasets — what exists where

### XPlainVerse (official challenge data)

| Split | Images | Path under `~/luka/data/XPlainVerse/` |
|-------|--------|----------------------------------------|
| Train | 450,000 | `train/` |
| Val | 110,000 | `val/` |
| Test | 200,000 | `test/` (multipart tar; see challenge docs) |

Val ground truth: `evaluation/data/val_ground_truth.jsonl` (**110,012 rows**).

### External training data (`~/luka/data/external/`)

| Dataset | Approx size | Used in |
|---------|-------------|---------|
| **OpenFake** | **2.5 TB** | all_v3/v4 manifests (~2M JPEG rows) |
| **GenImage** | 117 GB | early experiments |
| **DRCT-2M** | 40 GB | available, partial use |
| **DFBench** | 69 GB | `manifest_dfbench_train.parquet` |
| **SID_Set** | (see setup run) | `manifest_sid_set_trainval.parquet` |
| **NTIRE 2026 in-the-wild** | 277,643 rows | `manifest_ntire_train.parquet` |

### Training manifests (Pass-1)

Location: `research/experiments/02_pass1_classifier/manifests/`

| Manifest | Rows | Description |
|----------|-----:|-------------|
| `manifest_train*.parquet` | 450k | XPlainVerse train |
| `manifest_val*.parquet` | 110k | XPlainVerse val |
| `manifest_test.parquet` | 200,000 | Test images |
| `manifest_test_tta.parquet` | 400,000 | Test + horizontal flip (TTA) |
| `manifest_val_holdout.parquet` | 1,000 | Held-out for threshold calibration |
| `external/manifest_all_v3.parquet` | 2,965,261 | XP + OpenFake + DFBench + SID |
| `external/manifest_all_v4.parquet` | 3,242,904 | v3 + NTIRE |
| `external/manifest_all_v4_full.parquet` | 3,243,347 | v4 + former holdout rows |
| `external/manifest_ntire_train.parquet` | 277,643 | NTIRE only |

Build scripts: `build_manifest.py`, `build_manifest_external.py`, `finish_external_manifest_lj.sh`.

### VLM / compressor JSONL (Pass-2)

Location: `dataset/` (often **gitignored** — regenerate with `dataset/build_swift_jsonl.py`).

| File | Purpose |
|------|---------|
| `train_vlm_v2.jsonl` | VLM SFT (450k) |
| `val_vlm_v2.jsonl` | VLM val with labels |
| `val_vlm_infer_v2.jsonl` | 110k val, infer-only |
| `train_compressor*.jsonl` | Compressor SFT (fake only) |
| `train_grpo.jsonl` / `val_compressor_grpo.jsonl` | GRPO compressor training |

Prompts: `dataset/prompt_v2.txt`, `dataset/prompt.txt`.

---

## 5. Models & checkpoints

### Pass-1 — deployed / best

| Model | Checkpoint path | Notes |
|-------|-----------------|-------|
| **Warmstart v3** (2M ext, best pre-DINO) | `~/luka/runs/pass1_ensemble/external_all_v3_warmstart_20260615-105316/best_ckpt/ckpt.pt` | 41.2% fake @0.49, Det F1 0.849 |
| **DINOv3-MAC 2-epoch** | `~/luka/runs/pass1_dinov3_mac/dinov3_mac_20260616-205827/best_ckpt/ckpt.pt` | Used for fusion @0.360 |
| **DINOv3-MAC 3-epoch (final)** | `~/luka/runs/pass1_dinov3_mac/dinov3_mac_20260617-123523/best_ckpt/ckpt.pt` | **1.3 GB**, e3 TTA, @0.337 |
| Bombek SO400M + DINOv2 (original) | `~/luka/runs/pass1_ensemble/bombek_so400m_dinov2_20260528-225201/best_ckpt/ckpt.pt` | First pipeline, ~0.69 overall |
| Trainval resume | `~/luka/runs/pass1_ensemble/trainval_resume_e2-4_20260610-181626/best_ckpt/ckpt.pt` | 0.972 holdout (leaky) |

**DINOv3 architecture:** DINOv3-Large + LoRA r32/α64 on attn.qkv + MAC head (AVG+CLS+4 REG, last 4 layers), deep supervision, 512px, focal loss. Code: `research/experiments/02_pass1_classifier/models/dinov3_mac.py`. Train: `scripts/run_pass1_dinov3_mac_lj.sh`.

### Pass-1 — test predictions (TTA parquets)

| Run | Path |
|-----|------|
| Warmstart test TTA | `~/luka/runs/pass1_test_tta_warmstart/predictions.parquet` |
| DINOv3 fused test | `~/luka/runs/pass1_dinov3_test_tta/20260617-091636/{predictions,fused_test}.parquet` |
| DINOv3 e3 test TTA | `~/luka/runs/pass1_dinov3_test_tta/e3_full/predictions.parquet` |

Columns: `sample_id`, `p_fake_orig`, `p_fake_flip`, `p_fake_mean`, `pred_label_mean`.

### Pass-2 — VLM complex

| Asset | Path |
|-------|------|
| Base complex (200k test) | `~/luka/runs/pass2_test_complex/20260606-195337/complex_explanations.jsonl` |
| GRPO adapters (best) | `~/luka/runs/vlm_v2_grpo/job_48855/checkpoint-26400` |
| GRPO training dir | `~/luka/runs/vlm_v2_grpo/` |

Train: ms-swift GRPO, reward = complex metric proxy. Script: `scripts/train_vlm_v2_frida.sh` / lj variants.

### Compressor — simple explanations

| Asset | Path |
|-------|------|
| SFT checkpoint | `~/luka/runs/compressor_vl/checkpoint-10000` |
| **GRPO checkpoint (submitted)** | `~/luka/runs/compressor_grpo/20260613-201713/checkpoint-4800` |
| Base compressor infer | `~/luka/runs/compressor_test/20260607-080830/compressor_infer.jsonl` |
| GRPO infer (796442 build) | `~/luka/runs/compressor_grpo_infer_20260614-085756/` |

### Baseline VLMs (merged LoRA)

| Model | Path |
|-------|------|
| Qwen3-VL-8B-XPlainVerse | `baseline_models/Qwen3-VL-8B-XPlainVerse` |
| Qwen3-VL-8B-Instruct | `baseline_models/Qwen3-VL-8B-Instruct` |
| Pass-1 Bombek baseline | `baseline_models/pass1` |

HuggingFace refs: `kartik060702/Qwen3-VL-8B-XPlainVerse`, etc.

---

## 6. Submissions (CodaBench history)

| ID | File | Date | Overall | Description |
|----|------|------|--------:|-------------|
| 783968 | submission.zip | 2026-06-07 | 0.690 | First full pipeline |
| 786313 | sub2.zip | 2026-06-08 | 0.689 | Calibrated @0.11 |
| 788640 | sub3.zip | 2026-06-09 | 0.738 | New model |
| 795187 | sub4.zip | 2026-06-13 | 0.849 Det F1 | Warmstart @0.49 flip-patch |
| 796442 | sub5.zip | 2026-06-14 | 0.785 | GRPO simple only |
| 800238 | sub6.zip | 2026-06-16 | 0.787 | all_v3 ensemble |
| **801131** | **sub7.zip** | **2026-06-17** | **0.791** | **DINOv3 fused @0.360 flip-patch (FINAL PUBLIC)** |

### Submission zips on disk (`~/luka/runs/`)

| Directory | Pred fake % | Uploaded? |
|-----------|------------:|-----------|
| `submission_warmstart_flippatch_049/` | 41.2% | yes (#4 795187) |
| `submission_grpo_simple_20260614-093823/` | 41.2% | yes (#5 796442) |
| `submission_dinov3fused_036_flippatch/` | 42.8% | yes (#7 **801131**) |
| `submission_dinov3alone_338/` | ~43.5% | no |
| `submission_dinov3e3_0337_dino/` | 43.53% | **no (deadline blocked)** |

Each contains `submission.jsonl` + `submission.zip` (200k rows: `sample_id`, `label`, `complex_explanation`, `simple_explanation`).

**Rebuild fused submission:** `scripts/build_fused_detection_submission.py`  
**Rebuild flip-patch:** `scripts/run_calibrated_resubmit_lj.sh`  
**DINOv3 e3 alone:** `scripts/run_dinov3alone_e3_submit_lj.sh`

---

## 7. How we trained (timeline)

1. **Pass-2 VLM SFT** on 450k XPlainVerse → complex explanations (deployed Bombek Pass-1 labels).
2. **Compressor SFT** on fake rows → simple field.
3. **Pass-1 Bombek ensemble** (SigLIP + DINOv2) on balanced XP train → weak test generalization (0.69).
4. **External warmstart:** mix OpenFake + DFBench + SID into trainval → **Det F1 0.849** @41.2% fake.
5. **GRPO** Pass-2 complex + compressor simple → +0.013 overall, #6→#4.
6. **DINOv3-MAC** on all_v4 (3.24M, +NTIRE), 512px, 2 then 3 epochs → 0.9788 holdout macro-F1.
7. **Logit-fuse** DINOv3 0.6 + ensemble v3 0.4, threshold 0.360, flip-patch 15,101 rows → **sub7 0.790824**.
8. **DINOv3 e3 alone** @0.337, 4,514 flip-patch — built, not uploaded (CodaBench daily limit).

---

## 8. Key scripts (reproduce from scratch)

| Step | Script |
|------|--------|
| Build XP JSONL | `dataset/build_swift_jsonl.py` |
| Build Pass-1 manifests | `research/experiments/02_pass1_classifier/build_manifest*.py` |
| Train warmstart ensemble | `scripts/run_pass1_ensemble_external_all_warmstart_lj.sh` |
| Train DINOv3-MAC | `scripts/run_pass1_dinov3_mac_lj.sh` |
| Test TTA inference | `scripts/run_pass1_test_tta_*.sh`, `scripts/run_dinov3_test_tta_lj.sh` |
| Pass-2 complex infer | `scripts/run_pass2_test_complex_lj.sh` |
| GRPO compressor | `scripts/train_compressor_grpo_lj.sh` |
| Calibrated resubmit | `scripts/run_calibrated_resubmit_lj.sh` |
| Local evaluation | `scripts/run_codabench_scorer_lj.sh` |

---

## 9. Environment

| Component | Version / path |
|-----------|----------------|
| Lj container | Ubuntu 22.04, Python 3.10, Torch 2.2.2+cu121 |
| ms-swift | 4.3.0.dev0 editable `/opt/ms-swift` |
| vLLM eval (Frida) | 0.21.0, CUDA 13, torch 2.11 |
| CodaBench scorer Docker | `abhijeet1317/xdd-scorer:2026-v6` |

Required vLLM env (Frida): `VLLM_USE_FLASHINFER_SAMPLER=0`, `LD_LIBRARY_PATH` for cu13.

---

## 10. Backup priority (if disks may be deleted)

**Copy these first:**

1. **All of `~/luka/runs/pass1_dinov3_mac/`** — DINOv3 checkpoints (~5 GB total)
2. **`~/luka/runs/pass1_ensemble/external_all_v3_warmstart_*/best_ckpt/`**
3. **`~/luka/runs/vlm_v2_grpo/job_48855/checkpoint-26400`**
4. **`~/luka/runs/compressor_grpo/20260613-201713/checkpoint-4800`**
5. **Submission zips:** `submission_dinov3fused_036_flippatch/`, `submission_dinov3e3_0337_dino/`, `submission_warmstart_flippatch_049/`
6. **Parquets:** `pass1_dinov3_test_tta/e3_full/predictions.parquet`, `pass2_test_complex/.../complex_explanations.jsonl`
7. **Manifests:** `research/experiments/02_pass1_classifier/manifests/external/manifest_all_v4_full.parquet`
8. **Git repo** (already on GitHub)

**Too large to archive easily (re-download if needed):**

- `~/luka/data/XPlainVerse/` (367 GB)
- `~/luka/data/external/OpenFake/` (2.5 TB)

**Can regenerate from XP + external scripts:**

- `dataset/*.jsonl` via `build_swift_jsonl.py`
- Pass-1 manifests via `build_manifest_external.py` + OpenFake JPEG shards

---

## 11. Paper / post-challenge

- Paper deadline: **30 June 2026** (ACM-MM on-site presentation for prizes)
- Experiment notes: `research/experiments/*/RESULTS.md`, `notes/submissions.md`
- Outside-box ideas: `research/experiments/05_outside_box/OUTSIDE_BOX_IDEAS.md`

---

*Generated 2026-06-18. Update this file if you move checkpoints or re-upload submissions.*
