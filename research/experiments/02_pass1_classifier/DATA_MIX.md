# Pass-1 external data mix — strategy

## Is OpenFake shards 0–14 enough?

**Yes, for a Bombek-style supplemental stage.** Groups 0–14 = 285 shards ≈ **1.1M rows** (same range Bombek used). You do not need the remaining 1.8 TB.

## Current manifest outputs

| Manifest | Rows (approx) | Ready to train? |
|----------|---------------|-----------------|
| `manifest_openfake_core_0-14.parquet` | ~1.1M | Yes (parquet loader needed) |
| `manifest_dfbench_train.parquet` | ~436k | **No** — ZIPs not extracted yet |
| `manifest_genimage_train.parquet` | — | **No** — ZIPs not extracted |
| `manifest_external_mix_v1.parquet` | XP + capped fakes | After DFBench unzip + loader |

Build (CPU only — does not use GPUs):

```bash
# OpenFake — login node CPU
python3 research/experiments/02_pass1_classifier/build_manifest_external.py --only openfake

# DFBench — CPU on gpu node (/primoz), while GPUs train
./scripts/lj_cpu_primoz_exec.sh python3 research/experiments/02_pass1_classifier/build_manifest_external.py --only dfbench

# Combined mix (after above)
python3 research/experiments/02_pass1_classifier/build_manifest_external.py --only mix
```

## Recommended training recipe

**Do not replace the winning trainval-pooled recipe in one shot.** Stage it:

### Stage A — done / in flight
- `trainval_scratch` → `trainval_resume_e2-4`
- XPlainVerse pooled only, focal loss, macro-F1 holdout select
- Holdout already **0.966** macro F1 after resume epoch 1

### Stage B — external fine-tune (next)
Warm-start from best resume ckpt. One epoch max initially.

**Mix v1 (conservative, recommended first try):**

| Source | What to use | Cap | Why |
|--------|-------------|-----|-----|
| XPlainVerse pooled | all 559k | — | anchor distribution (competition domain) |
| OpenFake 0–14 | **fakes only** | 150k sample | generator diversity, matches Bombek pretrain |
| DFBench train | **fakes only** | 100k sample | 21-source fake diversity |
| GenImage | defer | — | needs 635 GB unzip; many 128–256 px natives |

- Keep **all XPlainVerse reals** — protects real recall (your val→test gap is partly real-F1).
- Add external **fakes only** so fake precision/diversity improves without swamping reals.
- Target ratio after mix: ~**45% real / 55% fake** (vs 18% real in raw pooled). Focal loss still on.
- Lower LR: `lr_head=5e-5`, `lr_lora=2e-5`, **1 epoch**, same 1k holdout.

### Stage C — if Stage B helps holdout
- Try 2 more epochs
- Add DFBench **reals** capped at 30k (IQA datasets — different domain, small cap)
- Optionally add GenImage SD1.5/Wukong 512px subset only (skip BigGAN 128)

### What to avoid
- Full 1.1M OpenFake + 436k DFBench + 559k XP without caps → fake-heavy, crushes real F1
- Training on DFBench/GenImage **test** splits
- GenImage BigGAN 128px at large weight (upscale artifacts ≠ XPlainVerse)

## Prep + training (in progress)

**CPU extraction** (tmux: `extract-dfbench`, `extract-openfake`):
- DFBench → `/primoz/luka/external/DFBench/DFBench/`
- OpenFake JPEG → `/primoz/luka/external/OpenFake_jpeg/`

**Manifest** `manifest_all_v1.parquet` — **2,076,717 rows** (~37% real):
XP 559k + OpenFake 1.08M + DFBench 437k

```bash
# After extraction completes:
./scripts/finish_external_manifest_lj.sh

# Train FROM SCRATCH on combined data (1 epoch, no resume ckpt):
LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=120:00:00 \
  ./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_ensemble_external_all_lj.sh
```

Fresh LoRA + head; pretrained SigLIP/DINOv2 backbones only. Do **not** warm-start from trainval resume.
