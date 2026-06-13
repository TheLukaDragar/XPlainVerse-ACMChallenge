# External Tier-1 datasets — paths and formats

Canonical locations after download:

| Dataset | Path | Size (full) | Status |
|---------|------|-------------|--------|
| DFBench | `/primoz/luka/external/DFBench/` | ~133 GB | **Complete** — extracted + manifest |
| GenImage | `/primoz/luka/external/GenImage/` | ~635 GB | **Complete** (ZIPs, not extracted) |
| OpenFake | `/home/jakob/luka/data/external/OpenFake/` | ~3.4 TB | **Downloading** |
| DRCT-2M | — | — | **Skipped** |
| SID_Set | `/primoz/luka/external/SID_Set/` | ~131 GB | **Complete** — extract + manifest pending |

---

## SID_Set (`saberzl/SID_Set`)

Open Images v7–anchored dataset for social-media deepfake detection (SIDA, CVPR 2025).

### On disk (after download)
```
SID_Set/
├── README.md
├── config.json
└── data/
    ├── train-*-of-00249.parquet    # 210k rows
    └── validation-*-of-00034.parquet  # 30k rows
```

**Test split (60k) is not on HuggingFace** — requires separate request per dataset card.

### Parquet schema
| Column | Type | Notes |
|--------|------|-------|
| `img_id` | string | Real IDs match Open Images v7 |
| `image` | image bytes | Embedded JPEG/PNG |
| `mask` | image bytes | Tampered-region mask (label=2 only) |
| `label` | int | **0**=real, **1**=full synthetic, **2**=tampered |

### Splits on HuggingFace
| Split | Rows | In Pass-1 manifest |
|-------|------|-------------------|
| train | 210,000 | yes |
| validation | 30,000 | yes |
| test | 60,000 | **no** (not on HF) |

Balanced: **80k real / 80k full_synthetic / 80k tampered**.

### Pass-1 manifest mapping
```python
# int label -> binary Pass-1
# 0 -> real; 1 full_synthetic + 2 tampered -> fake
label, generator = SID_SET_LABEL_MAP[row["label"]]
# label_int: real=0 fake=1
image_path = f"/primoz/luka/external/SID_Set_jpeg/data/{shard_stem}/{row_idx:06d}.jpg"
```

Output: `manifests/external/manifest_sid_set_trainval.parquet`  
Combined: `manifest_all_v2.parquet` (= all_v1 + SID_Set)

### Setup (after download)
```bash
./scripts/setup_sid_set_lj.sh
# audit only:
SKIP_EXTRACT=1 MANIFEST_ONLY=1 ./scripts/setup_sid_set_lj.sh  # needs JPEGs done first
```

### Download
```bash
./scripts/run_download_sid_set_lj.sh
```

---

## DFBench (`IntMeGroup/DFBench`)

### On disk
```
DFBench/
├── img_train.jsonl          # 436,506 rows
├── img_test.jsonl           # 109,131 rows
├── img_train_shuffled.jsonl # same train, shuffled
├── img_train.json / img_test.json   # alternate messages/images format
├── README.md
└── *.zip (21 archives)      # images — must UNZIP before use
    └── after unzip → DFBench/<source>/000002.jpg
```

### Annotation format (JSONL)
```json
{
  "image": "DFBench/Kolors/000002.jpg",
  "conversations": [
    {"from": "human", "value": "Is this a real image or a generated image? ... A: 'real' or B: 'generated'."},
    {"from": "gpt", "value": [{"role": "assistant", "content": "B"}]}
  ]
}
```

| Field | Meaning |
|-------|---------|
| `image` | Relative path under local dir (after ZIP extract) |
| Label | **A** = real, **B** = generated (parse from assistant content) |

### Splits
| Split | Rows | Real | Fake |
|-------|------|------|------|
| train | 436,506 | 37,410 (8.6%) | 399,096 |
| test | 109,131 | 9,357 | 99,774 |

### Sources (21 folders after unzip)
- **Real (8):** CLIVE, CSIQ, Flick8k, LIVE, TID2013, kadid10k, koniq10k, partial_source
- **Fake (13):** Janus, Kandinsky-3, Kolors, LaVi-Bridge, NOVA, PixArt, Playground, ali_flux_dev, ali_flux_schnell, edit, infinity, sd3_5_large, sd3_medium

### Pass-1 manifest mapping
```python
label = "real" if assistant == "A" else "fake"
image_path = f"/primoz/luka/external/DFBench/{row['image']}"  # after unzip
```

### Resolutions

**No single fixed size** — varies by source ZIP (sampled from archives on `/primoz`).

| Source type | Examples | Typical native size |
|-------------|----------|---------------------|
| Real IQA datasets | CLIVE, CSIQ, LIVE, Flick8k, TID2013, kadid10k, koniq10k | **384×384 – 768×512** (often ~500×500 or 512×384) |
| Fake @ 384 | Janus | **384×384** |
| Fake @ 512 | LaVi-Bridge, NOVA | **512×512** |
| Fake @ 1024 | Kolors, Kandinsky-3, PixArt, Playground, Flux dev/schnell, SD3.5, infinity | **1024×1024** |
| Mixed | edit, partial_source | **~333–640** (wide range) |

**Overall range (sampled):** width and height **384–1024 px** for most rows; some reals smaller (e.g. Flick8k ~500×332).

---

## GenImage (`ENSTA-U2IS/GenImage`)

### On disk (current: compressed multi-part ZIPs per generator)
```
GenImage/
├── ADM/           imagenet_ai_0508_adm.z01 … .zip
├── BigGAN/
├── Midjourney/
├── VQDM/
├── glide/
├── stable_diffusion_v_1_4/
├── stable_diffusion_v_1_5/
└── wukong/
```

**No JSON/CSV metadata** — labels are **folder names** after unzip.

### Layout after unzip (official convention)
```
{generator}/
├── train/
│   ├── ai/       ← fake
│   └── nature/   ← real (ImageNet)
└── val/
    ├── ai/
    └── nature/
```

| Folder | Label |
|--------|-------|
| `ai/` | `fake` |
| `nature/` | `real` |

~2.68M images total, 8 generators, 1000 ImageNet classes.

### Pass-1 manifest mapping
```python
# Walk extracted tree:
# .../train/ai/foo.png     → label=fake,  source=generator_name
# .../train/nature/foo.png → label=real,  source=generator_name
```

**Post-download step required:** unzip all archives (and merge split `.z01` parts where needed).

### Resolutions

**Fixed per generator** at synthesis time (GenImage paper; archives are split ZIPs so sizes were not re-verified file-by-file after download).

| Generator | Native generation resolution |
|-----------|------------------------------|
| **BigGAN** | **128×128** |
| **ADM, GLIDE, VQDM** | **256×256** |
| **Stable Diffusion v1.4 / v1.5** | **512×512** |
| **Wukong** | **512×512** |
| **Midjourney v5** | **1024×1024** |
| **Real (`nature/`)** | Native **ImageNet** JPEGs — varied, often ~256–500 px short side |

Real and fake images are **not shared across generators** — each subset has its own paired `nature/` reals at ImageNet-native sizes.

---

## OpenFake (`ComplexDataLab/OpenFake`)

### On disk
```
OpenFake/
├── README.md
├── prompt-image_bank.csv    # 1.33M text prompts (no images)
├── core/
│   ├── train-*-of-*.parquet     # ~608 shards, ~2.3M rows
│   ├── validation-*-of-*.parquet  # 15 shards
│   └── test-*-of-*.parquet        # 13 shards (OOD generators)
└── reddit/                  # optional config (in-the-wild test)
    └── test-*-of-*.parquet
```

**Images are embedded in parquet** — no separate image files.

### Parquet schema
| Column | Type | Notes |
|--------|------|-------|
| `image` | struct | `{"bytes": binary, "path": "2284969.jpg"}` — JPEG bytes inline |
| `label` | string | `real` or `fake` |
| `model` | string | Generator (`sd-3.5`, `flux.2-dev`, …) or real source (`laion`, `pexels`, …) |
| `prompt` | string | T2I prompt or LAION caption; NaN for some reals |
| `type` | string | `real`, `base`, `finetune`, `lora`, `image`, `video`, … |
| `release_date` | string | `YYYY-MM` or `YYYY-MM-DD` |

### Configs / splits (HF dataset card)
| Config | Splits | Use |
|--------|--------|-----|
| `core` | train + validation + test | Main training mix |
| `reddit` | test only | In-the-wild Reddit photos vs AI subs |

Train/val reals: LAION + Pexels. Test reals: DOCCI + ImageNet. Test fakes: held-out generators.

### Pass-1 manifest mapping
```python
label = row["label"]  # already "real" / "fake"
# Option A: extract bytes to disk at infer/train time
# Option B: custom Dataset that reads parquet row["image"]["bytes"]
image_path = f"/home/jakob/luka/data/external/OpenFake/extracted/{row['image']['path']}"
model_tag = row["model"]
```

### Resolutions

**Highly variable** — no standard size (sampled from `core/validation-*` and `core/train-*` parquet shards on `$HOME`).

| What | Common sizes (sample) | Full range (sample) |
|------|----------------------|---------------------|
| Validation shard | **1024×1024**, **512×512** | **197×256 – 6492×8192** |
| Train shard 0 | **768×1152**, **1024×1024** | **116×160 – 7573×6960** |

Mix includes:
- **Model-native squares:** 512, 1024
- **LAION / Pexels reals:** large web photos (2k–7k px)
- **Portrait / aspect ratios:** e.g. 832×1216, 1152×768

---

## Image resolutions — summary

None of the three datasets uses one global resolution. For Pass-1 training they are all **resized to `image_size=392`** in the Bombek ensemble pipeline (same as XPlainVerse).

| Dataset | Native resolution story | Note for 392 training |
|---------|-------------------------|------------------------|
| **DFBench** | Mostly **384–1024** | Good overlap with 392 |
| **GenImage** | **128–1024** by generator | Many natives **smaller** than 392 → upscaled |
| **OpenFake** | **~200 – 7500+** px | Many natives **much larger** → downscaled |

Native size still matters for compression/quality artifacts even after resize.

---

## Next steps (after OpenFake finishes)

1. Verify shard count: 644 files total under `core/` + `reddit/`
2. Build per-dataset manifest builders → unified `image_path, label, source, sample_id`
3. Decide mix ratio with XPlainVerse pooled manifest (start small, fakes only from external)

### DFBench setup (done)

```bash
./scripts/setup_dfbench_lj.sh --repair   # if LIVE / ali_flux_schnell need re-extract
./scripts/setup_dfbench_lj.sh            # full extract + manifest
```

Output: `manifests/external/manifest_dfbench_train.parquet` (~436k rows, 121 skipped — macOS `._` junk + 1 missing PixArt).
