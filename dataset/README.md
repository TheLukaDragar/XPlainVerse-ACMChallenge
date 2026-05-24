# XPlainVerse dataset preparation

This directory turns the raw XPlainVerse manifests into ms-swift JSONL files
for the two stages of our pipeline:

1. **VLM** (`Qwen3-VL-8B-Instruct`) — image → forensic complex paragraph + `Verdict: real|fake`.
2. **Compressor** (`Qwen3.5-4B`) — complex paragraph → short simple sentence.

This README explains *why* the prompts, label format, train/val split,
and hypothetical-prompt mix look the way they do. The design is informed by the
official challenge scoring (see `../.cursor/rules/xplainverse-evaluation-metrics.mdc`)
and by recent papers on VLM-based AI-image detection.

## Contents

| File | Purpose |
|------|---------|
| `prompt.txt` | All prompt templates, in `=== NAME === ... === END ===` sections. Edit this to change prompts. |
| `build_swift_jsonl.py` | Reads `prompt.txt` + raw manifests, optional class cap, train-only hypotheticals, writes JSONL. |
| `sanity.py` | Verifies that every path referenced by the manifests exists on disk. |
| `train_vlm.jsonl` | VLM SFT data (user + assistant + `Verdict:`). Default: all 450k train rows. |
| `train_vlm_infer.jsonl` | Same train rows but user-only (rarely used). |
| `train_compressor.jsonl` | Fake-only compressor SFT data (complex → simple). |
| `val_vlm.jsonl` | Val rows with assistant targets (`--val_dataset` during SFT). **Primary prompts only.** |
| `val_vlm_infer.jsonl` | **All 110k** val rows, user-only — used to produce the submission. |
| `val_compressor.jsonl` | Fake-only val for offline compressor evaluation. |

JSONL rows follow the ms-swift custom-dataset format
([docs](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md)):

```json
{"messages": [{"role": "user", "content": "<image>\nTASK: ..."},
              {"role": "assistant", "content": "... paragraph ...\n\nVerdict: fake"}],
 "images": ["/abs/path/to/img.png"]}
```

## How we arrived at the prompt format

### TL;DR — what the scoring rewards

`complex_overall = 0.3 · BERTScore + 0.4 · EntityF1 + 0.3 · FactsF1`,
`simple_overall = 0.7 · BERTScore + 0.3 · SLE_norm`.

`EntityF1 + FactsF1 = 70 %` of the complex score. Both are computed by
Qwen3.5-4B comparing the *evidence objects* and *atomic visual claims* in
your text against the ground-truth paragraph. So the only way to win is to
**name several specific objects in the image and describe what is wrong (or
authentic) about each**, in roughly GT style.

GT statistics from val:

- Complex paragraphs median ≈ **109 words**, 4–6 named objects, use
  connectives like *Additionally* / *Furthermore* / *On the left*.
- Common opener: *"The image contains several visual inconsistencies that
  suggest manipulation."* (348/2000 val rows).
- Real GT has **no separate simple text** — the simple field equals the
  complex field for real images.

### Decisions and the evidence behind them

1. **Drop the baseline `<reasoning>/<answer>` tags.**
   The XPlainVerse baseline trains on a single
   `<reasoning>...</reasoning><answer>real|fake</answer>` target and copies
   the reasoning into both `complex_explanation` and `simple_explanation`.
   That destroys the SLE simplicity score and produces poor entity
   coverage. Instead we use natural prose + a `Verdict:` footer.

2. **`Verdict:` is on its own line, last.** *Evidence before judgment*
   matches GT style (which never ends with "Verdict:" — we add it for
   parsing only) and lets us split with one trivial regex at submission
   time. Putting the label first encourages the model to commit before
   examining the image — exactly the over-confidence failure documented in
   PGT [§3, Fig. 2].

3. **Prompt asks for "the style and the synthesis artifacts" — verbatim.**
   The Prefill-Guided Thinking paper [2] shows that prefilling Qwen2.5-VL,
   LLaVA and Llama with exactly that phrase boosts zero-shot Macro F1 by
   up to **+24 %** across 16 generators, and that phrasing is brittle —
   variants like *"Let's examine the style"* alone lose ~10 points
   ([2, Table 2]).

4. **Ask for 4–6 named objects in one paragraph.** This is direct
   shaping toward GT style and toward `entity_f1` / `facts_f1`, which
   reward enumeration. FakeVLM [1] confirms that
   *label + free-form explanation* beats *label-only* even for the
   classification accuracy itself (§4.1, "Does the image look real/fake?"
   ablation).

5. **No chain-of-thought.** "Let's think step by step" underperforms the
   forensic phrasing in PGT ([2, Table 1, "CoT Prefill" vs "S2 Prefill"]),
   and the GT explanations don't contain step-by-step reasoning. Asking
   for it teaches the model to produce text that the metric will not
   score well.

6. **Mix in FFAA-style hypothetical prompts on train only (`--hypothetical-ratio 0.33`).**
   FFAA [3] randomly converts ~1/3 of **training** prompts to a hypothetical
   form (`"This image is fake. Identify the evidence."`). They showed it
   stabilizes the model when the image is borderline. We use exactly that
   ratio on **train**; **val and inference always use the primary prompt**
   so eval metrics match submission.

7. **No long category-specific rubric in the user prompt.** FakeVLM's
   14 category prompts ([1, §3.2 Label Prompt Design]) are used only for
   *annotation* by teacher models — never for the student VLM at
   inference. Putting that text in our user prompt would bloat the
   context and likely hurt; the GT paragraph itself already encodes the
   category-specific style.

8. **No system prompt.** ms-swift's Qwen3-VL template adds a sensible
   default; an extra system prompt is just more tokens with no measured
   benefit. We can revisit if the model breaks the output format.

### Output target format

VLM assistant message:

```
{ground-truth complex_explanation verbatim}

Verdict: {real|fake}
```

Compressor user message (assistant target is GT simple text verbatim):

```
TASK: Rewrite the explanation below in plain everyday language ...
Avoid technical vocabulary such as "artifact", "synthesis", ...

Explanation:
{complex paragraph from GT}
```

The forbidden-word list in the compressor prompt is deliberate: SLE
penalises technical vocabulary, and copying the complex paragraph into the
simple field is the most common single failure of one-pass approaches.

## Train / val split and sampling

XPlainVerse ships a **fixed official split** — we do not randomly re-split:

| Split | Source | Fake | Real | Total |
|-------|--------|------|------|-------|
| Train | `data/XPlainVerse/train/manifest.jsonl` | 320,000 | 130,000 | **450,000** |
| Val   | `data/XPlainVerse/val/manifest.jsonl`   |  60,000 |  50,000 | **110,000** |

Train and val `sample_id`s do not overlap. Val is used for monitoring during
SFT (typically a slice) and for the final 110k submission infer + official
score (`evaluation/data/val_ground_truth.jsonl`).

### VLM train rows (default: use all 450k)

Raw train is **2.46 : 1** fake-heavy. Default build keeps **every row**:

- **320k fake + 130k real = 450k** VLM training rows
- Controlled by `--train-max-per-class` (default **`0`** = no cap)

We accept mild imbalance to maximize training data. The official baseline
uses a balanced **260k** subset (130k per class); you can reproduce that
for ablations:

```bash
python3 dataset/build_swift_jsonl.py --train-max-per-class 130000 --compressor-max-train 130000
```

### Hypothetical prompts (train only)

`--hypothetical-ratio 0.33` applies to **train** only (~149k hypothetical +
~301k primary on 450k). Val `val_vlm.jsonl` is **100% primary** — eval must
mirror inference, not leak the label in the prompt.

### Val

Val (1.2 : 1 fake:real) is close to balanced; we keep **all 110k** rows
(`--val-max-per-class 0`). During SFT, scripts slice this for cheap eval
(e.g. `val_vlm.jsonl#100` sanity, `#2000` full) without touching the full
110k held-out set.

### Compressor data

- Train default: **all 320k fake** rows (`--compressor-max-train 0`).
- Real images are **excluded** from compressor training because their GT
  simple text equals the GT complex text. Including them teaches the
  compressor to copy, which kills SLE.
- Val: **60k fake** (`val_compressor.jsonl`).

## Files written

```
dataset/
├── prompt.txt
├── build_swift_jsonl.py
├── sanity.py
├── README.md
├── train_vlm.jsonl            # 450k rows (320k fake + 130k real)
├── train_vlm_infer.jsonl      # 450k rows
├── train_compressor.jsonl     # 320k rows (fake)
├── val_vlm.jsonl              # 110k rows, primary prompts only
├── val_vlm_infer.jsonl        # ALWAYS 110k rows, primary prompts only
└── val_compressor.jsonl       # 60k rows (fake val)
```

`val_vlm_infer.jsonl` is deliberately **not** balanced — at submission
time we need every val image regardless of label.

Each row carries:

| Field | Notes |
|-------|-------|
| `id` | `{label}__{sample_id}` |
| `sample_id` | Original `image_path` stem; used to align against the GT. |
| `label` | Kept for analysis; ms-swift ignores it. |
| `prompt_kind` | `primary` or `hypothetical` on **train**; always `primary` on **val**. |
| `messages` | ms-swift conversation. |
| `images` | Absolute path list (single image per row). |

## How to run

Run from the repo root:

```bash
cd /shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/code/XPlainVerse-ACMChallenge

python3 dataset/sanity.py

python3 dataset/build_swift_jsonl.py
```

That produces six JSONL files in `dataset/`:

```
dataset/train_vlm.jsonl            # 450k rows
dataset/train_vlm_infer.jsonl      # 450k rows
dataset/train_compressor.jsonl     # 320k rows (fake)
dataset/val_vlm.jsonl              # 110k rows (primary only)
dataset/val_vlm_infer.jsonl        # 110k rows  ← submission input
dataset/val_compressor.jsonl       # 60k rows (fake)
```

Full build takes ~12 min. Defaults: **all 450k train** (~2.5:1 fake:real),
33% hypothetical on train only, all 320k fake for compressor, seed 42.

After editing `prompt.txt`, rerun only:

```bash
python3 dataset/build_swift_jsonl.py
```

**Training:** see [`TRAINING.md`](TRAINING.md) for full flag rationale. Quick start:

```bash
chmod +x scripts/train_vlm_sanity.sh scripts/train_vlm_full.sh
./scripts/train_vlm_sanity.sh          # ~100 steps, 500 train rows, 1× A100
./scripts/train_vlm_full.sh            # full 450k, 1× A100
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_vlm_full.sh  # 4× GPU
```

---

### Verify manifests (one-time)

```bash
cd /shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/code/XPlainVerse-ACMChallenge
python3 dataset/sanity.py
```

Expected: `OK: all manifest paths exist for requested splits.` (≈ 3 min on
train + val; train has 450k rows).

### Smoke build (scratch dir, ≈ 30 s)

Use before the full build to confirm prompts parse and rows look right.

```bash
python3 dataset/build_swift_jsonl.py \
  --train-max-per-class 50 \
  --val-max-per-class 50 \
  --compressor-max-train 50 \
  --compressor-max-val 50 \
  --hypothetical-ratio 0.33 \
  --output-dir /tmp/xpv_smoke

head -n 1 /tmp/xpv_smoke/train_vlm.jsonl | python3 -m json.tool
wc -l /tmp/xpv_smoke/*.jsonl
```

### Common overrides

```bash
# Val only (skip regenerating 450k train):
python3 dataset/build_swift_jsonl.py --splits val

# Disable hypothetical augmentation on train:
python3 dataset/build_swift_jsonl.py --hypothetical-ratio 0.0

# Baseline-style balanced 260k train (130k per class):
python3 dataset/build_swift_jsonl.py --train-max-per-class 130000 --compressor-max-train 130000

# Custom output dir:
python3 dataset/build_swift_jsonl.py \
  --output-dir /shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/swift_datasets

python3 dataset/build_swift_jsonl.py --help
```

Prompts are baked into each row's `messages` — any `prompt.txt` change
requires a rebuild. Same `--seed` (default 42) gives deterministic output.

## Citations

1. **FakeVLM** — *Spot the Fake: Large Multimodal Model-Based Synthetic
   Image Detection with Artifact Explanation* (NeurIPS 2025).
   [arXiv:2503.14905](https://arxiv.org/abs/2503.14905) ·
   [code](https://github.com/opendatalab/FakeVLM)
   - Source of the *short VQA prompt + free-form explanation* recipe.
   - Documents that `label-only` answers hurt both detection and
     interpretability vs `label + explanation`.

2. **Prefill-Guided Thinking (PGT)** — *Prefill-Guided Thinking for
   zero-shot detection of AI-generated images*.
   [arXiv:2506.11031](https://arxiv.org/abs/2506.11031)
   - Source of `"Examine the style and the synthesis artifacts"`.
   - +24 % Macro F1 zero-shot over baseline across Qwen, LLaVA, Llama.
   - Phrasing-sensitivity table (Table 2) — dropping
     "synthesis artifacts" loses 5–10 points.

3. **FFAA** — *Face Forgery Analysis Assistant*
   ([arXiv:2408.10072](https://arxiv.org/abs/2408.10072),
   [code](https://github.com/thu-huangzc/FFAA))
   - Source of the hypothetical-prompt augmentation (~1/3 of training
     samples). Stabilises judgments on borderline images.

4. **AntifakePrompt** —
   [arXiv:2310.17419](https://arxiv.org/abs/2310.17419) — independent
   evidence that simple VQA-style prompts ("Is this photo real?") suffice
   with a tuned prompt token.

5. **ms-swift docs** — Qwen3 / Qwen3-VL / Qwen3.5 best practices.
   - [Qwen3-VL Best Practice](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3-VL-Best-Practice.md)
   - [Qwen3 Best Practice](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3-Best-Practice.md)
     — `--loss_scale ignore_empty_think`, `--response_prefix`,
     `--enable_thinking false` for hybrid models.
   - [Custom dataset format](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md)
     — `chat_template_kwargs.enable_thinking`, per-message `"loss"` field.
