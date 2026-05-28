# Codabench final evaluator (`xdd-scorer`)

Extracted from the official CodaBench scoring Docker image used on the [Explainable Deepfake Detection Challenge 2026](https://explainable-deepfake-detection.github.io/) leaderboard.

| | |
|---|---|
| Docker image | `abhijeet1317/xdd-scorer:2026-v5` |
| Organizer | abhijeet8901 (abhijeet.narang1@monash.edu) |
| Phase deadline | 16 June 2026, 01:59 CEST |
| Extracted | 28 May 2026 from `/app/program/` inside the image |

See [PROVENANCE.md](./PROVENANCE.md) for pull commands and checksum notes.

---

## Not the same as `evaluation/`

This repo has **two** evaluators:

| | `evaluation/` (val dev) | `evaluator_codabench/` (final test) |
|---|---|---|
| Ground truth | Public `val_ground_truth.jsonl` (110k) | Hidden reference on CodaBench only |
| Submission | Single `submission.jsonl` | **Zip** with 3 JSONL files |
| Task 1 detection | Not scored | **Scored** (macro F1, acc, fake/real F1) |
| Complex explanation | 0.3 BERT + **0.4 entity + 0.3 facts** (Qwen) | **BERT only** |
| Simple explanation | 0.7 BERT + 0.3 SLE | 0.7 BERT + 0.3 SLE |
| Combined explanation | Separate per-field scores | `explanation_score = 0.5·(complex_bert + simple_overall)` |

Optimizing for local Qwen entity/facts metrics helps validation R&D; the **leaderboard scorer** (as shipped in v5) does not run Qwen entity/facts.

---

## Submission format (CodaBench)

Upload a **zip** containing exactly these files (names must match):

### `detection.jsonl`

```json
{"id": "000048223b6a3cdfdaa90c26", "pred_label": 1}
```

- `id`: sample id (string)
- `pred_label`: `0` = real, `1` = fake (strict integers, not booleans)

### `complex.jsonl`

```json
{"id": "000048223b6a3cdfdaa90c26", "complex_explanation": "The puppy's fur appears overly soft..."}
```

### `simple.jsonl`

```json
{"id": "000048223b6a3cdfdaa90c26", "simple_explanation": "The puppy looks pasted onto her shirt."}
```

Empty explanation strings are skipped (row not scored for that field). Unknown or duplicate ids fail validation.

---

## Scored outputs (`scores.json`)

| Key | Meaning |
|---|---|
| `detection_macro_f1` | Macro F1 over real/fake (coverage-weighted) |
| `detection_accuracy` | Accuracy (coverage-weighted) |
| `detection_fake_f1` | Fake class F1 |
| `detection_real_f1` | Real class F1 |
| `complex_bert_f1` | BERTScore F1 vs hidden complex reference |
| `simple_bert_f1` | BERTScore F1 vs hidden simple reference |
| `simple_sle_raw` | Raw SLE simplicity score |
| `simple_sle_norm` | SLE clipped to [-1, 4] then mapped to [0, 1] |
| `simple_overall_score` | `0.7·simple_bert + 0.3·simple_sle_norm` |
| `explanation_score` | `0.5·(complex_bert + simple_overall)` |

Missing explanation rows reduce scores via a coverage factor (partial submissions get proportionally lower explanation scores).

---

## Directory layout

```
evaluator_codabench/
├── README.md                 # this file
├── PROVENANCE.md             # image pull / extraction notes
├── data/xdd/                 # placeholder paths (SLE weights live in Docker image)
└── program/
    ├── scoring.py            # CodaBench entry point
    ├── requirements.txt
    ├── metadata.yaml
    └── metrics/
        ├── bertscore_metric.py
        ├── detection.py
        ├── sle_metric.py
        └── validation.py
```

CodaBench invokes:

```bash
python3 /app/program/scoring.py /app/input /app/output
```

Expected input layout:

```
input/
├── ref/
│   └── config.json          # {"reference_path": "/path/to/hidden_reference.jsonl", ...}
└── res/
    └── submission.zip       # or loose detection/complex/simple.jsonl
```

Reference rows (hidden on leaderboard) include fields like `id`, `label`, `complex_reference`, `simple_reference`, and `score_explanations: true|false`.

---

## Convert val submission → CodaBench zip

From repo root, turn a single-file val submission into the 3-file zip:

```bash
python3 scripts/build_codabench_submission.py \
  --input /path/to/submission.jsonl \
  --output /path/to/submission.zip
```

Input lines use the val format: `sample_id`, `label` (`real`|`fake`), `complex_explanation`, `simple_explanation`.

---

## Run locally (Docker)

```bash
docker pull abhijeet1317/xdd-scorer:2026-v5

# Dry run with mocks (no GPU models):
docker run --rm \
  -v "$PWD/scorer_input:/app/input" \
  -v "$PWD/scorer_output:/app/output" \
  abhijeet1317/xdd-scorer:2026-v5 \
  python3 /app/program/scoring.py /app/input /app/output --mock-bert --mock-sle
```

On CodaBench, mocks are not used; BERTScore and SLE run on GPU when available.

---

## Run on Elixir Lj (Apptainer)

Docker is not on the Slurm login node. Use Apptainer on the GPU node:

```bash
# One-time pull (already done if ~/containers/xdd-scorer_2026-v5.sif exists):
apptainer pull ~/containers/xdd-scorer_2026-v5.sif docker://abhijeet1317/xdd-scorer:2026-v5

# Score via wrapper (uses --no-home to avoid host ~/.local numpy/sklearn conflicts):
./scripts/run_codabench_scorer_lj.sh /path/to/input /path/to/output
```

Add `--mock-bert --mock-sle` for a fast format check without downloading DeBERTa/SLE weights.

---

## Run extracted Python directly

Requires the same deps as `program/requirements.txt` plus SLE model weights at `/app/data/xdd/sle` (bundled in Docker, not in this git tree).

```bash
cd evaluator_codabench/program
pip install -r requirements.txt
python3 scoring.py /path/to/input /path/to/output --mock-bert --mock-sle
```

Use mocks unless you have copied SLE weights from the container and set `reference_path` in `input/ref/config.json`.

---

## Lj container cache

| Path | Description |
|---|---|
| `~/containers/xdd-scorer_2026-v5.sif` | Apptainer image (4.4 GB, pulled 28 May 2026) |

Pull command:

```bash
apptainer pull ~/containers/xdd-scorer_2026-v5.sif docker://abhijeet1317/xdd-scorer:2026-v5
```
