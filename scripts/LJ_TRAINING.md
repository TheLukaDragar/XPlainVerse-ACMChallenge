# Ljubljana multi-GPU VLM training (elixir-lj-gpu-01)

Use the Apptainer SIF via `~/xplainverse_exec.sh` — not the host `(base)` conda shell.

**Slurm login node (Cursor worker):** dispatch GPU/container commands with `./scripts/lj_gpu_exec.sh …` from the repo root (see `scripts/lj_gpu_exec.sh`). **Already on `elixir-lj-gpu-*`:** run `~/xplainverse_exec.sh …` directly.

## One-time: build JSONL manifests

`dataset/*.jsonl` is **gitignored**; generate it on the GPU node (or via `lj_gpu_exec`).

**Full challenge train/val** (450k / 110k manifests — long I/O pass, ~45+ minutes typical):

```bash
./scripts/lj_gpu_exec.sh python3 dataset/build_swift_jsonl.py \
  --data-root /home/jakob/luka/data/XPlainVerse --output-dir dataset
```

Detached Slurm batch (survives logout; same node/partition defaults as `lj_gpu_exec`):

```bash
./scripts/sbatch_jsonl_build_lj.sh
```

**Small capped split** (smoke / layout check only; same script):

```bash
./scripts/lj_gpu_exec.sh python3 dataset/build_swift_jsonl.py \
  --data-root /home/jakob/luka/data/XPlainVerse --output-dir dataset \
  --train-max-per-class 2000 --val-max-per-class 1000
```

On the GPU node / inside an interactive `~/xplainverse_exec.sh` shell:

```bash
cd /workspace/XPlainVerse-ACMChallenge
python3 dataset/build_swift_jsonl.py \
  --data-root /home/jakob/luka/data/XPlainVerse \
  --output-dir dataset
```

Baseline-style balanced train (130k/class):

```bash
  ... --train-max-per-class 130000
```

## Full 4-GPU training (450k train, default hyperparams)

Login node:

```bash
cd /home/jakob/luka/code/XPlainVerse-ACMChallenge
./scripts/lj_gpu_exec.sh bash scripts/train_vlm_full_lj.sh
```

GPU node (container shell):

```bash
~/xplainverse_exec.sh bash /workspace/XPlainVerse-ACMChallenge/scripts/train_vlm_full_lj.sh
```

Defaults: `NPROC_PER_NODE=4`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `PER_DEVICE_BS=2`, `GRAD_ACCUM=4`, `--deepspeed zero2` **when the `deepspeed` package is installed**, else **DDP only** (see DeepSpeed section below), output `/home/jakob/luka/runs/vlm_full`.

## Fallback (no flash_attn / packing issues)

```bash
ATTN_IMPL=sdpa PACKING=false PADDING_FREE=false REPORT_TO=tensorboard \
  ~/xplainverse_exec.sh bash /workspace/XPlainVerse-ACMChallenge/scripts/train_vlm_full_lj.sh
```

## DeepSpeed ZeRO (optional)

The lj Apptainer image may not include the `deepspeed` Python package. If `import deepspeed` fails, `train_vlm_full_lj.sh` **drops `--deepspeed`** and uses plain **DDP** across 4 GPUs (still correct, usually more VRAM).

To enable ZeRO-2 after installing inside the SIF (or a rebuilt image):

```bash
pip install 'deepspeed>=0.14.0'
```

## Smoke / debug (do not run full 450k)

Prefer `MAX_STEPS` + `TRAIN_SLICE` so training stops after a few optimizer steps (multi-GPU **torchrun / DDP**; ZeRO-2 only if `deepspeed` is installed in the SIF):

```bash
cd /home/jakob/luka/code/XPlainVerse-ACMChallenge
LJ_GPU_TIME=01:00:00 ./scripts/lj_gpu_exec.sh bash -lc \
  'REPORT_TO=tensorboard MAX_STEPS=4 TRAIN_SLICE=32 VAL_SLICE=4 \
   SAVE_STEPS=999999 EVAL_STEPS=999999 PREDICT_WITH_GENERATE=false \
   OUTPUT_DIR=/home/jakob/luka/runs/vlm_smoke_lj bash scripts/train_vlm_full_lj.sh'
```

(`TRAIN_SLICE` appends ms-swift `path.jsonl#N` cap; `MAX_STEPS` maps to `--max_steps` and avoids a full epoch.)

## Paths

| | Path |
|---|---|
| Host code | `/home/jakob/luka/code/XPlainVerse-ACMChallenge` |
| Container code | `/workspace/XPlainVerse-ACMChallenge` |
| Images / manifests | `/home/jakob/luka/data/XPlainVerse` |
| Checkpoints | `/home/jakob/luka/runs/vlm_full` |

## Verify environment

```bash
~/xplainverse_exec.sh python3 -c "import torch; print(torch.__version__, torch.cuda.device_count())"
~/xplainverse_exec.sh which swift
~/xplainverse_exec.sh python3 -c "import importlib.util as u; print('flash_attn', 'ok' if u.find_spec('flash_attn') else 'missing')"
```

Expect **4** GPUs and `swift` on `/usr/local/bin/swift`. `flash_attn` may be absent; the training script falls back to `sdpa` and disables packing.
