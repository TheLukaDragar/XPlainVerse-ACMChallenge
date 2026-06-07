# Container image (NVIDIA CUDA / Slurm Pyxis)

This image installs:

1. **Evaluator stack** — everything in `evaluation/env/xplainverse_eval_env.txt` (PyTorch `+cu124`, pinned deps, `transformers` from the git revision in that file).
2. **Baseline stack** — [ms-swift](https://github.com/modelscope/ms-swift) (`pip install -e`) and `vllm`, matching `baselines/README.md`, plus `qwen_vl_utils` and `decord` for Qwen-VL–style inference paths.

Base image: `nvidia/cuda:12.4.1-devel-ubuntu22.04` so the container has **nvcc** (CUDA toolkit) as well as user-space libraries matching **cu124** wheels. vLLM’s engine can JIT-compile FlashInfer ops; a **runtime-only** image often fails with `nvcc: not found` even though PyTorch runs fine.

## Local build

From the repository root:

```bash
docker build -f docker/Dockerfile -t xplainverse:dev .
```

Pin ms-swift to a branch or tag:

```bash
docker build -f docker/Dockerfile --build-arg MS_SWIFT_REF=vX.Y.Z -t xplainverse:dev .
```

## Ljubljana training image (`Dockerfile.lj`)

**`docker/Dockerfile.lj`** mirrors the eval image stack: **CUDA 13 + torch 2.11 (cu130) + vLLM** on `elixir-lj-gpu-01`. Use this for ms-swift training, GRPO/vLLM, and **Pass-1** (Bombek SigLIP2+DINOv2 ensemble needs **`timm>=1.0.15`** — added in branch `cursor/lj-docker-timm-a544`).

Do **not** use `~/xplainverse_exec.sh` (local Apptainer SIF) for Pass-1/timm — that image has no `timm` and is read-only. Use:

```bash
./scripts/lj_ghcr_image_exec.sh bash scripts/run_pass1_ensemble_bombek_lj.sh
```

```bash
docker build -f docker/Dockerfile.lj -t xplainverse-lj-train:latest .
```

Override the wheel only if you change the Torch/CUDA line (pick a filename that matches `torch.__version__` and `torch.version.cuda`).

### CI-built image (GitHub Actions)

On pushes (when `docker/Dockerfile.lj` or related paths change), workflow **`.github/workflows/container-lj.yml`** pushes to a **separate GHCR package** from the CUDA 13 / vLLM eval image:

| Package | Dockerfile | Tags |
|---------|------------|------|
| `ghcr.io/<owner>/<repo>-lj` | `docker/Dockerfile.lj` | `latest`, `sha-<7hex>` |
| `ghcr.io/<owner>/<repo>` | `docker/Dockerfile` | `latest`, `sha-<7hex>` |

Apptainer example (Lj training):

```bash
apptainer pull docker://ghcr.io/<owner>/<repo>-lj:latest
```

Legacy note: an early Lj build was pushed to the eval package with tag `sha-6224dd3-slurm`; use that tag only until the next `-lj` CI run.

## CI-published image

GitHub Actions builds and pushes to GHCR when relevant paths change on the default branch (see `.github/workflows/container.yml`). Image name:

`ghcr.io/<owner>/<repo>` (lowercased), tags `latest` (default branch) and `sha-<full>`.

### PyTorch / NVRTC (`libnvrtc-builtins.so.13.0`)

Some installs end up with **CUDA 13 PyTorch** while the container only has **CUDA 12.x** user libraries (or vice versa). `torch.compile` then fails opening `libnvrtc-builtins.so.*`.

- **Docker:** After `ms-swift`, the Dockerfile **reinstalls** `torch==2.6.0+cu124` and installs **vLLM** with `docker/torch-cu124-constraints.txt` so pip does not silently upgrade the stack.
- **Smoke script:** Sets `TORCH_COMPILE_DISABLE=1` by default; override with `TORCH_COMPILE_DISABLE=0` if your environment is consistent.
- **Manual fix:** `python -c "import torch; print(torch.__version__, torch.version.cuda)"` — if this does not match your intended CUDA line, reinstall torch from [pytorch.org](https://pytorch.org) for the same major as your image/driver.

## Slurm + Pyxis example

Exact flags depend on your site; a typical pattern:

```bash
srun --gpus=1 \
  --container-image=ghcr.io/your-org/your-repo:latest \
  --container-mounts=/path/on/host:/workspace/mount \
  bash -lc 'cd /workspace/XPlainVerse-ACMChallenge/evaluation && python evaluate_val.py --help'
```

Mount HF cache or datasets as needed. For private GHCR images, configure registry credentials on the cluster (Enroot/Pyxis auth).

## Dependency overlaps

ms-swift and vLLM may pull versions that differ from the evaluator pins. For strict evaluator reproducibility, run evaluation in an environment created only from `evaluation/env/`; this image optimizes for **one container that includes baselines + evaluator** for cluster convenience.
