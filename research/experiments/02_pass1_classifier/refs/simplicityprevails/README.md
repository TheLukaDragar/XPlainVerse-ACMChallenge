---
license: apache-2.0
pipeline_tag: image-classification
tags:
  - computer-vision
  - image-classification
  - pytorch
library_name: pytorch
---

Official repository for the paper "Simplicity Prevails: The Emergence of Generalizable AIGI Detection in Visual Foundation Models"(https://arxiv.org/pdf/2602.01738)

If you have any questions, please feel free to open a discussion in the Community tab. For direct inquiries, you can also reach out to us via email at 2450042008@mails.szu.edu.cn.

# VFM Baselines Release

This directory contains the 7 vision foundation model baselines used in the paper:

- `MetaCLIP-Linear`
- `MetaCLIP2-Linear`
- `SigLIP-Linear`
- `SigLIP2-Linear`
- `PE-CLIP-Linear`
- `DINOv2-Linear`
- `DINOv3-Linear`

## Contents

- `models.py`: unified model-loading code for all 7 baselines
- `test_vfm_baselines.py`: unified evaluation script
- `weights/`: released checkpoints
- `core/vision_encoder/`: vendored PE vision encoder code required by `PE-CLIP-Linear`

## Model Names

The unified loader and test script accept these names:

- `metacliplin`
- `metaclip2lin`
- `sigliplin`
- `siglip2lin`
- `pelin`
- `dinov2lin`
- `dinov3lin`

The paper names such as `MetaCLIP-Linear` and `DINOv3-Linear` are also accepted.

## Usage

Evaluate a single model:

```bash
python test_vfm_baselines.py \
  --model sigliplin \
  --real-dir /path/to/0_real \
  --fake-dir /path/to/1_fake \
  --max-samples 100
```

Evaluate all 7 models:

```bash
python test_vfm_baselines.py \
  --model all \
  --real-dir /path/to/0_real \
  --fake-dir /path/to/1_fake \
  --max-samples 100
```

Optional arguments:

- `--checkpoint`: override the default checkpoint for single-model evaluation
- `--batch-size`: batch size for evaluation
- `--num-workers`: dataloader workers
- `--device`: explicit device such as `cuda:0` or `cpu`
- `--save-json`: save results to a JSON file

## Dependencies

The release code expects these Python packages:

- `torch`
- `torchvision`
- `transformers`
- `scikit-learn`
- `Pillow`
- `timm`
- `einops`
- `ftfy`
- `regex`
- `huggingface_hub`

## Notes

- The clip-family and DINO-family baselines instantiate the backbone from Hugging Face model configs and then load the released checkpoint.
- `PE-CLIP-Linear` uses the vendored `core/vision_encoder` code in this directory.
- The checkpoints in `weights/` are arranged locally for packaging convenience. For public release, they can be uploaded as the same filenames.
