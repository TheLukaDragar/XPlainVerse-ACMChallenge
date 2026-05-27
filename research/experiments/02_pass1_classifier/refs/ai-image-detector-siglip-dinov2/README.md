---
license: mit
tags:
  - image-classification
  - ai-detection
  - deepfake-detection
  - siglip
  - dinov2
  - lora
  - pytorch
  - quality-agnostic
datasets:
  - nebula-9000/OpenFake
metrics:
  - accuracy
  - roc_auc
pipeline_tag: image-classification
---

# AI Image Detector (SigLIP2 + DINOv2 Ensemble)

A high-accuracy, **quality-agnostic** model for detecting AI-generated images, achieving **0.9997 AUC** on validation and strong cross-dataset generalization.

## Key Features

- **Quality-agnostic**: Performs consistently on both pristine and degraded images (JPEG compression, blur, noise)
- **Dual-encoder architecture**: Combines SigLIP2's semantic understanding with DINOv2's self-supervised features
- **Efficient fine-tuning**: Uses LoRA adapters (~8M trainable params out of ~740M total)
- **Production-ready**: Tested on 10+ external datasets

## Performance

### Validation Results (OpenFake, 5K images)

| Metric | Clean Images | Degraded Images | Average |
|--------|--------------|-----------------|---------|
| AUC | 0.9998 | 0.9995 | **0.9997** |
| Accuracy | 99.24% | 98.96% | 99.10% |

**Quality-agnostic verification**: AUC gap between clean and degraded images is only **0.0003**, confirming robust performance across image quality levels.

### Cross-Dataset Generalization

#### Real Image Datasets (Target: Classify as Real)

| Dataset | Samples | Accuracy | Mean P(AI) |
|---------|---------|----------|------------|
| Food-101 | 300 | **100.00%** | 0.032 |
| COCO 2017 | 300 | 90.67% | 0.135 |
| Cats vs Dogs | 300 | **99.67%** | 0.036 |
| Stanford Cars | 300 | 94.67% | 0.110 |
| Oxford Flowers | 300 | 95.67% | 0.115 |
| **Average** | — | **96.13%** | — |

#### AI-Generated Image Datasets (Target: Classify as AI)

| Dataset | Generator | Samples | Accuracy | Mean P(AI) |
|---------|-----------|---------|----------|------------|
| DALL-E 3 | OpenAI | 300 | **100.00%** | 0.993 |
| Midjourney V6 | Midjourney | 300 | 96.33% | 0.936 |
| **Average** | — | — | **98.17%** | — |

#### Mixed Benchmark Datasets

| Dataset | Samples | Accuracy | AUC | F1 |
|---------|---------|----------|-----|-----|
| AI-or-Not | 500 | **96.80%** | **0.9986** | 97.04% |

**Overall cross-dataset accuracy: 97.15%**

### Supported AI Generators

Trained on OpenFake dataset which includes images from 25+ generators:

- **Diffusion models**: Stable Diffusion (1.5, 2.1, XL, 3.5), Flux (1.0, 1.1 Pro), DALL-E 3, Midjourney (v5, v6), Imagen, Kandinsky
- **GANs**: StyleGAN, ProGAN, BigGAN
- **Other**: GPT-Image-1, Firefly, Ideogram, and more

## Usage

### Installation

```bash
pip install torch torchvision transformers timm peft pillow
```

### Quick Start

```python
from huggingface_hub import hf_hub_download
from model import AIImageDetector

# Download model
model_path = hf_hub_download(
    repo_id="Bombek1/ai-image-detector-siglip-dinov2",
    filename="pytorch_model.pt"
)

# Initialize detector
detector = AIImageDetector(model_path)

# Predict single image
result = detector.predict("path/to/image.jpg")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"P(AI): {result['probability']:.4f}")
```

### Batch Processing

```python
from pathlib import Path

images = list(Path("./images").glob("*.jpg"))
for img_path in images:
    result = detector.predict(img_path)
    print(f"{img_path.name}: {result['prediction']} ({result['confidence']:.1%})")
```

## Model Architecture

```
EnsembleAIDetector (~740M parameters, ~8M trainable)
├── SigLIP2-SO400M-patch14-384 (with LoRA r=32 on q_proj, v_proj)
│   └── Output: 1152-dim features
├── DINOv2-Large-patch14 (with LoRA r=32 on qkv)
│   └── Output: 1024-dim features
└── ClassificationHead
    ├── LayerNorm(2176)
    ├── Linear(2176 → 512) + GELU + Dropout(0.3)
    ├── Linear(512 → 256) + GELU + Dropout(0.3)
    └── Linear(256 → 1) → Sigmoid
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | OpenFake (~95K train, 5K val) |
| Image Size | 392×392 |
| Epochs | 5 |
| Batch Size | 16 (effective: 144 with grad accum) |
| Learning Rate | 2e-4 (head), 5e-5 (LoRA) |
| Scheduler | Cosine with warmup |
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| Loss | Focal Loss (γ=2, α=0.25) |

### Quality-Agnostic Augmentations

The model is trained with aggressive image degradation to ensure robustness:

- JPEG compression (quality 30-95)
- Gaussian blur (σ up to 2.0)
- Gaussian noise (σ up to 0.05)
- Resize artifacts (down to 50% then back up)
- Color jitter, random crops, flips

## Limitations

| Limitation | Details |
|------------|---------|
| **Low-resolution images** | Performance degrades on images <128×128 (e.g., CIFAKE 32×32 dataset shows ~50% accuracy) |
| **COCO-style images** | ~9% false positive rate on casual/cluttered real photos |
| **Artistic macro photography** | Professional studio/macro shots may occasionally trigger false positives (~5%) |
| **Non-photographic content** | Designed for photographs; screenshots, graphics, and illustrations may not work well |

## Files

- `pytorch_model.pt` — Full checkpoint with LoRA weights
- `model.py` — Inference code with `AIImageDetector` class
- `config.json` — Model configuration

## Citation

```bibtex
@misc{ai-image-detector-2025,
  author = {Bombek1},
  title = {AI Image Detector (SigLIP2 + DINOv2 Ensemble)},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2}
}
```

## License

MIT License
