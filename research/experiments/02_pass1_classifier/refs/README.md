# Pass-1 reference implementations (code only, no weights)

Cloned from Hugging Face for recipe reference when writing `train.py`.

| Directory | Source | What to read |
|-----------|--------|--------------|
| `simplicityprevails/` | [Lunahera/simplicityprevails](https://huggingface.co/Lunahera/simplicityprevails) | `models.py` — frozen VFM + linear head loading; paper recipe: AdamW 1e-3, batch 128, 2 epochs, no aug |
| `ai-image-detector-siglip-dinov2/` | [Bombek1/ai-image-detector-siglip-dinov2](https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2) | `model.py` — dual SigLIP2+DINOv2 ensemble + LoRA (future Step 4) |

Refresh (skip LFS weights):

```bash
cd research/experiments/02_pass1_classifier/refs
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/Lunahera/simplicityprevails
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2
rm -rf */.git simplicityprevails/weights ai-image-detector-siglip-dinov2/pytorch_model.pt
```

Our training entrypoint: `../train.py` (XPlainVerse 450k, balanced 260k subset).

**Note:** HF repos ship `.gitattributes` that mark `*.gz` etc. as Git LFS. Do not commit those
files into this git repo — GitHub blocks **new LFS objects on public forks** (see push error
`can not upload new objects to public fork`). The PE-CLIP BPE vocab gzip is omitted here;
only `models.py` / `model.py` are needed for recipe reference.
