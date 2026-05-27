"""Minimal model-loading code for the 7 VFM baselines in the paper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoConfig, AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"

MODEL_SPECS = {
    "metacliplin": {
        "paper_name": "MetaCLIP-Linear",
        "checkpoint": "metacliplin0.pth",
        "hf_model": "facebook/metaclip-h14-fullcc2.5b",
        "feature_dim": 1280,
        "image_size": 224,
        "pooler_output": True,
    },
    "metaclip2lin": {
        "paper_name": "MetaCLIP2-Linear",
        "checkpoint": "metaclip2lin0.pth",
        "hf_model": "facebook/metaclip-2-worldwide-giant",
        "feature_dim": 1280,
        "image_size": 224,
        "pooler_output": True,
    },
    "sigliplin": {
        "paper_name": "SigLIP-Linear",
        "checkpoint": "sigliplin0.pth",
        "hf_model": "google/siglip-large-patch16-384",
        "feature_dim": 1024,
        "image_size": 384,
        "pooler_output": True,
    },
    "siglip2lin": {
        "paper_name": "SigLIP2-Linear",
        "checkpoint": "siglip2lin0.pth",
        "hf_model": "google/siglip2-giant-opt-patch16-384",
        "feature_dim": 1536,
        "image_size": 384,
        "pooler_output": True,
    },
    "pelin": {
        "paper_name": "PE-CLIP-Linear",
        "checkpoint": "pelin0.pth",
        "feature_dim": 1024,
        "image_size": 336,
        "pooler_output": False,
    },
    "dinov2lin": {
        "paper_name": "DINOv2-Linear",
        "checkpoint": "dinov2lin0.pth",
        "feature_dim": 1024,
        "pooler_output": False,
    },
    "dinov3lin": {
        "paper_name": "DINOv3-Linear",
        "checkpoint": "dinov3lin0.pth",
        "hf_model": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "feature_dim": 4096,
        "pooler_output": False,
    },
}

ALIASES = {
    "MetaCLIP-Linear": "metacliplin",
    "MetaCLIP2-Linear": "metaclip2lin",
    "SigLIP-Linear": "sigliplin",
    "SigLIP2-Linear": "siglip2lin",
    "PE-CLIP-Linear": "pelin",
    "DINOv2-Linear": "dinov2lin",
    "DINOv3-Linear": "dinov3lin",
}


def canonical_model_name(name: str) -> str:
    if name in MODEL_SPECS:
        return name
    if name in ALIASES:
        return ALIASES[name]
    raise KeyError(f"Unknown model: {name}")


def default_checkpoint_path(model_name: str) -> Path:
    model_name = canonical_model_name(model_name)
    return WEIGHTS_DIR / MODEL_SPECS[model_name]["checkpoint"]


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_checkpoint(checkpoint_path: str | Path) -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    normalized = {}
    for key, value in checkpoint.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def _infer_feature_dim(state_dict: dict, default_dim: int) -> int:
    head_weight = state_dict.get("head.weight")
    if isinstance(head_weight, torch.Tensor) and head_weight.ndim == 2:
        return int(head_weight.shape[1])
    return default_dim


def _load_image_processor(model_name: str):
    try:
        return AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
    except Exception:
        try:
            return AutoImageProcessor.from_pretrained(model_name)
        except Exception:
            return None


def _load_backbone(model_name: str):
    try:
        return AutoModel.from_pretrained(model_name, local_files_only=True)
    except Exception:
        config = AutoConfig.from_pretrained(model_name)
        return AutoModel.from_config(config)


class _PoolerLinearModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, 2)

    def forward(self, x):
        with torch.no_grad():
            outputs = self.backbone(x)
            features = outputs.pooler_output.float()
        return self.head(features)


class _ClsTokenLinearModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, 2)

    def forward(self, x):
        with torch.no_grad():
            outputs = self.backbone(x)
            features = outputs.last_hidden_state[:, 0].float()
        return self.head(features)


class _PELinearModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, 2)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            if isinstance(features, torch.Tensor):
                features = features.float()
        return self.head(features)


def _finalize_model(model: nn.Module, state_dict: dict, device=None) -> nn.Module:
    model.load_state_dict(state_dict, strict=False)
    model.to(_resolve_device(device))
    model.eval()
    return model


def _build_clip_transform(image_size: int, image_processor=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if image_processor is not None:
        mean = getattr(image_processor, "image_mean", mean)
        std = getattr(image_processor, "image_std", std)
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _build_dino_transform():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_metacliplin(checkpoint_path: str | Path | None = None, device=None):
    spec = MODEL_SPECS["metacliplin"]
    checkpoint_path = checkpoint_path or default_checkpoint_path("metacliplin")
    state_dict = _load_checkpoint(checkpoint_path)
    feature_dim = _infer_feature_dim(state_dict, spec["feature_dim"])
    image_processor = _load_image_processor(spec["hf_model"])
    backbone = _load_backbone(spec["hf_model"])
    model = _PoolerLinearModel(backbone.vision_model, feature_dim)
    model = _finalize_model(model, state_dict, device=device)
    return model, _build_clip_transform(spec["image_size"], image_processor)


def load_metaclip2lin(checkpoint_path: str | Path | None = None, device=None):
    spec = MODEL_SPECS["metaclip2lin"]
    checkpoint_path = checkpoint_path or default_checkpoint_path("metaclip2lin")
    state_dict = _load_checkpoint(checkpoint_path)
    feature_dim = _infer_feature_dim(state_dict, spec["feature_dim"])
    image_processor = _load_image_processor(spec["hf_model"])
    backbone = _load_backbone(spec["hf_model"])
    model = _PoolerLinearModel(backbone.vision_model, feature_dim)
    model = _finalize_model(model, state_dict, device=device)
    return model, _build_clip_transform(spec["image_size"], image_processor)


def load_sigliplin(checkpoint_path: str | Path | None = None, device=None):
    spec = MODEL_SPECS["sigliplin"]
    checkpoint_path = checkpoint_path or default_checkpoint_path("sigliplin")
    state_dict = _load_checkpoint(checkpoint_path)
    feature_dim = _infer_feature_dim(state_dict, spec["feature_dim"])
    image_processor = _load_image_processor(spec["hf_model"])
    backbone = _load_backbone(spec["hf_model"])
    model = _PoolerLinearModel(backbone.vision_model, feature_dim)
    model = _finalize_model(model, state_dict, device=device)
    return model, _build_clip_transform(spec["image_size"], image_processor)


def load_siglip2lin(checkpoint_path: str | Path | None = None, device=None):
    spec = MODEL_SPECS["siglip2lin"]
    checkpoint_path = checkpoint_path or default_checkpoint_path("siglip2lin")
    state_dict = _load_checkpoint(checkpoint_path)
    feature_dim = _infer_feature_dim(state_dict, spec["feature_dim"])
    image_processor = _load_image_processor(spec["hf_model"])
    backbone = _load_backbone(spec["hf_model"])
    model = _PoolerLinearModel(backbone.vision_model, feature_dim)
    model = _finalize_model(model, state_dict, device=device)
    return model, _build_clip_transform(spec["image_size"], image_processor)


def load_dinov2lin(checkpoint_path: str | Path | None = None, device=None):
    checkpoint_path = checkpoint_path or default_checkpoint_path("dinov2lin")
    state_dict = _load_checkpoint(checkpoint_path)
    feature_dim = _infer_feature_dim(state_dict, MODEL_SPECS["dinov2lin"]["feature_dim"])
    if feature_dim == 1536:
        candidates = ["facebook/dinov2-giant", "facebook/dinov2-large"]
    elif feature_dim == 1024:
        candidates = ["facebook/dinov2-large", "facebook/dinov2-base"]
    elif feature_dim == 768:
        candidates = ["facebook/dinov2-base", "facebook/dinov2-small"]
    else:
        candidates = ["facebook/dinov2-large"]

    last_error = None
    backbone = None
    for candidate in candidates:
        try:
            backbone = _load_backbone(candidate)
            break
        except Exception as exc:
            last_error = exc
    if backbone is None:
        raise RuntimeError(f"Failed to load DINOv2 backbone: {last_error}")

    model = _ClsTokenLinearModel(backbone, feature_dim)
    model = _finalize_model(model, state_dict, device=device)
    return model, _build_dino_transform()


def load_dinov3lin(checkpoint_path: str | Path | None = None, device=None):
    checkpoint_path = checkpoint_path or default_checkpoint_path("dinov3lin")
    state_dict = _load_checkpoint(checkpoint_path)
    feature_dim = _infer_feature_dim(state_dict, MODEL_SPECS["dinov3lin"]["feature_dim"])
    backbone = _load_backbone(MODEL_SPECS["dinov3lin"]["hf_model"])
    model = _ClsTokenLinearModel(backbone, feature_dim)
    model = _finalize_model(model, state_dict, device=device)
    return model, _build_dino_transform()


def load_pelin(checkpoint_path: str | Path | None = None, device=None):
    checkpoint_path = checkpoint_path or default_checkpoint_path("pelin")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as pe_transforms

    state_dict = _load_checkpoint(checkpoint_path)
    feature_dim = _infer_feature_dim(state_dict, MODEL_SPECS["pelin"]["feature_dim"])
    clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=False)
    model = _PELinearModel(clip_model.visual, feature_dim)
    model = _finalize_model(model, state_dict, device=device)
    return model, pe_transforms.get_image_transform(MODEL_SPECS["pelin"]["image_size"])


LOADERS: dict[str, Callable] = {
    "metacliplin": load_metacliplin,
    "metaclip2lin": load_metaclip2lin,
    "sigliplin": load_sigliplin,
    "siglip2lin": load_siglip2lin,
    "pelin": load_pelin,
    "dinov2lin": load_dinov2lin,
    "dinov3lin": load_dinov3lin,
}


def load_model(model_name: str, checkpoint_path: str | Path | None = None, device=None):
    model_name = canonical_model_name(model_name)
    return LOADERS[model_name](checkpoint_path=checkpoint_path, device=device)
