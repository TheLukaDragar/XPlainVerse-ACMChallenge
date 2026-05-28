"""Bombek1 SigLIP2-SO400M + DINOv2-Large ensemble (recipe reference in refs/ai-image-detector-siglip-dinov2/)."""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import SiglipVisionModel
from torchvision import transforms


DEFAULT_SIGLIP = "google/siglip2-so400m-patch14-384"
DEFAULT_DINOV2 = "vit_large_patch14_dinov2.lvd142m"
DEFAULT_IMAGE_SIZE = 392


class LoRALinear(nn.Module):
    """Custom LoRA for DINOv2 QKV layers (matches Bombek1 reference)."""

    def __init__(self, original: nn.Linear, rank: int, alpha: float, dropout: float = 0.1):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        for param in self.original.parameters():
            param.requires_grad = False
        self.lora_A = nn.Linear(original.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class EnsembleAIDetector(nn.Module):
    def __init__(
        self,
        siglip_model_name: str,
        dinov2_model_name: str,
        image_size: int = DEFAULT_IMAGE_SIZE,
        hidden_dim: int = 512,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.siglip = SiglipVisionModel.from_pretrained(siglip_model_name, torch_dtype=torch.bfloat16)
        self.siglip_dim = self.siglip.config.hidden_size
        self.dinov2 = timm.create_model(
            dinov2_model_name,
            pretrained=True,
            num_classes=0,
            img_size=image_size,
        )
        self.dinov2_dim = self.dinov2.num_features
        self.classifier = ClassificationHead(self.siglip_dim + self.dinov2_dim, hidden_dim, head_dropout)

    def forward(self, siglip_pixels: torch.Tensor, dinov2_pixels: torch.Tensor) -> torch.Tensor:
        siglip_features = self.siglip(pixel_values=siglip_pixels).pooler_output
        dinov2_features = self.dinov2(dinov2_pixels)
        combined = torch.cat([siglip_features.float(), dinov2_features], dim=-1)
        return self.classifier(combined)


def apply_dinov2_lora(model: EnsembleAIDetector, rank: int, alpha: int, dropout: float) -> None:
    for module in model.dinov2.modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, rank, alpha, dropout)


def create_model_with_lora(
    siglip_model_name: str = DEFAULT_SIGLIP,
    dinov2_model_name: str = DEFAULT_DINOV2,
    image_size: int = DEFAULT_IMAGE_SIZE,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    hidden_dim: int = 512,
    head_dropout: float = 0.3,
) -> EnsembleAIDetector:
    model = EnsembleAIDetector(
        siglip_model_name,
        dinov2_model_name,
        image_size=image_size,
        hidden_dim=hidden_dim,
        head_dropout=head_dropout,
    )
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    model.siglip = get_peft_model(model.siglip, lora_config)
    apply_dinov2_lora(model, lora_rank, lora_alpha, lora_dropout)
    return model


def dinov2_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def optimizer_param_groups(model: EnsembleAIDetector, lr_head: float, lr_lora: float) -> list[dict]:
    head_params = list(model.classifier.parameters())
    lora_params = [p for p in model.parameters() if p.requires_grad and not any(p is hp for hp in head_params)]
    return [
        {"params": head_params, "lr": lr_head},
        {"params": lora_params, "lr": lr_lora},
    ]
