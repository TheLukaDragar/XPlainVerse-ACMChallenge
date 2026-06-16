"""DINOv3-Large + Multi-Aspect Classification (MAC) head — reproduces DINO-MAC.

DINO-MAC (ShallowReal, 1st place NTIRE 2026 Robust Deepfake Detection, github.com/qcf-568/DINOMAC):
  - backbone: timm `vit_large_patch16_dinov3.lvd1689m` (gated; needs HF access)
  - LoRA r=32, alpha=64 on attn.qkv
  - MAC head: for each of the last 4 transformer layers, concat
      mean(patch_tokens) [AVG] + CLS + 4 REG prefix tokens = 6 x 1024 = 6144-dim
    -> MLP -> logit. Deep supervision: loss on all 4, infer on the last layer.

We output ONE logit per head (binary, focal loss) to match the rest of the Pass-1
pipeline (sigmoid -> p_fake), rather than DINO-MAC's 2-class CE.
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
from torchvision import transforms

DEFAULT_DINOV3 = "vit_large_patch16_dinov3.lvd1689m"
DEFAULT_LAYERS = (20, 21, 22, 23)
DEFAULT_IMAGE_SIZE = 384


class LoRALinear(nn.Module):
    """Custom LoRA wrapper for the fused attn.qkv Linear (matches our DINOv2 recipe)."""

    def __init__(self, original: nn.Linear, rank: int, alpha: float, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        for p in self.original.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(original.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class MACHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DINOv3MAC(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_DINOV3,
        layers: tuple[int, ...] = DEFAULT_LAYERS,
        head_hidden: int = 1024,
        head_dropout: float = 0.2,
        deep_supervision: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        # Freeze the backbone; only LoRA adapters (added later) + MAC heads train.
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.layers = list(layers)
        self.deep_supervision = deep_supervision
        c = self.backbone.embed_dim
        self.n_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
        in_dim = c * (self.n_prefix + 1)  # +1 for [AVG] mean-patch token
        self.heads = nn.ModuleList(
            [MACHead(in_dim, head_hidden, head_dropout) for _ in self.layers]
        )

    def forward(self, x: torch.Tensor):
        bs = x.size(0)
        feats = self.backbone.forward_intermediates(
            x, self.layers, return_prefix_tokens=True, norm=True
        )[1]
        logits = []
        for i, layer in enumerate(feats):
            patch, prefix = layer[0], layer[1]  # (B,C,H,W), (B,n_prefix,C)
            avg = patch.flatten(2).mean(2).unsqueeze(1)  # (B,1,C)
            cat = torch.cat((avg, prefix), 1).reshape(bs, -1).float()  # (B, (n_prefix+1)*C)
            logits.append(self.heads[i](cat))
        if self.training and self.deep_supervision:
            return logits  # list of 4 (final = logits[-1])
        return logits[-1]


def apply_dinov3_lora(model: DINOv3MAC, rank: int, alpha: int, dropout: float) -> None:
    for module in model.backbone.modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, rank, alpha, dropout)


def create_dinov3_mac(
    model_name: str = DEFAULT_DINOV3,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    head_hidden: int = 1024,
    head_dropout: float = 0.2,
    deep_supervision: bool = True,
    pretrained: bool = True,
) -> DINOv3MAC:
    model = DINOv3MAC(
        model_name=model_name,
        layers=layers,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
        deep_supervision=deep_supervision,
        pretrained=pretrained,
    )
    apply_dinov3_lora(model, lora_rank, lora_alpha, lora_dropout)
    return model


def dinov3_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def dinov3_param_groups(model: DINOv3MAC, lr_head: float, lr_lora: float) -> list[dict]:
    head_params = list(model.heads.parameters())
    head_ids = {id(p) for p in head_params}
    lora_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return [
        {"params": head_params, "lr": lr_head},
        {"params": lora_params, "lr": lr_lora},
    ]
