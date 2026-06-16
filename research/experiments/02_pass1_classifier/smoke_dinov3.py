#!/usr/bin/env python3
"""Smoke test: confirm DINOv3-Large (gated) loads in container + MAC token API works."""
import os
import torch
import timm

print("timm version:", timm.__version__)
matches = timm.list_models("*dinov3*")
print("dinov3 models in timm:", matches[:12], "..." if len(matches) > 12 else "")
MODEL = "vit_large_patch16_dinov3.lvd1689m"
print("creating:", MODEL)
# num_classes=0 gives the raw Eva backbone with forward_intermediates available.
m = timm.create_model(MODEL, pretrained=True, num_classes=0)
m = m.eval().cuda()
print("created ok; num params:", sum(p.numel() for p in m.parameters()))
print("has forward_intermediates:", hasattr(m, "forward_intermediates"))
print("num_prefix_tokens:", getattr(m, "num_prefix_tokens", "?"))

x = torch.randn(2, 3, 384, 384).cuda()
with torch.no_grad():
    out = m.forward_intermediates(
        x, [20, 21, 22, 23], return_prefix_tokens=True, norm=True
    )[1]
print("forward_intermediates ok; n_layers returned:", len(out))
for i, layer in enumerate(out):
    patch, prefix = layer[0], layer[1]
    print(f"  layer{i}: patch={tuple(patch.shape)} prefix={tuple(prefix.shape)}")
# MAC concat per layer: mean(patch) + prefix tokens
avg = out[0][0].flatten(2).mean(2).unsqueeze(1)
cat = torch.cat((avg, out[0][1]), 1)
print("MAC concat shape (bs, n_tok, c):", tuple(cat.shape), "-> flatten:", cat.reshape(2, -1).shape)
print("SMOKE OK")
