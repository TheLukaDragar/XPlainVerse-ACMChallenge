#!/usr/bin/env python3
"""Find max per-GPU batch size for Pass-1 ensemble (forward+backward)."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1] / "research/experiments/02_pass1_classifier"
sys.path.insert(0, str(EXP))

from ensemble_model import create_model_with_lora, dinov2_transform  # noqa: E402
from load_ensemble_ckpt import init_model_from_bombek  # noqa: E402
from transformers import AutoProcessor  # noqa: E402


def try_batch(model, device, batch_size: int, siglip_hw: int, dinov2_hw: int) -> tuple[bool, float]:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    siglip = torch.randn(batch_size, 3, siglip_hw, siglip_hw, device=device, dtype=torch.bfloat16)
    dinov2 = torch.randn(batch_size, 3, dinov2_hw, dinov2_hw, device=device, dtype=torch.bfloat16)
    labels = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.float32)
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(siglip, dinov2)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        model.zero_grad(set_to_none=True)
        del siglip, dinov2, labels, logits, loss
        torch.cuda.empty_cache()
        return True, peak_gb
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            gc.collect()
            return False, 0.0
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip", required=True)
    parser.add_argument("--dinov2", default="vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--init-bombek", required=True)
    parser.add_argument("--image-size", type=int, default=392)
    parser.add_argument("--max-batch", type=int, default=48)
    parser.add_argument("--margin-gb", type=float, default=5.0)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    model = create_model_with_lora(
        siglip_model_name=args.siglip,
        dinov2_model_name=args.dinov2,
        image_size=args.image_size,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.1,
    )
    init_model_from_bombek(model, args.init_bombek)
    model = model.to(device)
    model.train()

    proc = AutoProcessor.from_pretrained(args.siglip)
    siglip_hw = int(proc.image_processor.size["height"])
    dinov2_hw = args.image_size

    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    cap_gb = total_gb - args.margin_gb
    print(f"GPU total={total_gb:.1f}GB  cap={cap_gb:.1f}GB (margin {args.margin_gb}GB)")
    print(f"tensor shapes: siglip={siglip_hw}x{siglip_hw}  dinov2={dinov2_hw}x{dinov2_hw}")

    ok_sizes: list[tuple[int, float]] = []
    for bs in range(8, args.max_batch + 1, 4):
        ok, peak = try_batch(model, device, bs, siglip_hw, dinov2_hw)
        status = f"peak={peak:.1f}GB" if ok else "OOM"
        print(f"  batch={bs:2d}  {status}")
        if ok and peak <= cap_gb:
            ok_sizes.append((bs, peak))
        elif ok:
            print(f"    (over cap {cap_gb:.1f}GB — stopping)")
            break
        else:
            break

    if not ok_sizes:
        print("ERROR: no batch size succeeded")
        sys.exit(1)

    best_bs, best_peak = ok_sizes[-1]
    print(f"RECOMMENDED batch_size={best_bs}  (peak {best_peak:.1f}GB / cap {cap_gb:.1f}GB)")


if __name__ == "__main__":
    main()
