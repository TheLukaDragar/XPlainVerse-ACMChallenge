#!/usr/bin/env python3
"""Evaluate Bombek1-style ensemble checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor

from ensemble_model import create_model_with_lora, dinov2_transform
from train_ensemble import EnsembleManifestDataset, collate_ensemble, run_validation
from train import load_manifest


def load_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config") or ckpt.get("args", {})
    model = create_model_with_lora(
        cfg.get("siglip_model") or cfg.get("siglip", "google/siglip2-so400m-patch14-384"),
        cfg.get("dinov2_model") or cfg.get("dinov2", "vit_large_patch14_dinov2.lvd142m"),
        image_size=int(cfg.get("image_size", 392)),
        lora_rank=int(cfg.get("lora_rank", cfg.get("lora_r", 32))),
        lora_alpha=int(cfg.get("lora_alpha", cfg.get("lora_alpha", 64))),
        lora_dropout=float(cfg.get("lora_dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--slice", type=int, default=0)
    args = parser.parse_args()

    model, cfg = load_checkpoint(Path(args.ckpt), args.device)
    image_size = int(cfg.get("image_size", 392))
    siglip_id = cfg.get("siglip_model") or cfg.get("siglip", "google/siglip2-so400m-patch14-384")

    processor = AutoProcessor.from_pretrained(siglip_id)
    size = {"height": image_size, "width": image_size}
    if hasattr(processor, "size"):
        processor.size = size
    if hasattr(processor, "crop_size") and processor.crop_size:
        processor.crop_size = size

    df = load_manifest(args.manifest, args.slice, seed=0)
    loader = torch.utils.data.DataLoader(
        EnsembleManifestDataset(df, processor, dinov2_transform(image_size), None),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_ensemble,
        pin_memory=True,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, y_true, y_score, sample_ids = run_validation(model, loader, args.device)
    metrics["n"] = len(df)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"sample_id": sample_ids, "label_int": y_true, "p_fake": y_score}).to_parquet(
        out_dir / "predictions.parquet"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
