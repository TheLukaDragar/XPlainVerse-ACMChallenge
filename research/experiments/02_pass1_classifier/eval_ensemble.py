#!/usr/bin/env python3
"""Evaluate Bombek1-style ensemble checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor

from ensemble_model import dinov2_transform
from load_ensemble_ckpt import load_ensemble_checkpoint
from train_ensemble import EnsembleManifestDataset, collate_ensemble, run_validation
from train import load_manifest


def load_checkpoint(ckpt_path: Path, device: str):
    return load_ensemble_checkpoint(ckpt_path, device=device)


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
