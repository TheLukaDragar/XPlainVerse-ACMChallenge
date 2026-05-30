#!/usr/bin/env python3
"""Evaluate a timm Pass-1 full fine-tune checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import timm
import torch
from timm.data import create_transform, resolve_model_data_config

from train import load_manifest, run_validation
from train_timm import TimmManifestDataset, build_timm_model, timm_data_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--slice", type=int, default=0)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    run_args = ckpt["args"]
    model_name = run_args["model"]
    image_size = int(run_args["image_size"])

    model = build_timm_model(model_name, 2, args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data_config = timm_data_config(model_name, image_size)
    transform = create_transform(**data_config, is_training=False)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_manifest(args.manifest, args.slice, seed=0)
    loader = torch.utils.data.DataLoader(
        TimmManifestDataset(df, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),
            torch.tensor([b[1] for b in batch], dtype=torch.long),
            [b[2] for b in batch],
        ),
        pin_memory=True,
    )

    metrics, y_true, y_score, sample_ids = run_validation(model, loader, args.device)
    metrics["n"] = len(df)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"sample_id": sample_ids, "label_int": y_true, "p_fake": y_score}).to_parquet(
        out_dir / "predictions.parquet"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
