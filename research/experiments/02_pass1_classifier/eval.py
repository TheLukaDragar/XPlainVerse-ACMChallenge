#!/usr/bin/env python3
"""Evaluate a trained Pass-1 checkpoint on a manifest parquet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoImageProcessor

from train import (
    RealFakeClassifier,
    ManifestDataset,
    build_model,
    collate_batch,
    load_manifest,
    run_validation,
)


def load_checkpoint(ckpt_path: Path, device: str) -> tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    run_args = ckpt["args"]
    backbone_id = run_args["backbone"]

    model = build_model(
        backbone_id,
        bool(run_args.get("lora")),
        run_args.get("lora_r", 16),
        run_args.get("lora_alpha", 32),
        device,
    )
    if run_args.get("lora"):
        model.encoder.load_state_dict(ckpt["backbone_state"])
    model.head.load_state_dict(ckpt["head"])
    model.eval()
    return model, run_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to best_ckpt/ckpt.pt")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--slice", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, run_args = load_checkpoint(Path(args.ckpt), args.device)
    processor = AutoImageProcessor.from_pretrained(run_args["backbone"])
    df = load_manifest(args.manifest, args.slice, seed=0)

    loader = torch.utils.data.DataLoader(
        ManifestDataset(df, processor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
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
