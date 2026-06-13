#!/usr/bin/env python3
"""Run Pass-1 ensemble inference on the test split (optional horizontal-flip TTA).

Writes:
  predictions_long.parquet   — sample_id, view, p_fake (one row per forward pass)
  predictions.parquet        — wide: p_fake_orig, p_fake_flip, p_fake_mean, pred_label_mean
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from ensemble_model import create_model_with_lora, dinov2_transform
from eval_ensemble import load_checkpoint
from train import load_manifest


class EnsembleTestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        siglip_processor,
        dinov2_tf,
    ):
        self.df = df.reset_index(drop=True)
        self.siglip_processor = siglip_processor
        self.dinov2_tf = dinov2_tf
        self.has_view = "view" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.has_view and str(row["view"]) == "flip":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        siglip = self.siglip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        dinov2 = self.dinov2_tf(image)
        label = float(int(row.get("label_int", -1)))
        view = str(row["view"]) if self.has_view else "orig"
        return siglip, dinov2, label, str(row["sample_id"]), view


def collate_test(batch):
    siglip = torch.stack([item[0] for item in batch])
    dinov2 = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    sample_ids = [item[3] for item in batch]
    views = [item[4] for item in batch]
    return siglip, dinov2, labels, sample_ids, views


@torch.no_grad()
def run_inference(model, loader: DataLoader, device: str) -> pd.DataFrame:
    model.eval()
    rows: list[dict] = []
    for siglip, dinov2, _labels, sample_ids, views in loader:
        siglip = siglip.to(device, non_blocking=True)
        dinov2 = dinov2.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(siglip, dinov2)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        for sid, view, p in zip(sample_ids, views, probs):
            rows.append({"sample_id": sid, "view": view, "p_fake": float(p)})
    return pd.DataFrame(rows)


def to_wide(long_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if long_df.empty:
        raise ValueError("empty predictions")

    views = sorted(long_df["view"].unique())
    if views == ["orig"]:
        wide = long_df.rename(columns={"p_fake": "p_fake_orig"})[["sample_id", "p_fake_orig"]]
        wide["p_fake_flip"] = np.nan
        wide["p_fake_mean"] = wide["p_fake_orig"]
    else:
        pivot = long_df.pivot(index="sample_id", columns="view", values="p_fake")
        pivot = pivot.rename(columns={"orig": "p_fake_orig", "flip": "p_fake_flip"})
        if "p_fake_orig" not in pivot.columns or "p_fake_flip" not in pivot.columns:
            raise ValueError(f"expected orig+flip views, got columns={list(pivot.columns)}")
        wide = pivot.reset_index()
        wide["p_fake_mean"] = (wide["p_fake_orig"] + wide["p_fake_flip"]) / 2.0

    wide["pred_label_mean"] = (wide["p_fake_mean"] >= threshold).astype(int)
    wide = wide.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    return wide


def main() -> None:
    parser = argparse.ArgumentParser(description="Pass-1 ensemble inference on test split")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--manifest", default="", help="manifest_test.parquet or manifest_test_tta.parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--slice", type=int, default=0, help="0 = all rows")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="merge predictions_long_shard*.parquet -> wide predictions.parquet",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.129,
        help="decision threshold for pred_label_mean (val F1-opt default)",
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        shard_paths = sorted(out_dir.glob("predictions_long_shard*.parquet"))
        if not shard_paths:
            raise FileNotFoundError(f"no shard files in {out_dir}")
        long_df = pd.concat([pd.read_parquet(p) for p in shard_paths], ignore_index=True)
        long_path = out_dir / "predictions_long.parquet"
        long_df.to_parquet(long_path, index=False)
        print(f"merged {len(shard_paths)} shards -> {long_path} ({len(long_df)} rows)")
        wide_df = to_wide(long_df, args.threshold)
        wide_path = out_dir / "predictions.parquet"
        wide_df.to_parquet(wide_path, index=False)
        summary = {
            "n_images": int(wide_df["sample_id"].nunique()),
            "n_forward_passes": int(len(long_df)),
            "threshold": args.threshold,
            "pred_fake_rate_mean": float(wide_df["pred_label_mean"].mean()),
            "p_fake_mean_avg": float(wide_df["p_fake_mean"].mean()),
            "views": sorted(long_df["view"].unique().tolist()),
            "shard_count": len(shard_paths),
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        print(f"wrote {wide_path}")
        return

    if not args.ckpt or not args.manifest:
        raise SystemExit("--ckpt and --manifest required unless --merge-only")
    model, cfg = load_checkpoint(Path(args.ckpt), args.device)
    image_size = int(cfg.get("image_size", 392))
    siglip_id = cfg.get("siglip_model") or cfg.get("siglip", "google/siglip2-so400m-patch14-384")
    processor = AutoProcessor.from_pretrained(siglip_id)

    df = load_manifest(args.manifest, args.slice, seed=0)
    if args.shard_count > 1:
        df = df.iloc[args.shard_id :: args.shard_count].reset_index(drop=True)
        print(f"shard {args.shard_id}/{args.shard_count}: {len(df)} rows")
    loader = DataLoader(
        EnsembleTestDataset(df, processor, dinov2_transform(image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_test,
        pin_memory=True,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.shard_count > 1:
        long_path = out_dir / f"predictions_long_shard{args.shard_id}.parquet"
    else:
        long_path = out_dir / "predictions_long.parquet"
    if long_path.is_file():
        print(f"[resume] loading existing long predictions: {long_path}")
        long_df = pd.read_parquet(long_path)
    else:
        print(f"running inference on {len(df)} rows (manifest={args.manifest})")
        long_df = run_inference(model, loader, args.device)
        long_df.to_parquet(long_path, index=False)
        print(f"wrote {long_path} ({len(long_df)} rows)")

    if args.shard_count > 1:
        print(f"shard {args.shard_id} done")
        return

    wide_df = to_wide(long_df, args.threshold)
    wide_path = out_dir / "predictions.parquet"
    wide_df.to_parquet(wide_path, index=False)

    summary = {
        "n_images": int(wide_df["sample_id"].nunique()),
        "n_forward_passes": int(len(long_df)),
        "threshold": args.threshold,
        "pred_fake_rate_mean": float(wide_df["pred_label_mean"].mean()),
        "p_fake_mean_avg": float(wide_df["p_fake_mean"].mean()),
        "views": sorted(long_df["view"].unique().tolist()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {wide_path}")


if __name__ == "__main__":
    main()
