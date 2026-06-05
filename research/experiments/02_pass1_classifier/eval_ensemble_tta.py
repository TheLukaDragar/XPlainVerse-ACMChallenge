#!/usr/bin/env python3
"""Pass-1 ensemble inference with orig + horizontal-flip TTA.

Writes:
  predictions_long.parquet  — sample_id, view, p_fake
  predictions.parquet       — p_fake_orig, p_fake_flip, p_fake_mean, pred_label_mean
  metrics.json              — if GT labels present: AUC + thr_best_f1 on full split
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from ensemble_model import create_model_with_lora, dinov2_transform
from eval_ensemble import load_checkpoint
from train import load_manifest


class EnsembleTTADataset(Dataset):
    def __init__(self, df: pd.DataFrame, siglip_processor, dinov2_tf):
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


def collate_tta(batch):
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


def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    pivot = long_df.pivot(index="sample_id", columns="view", values="p_fake")
    pivot = pivot.rename(columns={"orig": "p_fake_orig", "flip": "p_fake_flip"})
    wide = pivot.reset_index()
    wide["p_fake_mean"] = (wide["p_fake_orig"] + wide["p_fake_flip"]) / 2.0
    return wide.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def val_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_curve = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.nanargmax(f1_curve[:-1]))
    thr = float(thresholds[best_idx])
    pred = (y_score >= thr).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
        "thr_best_f1": thr,
        "acc_at_0.5": float(accuracy_score(y_true, (y_score >= 0.5).astype(int))),
        "acc_at_best": float(accuracy_score(y_true, pred)),
        "real_acc_at_best": float(accuracy_score(y_true[y_true == 0], pred[y_true == 0])),
        "fake_acc_at_best": float(accuracy_score(y_true[y_true == 1], pred[y_true == 1])),
        "f1_fake_at_best": float(f1_score(y_true, pred, pos_label=1)),
        "n": int(len(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True, help="manifest_*_tta.parquet")
    parser.add_argument("--labels-manifest", type=Path, default=None,
                        help="one-row-per-image manifest with label_int (for val metrics)")
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
    loader = DataLoader(
        EnsembleTTADataset(df, processor, dinov2_transform(image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_tta,
        pin_memory=True,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "predictions_long.parquet"

    if long_path.is_file():
        print(f"[resume] {long_path}")
        long_df = pd.read_parquet(long_path)
    else:
        print(f"running inference on {len(df)} rows")
        long_df = run_inference(model, loader, args.device)
        long_df.to_parquet(long_path, index=False)
        print(f"wrote {long_path} ({len(long_df)} rows)")

    wide_df = to_wide(long_df)

    labels_path = args.labels_manifest
    if labels_path and labels_path.is_file():
        labels_df = pd.read_parquet(labels_path).set_index("sample_id")
        wide_df = wide_df.merge(
            labels_df[["label_int"]].reset_index(),
            on="sample_id",
            how="left",
        )
        y_true = wide_df["label_int"].astype(int).to_numpy()
        y_score = wide_df["p_fake_mean"].astype(float).to_numpy()
        metrics = val_metrics(y_true, y_score)
        thr = metrics["thr_best_f1"]
        wide_df["pred_label_mean"] = (wide_df["p_fake_mean"] >= thr).astype(int)
        metrics["pred_fake_rate_at_thr"] = float(wide_df["pred_label_mean"].mean())
        metrics["p_fake_mean_avg"] = float(wide_df["p_fake_mean"].mean())
        metrics["views"] = sorted(long_df["view"].unique().tolist())
        metrics["n_forward_passes"] = int(len(long_df))
    else:
        thr = 0.5
        wide_df["pred_label_mean"] = (wide_df["p_fake_mean"] >= thr).astype(int)
        metrics = {
            "n_images": int(wide_df["sample_id"].nunique()),
            "n_forward_passes": int(len(long_df)),
            "threshold": thr,
            "pred_fake_rate_mean": float(wide_df["pred_label_mean"].mean()),
            "p_fake_mean_avg": float(wide_df["p_fake_mean"].mean()),
            "views": sorted(long_df["view"].unique().tolist()),
            "note": "no labels manifest — thr_best_f1 not computed",
        }

    wide_path = out_dir / "predictions.parquet"
    wide_df.to_parquet(wide_path, index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"wrote {wide_path}")


if __name__ == "__main__":
    main()
