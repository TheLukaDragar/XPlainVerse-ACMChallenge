#!/usr/bin/env python3
"""DINOv3-MAC inference on a manifest with horizontal-flip TTA (multi-GPU sharded).

Each rank processes manifest rows where (idx % world_size == rank), writes a per-rank
parquet shard; rank 0 merges into <out>/predictions.parquet with columns:
  sample_id, p_fake_orig, p_fake_flip, p_fake_mean, pred_label_mean
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dinov3_mac_model import create_dinov3_mac, dinov3_transform


class InferDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["image_path"]).convert("RGB")
        return self.tf(img), str(row["sample_id"])


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.manifest)
    df = df.iloc[rank::world].reset_index(drop=True)
    if rank == 0:
        print(f"world={world} total≈{len(df)*world} this_rank={len(df)} img_size={args.image_size}")

    model = create_dinov3_mac(pretrained=False)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd.get("model_state", sd), strict=True)
    model = model.to(device).eval()

    tf = dinov3_transform(args.image_size)
    loader = DataLoader(
        InferDataset(df, tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    ids, p_orig, p_flip = [], [], []
    for bi, (pixels, sids) in enumerate(loader):
        pixels = pixels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lo = model(pixels)
            lf = model(torch.flip(pixels, dims=[3]))
        p_orig.append(torch.sigmoid(lo.float()).cpu().numpy())
        p_flip.append(torch.sigmoid(lf.float()).cpu().numpy())
        ids.extend(sids)
        if rank == 0 and bi % 50 == 0:
            print(f"  rank0 batch {bi}/{len(loader)}", flush=True)

    po = np.concatenate(p_orig)
    pf = np.concatenate(p_flip)
    pm = (po + pf) / 2.0
    shard = pd.DataFrame({
        "sample_id": ids,
        "p_fake_orig": po,
        "p_fake_flip": pf,
        "p_fake_mean": pm,
        "pred_label_mean": (pm >= args.threshold).astype(int),
    })
    shard_path = out_dir / f"shard_rank{rank}.parquet"
    shard.to_parquet(shard_path)
    print(f"rank {rank} wrote {len(shard)} rows -> {shard_path}", flush=True)

    if world > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        dist.barrier()

    if rank == 0:
        shards = sorted(out_dir.glob("shard_rank*.parquet"))
        merged = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
        merged = merged.drop_duplicates("sample_id").reset_index(drop=True)
        merged.to_parquet(out_dir / "predictions.parquet")
        print(f"merged {len(merged)} rows -> {out_dir/'predictions.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
