#!/usr/bin/env python3
"""Pass-1 binary classifier — timm backbone, full fine-tune (all params trainable).

Designed for native 512×512 CNN/hybrid models (ConvNeXt, CSATv2) to reduce shortcut
learning on generator-specific artifacts vs frozen VFM + LoRA.

Usage:
    python3 train_timm.py \\
        --train manifests/manifest_train_balanced.parquet \\
        --val manifests/manifest_val.parquet \\
        --out ~/luka/runs/pass1_timm/convnext_small_512 \\
        --model convnext_small.fb_in22k_ft_in1k \\
        --image-size 512
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import get_cosine_schedule_with_warmup

from train import (
    cleanup_distributed,
    collate_batch,
    finish_wandb,
    init_distributed,
    init_wandb,
    is_main_process,
    load_manifest,
    loader_batch_size,
    run_validation,
    setup_parallel_model,
    unwrap_model,
    wandb_log,
)


class TimmManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        tensor = self.transform(image)
        return tensor, int(row["label_int"]), row["sample_id"]


def build_timm_model(model_name: str, num_classes: int, image_size: int, device: str) -> nn.Module:
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        img_size=image_size,
    )
    return model.to(device)


def save_best_checkpoint(
    out_dir: Path,
    model: nn.Module,
    run_args: dict,
    metrics: dict,
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_ids: list[str],
) -> None:
    ckpt_dir = out_dir / "best_ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "args": run_args,
            "metrics": metrics,
        },
        ckpt_dir / "ckpt.pt",
    )
    pd.DataFrame({"sample_id": sample_ids, "label_int": y_true, "p_fake": y_score}).to_parquet(
        out_dir / "val_predictions.parquet"
    )


def train(args: argparse.Namespace) -> None:
    local_rank, rank, world_size, is_distributed = init_distributed()
    if is_distributed:
        args.device = f"cuda:{local_rank}"
    main = is_main_process(rank)

    out_dir = Path(args.out)
    if main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed:
        import torch.distributed as dist

        dist.barrier()

    run_args = vars(args).copy()
    run_args["trainer"] = "timm_fullft"
    run_args["world_size"] = world_size
    if main:
        (out_dir / "run_args.json").write_text(json.dumps(run_args, indent=2))

    if main:
        print("== Pass-1 train_timm.py (full fine-tune) ==")
        print(f"  model      : {args.model}")
        print(f"  image_size : {args.image_size}")
        print(f"  out        : {out_dir}")
        print(f"  lr         : {args.lr}")
        print(f"  aug        : {args.augment}")
        if is_distributed:
            print(f"  gpus       : {world_size} (local rank {local_rank})")
            print(f"  eff_batch  : {args.batch_size * world_size}")

    df_train = load_manifest(args.train, args.train_slice, args.seed)
    df_val = load_manifest(args.val, args.val_slice, args.seed)
    if main:
        print(f"  train      : {len(df_train)} rows")
        print(f"  val        : {len(df_val)} rows")

    probe = build_timm_model(args.model, 2, args.image_size, "cpu")
    data_config = resolve_model_data_config(probe)
    data_config["input_size"] = (3, args.image_size, args.image_size)
    train_tf = create_transform(**data_config, is_training=bool(args.augment))
    val_tf = create_transform(**data_config, is_training=False)
    del probe

    model = build_timm_model(args.model, 2, args.image_size, args.device)
    for param in model.parameters():
        param.requires_grad = True

    args.lora = 0
    model, parallel_gpus, parallel_mode = setup_parallel_model(model, args, is_distributed)
    batch_size = loader_batch_size(args, parallel_gpus, parallel_mode)
    wb = init_wandb(args, out_dir, rank)

    if main:
        n_params = sum(p.numel() for p in unwrap_model(model).parameters())
        n_train = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
        print(f"  params     : {n_train:,} trainable / {n_params:,} total")
        print(f"  parallel   : {parallel_mode}")
        print(f"  eff_batch  : {batch_size}")

    train_sampler = DistributedSampler(TimmManifestDataset(df_train, train_tf), shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        TimmManifestDataset(df_train, train_tf),
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=is_distributed,
    )
    val_loader = None
    if main:
        val_loader = DataLoader(
            TimmManifestDataset(df_val, val_tf),
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
        )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_auc = -1.0
    epoch_logs: list[dict] = []
    global_step = 0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for images, labels, _ in train_loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(images), labels)

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % args.log_every == 0 and main:
                print(
                    f"epoch {epoch} step {global_step}/{total_steps} "
                    f"loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                wandb_log(
                    wb,
                    {
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    },
                    global_step,
                )

        if is_distributed:
            import torch.distributed as dist

            dist.barrier()

        if main and val_loader is not None:
            metrics, y_true, y_score, sample_ids = run_validation(
                unwrap_model(model), val_loader, args.device
            )
            metrics["epoch"] = epoch
            epoch_logs.append(metrics)
            print(f"\n=== val epoch {epoch} ===")
            print(json.dumps(metrics, indent=2))
            wandb_log(wb, {f"val/{k}": v for k, v in metrics.items() if k != "epoch"}, global_step)

            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                save_best_checkpoint(
                    out_dir, unwrap_model(model), run_args, metrics, y_true, y_score, sample_ids
                )
                print(f"  saved best checkpoint (AUC {best_auc:.4f})")
                wandb_log(wb, {"val/best_auc": best_auc}, global_step)

        if is_distributed:
            import torch.distributed as dist

            dist.barrier()

    if main:
        (out_dir / "metrics.json").write_text(json.dumps(epoch_logs, indent=2))
        print(f"\nfinished — best val AUC {best_auc:.4f}")
    finish_wandb(wb, {"best_val_auc": best_auc, "epochs": args.epochs}, rank)
    cleanup_distributed(is_distributed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pass-1 timm full fine-tune @ 512px")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model",
        default="convnext_small.fb_in22k_ft_in1k",
        help="timm model name (use csatv2_21m.sw_r512_in1k for native 512 ViT)",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24, help="per GPU")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--augment", type=int, default=1, help="1 = timm train transforms (recommended)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-slice", type=int, default=0)
    parser.add_argument("--val-slice", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--report-to", default=os.environ.get("REPORT_TO", "wandb"), choices=("wandb", "none"))
    parser.add_argument("--lora", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--num-gpus", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
