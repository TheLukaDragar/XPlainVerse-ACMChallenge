#!/usr/bin/env python3
"""Train DINOv3-Large + MAC head (DINO-MAC reproduction) on XPlainVerse + external mix.

Single-tower detector to be LOGIT-FUSED with the SigLIP2+DINOv2 ensemble. DINOv3 is a
decorrelated backbone (self-supervised LVD-1689m) that the NTIRE 2026 winners showed
beats SigLIP/DINOv2; fusing it adds the diversity our v3/warmstart pair lacks.

Mirrors train_ensemble.py (reuses its dist/wandb/manifest helpers + checkpoint format).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import get_cosine_schedule_with_warmup

from ensemble_augment import QualityAgnosticAugment
from dinov3_mac_model import (
    DEFAULT_DINOV3,
    DEFAULT_IMAGE_SIZE,
    create_dinov3_mac,
    dinov3_param_groups,
    dinov3_transform,
)
from train_ensemble import (
    FocalLoss,
    _best_macro_threshold,
)
from train import (
    cleanup_distributed,
    finish_wandb,
    init_distributed,
    init_wandb,
    is_main_process,
    load_manifest,
    loader_batch_size,
    setup_parallel_model,
    unwrap_model,
    wandb_log,
)
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_curve, roc_auc_score


class DINOv3Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, tf, train_augment: QualityAgnosticAugment | None = None):
        self.df = df.reset_index(drop=True)
        self.tf = tf
        self.train_augment = train_augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.train_augment is not None:
            image = self.train_augment(image)
        pixels = self.tf(image)
        label = float(int(row["label_int"]))
        return pixels, label, row["sample_id"]


def collate(batch):
    pixels = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    ids = [b[2] for b in batch]
    return pixels, labels, ids


@torch.no_grad()
def run_validation(model, loader, device):
    model.eval()
    probs, labels, ids = [], [], []
    for pixels, y, sid in loader:
        pixels = pixels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logit = model(pixels)
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        labels.append(y.numpy())
        ids.extend(sid)
    y_true = np.concatenate(labels).astype(int)
    y_score = np.concatenate(probs)
    precision, recall, thr = precision_recall_curve(y_true, y_score)
    f1c = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.nanargmax(f1c[:-1]))
    thr_best = float(thr[best_idx])
    pred_best = (y_score >= thr_best).astype(int)
    thr_macro, macro_best = _best_macro_threshold(y_true, y_score)
    pred_macro = (y_score >= thr_macro).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
        "thr_best_f1": thr_best,
        "acc_at_0.5": float(accuracy_score(y_true, (y_score >= 0.5).astype(int))),
        "acc_at_best": float(accuracy_score(y_true, pred_best)),
        "thr_best_macro_f1": float(thr_macro),
        "macro_f1_at_best": float(macro_best),
        "real_f1_at_best": float(f1_score(y_true, pred_macro, pos_label=0)),
        "fake_f1_at_macro": float(f1_score(y_true, pred_macro, pos_label=1)),
        "real_acc_at_macro": float(accuracy_score(y_true[y_true == 0], pred_macro[y_true == 0])),
        "fake_acc_at_macro": float(accuracy_score(y_true[y_true == 1], pred_macro[y_true == 1])),
    }
    return metrics, y_true, y_score, ids


def save_best(out_dir, model, run_args, metrics, y_true, y_score, ids):
    ckpt_dir = out_dir / "best_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": unwrap_model(model).state_dict(),
            "metrics": metrics,
            "run_args": {**run_args, "dinov3": run_args.get("dinov3")},
        },
        ckpt_dir / "ckpt.pt",
    )
    pd.DataFrame({"sample_id": ids, "label_int": y_true, "p_fake": y_score}).to_parquet(
        out_dir / "val_predictions.parquet"
    )


def deep_supervision_loss(logits, labels, loss_fn):
    if isinstance(logits, list):
        final = loss_fn(logits[-1], labels)
        aux = sum(loss_fn(lg, labels) for lg in logits[:-1]) / max(1, len(logits) - 1)
        return final + aux
    return loss_fn(logits, labels)


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
    run_args["trainer"] = "dinov3_mac"
    run_args["world_size"] = world_size
    if main:
        (out_dir / "run_args.json").write_text(json.dumps(run_args, indent=2))
        print("== Pass-1 train_dinov3_mac.py (DINO-MAC reproduction) ==")
        print(f"  dinov3     : {args.dinov3}")
        print(f"  image_size : {args.image_size}")
        print(f"  out        : {out_dir}")

    df_train = load_manifest(args.train, args.train_slice, args.seed)
    df_val = load_manifest(args.val, args.val_slice, args.seed)
    if main:
        print(f"  train      : {len(df_train)} rows")
        print(f"  val        : {len(df_val)} rows")

    tf = dinov3_transform(args.image_size)
    train_aug = (
        QualityAgnosticAugment(
            p_jpeg=float(os.environ.get("AUG_P_JPEG", "0.5")),
            p_blur=float(os.environ.get("AUG_P_BLUR", "0.3")),
            p_noise=float(os.environ.get("AUG_P_NOISE", "0.3")),
            p_resize=float(os.environ.get("AUG_P_RESIZE", "0.3")),
        )
        if args.augment
        else None
    )

    model = create_dinov3_mac(
        model_name=args.dinov3,
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        deep_supervision=bool(args.deep_supervision),
    )
    if str(args.init_ckpt).strip():
        sd = torch.load(args.init_ckpt, map_location="cpu")
        state = sd.get("model_state", sd)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if main:
            print(f"  init_ckpt  : {args.init_ckpt} (missing={len(missing)} unexpected={len(unexpected)})")
    model = model.to(args.device)
    args.lora = 1
    model, parallel_gpus, parallel_mode = setup_parallel_model(model, args, is_distributed)
    batch_size = loader_batch_size(args, parallel_gpus, parallel_mode)
    wb = init_wandb(args, out_dir, rank)

    if main:
        trainable = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
        total = sum(p.numel() for p in unwrap_model(model).parameters())
        print(f"  params     : {trainable:,} trainable / {total:,} total")
        print(f"  batch/gpu  : {batch_size}  grad_accum: {args.grad_accum}  eff: {batch_size * args.grad_accum}")

    train_ds = DINOv3Dataset(df_train, tf, train_aug)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=is_distributed,
    )
    val_loader = None
    if main:
        val_loader = DataLoader(
            DINOv3Dataset(df_val, tf, None),
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

    optimizer = AdamW(
        dinov3_param_groups(unwrap_model(model), args.lr_head, args.lr_lora),
        weight_decay=args.weight_decay,
    )
    optimizer_steps = max(1, (len(train_loader) // args.grad_accum) * args.epochs)
    warmup_steps = int(optimizer_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, optimizer_steps)
    loss_fn = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)

    best_score = -1.0
    epoch_logs: list[dict] = []
    global_step = 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        for batch_idx, (pixels, labels, _) in enumerate(train_loader):
            pixels = pixels.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pixels)
                loss = deep_supervision_loss(logits, labels, loss_fn) / args.grad_accum
            loss.backward()
            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.log_every == 0 and main:
                    print(
                        f"epoch {epoch} step {global_step}/{optimizer_steps} "
                        f"loss={loss.item() * args.grad_accum:.4f} "
                        f"lr_head={optimizer.param_groups[0]['lr']:.2e}"
                    )
                    wandb_log(
                        wb,
                        {
                            "train/loss": loss.item() * args.grad_accum,
                            "train/lr_head": optimizer.param_groups[0]["lr"],
                            "train/lr_lora": optimizer.param_groups[1]["lr"],
                            "train/epoch": epoch,
                        },
                        global_step,
                    )
        if is_distributed:
            import torch.distributed as dist

            dist.barrier()
        if main and val_loader is not None:
            metrics, y_true, y_score, ids = run_validation(unwrap_model(model), val_loader, args.device)
            metrics["epoch"] = epoch
            epoch_logs.append(metrics)
            print(f"\n=== val epoch {epoch} ===")
            print(json.dumps(metrics, indent=2))
            wandb_log(wb, {f"val/{k}": v for k, v in metrics.items() if k != "epoch"}, global_step)
            current = metrics["macro_f1_at_best" if args.select_metric == "macro_f1" else "auc"]
            if current > best_score:
                best_score = current
                save_best(out_dir, model, run_args, metrics, y_true, y_score, ids)
                print(f"  saved best checkpoint ({args.select_metric} {best_score:.4f})")
        if is_distributed:
            import torch.distributed as dist

            dist.barrier()

    if main:
        (out_dir / "metrics.json").write_text(json.dumps(epoch_logs, indent=2))
        print(f"\nfinished — best val {args.select_metric} {best_score:.4f}")
    finish_wandb(wb)
    cleanup_distributed(is_distributed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DINOv3-MAC Pass-1 detector")
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dinov3", default=DEFAULT_DINOV3)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr-head", type=float, default=2e-4)
    p.add_argument("--lr-lora", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--head-hidden", type=int, default=1024)
    p.add_argument("--head-dropout", type=float, default=0.2)
    p.add_argument("--deep-supervision", type=int, default=1)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--augment", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-slice", type=int, default=0)
    p.add_argument("--val-slice", type=int, default=0)
    p.add_argument("--select-metric", default="macro_f1", choices=("auc", "macro_f1"))
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--report-to", default=os.environ.get("REPORT_TO", "wandb"), choices=("wandb", "none"))
    p.add_argument("--init-ckpt", default="")
    p.add_argument("--lora", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--num-gpus", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--backbone", default="", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
