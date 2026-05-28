#!/usr/bin/env python3
"""Train Bombek1-style SigLIP2-SO400M + DINOv2-Large ensemble on XPlainVerse.

Recipe: refs/ai-image-detector-siglip-dinov2/ (OpenFake hyperparameters).
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor
from transformers.optimization import get_cosine_schedule_with_warmup

from ensemble_augment import QualityAgnosticAugment
from ensemble_model import (
    DEFAULT_DINOV2,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SIGLIP,
    EnsembleAIDetector,
    create_model_with_lora,
    dinov2_transform,
    optimizer_param_groups,
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


class EnsembleManifestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        siglip_processor,
        dinov2_tf,
        train_augment: QualityAgnosticAugment | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.siglip_processor = siglip_processor
        self.dinov2_tf = dinov2_tf
        self.train_augment = train_augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.train_augment is not None:
            image = self.train_augment(image)
        siglip = self.siglip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        dinov2 = self.dinov2_tf(image)
        label = float(int(row["label_int"]))
        return siglip, dinov2, label, row["sample_id"]


def collate_ensemble(batch):
    siglip = torch.stack([item[0] for item in batch])
    dinov2 = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    sample_ids = [item[3] for item in batch]
    return siglip, dinov2, labels, sample_ids


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - p_t).pow(self.gamma) * bce).mean()


@torch.no_grad()
def run_validation(model: EnsembleAIDetector, loader: DataLoader, device: str) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    prob_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    sample_ids: list[str] = []

    for siglip, dinov2, labels, ids in loader:
        siglip = siglip.to(device, non_blocking=True)
        dinov2 = dinov2.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(siglip, dinov2)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        prob_chunks.append(probs)
        label_chunks.append(labels.numpy())
        sample_ids.extend(ids)

    y_true = np.concatenate(label_chunks).astype(int)
    y_score = np.concatenate(prob_chunks)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_curve = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.nanargmax(f1_curve[:-1]))
    threshold_best = float(thresholds[best_idx])
    pred_default = (y_score >= 0.5).astype(int)
    pred_best = (y_score >= threshold_best).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
        "thr_best_f1": threshold_best,
        "acc_at_0.5": float(accuracy_score(y_true, pred_default)),
        "acc_at_best": float(accuracy_score(y_true, pred_best)),
        "real_acc_at_best": float(accuracy_score(y_true[y_true == 0], pred_best[y_true == 0])),
        "fake_acc_at_best": float(accuracy_score(y_true[y_true == 1], pred_best[y_true == 1])),
        "f1_fake_at_best": float(f1_score(y_true, pred_best, pos_label=1)),
    }
    return metrics, y_true, y_score, sample_ids


def save_best_checkpoint(
    out_dir: Path,
    model: EnsembleAIDetector,
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
            "model_state_dict": unwrap_model(model).state_dict(),
            "config": {
                "siglip_model": run_args["siglip"],
                "dinov2_model": run_args["dinov2"],
                "image_size": run_args["image_size"],
                "lora_rank": run_args["lora_r"],
                "lora_alpha": run_args["lora_alpha"],
                "lora_dropout": run_args["lora_dropout"],
            },
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
    run_args["trainer"] = "bombek_ensemble"
    run_args["world_size"] = world_size
    if main:
        (out_dir / "run_args.json").write_text(json.dumps(run_args, indent=2))
        print("== Pass-1 train_ensemble.py (Bombek1 recipe) ==")
        print(f"  siglip     : {args.siglip}")
        print(f"  dinov2     : {args.dinov2}")
        print(f"  image_size : {args.image_size}")
        print(f"  out        : {out_dir}")

    df_train = load_manifest(args.train, args.train_slice, args.seed)
    df_val = load_manifest(args.val, args.val_slice, args.seed)
    if main:
        print(f"  train      : {len(df_train)} rows")
        print(f"  val        : {len(df_val)} rows")

    # SigLIP2-SO400M is patch14-384 — keep native 384 preprocessing (Bombek uses
    # AutoProcessor defaults for SigLIP; only DINOv2 is resized to image_size).
    siglip_processor = AutoProcessor.from_pretrained(args.siglip)

    dinov2_tf = dinov2_transform(args.image_size)
    train_aug = QualityAgnosticAugment() if args.augment else None

    model = create_model_with_lora(
        args.siglip,
        args.dinov2,
        image_size=args.image_size,
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        hidden_dim=args.head_hidden,
        head_dropout=args.head_dropout,
    )
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

    train_ds = EnsembleManifestDataset(df_train, siglip_processor, dinov2_tf, train_aug)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_ensemble,
        pin_memory=True,
        drop_last=is_distributed,
    )
    val_loader = None
    if main:
        val_loader = DataLoader(
            EnsembleManifestDataset(df_val, siglip_processor, dinov2_tf, None),
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_ensemble,
            pin_memory=True,
        )

    optimizer = AdamW(
        optimizer_param_groups(unwrap_model(model), args.lr_head, args.lr_lora),
        weight_decay=args.weight_decay,
    )
    optimizer_steps = max(1, (len(train_loader) // args.grad_accum) * args.epochs)
    warmup_steps = int(optimizer_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=optimizer_steps,
    )
    loss_fn = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)

    best_auc = -1.0
    epoch_logs: list[dict] = []
    global_step = 0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        for batch_idx, (siglip, dinov2, labels, _) in enumerate(train_loader):
            siglip = siglip.to(args.device, non_blocking=True)
            dinov2 = dinov2.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(siglip, dinov2)
                loss = loss_fn(logits, labels) / args.grad_accum

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
            metrics, y_true, y_score, sample_ids = run_validation(unwrap_model(model), val_loader, args.device)
            metrics["epoch"] = epoch
            epoch_logs.append(metrics)
            print(f"\n=== val epoch {epoch} ===")
            print(json.dumps(metrics, indent=2))
            wandb_log(wb, {f"val/{k}": v for k, v in metrics.items() if k != "epoch"}, global_step)
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                save_best_checkpoint(out_dir, unwrap_model(model), run_args, metrics, y_true, y_score, sample_ids)
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
    parser = argparse.ArgumentParser(description="Bombek1 SigLIP2+DINOv2 ensemble on XPlainVerse")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--siglip", default=DEFAULT_SIGLIP)
    parser.add_argument("--dinov2", default=DEFAULT_DINOV2)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16, help="per GPU (Bombek: 16)")
    parser.add_argument("--grad-accum", type=int, default=4, help="2 GPU ×16 ×4 = 128 eff; use 9 on 1 GPU for 144")
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--lr-lora", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--augment", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-slice", type=int, default=0)
    parser.add_argument("--val-slice", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--report-to", default=os.environ.get("REPORT_TO", "wandb"), choices=("wandb", "none"))
    parser.add_argument("--lora", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--num-gpus", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--backbone", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
