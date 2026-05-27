#!/usr/bin/env python3
"""Pass-1 binary classifier — frozen VFM + linear head (+ optional LoRA).

Recipe follows Simplicity Prevails (arXiv 2602.01738):
  frozen backbone → pooled features → linear head → {real, fake}
  Paper: AdamW lr=1e-3, batch=128, 2 epochs, no augmentation, GenImage train set.

We adapt for XPlainVerse:
  - 260k balanced train manifest (130k real + 130k fake)
  - LayerNorm + Dropout before head (slightly richer than pure linear)
  - Native resolution via each model's AutoImageProcessor (224 DINOv3, 384 SigLIP2)

Reference code (no training script in upstream): refs/simplicityprevails/models.py

Usage:
    python3 train.py \\
        --train manifests/manifest_train_balanced.parquet \\
        --val   manifests/manifest_val.parquet \\
        --out   ~/luka/runs/pass1/v1 \\
        --backbone baseline_models/pass1/siglip2-so400m
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class ManifestDataset(Dataset):
    """Parquet manifest columns: image_path, label_int (fake=1), sample_id."""

    def __init__(self, df: pd.DataFrame, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, int(row["label_int"]), row["sample_id"]


def collate_batch(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    sample_ids = [item[2] for item in batch]
    return images, labels, sample_ids


def load_manifest(path: str, max_rows: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if max_rows > 0 and max_rows < len(df):
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Model  (see refs/simplicityprevails/models.py for upstream loading pattern)
# ---------------------------------------------------------------------------


def lora_target_modules(backbone_id: str) -> list[str]:
    if "siglip" in backbone_id.lower():
        return ["q_proj", "k_proj", "v_proj", "out_proj"]
    return ["query", "key", "value"]


def load_vision_encoder(backbone_id: str, dtype: torch.dtype) -> tuple[nn.Module, int]:
    """Load HF vision encoder; unwrap SigLIP .vision_model when present."""
    full = AutoModel.from_pretrained(backbone_id, dtype=dtype)
    if hasattr(full, "vision_model"):
        encoder = full.vision_model
    else:
        encoder = full
    return encoder, encoder.config.hidden_size


def pool_encoder_features(encoder: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    outputs = encoder(pixel_values=pixel_values)
    if getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state[:, 0]


class RealFakeClassifier(nn.Module):
    """Frozen (or LoRA) encoder + trainable classification head."""

    def __init__(self, encoder: nn.Module, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = pool_encoder_features(self.encoder, pixel_values)
        return self.head(features)


def build_model(backbone_id: str, use_lora: bool, lora_r: int, lora_alpha: int, device: str) -> RealFakeClassifier:
    encoder, hidden_size = load_vision_encoder(backbone_id, torch.bfloat16)

    if use_lora:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules(backbone_id),
            lora_dropout=0.05,
            bias="none",
        )
        encoder = get_peft_model(encoder, config)
        encoder.print_trainable_parameters()
    else:
        for param in encoder.parameters():
            param.requires_grad = False

    return RealFakeClassifier(encoder, hidden_size).to(device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_validation(
    model: RealFakeClassifier,
    loader: DataLoader,
    device: str,
) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    prob_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    sample_ids: list[str] = []

    for images, labels, ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        prob_chunks.append(probs)
        label_chunks.append(labels.numpy())
        sample_ids.extend(ids)

    y_true = np.concatenate(label_chunks)
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
    model: RealFakeClassifier,
    processor,
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
            "head": model.head.state_dict(),
            "backbone_state": model.encoder.state_dict() if run_args["lora"] else None,
            "args": run_args,
            "metrics": metrics,
        },
        ckpt_dir / "ckpt.pt",
    )
    processor.save_pretrained(ckpt_dir / "processor")
    pd.DataFrame({"sample_id": sample_ids, "label_int": y_true, "p_fake": y_score}).to_parquet(
        out_dir / "val_predictions.parquet"
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_args = vars(args)
    (out_dir / "run_args.json").write_text(json.dumps(run_args, indent=2))

    print(f"== Pass-1 train.py ==")
    print(f"  backbone : {args.backbone}")
    print(f"  out      : {out_dir}")
    print(f"  lora     : {bool(args.lora)}")

    df_train = load_manifest(args.train, args.train_slice, args.seed)
    df_val = load_manifest(args.val, args.val_slice, args.seed)
    print(f"  train    : {len(df_train)} rows")
    print(f"  val      : {len(df_val)} rows")

    processor = AutoImageProcessor.from_pretrained(args.backbone)
    model = build_model(args.backbone, bool(args.lora), args.lora_r, args.lora_alpha, args.device)

    train_loader = DataLoader(
        ManifestDataset(df_train, processor),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        ManifestDataset(df_val, processor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    head_params = list(model.head.parameters())
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer_groups = [{"params": head_params, "lr": args.lr_head}]
    if encoder_params:
        optimizer_groups.append({"params": encoder_params, "lr": args.lr_backbone})
    optimizer = AdamW(optimizer_groups, weight_decay=0.01)

    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    best_auc = -1.0
    epoch_logs: list[dict] = []
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        for images, labels, _ in train_loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(images), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % args.log_every == 0:
                print(
                    f"epoch {epoch} step {global_step}/{total_steps} "
                    f"loss={loss.item():.4f} lr_head={optimizer.param_groups[0]['lr']:.2e}"
                )

        metrics, y_true, y_score, sample_ids = run_validation(model, val_loader, args.device)
        metrics["epoch"] = epoch
        epoch_logs.append(metrics)
        print(f"\n=== val epoch {epoch} ===")
        print(json.dumps(metrics, indent=2))

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            save_best_checkpoint(out_dir, model, processor, run_args, metrics, y_true, y_score, sample_ids)
            print(f"  saved best checkpoint (AUC {best_auc:.4f})")

    (out_dir / "metrics.json").write_text(json.dumps(epoch_logs, indent=2))
    print(f"\nfinished — best val AUC {best_auc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pass-1 real/fake classifier (Simplicity Prevails recipe)")
    parser.add_argument("--train", required=True, help="train manifest parquet")
    parser.add_argument("--val", required=True, help="val manifest parquet")
    parser.add_argument("--out", required=True, help="output run directory")
    parser.add_argument("--backbone", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Simplicity Prevails uses 1e-3")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="only when --lora 1")
    parser.add_argument("--lora", type=int, default=0, help="1 = LoRA encoder (Bombek1 / DINOv3-Forensics style)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-slice", type=int, default=0, help="debug: cap train rows")
    parser.add_argument("--val-slice", type=int, default=0, help="debug: cap val rows")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
