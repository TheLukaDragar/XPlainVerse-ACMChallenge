#!/usr/bin/env python3
"""Pass-1 binary classifier — frozen VFM + linear head (+ optional LoRA).

Recipe follows Simplicity Prevails (arXiv 2602.01738):
  frozen backbone → pooled features → linear head → {real, fake}
  Paper: AdamW lr=1e-3, batch=128, 2 epochs, no augmentation, GenImage train set.

We adapt for XPlainVerse:
  - Same backbones as Lunahera/simplicityprevails: DINOv3-7B + SigLIP2-giant
  - 260k balanced train manifest (130k real + 130k fake)
  - LayerNorm + Dropout before head (slightly richer than pure linear)
  - Native resolution via each model's AutoImageProcessor (224 DINOv3, 384 SigLIP2)

Reference: refs/simplicityprevails/models.py (dinov3lin → vit7b16, siglip2lin → giant)

Usage:
    python3 train.py \\
        --train manifests/manifest_train_balanced.parquet \\
        --val   manifests/manifest_val.parquet \\
        --out   ~/luka/runs/pass1/v1 \\
        --backbone baseline_models/pass1/dinov3-7b
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch.distributed as dist
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
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoImageProcessor, AutoModel
from transformers.optimization import get_cosine_schedule_with_warmup

# ---------------------------------------------------------------------------
# Distributed
# ---------------------------------------------------------------------------


def init_distributed() -> tuple[int, int, int, bool]:
    """Return (local_rank, rank, world_size, is_distributed)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size, True
    return 0, 0, 1, False


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
        return model.module
    return model


def setup_parallel_model(
    model: nn.Module,
    args: argparse.Namespace,
    is_distributed: bool,
) -> tuple[nn.Module, int, str]:
    """Return (model, parallel_gpus, parallel_mode)."""
    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(args.lora),
        )
        return model, int(os.environ["WORLD_SIZE"]), "ddp"

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        model = nn.DataParallel(model, device_ids=device_ids)
        args.device = "cuda:0"
        return model, num_gpus, "dataparallel"

    return model, 1, "single"


def loader_batch_size(args: argparse.Namespace, parallel_gpus: int, parallel_mode: str) -> int:
    """batch-size CLI is per-GPU; DataParallel splits a combined batch across devices."""
    if parallel_mode == "dataparallel" and parallel_gpus > 1:
        return args.batch_size * parallel_gpus
    return args.batch_size


def is_main_process(rank: int) -> bool:
    return rank == 0


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


def _encoder_supports_interpolate(encoder: nn.Module) -> bool:
    """Cache whether the encoder forward accepts interpolate_pos_encoding."""
    cached = getattr(encoder, "_supports_interp_pos_enc", None)
    if cached is not None:
        return cached
    import inspect

    target = encoder.module if hasattr(encoder, "module") else encoder
    try:
        sig = inspect.signature(target.forward)
        supported = "interpolate_pos_encoding" in sig.parameters
    except (TypeError, ValueError):
        supported = False
    encoder._supports_interp_pos_enc = supported
    return supported


def pool_encoder_features(encoder: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    kwargs = {}
    if _encoder_supports_interpolate(encoder):
        kwargs["interpolate_pos_encoding"] = True
    outputs = encoder(pixel_values=pixel_values, **kwargs)
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


def build_model(
    backbone_id: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    device: str,
) -> RealFakeClassifier:
    encoder, hidden_size = load_vision_encoder(backbone_id, torch.bfloat16)

    if use_lora:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules(backbone_id),
            lora_dropout=lora_dropout,
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
# Weights & Biases (optional)
# ---------------------------------------------------------------------------


def init_wandb(args: argparse.Namespace, out_dir: Path, rank: int):
    if args.report_to != "wandb" or not is_main_process(rank):
        return None
    try:
        import wandb
    except ImportError:
        print("warning: wandb not installed; disable with --report-to none")
        return None

    run_name = os.environ.get("WANDB_RUN_NAME") or out_dir.name
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "XPlainVerse-ACMChallenge"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=run_name,
        config=vars(args),
        dir=str(out_dir),
        tags=[t for t in os.environ.get("WANDB_TAGS", "pass1").split(",") if t],
    )
    print(f"  wandb    : {wandb.run.url}")
    return wandb


def wandb_log(wb, metrics: dict, step: int) -> None:
    if wb is not None:
        wb.log(metrics, step=step)


def finish_wandb(wb, summary: dict, rank: int) -> None:
    if wb is not None and is_main_process(rank):
        for key, value in summary.items():
            wb.run.summary[key] = value
        wb.finish()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    local_rank, rank, world_size, is_distributed = init_distributed()
    if is_distributed:
        args.device = f"cuda:{local_rank}"
    main = is_main_process(rank)

    out_dir = Path(args.out)
    if main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed:
        dist.barrier()

    run_args = vars(args)
    run_args["world_size"] = world_size
    if main:
        (out_dir / "run_args.json").write_text(json.dumps(run_args, indent=2))

    if main:
        print("== Pass-1 train.py ==")
        print(f"  backbone : {args.backbone}")
        print(f"  out      : {out_dir}")
        print(f"  lora     : {bool(args.lora)}")
        if args.lora:
            print(f"  lora_r/a : {args.lora_r}/{args.lora_alpha}  dropout={args.lora_dropout}")
            print(f"  lr_back  : {args.lr_backbone}")
        if is_distributed:
            print(f"  gpus     : {world_size} (local rank {local_rank})")
            print(f"  eff_batch: {args.batch_size * world_size}")

    df_train = load_manifest(args.train, args.train_slice, args.seed)
    df_val = load_manifest(args.val, args.val_slice, args.seed)
    if main:
        print(f"  train    : {len(df_train)} rows")
        print(f"  val      : {len(df_val)} rows")

    processor = AutoImageProcessor.from_pretrained(args.backbone)
    if args.image_size and args.image_size > 0:
        new_size = {"height": args.image_size, "width": args.image_size}
        if hasattr(processor, "size"):
            processor.size = new_size
        if hasattr(processor, "crop_size") and processor.crop_size:
            processor.crop_size = new_size
        if main:
            print(f"  image_size override: {args.image_size}x{args.image_size}")
    model = build_model(
        args.backbone,
        bool(args.lora),
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.device,
    )
    model, parallel_gpus, parallel_mode = setup_parallel_model(model, args, is_distributed)
    batch_size = loader_batch_size(args, parallel_gpus, parallel_mode)
    wb = init_wandb(args, out_dir, rank)

    if main:
        print(f"  parallel : {parallel_mode}")
        print(f"  gpus     : {parallel_gpus}")
        print(f"  eff_batch: {batch_size}")

    train_dataset = ManifestDataset(df_train, processor)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=is_distributed,
        persistent_workers=False,
    )
    val_loader = None
    if main:
        val_loader = DataLoader(
            ManifestDataset(df_val, processor),
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    core = unwrap_model(model)
    head_params = list(core.head.parameters())
    encoder_params = [p for p in core.encoder.parameters() if p.requires_grad]
    optimizer_groups = [{"params": head_params, "lr": args.lr_head}]
    if encoder_params:
        optimizer_groups.append({"params": encoder_params, "lr": args.lr_backbone})
    optimizer = AdamW(optimizer_groups, weight_decay=0.01)

    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)
    if args.lr_schedule == "cosine":
        if warmup_steps > 0:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    elif warmup_steps > 0:
        warmup_sched = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        const_sched = ConstantLR(
            optimizer, factor=1.0, total_iters=max(1, total_steps - warmup_steps)
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_sched, const_sched], milestones=[warmup_steps]
        )
    else:
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    best_auc = -1.0
    epoch_logs: list[dict] = []
    global_step = 0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        if not args.lora:
            unwrap_model(model).encoder.eval()
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

            if global_step % args.log_every == 0 and main:
                print(
                    f"epoch {epoch} step {global_step}/{total_steps} "
                    f"loss={loss.item():.4f} lr_head={optimizer.param_groups[0]['lr']:.2e}"
                )
                log_payload = {
                    "train/loss": loss.item(),
                    "train/lr_head": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                }
                if len(optimizer.param_groups) > 1:
                    log_payload["train/lr_backbone"] = optimizer.param_groups[1]["lr"]
                wandb_log(wb, log_payload, global_step)

        if is_distributed:
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
            wandb_log(wb, {"val/epoch": epoch}, global_step)

            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                save_best_checkpoint(
                    out_dir, unwrap_model(model), processor, run_args, metrics, y_true, y_score, sample_ids
                )
                print(f"  saved best checkpoint (AUC {best_auc:.4f})")
                wandb_log(wb, {"val/best_auc": best_auc}, global_step)

        if is_distributed:
            dist.barrier()

    if main:
        (out_dir / "metrics.json").write_text(json.dumps(epoch_logs, indent=2))
        print(f"\nfinished — best val AUC {best_auc:.4f}")
    finish_wandb(wb, {"best_val_auc": best_auc, "epochs": args.epochs}, rank)
    cleanup_distributed(is_distributed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pass-1 real/fake classifier (Simplicity Prevails recipe)")
    parser.add_argument("--train", required=True, help="train manifest parquet")
    parser.add_argument("--val", required=True, help="val manifest parquet")
    parser.add_argument("--out", required=True, help="output run directory")
    parser.add_argument("--backbone", default="facebook/dinov3-vit7b16-pretrain-lvd1689m")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128, help="Simplicity Prevails uses 128")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Simplicity Prevails uses 1e-3")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="only when --lora 1")
    parser.add_argument(
        "--lr-schedule",
        default="constant",
        choices=("constant", "cosine"),
        help="paper uses constant lr=1e-3 for the linear head",
    )
    parser.add_argument("--lora", type=int, default=0, help="1 = LoRA encoder (Bombek1 / DINOv3-Forensics style)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="fraction of total steps for LR warmup (used with cosine or constant)",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="override processor resize (HxW square). 0 = use backbone default. "
        "ViT pos-embed is auto-interpolated via interpolate_pos_encoding=True.",
    )
    parser.add_argument("--device", default="cuda:0", help="ignored under torchrun (uses LOCAL_RANK)")
    parser.add_argument("--local-rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-slice", type=int, default=0, help="debug: cap train rows")
    parser.add_argument("--val-slice", type=int, default=0, help="debug: cap val rows")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--report-to",
        default=os.environ.get("REPORT_TO", "wandb"),
        choices=("wandb", "none"),
        help="experiment tracking (env REPORT_TO; use none to disable wandb)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
