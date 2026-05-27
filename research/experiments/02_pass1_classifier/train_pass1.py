#!/usr/bin/env python3
"""Pass-1 binary classifier — DINOv3-Large + linear head (+ optional LoRA).

Why this design (citations in research/02_literature_review.md):
- DINOv3 is the strongest in-the-wild AIGC backbone (Simplicity Prevails 2026,
  arXiv:2602.01738) — DINOv3-Linear hits 96.5% on GenImage / 94.0% in-the-wild.
- DINOv3-Large (300M) is ungated; the 7B variant requires HF gating acceptance.
- Linear head first to match the published "linear probe" recipe.
- LoRA optional for additional capacity on the backbone (DINOv3-Forensics 2026
  arXiv:2604.16083 reports +10 absolute points using DINOv3 + LoRA).

Usage:
    python3 train_pass1.py \
        --train manifest_train_balanced.parquet \
        --val   manifest_val.parquet \
        --out   /shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/pass1_dinov3 \
        --backbone facebook/dinov3-vitl16-pretrain-lvd1689m \
        --lora 0    # set to 1 to also LoRA the backbone
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                              precision_recall_curve, roc_auc_score)


class ImgDataset(Dataset):
    def __init__(self, df, processor, mean, std):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.tf(img)
        return x, int(row["label_int"]), row["sample_id"]


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    sids = [b[2] for b in batch]
    return xs, ys, sids


class Classifier(nn.Module):
    def __init__(self, backbone, hidden_dim, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        feat = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return self.head(feat)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backbone", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--lora", type=int, default=0, help="1 to LoRA the backbone")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"== Pass-1 trainer == out={out_dir} backbone={args.backbone}")

    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)
    print(f"train={len(df_train)} val={len(df_val)}")

    processor = AutoImageProcessor.from_pretrained(args.backbone)
    mean = processor.image_mean if hasattr(processor, "image_mean") else [0.485, 0.456, 0.406]
    std = processor.image_std if hasattr(processor, "image_std") else [0.229, 0.224, 0.225]

    backbone = AutoModel.from_pretrained(args.backbone, dtype=torch.bfloat16)
    hidden_dim = backbone.config.hidden_size

    if args.lora:
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["query", "key", "value"],
            lora_dropout=0.05, bias="none",
        )
        backbone = get_peft_model(backbone, lc)
        backbone.print_trainable_parameters()
    else:
        for p_ in backbone.parameters():
            p_.requires_grad = False

    model = Classifier(backbone, hidden_dim).to(args.device)

    train_ds = ImgDataset(df_train, processor, mean, std)
    val_ds = ImgDataset(df_val, processor, mean, std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    head_params = list(model.head.parameters())
    backbone_params = [p_ for p_ in model.backbone.parameters() if p_.requires_grad]
    optim_groups = [{"params": head_params, "lr": args.lr_head}]
    if backbone_params:
        optim_groups.append({"params": backbone_params, "lr": args.lr_backbone})
    optim = AdamW(optim_groups, weight_decay=0.01)
    total_steps = max(1, args.epochs * len(train_loader))
    sched = CosineAnnealingLR(optim, T_max=total_steps)

    loss_fn = nn.CrossEntropyLoss()
    best_auc = -1.0
    log = []
    step = 0

    for epoch in range(args.epochs):
        model.train()
        for xs, ys, _ in train_loader:
            xs = xs.to(args.device, non_blocking=True)
            ys = ys.to(args.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xs)
                loss = loss_fn(logits, ys)
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} lr_head {optim.param_groups[0]['lr']:.2e}")

        # eval
        model.eval()
        all_probs = []
        all_labels = []
        all_sids = []
        with torch.no_grad():
            for xs, ys, sids in val_loader:
                xs = xs.to(args.device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(xs)
                probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
                all_probs.append(probs)
                all_labels.append(ys.numpy())
                all_sids.extend(sids)
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        auc = float(roc_auc_score(all_labels, all_probs))
        ap = float(average_precision_score(all_labels, all_probs))
        prec, rec, thr = precision_recall_curve(all_labels, all_probs)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = int(np.nanargmax(f1[:-1]))
        thr_best = float(thr[best_idx])
        pred05 = (all_probs >= 0.5).astype(int)
        pred_best = (all_probs >= thr_best).astype(int)
        m = {
            "epoch": epoch,
            "auc": auc,
            "ap": ap,
            "thr_best_f1": thr_best,
            "acc_at_0.5": float(accuracy_score(all_labels, pred05)),
            "acc_at_best": float(accuracy_score(all_labels, pred_best)),
            "real_acc_at_best": float(accuracy_score(all_labels[all_labels==0], pred_best[all_labels==0])),
            "fake_acc_at_best": float(accuracy_score(all_labels[all_labels==1], pred_best[all_labels==1])),
            "f1_fake_at_best": float(f1_score(all_labels, pred_best, pos_label=1)),
        }
        log.append(m)
        print(f"\n=== epoch {epoch} val ===")
        print(json.dumps(m, indent=2))

        if auc > best_auc:
            best_auc = auc
            ckpt_dir = out_dir / "best_ckpt"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({"head": model.head.state_dict(),
                        "backbone_state": (model.backbone.state_dict() if args.lora else None),
                        "args": vars(args),
                        "metrics": m},
                       ckpt_dir / "ckpt.pt")
            preds_df = pd.DataFrame({"sample_id": all_sids,
                                     "label_int": all_labels,
                                     "p_fake": all_probs})
            preds_df.to_parquet(out_dir / "val_predictions.parquet")

    (out_dir / "metrics.json").write_text(json.dumps(log, indent=2))
    print(f"\ndone. best AUC {best_auc:.4f}")


if __name__ == "__main__":
    main()
