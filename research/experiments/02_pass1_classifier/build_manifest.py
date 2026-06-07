#!/usr/bin/env python3
"""Build image-classifier manifests from XPlainVerse VLM jsonls.

Outputs (parquet):
- manifest_train.parquet               — all 450k train rows
- manifest_train_balanced.parquet      — 1:1 balanced subset (260k; undersamples fake)
- manifest_train_full_balanced.parquet — 1:1 balanced over ALL fakes (oversamples real)
- manifest_val.parquet                 — full val set (~110k) with GT labels

`full_balanced` keeps every fake (max diversity) and oversamples the minority real
class with replacement to a 1:1 ratio. Combined with quality-agnostic augmentation,
repeated reals are not pure duplicates. Use this to "train on full data" without
biasing the model toward the majority (fake) class — which would crush real recall.

Schema: image_path, label ("real"/"fake"), label_int (fake=1), sample_id.

Usage:
  python3 build_manifest.py
  CODE_ROOT=/workspace/XPlainVerse-ACMChallenge python3 build_manifest.py
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pandas as pd

LABEL2INT = {"real": 0, "fake": 1}


def repo_root() -> Path:
    env = os.environ.get("CODE_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_train(train_vlm: Path, out_dir: Path) -> None:
    rows = []
    for r in load_jsonl(train_vlm):
        label = r["label"]
        if label not in LABEL2INT:
            continue
        img = r.get("images", [None])[0]
        if img is None:
            continue
        rows.append({
            "sample_id": r["sample_id"],
            "image_path": img,
            "label": label,
            "label_int": LABEL2INT[label],
        })
    df = pd.DataFrame(rows)
    print(f"train: {len(df)} rows; class counts: {df.label.value_counts().to_dict()}")
    df.to_parquet(out_dir / "manifest_train.parquet")

    rng = random.Random(0)
    n_real = int((df.label == "real").sum())
    n_fake = int((df.label == "fake").sum())
    n = min(n_real, n_fake)
    real_idx = df[df.label == "real"].index.tolist()
    fake_idx = df[df.label == "fake"].index.tolist()
    rng.shuffle(fake_idx)
    keep = real_idx + fake_idx[:n]
    df_bal = df.loc[keep].sample(frac=1, random_state=0).reset_index(drop=True)
    print(f"train_balanced: {len(df_bal)} rows; class counts: {df_bal.label.value_counts().to_dict()}")
    df_bal.to_parquet(out_dir / "manifest_train_balanced.parquet")

    # full_balanced: keep ALL of the majority class, oversample the minority with
    # replacement to a 1:1 ratio. Maximizes (fake) diversity without class imbalance.
    df_real = df[df.label == "real"]
    df_fake = df[df.label == "fake"]
    minority, majority = (df_real, df_fake) if n_real <= n_fake else (df_fake, df_real)
    minority_os = minority.sample(n=len(majority), replace=True, random_state=0)
    df_full = (
        pd.concat([majority, minority_os], ignore_index=True)
        .sample(frac=1, random_state=0)
        .reset_index(drop=True)
    )
    print(
        f"train_full_balanced: {len(df_full)} rows "
        f"(majority={len(majority)}, minority oversampled {len(minority)}->{len(minority_os)}); "
        f"class counts: {df_full.label.value_counts().to_dict()}"
    )
    df_full.to_parquet(out_dir / "manifest_train_full_balanced.parquet")


def build_val(val_infer: Path, val_gt: Path, out_dir: Path) -> None:
    gt = {r["sample_id"]: r["label"] for r in load_jsonl(val_gt)}
    rows = []
    for r in load_jsonl(val_infer):
        sid = r["sample_id"]
        gt_label = gt.get(sid)
        if gt_label not in LABEL2INT:
            continue
        img = r.get("images", [None])[0]
        if img is None:
            continue
        rows.append({
            "sample_id": sid,
            "image_path": img,
            "label": gt_label,
            "label_int": LABEL2INT[gt_label],
        })
    df = pd.DataFrame(rows)
    print(f"val: {len(df)} rows; class counts: {df.label.value_counts().to_dict()}")
    df.to_parquet(out_dir / "manifest_val.parquet")


def main() -> None:
    root = repo_root()
    out_dir = Path(os.environ.get("MANIFEST_DIR", root / "research/experiments/02_pass1_classifier/manifests"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_vlm = Path(os.environ.get("TRAIN_VLM_JSONL", root / "dataset/train_vlm.jsonl"))
    val_infer = Path(os.environ.get("VAL_INFER_JSONL", root / "dataset/val_vlm_infer.jsonl"))
    val_gt = Path(os.environ.get("VAL_GT_JSONL", root / "evaluation/data/val_ground_truth.jsonl"))

    for path in (train_vlm, val_infer, val_gt):
        if not path.is_file():
            raise FileNotFoundError(f"missing input: {path}")

    print(f"code root : {root}")
    print(f"out dir   : {out_dir}")
    build_train(train_vlm, out_dir)
    build_val(val_infer, val_gt, out_dir)
    print(f"\nmanifests in: {out_dir}")


if __name__ == "__main__":
    main()
