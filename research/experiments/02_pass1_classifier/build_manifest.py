#!/usr/bin/env python3
"""Build image-classifier manifests from XPlainVerse VLM jsonls.

Outputs (parquet):
- manifest_train.parquet      — all 450k train rows
- manifest_train_balanced.parquet — 1:1 balanced subset (260k = 130k real + 130k fake)
- manifest_val.parquet        — full val set (~110k rows) with GT labels

Schema: image_path (str), label (str, "real"/"fake"), label_int (0/1, fake=1), sample_id (str).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path("/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/code/XPlainVerse-ACMChallenge")
TRAIN_VLM = ROOT / "dataset/train_vlm.jsonl"
VAL_INFER = ROOT / "dataset/val_vlm_infer.jsonl"
VAL_GT = ROOT / "evaluation/data/val_ground_truth.jsonl"
OUT_DIR = ROOT / "research/experiments/02_pass1_classifier/manifests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL2INT = {"real": 0, "fake": 1}


def load_jsonl(path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_train():
    rows = []
    for r in load_jsonl(TRAIN_VLM):
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
    df.to_parquet(OUT_DIR / "manifest_train.parquet")

    rng = random.Random(0)
    n_real = int((df.label == "real").sum())
    n_fake = int((df.label == "fake").sum())
    n = min(n_real, n_fake)
    real_idx = df[df.label == "real"].index.tolist()
    fake_idx = df[df.label == "fake"].index.tolist()
    rng.shuffle(fake_idx)
    keep = real_idx + fake_idx[:n]
    df_bal = df.loc[keep].reset_index(drop=True)
    df_bal = df_bal.sample(frac=1, random_state=0).reset_index(drop=True)
    print(f"train_balanced: {len(df_bal)} rows; class counts: {df_bal.label.value_counts().to_dict()}")
    df_bal.to_parquet(OUT_DIR / "manifest_train_balanced.parquet")


def build_val():
    gt = {r["sample_id"]: r["label"] for r in load_jsonl(VAL_GT)}
    rows = []
    for r in load_jsonl(VAL_INFER):
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
    df.to_parquet(OUT_DIR / "manifest_val.parquet")


if __name__ == "__main__":
    build_train()
    build_val()
    print(f"\nmanifests in: {OUT_DIR}")
