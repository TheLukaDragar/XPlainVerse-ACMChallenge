#!/usr/bin/env python3
"""Calibrate Pass-1 decision threshold on FULL val for macro F1, per score column.

Clean calibration only: tunes the threshold on labeled val (p_fake_orig / flip /
mean) and reports the resulting test pred-fake-rate + how many test verdicts would
FLIP vs the currently-deployed operating point. No test labels are used.

Usage (inside GHCR -lj container):
  python3 calibrate_macrof1.py \
    --val-pred  /home/jakob/luka/runs/pass1_val_tta/20260605-215505/predictions.parquet \
    --test-pred /home/jakob/luka/runs/pass1_test_tta/20260605-170149/predictions.parquet \
    --deployed-score-col p_fake_orig --deployed-threshold 0.0838903859257698 \
    --out /home/jakob/luka/runs/pass1_calibration/macrof1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int) -> float:
    y_pos = y_true == pos_label
    pred_pos = y_pred == pos_label
    tp = int((y_pos & pred_pos).sum())
    fp = int((~y_pos & pred_pos).sum())
    fn = int((y_pos & ~pred_pos).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / max(precision + recall, 1e-9))


def macro_f1(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict:
    pred = (y_score >= thr).astype(int)
    fake_f1 = _binary_f1(y_true, pred, 1)
    real_f1 = _binary_f1(y_true, pred, 0)
    return {
        "threshold": float(thr),
        "macro_f1": 0.5 * (fake_f1 + real_f1),
        "fake_f1": fake_f1,
        "real_f1": real_f1,
        "acc": float((pred == y_true).mean()),
        "pred_fake_rate": float(pred.mean()),
    }


def best_macro(y_true: np.ndarray, y_score: np.ndarray, grid: np.ndarray) -> dict:
    best = None
    for thr in grid:
        m = macro_f1(y_true, y_score, float(thr))
        if best is None or m["macro_f1"] > best["macro_f1"]:
            best = m
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-pred", required=True, type=Path)
    ap.add_argument("--test-pred", required=True, type=Path)
    ap.add_argument("--deployed-score-col", default="p_fake_orig")
    ap.add_argument("--deployed-threshold", type=float, default=0.0838903859257698)
    ap.add_argument("--cols", nargs="+", default=["p_fake_orig", "p_fake_flip", "p_fake_mean"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    val = pd.read_parquet(args.val_pred)
    test = pd.read_parquet(args.test_pred)
    y_val = val["label_int"].astype(int).to_numpy()
    grid = np.round(np.arange(0.005, 0.5 + 1e-9, 0.005), 4)

    # Currently-deployed test verdicts (the operating point behind submission.zip).
    dep_col = args.deployed_score_col
    deployed_test_pred = (test[dep_col].astype(float).to_numpy() >= args.deployed_threshold).astype(int)
    deployed_val = macro_f1(y_val, val[dep_col].astype(float).to_numpy(), args.deployed_threshold)

    report = {
        "deployed": {
            "score_col": dep_col,
            "threshold": args.deployed_threshold,
            "val": deployed_val,
            "test_pred_fake_rate": float(deployed_test_pred.mean()),
            "n_test": int(len(test)),
            "n_val": int(len(val)),
        },
        "candidates": {},
    }

    for col in args.cols:
        if col not in val.columns or col not in test.columns:
            continue
        best = best_macro(y_val, val[col].astype(float).to_numpy(), grid)
        test_pred = (test[col].astype(float).to_numpy() >= best["threshold"]).astype(int)
        n_flip = int((test_pred != deployed_test_pred).sum())
        report["candidates"][col] = {
            "val_best": best,
            "test_pred_fake_rate": float(test_pred.mean()),
            "test_flips_vs_deployed": n_flip,
            "test_flip_frac": n_flip / len(test),
            "val_macro_f1_gain_vs_deployed": best["macro_f1"] - deployed_val["macro_f1"],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
