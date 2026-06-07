#!/usr/bin/env python3
"""Compare Pass-1 TTA score columns (orig / flip / mean) on labeled val."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(int)
    s = y_score.astype(float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diffs = pos[:, None] - neg[None, :]
    return float((np.sum(diffs > 0) + 0.5 * np.sum(diffs == 0)) / diffs.size)


def thr_best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict]:
    y = y_true.astype(int)
    s = y_score.astype(float)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    fn_total = int((y == 1).sum())
    tp_total = fn_total
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(tp_total, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)

    best_idx = int(np.argmax(f1))
    thr = float(s_sorted[best_idx])
    pred = (s >= thr).astype(int)
    fake_f1 = _binary_f1(y, pred, pos_label=1)
    real_f1 = _binary_f1(y, pred, pos_label=0)
    macro_f1 = 0.5 * (fake_f1 + real_f1)
    return thr, {
        "auc": roc_auc(y, s),
        "thr_best_f1": thr,
        "acc_at_best": float((pred == y).mean()),
        "macro_f1_at_best": macro_f1,
        "fake_f1_at_best": fake_f1,
        "real_f1_at_best": real_f1,
        "pred_fake_rate": float(pred.mean()),
        "score_mean": float(s.mean()),
        "score_median": float(np.median(s)),
    }


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray, *, pos_label: int) -> float:
    y_pos = y_true == pos_label
    y_neg = ~y_pos
    pred_pos = y_pred == pos_label
    tp = int((y_pos & pred_pos).sum())
    fp = int((y_neg & pred_pos).sum())
    fn = int((y_pos & ~pred_pos).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / max(precision + recall, 1e-9))


def at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    pred = (y_score >= threshold).astype(int)
    return {
        "threshold": threshold,
        "acc": float((pred == y_true).mean()),
        "macro_f1": 0.5 * (_binary_f1(y_true, pred, pos_label=1) + _binary_f1(y_true, pred, pos_label=0)),
        "fake_f1": _binary_f1(y_true, pred, pos_label=1),
        "real_f1": _binary_f1(y_true, pred, pos_label=0),
        "pred_fake_rate": float(pred.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--deployed-threshold", type=float, default=0.0838903859257698)
    args = parser.parse_args()

    df = pd.read_parquet(args.predictions)
    y = df["label_int"].astype(int).to_numpy()

    cols = ["p_fake_orig", "p_fake_flip", "p_fake_mean"]
    report: dict = {"n": int(len(df)), "by_score_col": {}}
    for col in cols:
        _, metrics = thr_best_f1(y, df[col].astype(float).to_numpy())
        metrics["score_col"] = col
        metrics["at_deployed_thr"] = at_threshold(
            y, df[col].astype(float).to_numpy(), args.deployed_threshold
        )
        report["by_score_col"][col] = metrics

    orig = df["p_fake_orig"].astype(float).to_numpy()
    flip = df["p_fake_flip"].astype(float).to_numpy()
    diff = np.abs(orig - flip)
    thr_deploy = args.deployed_threshold
    report["flip_analysis"] = {
        "pearson_orig_flip": float(np.corrcoef(orig, flip)[0, 1]),
        "median_abs_diff": float(np.median(diff)),
        "mean_abs_diff": float(diff.mean()),
        "p95_abs_diff": float(np.quantile(diff, 0.95)),
        "disagree_at_deployed_thr": float(((orig >= thr_deploy) != (flip >= thr_deploy)).mean()),
    }

    out = args.output or args.predictions.parent / "metrics_by_score_col.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
