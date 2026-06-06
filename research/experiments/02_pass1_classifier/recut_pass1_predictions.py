#!/usr/bin/env python3
"""Recut Pass-1 wide predictions with a chosen score column + threshold.

Use after TTA inference to switch from p_fake_mean to p_fake_orig (or recalibrate
threshold from full val) without re-running the ensemble.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def thr_best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_curve = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.nanargmax(f1_curve[:-1]))
    thr = float(thresholds[best_idx])
    pred = (y_score >= thr).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
        "thr_best_f1": thr,
        "acc_at_best": float(accuracy_score(y_true, pred)),
        "f1_fake_at_best": float(f1_score(y_true, pred, pos_label=1)),
        "pred_fake_rate": float(pred.mean()),
        "n": int(len(y_true)),
    }
    return thr, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="wide predictions.parquet")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--score-col",
        default="p_fake_orig",
        help="column to use for detection (default p_fake_orig — no flip averaging)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="decision threshold; omit with --calibrate-from",
    )
    parser.add_argument(
        "--calibrate-from",
        type=Path,
        default=None,
        help="val wide parquet with label_int for F1-opt threshold",
    )
    parser.add_argument("--metrics-json", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if args.score_col not in df.columns:
        raise ValueError(f"missing --score-col {args.score_col!r}; have {list(df.columns)}")

    threshold = args.threshold
    cal_metrics = None
    if args.calibrate_from:
        cal = pd.read_parquet(args.calibrate_from)
        if "label_int" not in cal.columns:
            raise ValueError(f"{args.calibrate_from} missing label_int")
        y = cal["label_int"].astype(int).to_numpy()
        s = cal[args.score_col].astype(float).to_numpy()
        threshold, cal_metrics = thr_best_f1(y, s)

    if threshold is None:
        raise ValueError("set --threshold or --calibrate-from")

    out = df.copy()
    out["p_fake"] = out[args.score_col].astype(float)
    out["pred_label"] = (out["p_fake"] >= threshold).astype(int)
    out["threshold"] = threshold
    out["score_col"] = args.score_col

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "score_col": args.score_col,
        "threshold": threshold,
        "n_images": int(len(out)),
        "pred_fake_rate": float(out["pred_label"].mean()),
        "p_fake_avg": float(out["p_fake"].mean()),
    }
    if cal_metrics:
        summary["calibration"] = cal_metrics

    metrics_path = args.metrics_json or args.output.parent / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
