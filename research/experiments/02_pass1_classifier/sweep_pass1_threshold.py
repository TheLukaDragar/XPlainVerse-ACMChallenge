#!/usr/bin/env python3
"""Sweep Pass-1 decision thresholds on val (labeled) and test (pred rate only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_tta_scores import _binary_f1, thr_best_f1


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    pred = (y_score >= threshold).astype(int)
    fake_f1 = _binary_f1(y_true, pred, pos_label=1)
    real_f1 = _binary_f1(y_true, pred, pos_label=0)
    return {
        "threshold": threshold,
        "acc": float((pred == y_true).mean()),
        "macro_f1": 0.5 * (fake_f1 + real_f1),
        "fake_f1": fake_f1,
        "real_f1": real_f1,
        "pred_fake_rate": float(pred.mean()),
    }


def sweep_thresholds(
    thresholds: np.ndarray,
    y_true: np.ndarray | None,
    y_score: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    for thr in thresholds:
        row = {"threshold": float(thr), "pred_fake_rate": float((y_score >= thr).mean())}
        if y_true is not None:
            row.update(
                {
                    "acc": float(((y_score >= thr).astype(int) == y_true).mean()),
                    "macro_f1": metrics_at_threshold(y_true, y_score, float(thr))["macro_f1"],
                    "fake_f1": metrics_at_threshold(y_true, y_score, float(thr))["fake_f1"],
                    "real_f1": metrics_at_threshold(y_true, y_score, float(thr))["real_f1"],
                }
            )
        rows.append(row)
    return rows


def quantile_threshold(y_score: np.ndarray, target_fake_rate: float) -> float:
    return float(np.quantile(y_score, 1.0 - target_fake_rate))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-min", type=float, default=0.02)
    parser.add_argument("--threshold-max", type=float, default=0.15)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--target-fake-rates", type=float, nargs="+", default=[0.50, 0.545, 0.55, 0.559])
    args = parser.parse_args()

    val_df = pd.read_parquet(args.val_predictions)
    test_df = pd.read_parquet(args.test_predictions)
    y_val = val_df["label_int"].astype(int).to_numpy()

    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)
    cols = ["p_fake_orig", "p_fake_flip", "p_fake_mean"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"thresholds": thresholds.tolist(), "columns": {}, "candidates": []}

    for col in cols:
        val_scores = val_df[col].astype(float).to_numpy()
        test_scores = test_df[col].astype(float).to_numpy()

        val_sweep = sweep_thresholds(thresholds, y_val, val_scores)
        test_sweep = sweep_thresholds(thresholds, None, test_scores)

        _, best = thr_best_f1(y_val, val_scores)
        col_summary = {
            "val_best": best,
            "val_sweep_csv": str(args.output_dir / f"val_{col}.csv"),
            "test_sweep_csv": str(args.output_dir / f"test_{col}.csv"),
            "prior_matched_thresholds": {},
        }

        pd.DataFrame(val_sweep).to_csv(col_summary["val_sweep_csv"], index=False)
        pd.DataFrame(test_sweep).to_csv(col_summary["test_sweep_csv"], index=False)

        for rate in args.target_fake_rates:
            thr_val = quantile_threshold(val_scores, rate)
            thr_test = quantile_threshold(test_scores, rate)
            val_at = metrics_at_threshold(y_val, val_scores, thr_test)
            col_summary["prior_matched_thresholds"][str(rate)] = {
                "target_pred_fake_rate": rate,
                "thr_on_val_scores": thr_val,
                "thr_on_test_scores": thr_test,
                "val_metrics_if_apply_test_thr": val_at,
                "test_pred_fake_rate_at_thr": float((test_scores >= thr_test).mean()),
            }

        best_val_idx = int(np.argmax([r["macro_f1"] for r in val_sweep]))
        col_summary["val_sweep_best_macro_f1"] = val_sweep[best_val_idx]
        summary["columns"][col] = col_summary

    deployed_thr = 0.0838903859257698
    for col in cols:
        for tag, thr in [
            ("deployed", deployed_thr),
            ("val_f1_opt", summary["columns"][col]["val_best"]["thr_best_f1"]),
            ("prior_match_55_test", summary["columns"][col]["prior_matched_thresholds"]["0.55"]["thr_on_test_scores"]),
            ("prior_match_545_test", summary["columns"][col]["prior_matched_thresholds"]["0.545"]["thr_on_test_scores"]),
        ]:
            val_scores = val_df[col].astype(float).to_numpy()
            test_scores = test_df[col].astype(float).to_numpy()
            summary["candidates"].append(
                {
                    "name": f"{col}__{tag}",
                    "score_col": col,
                    "policy": tag,
                    "threshold": float(thr),
                    "val": metrics_at_threshold(y_val, val_scores, float(thr)),
                    "test_pred_fake_rate": float((test_scores >= thr).mean()),
                }
            )

    out_json = args.output_dir / "sweep_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"wrote {args.output_dir}")
    print("\n=== Candidate thresholds ===")
    for row in summary["candidates"]:
        v = row["val"]
        print(
            f"{row['name']:40s} thr={row['threshold']:.4f} "
            f"val_macro_f1={v['macro_f1']:.4f} val_acc={v['acc']:.4f} "
            f"val_pred_fake={v['pred_fake_rate']:.3f} test_pred_fake={row['test_pred_fake_rate']:.3f}"
        )
    print(f"\nfull summary: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
