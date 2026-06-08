#!/usr/bin/env python3
"""Recut Pass-1 test labels + identify verdict flips vs deployed operating point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-tta", required=True, type=Path, help="wide test TTA parquet")
    ap.add_argument("--test-manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--new-score-col", default="p_fake_mean")
    ap.add_argument("--new-threshold", type=float, default=0.11)
    ap.add_argument("--old-score-col", default="p_fake_orig")
    ap.add_argument("--old-threshold", type=float, default=0.0838903859257698)
    args = ap.parse_args()

    df = pd.read_parquet(args.test_tta)
    old_pred = (df[args.old_score_col].astype(float) >= args.old_threshold).astype(int)
    new_pred = (df[args.new_score_col].astype(float) >= args.new_threshold).astype(int)

    out = df.copy()
    out["p_fake"] = out[args.new_score_col].astype(float)
    out["pred_label"] = new_pred
    out["threshold"] = args.new_threshold
    out["score_col"] = args.new_score_col

    flip_mask = old_pred != new_pred
    flip_ids = out.loc[flip_mask, "sample_id"].astype(str).tolist()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pass1_out = args.out_dir / "pass1_test_predictions.parquet"
    out.to_parquet(pass1_out, index=False)

    manifest = pd.read_parquet(args.test_manifest)
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    flip_set = set(flip_ids)
    flip_manifest = manifest[manifest["sample_id"].isin(flip_set)].sort_values("sample_id").reset_index(drop=True)
    flip_manifest.to_parquet(args.out_dir / "manifest_test_flips.parquet", index=False)

    # Transition counts for logging.
    transitions: dict[str, int] = {}
    for o, n in zip(old_pred[flip_mask], new_pred[flip_mask], strict=True):
        key = f"{'fake' if o else 'real'}_to_{'fake' if n else 'real'}"
        transitions[key] = transitions.get(key, 0) + 1

    summary = {
        "pass1_out": str(pass1_out),
        "flip_manifest": str(args.out_dir / "manifest_test_flips.parquet"),
        "new_score_col": args.new_score_col,
        "new_threshold": args.new_threshold,
        "old_score_col": args.old_score_col,
        "old_threshold": args.old_threshold,
        "n_test": int(len(df)),
        "n_flips": len(flip_ids),
        "flip_frac": len(flip_ids) / len(df),
        "new_pred_fake_rate": float(new_pred.mean()),
        "old_pred_fake_rate": float(old_pred.mean()),
        "transitions": transitions,
        "flip_sample_ids": flip_ids,
    }
    (args.out_dir / "recalibration_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "flip_sample_ids"}, indent=2))
    print(f"wrote {pass1_out} ({len(out)} rows)")
    print(f"wrote flip manifest ({len(flip_manifest)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
