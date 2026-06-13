#!/usr/bin/env python3
"""Fast detection-only resubmission: override labels with a new Pass-1 operating
point while REUSING existing complex/simple explanations unchanged (no regen).

Detection and explanation are scored independently on the leaderboard, so this
lets us test a new detector's macro F1 without spending GPU on Pass-2/compressor.
Rows whose label flips vs the base get a label/explanation mismatch — acceptable
for a detection-focused submit.

Usage (CPU, login node OK):
  python3 scripts/build_warmstart_detection_submission.py \
    --pass1-tta /home/jakob/luka/runs/pass1_test_tta_warmstart/predictions.parquet \
    --score-col p_fake_mean --threshold 0.49 \
    --base-submission /home/jakob/luka/runs/submission_calibrated_mean011_20260608-105939/submission.jsonl \
    --out-dir /home/jakob/luka/runs/submission_warmstart_detonly_049
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pass1-tta", required=True, type=Path)
    ap.add_argument("--score-col", default="p_fake_mean")
    ap.add_argument("--threshold", type=float, default=0.49)
    ap.add_argument("--base-submission", required=True, type=Path,
                    help="existing submission.jsonl (sample_id,label,complex,simple) to reuse explanations from")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(args.pass1_tta)
    preds["sample_id"] = preds["sample_id"].astype(str)
    new_label = {
        sid: ("fake" if float(p) >= args.threshold else "real")
        for sid, p in zip(preds["sample_id"], preds[args.score_col].astype(float))
    }
    new_fake = sum(1 for v in new_label.values() if v == "fake")

    out_jsonl = args.out_dir / "submission.jsonl"
    n = 0
    flips = 0
    missing = 0
    transitions = {"fake_to_real": 0, "real_to_fake": 0}
    with args.base_submission.open() as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or row.get("id"))
            base_label = row.get("label")
            label = new_label.get(sid)
            if label is None:
                missing += 1
                label = base_label
            elif label != base_label:
                flips += 1
                key = f"{base_label}_to_{label}"
                transitions[key] = transitions.get(key, 0) + 1
            row["label"] = label
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    summary = {
        "pass1_tta": str(args.pass1_tta),
        "score_col": args.score_col,
        "threshold": args.threshold,
        "base_submission": str(args.base_submission),
        "n_rows": n,
        "n_missing_from_preds": missing,
        "new_pred_fake_rate": new_fake / max(len(new_label), 1),
        "submission_pred_fake_rate": sum(
            1 for _ in [0]
        ),  # placeholder, recomputed below
        "flips_vs_base": flips,
        "flip_frac": flips / max(n, 1),
        "transitions": transitions,
    }
    # Recompute fake rate over the written submission.
    fake_written = 0
    with out_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line and json.loads(line)["label"] == "fake":
                fake_written += 1
    summary["submission_pred_fake_rate"] = fake_written / max(n, 1)

    (args.out_dir / "detection_override_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_jsonl} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
