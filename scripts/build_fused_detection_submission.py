#!/usr/bin/env python3
"""Build a CodaBench submission by swapping ONLY detection labels with a logit-fused
DINOv3 + ensemble score, reusing a prior zip's complex/simple explanations verbatim.

Detection and explanation are scored independently, so a label-only swap is a pure
detection upgrade with zero explanation-score change.

Example:
  python3 scripts/build_fused_detection_submission.py \
    --fused /home/jakob/luka/runs/pass1_dinov3_test_tta/<ts>/fused_test.parquet \
    --prior-zip /home/jakob/luka/runs/submission_allv3_orig325_*/submission.zip \
    --threshold 0.360 \
    --out /home/jakob/luka/runs/submission_dinov3fused_036
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fused", required=True, type=Path, help="parquet: sample_id, p_fake")
    ap.add_argument("--prior-zip", required=True, type=Path)
    ap.add_argument("--threshold", required=True, type=float)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "_prior"
    work.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.prior_zip) as z:
        z.extractall(work)

    ft = pd.read_parquet(args.fused)
    lab = {str(s): (1 if p >= args.threshold else 0) for s, p in zip(ft["sample_id"], ft["p_fake"])}

    tot = nf = miss = 0
    with (work / "detection.jsonl").open() as fin, (args.out / "detection.jsonl").open("w") as fout:
        for line in fin:
            d = json.loads(line)
            sid = d["id"].rsplit(".", 1)[0]
            tot += 1
            if sid in lab:
                d["pred_label"] = str(lab[sid])
            else:
                miss += 1
            nf += int(d["pred_label"] == "1")
            fout.write(json.dumps(d) + "\n")

    shutil.copy(work / "complex.jsonl", args.out / "complex.jsonl")
    shutil.copy(work / "simple.jsonl", args.out / "simple.jsonl")

    zp = args.out / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("detection.jsonl", "complex.jsonl", "simple.jsonl"):
            z.write(args.out / f, f)

    print(f"rows={tot} fake={nf} ({nf/tot*100:.1f}%) missing_in_fused={miss}")
    print(f"wrote {zp} ({zp.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
