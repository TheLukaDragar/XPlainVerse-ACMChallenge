#!/usr/bin/env python3
"""Duplicate a Pass-1 manifest into orig + horizontal-flip TTA rows."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="manifest parquet (test or val)")
    parser.add_argument("--output", required=True, type=Path, help="manifest_*_tta.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.input).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    orig = df.copy()
    orig["view"] = "orig"
    flip = df.copy()
    flip["view"] = "flip"
    out = pd.concat([orig, flip], ignore_index=True)
    out = out.sort_values(["sample_id", "view"], kind="mergesort").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"wrote {args.output}")
    print(f"  input rows: {len(df)}  tta rows: {len(out)}  views: {out.view.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
