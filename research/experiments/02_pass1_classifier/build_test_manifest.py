#!/usr/bin/env python3
"""Build Pass-1 manifest parquet(s) from the XPlainVerse test image directory.

Test images live on Lj NVMe (flat dir, no labels):
  /primoz/luka/XPlainVerse/data/XPlainVerse/test/images/<sample_id>.{png,webp}

Outputs:
  manifest_test.parquet       — one row per image (200k)
  manifest_test_tta.parquet   — two rows per image: view=orig + view=flip (400k)

Schema: sample_id, image_path, view (optional), label_int (-1 = unknown)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".png", ".webp", ".jpg", ".jpeg"}


def repo_root() -> Path:
    env = os.environ.get("CODE_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


def scan_test_images(test_images_dir: Path) -> list[dict]:
    if not test_images_dir.is_dir():
        raise FileNotFoundError(f"test images dir not found: {test_images_dir}")

    rows: list[dict] = []
    for path in sorted(test_images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rows.append(
            {
                "sample_id": path.stem,
                "image_path": str(path.resolve()),
                "view": "orig",
                "label_int": -1,
            }
        )
    if not rows:
        raise RuntimeError(f"no images found under {test_images_dir}")
    return rows


def write_manifest(rows: list[dict], out_path: Path, *, tta: bool) -> None:
    df = pd.DataFrame(rows)
    if tta:
        flip_rows = []
        for row in rows:
            flip_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "view": "flip",
                    "label_int": -1,
                }
            )
        df = pd.concat([df, pd.DataFrame(flip_rows)], ignore_index=True)
        df = df.sort_values(["sample_id", "view"], kind="mergesort").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    n_images = df["sample_id"].nunique()
    print(f"wrote {out_path}")
    print(f"  rows: {len(df)}  unique sample_id: {n_images}")
    if tta:
        print(f"  views: {df.view.value_counts().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build test Pass-1 manifest parquet(s)")
    parser.add_argument(
        "--test-images-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "TEST_IMAGES_DIR",
                "/primoz/luka/XPlainVerse/data/XPlainVerse/test/images",
            )
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <repo>/research/experiments/02_pass1_classifier/manifests",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="also write manifest_test_tta.parquet (orig + horizontal flip rows)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (
        repo_root() / "research/experiments/02_pass1_classifier/manifests"
    )
    rows = scan_test_images(args.test_images_dir)

    write_manifest(rows, out_dir / "manifest_test.parquet", tta=False)
    if args.tta:
        write_manifest(rows, out_dir / "manifest_test_tta.parquet", tta=True)


if __name__ == "__main__":
    main()
