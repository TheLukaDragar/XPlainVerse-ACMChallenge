#!/usr/bin/env python3
"""Extract SID_Set parquet images to JPEG on /primoz for Pass-1 training."""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

SID_ROOT = Path(os.environ.get("SID_SET_ROOT", "/primoz/luka/external/SID_Set"))
OUT_ROOT = Path(os.environ.get("SID_SET_JPEG_ROOT", "/primoz/luka/external/SID_Set_jpeg"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def shard_list(splits: list[str]) -> list[Path]:
    data = SID_ROOT / "data"
    out: list[Path] = []
    for split in splits:
        if split == "train":
            out.extend(sorted(data.glob("train-*.parquet")))
        elif split in {"validation", "val"}:
            out.extend(sorted(data.glob("validation-*.parquet")))
        else:
            raise ValueError(f"unknown split: {split}")
    return [p for p in out if p.stat().st_size > 1024]


def save_jpeg(raw: bytes, out_path: Path) -> bool:
    if not raw or raw.lstrip().startswith((b"<!", b"<html", b"<HTML", b"<?xml")):
        return False
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG", quality=95)
        return out_path.stat().st_size > 64
    except Exception:
        return False


def extract_shard(shard: Path, skip_existing: bool) -> tuple[str, int, int, int]:
    rel = shard.relative_to(SID_ROOT)
    out_dir = OUT_ROOT / rel.parent / shard.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(shard, columns=["image"])
    col = table.column("image")
    wrote = skipped = failed = 0
    for i in range(len(table)):
        out_path = out_dir / f"{i:06d}.jpg"
        if skip_existing and out_path.is_file() and out_path.stat().st_size > 64:
            skipped += 1
            continue
        raw = col[i].as_py().get("bytes") or b""
        if save_jpeg(raw, out_path):
            wrote += 1
        else:
            failed += 1
    return shard.name, wrote, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    shards = shard_list(args.splits)
    if not shards:
        raise RuntimeError(f"no SID_Set shards under {SID_ROOT}/data")

    log(f"SID_Set JPEG extract: {len(shards)} shards -> {OUT_ROOT}  workers={args.workers}")
    total_wrote = total_skip = total_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_shard, shard, args.skip_existing): shard
            for shard in shards
        }
        for n, fut in enumerate(as_completed(futures), 1):
            name, wrote, skipped, failed = fut.result()
            total_wrote += wrote
            total_skip += skipped
            total_fail += failed
            log(f"  [{n}/{len(shards)}] {name}: wrote={wrote} skip={skipped} fail={failed}")

    log(f"done — wrote={total_wrote} skip={total_skip} fail={total_fail}")


if __name__ == "__main__":
    main()
