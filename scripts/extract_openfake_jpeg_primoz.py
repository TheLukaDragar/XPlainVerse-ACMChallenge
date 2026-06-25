#!/usr/bin/env python3
"""Extract OpenFake parquet embedded JPEGs to primoz for fast Pass-1 training IO."""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import pyarrow.parquet as pq

OPENFAKE_ROOT = Path(os.environ.get("OPENFAKE_ROOT", "/home/jakob/luka/data/external/OpenFake"))
OUT_ROOT = Path(os.environ.get("OPENFAKE_JPEG_ROOT", "/primoz/luka/external/OpenFake_jpeg"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def shards(max_group: int) -> list[Path]:
    core = OPENFAKE_ROOT / "core"
    out: list[Path] = []
    for p in sorted(core.glob("train-*-of-*.parquet")):
        m = re.match(r"train-(\d+)-of-", p.name)
        if not m or int(m.group(1)) > max_group:
            continue
        if p.stat().st_size > 1024:
            out.append(p)
    return out


def extract_shard(shard: Path, skip_existing: bool) -> int:
    rel = shard.relative_to(OPENFAKE_ROOT)
    out_dir = OUT_ROOT / rel.parent / rel.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(shard, columns=["image"])
    col = table.column("image")
    n = 0
    for i in range(len(table)):
        out_path = out_dir / f"{i:06d}.jpg"
        if skip_existing and out_path.is_file() and out_path.stat().st_size > 1024:
            continue
        img_struct = col[i].as_py()
        raw = img_struct.get("bytes")
        if not raw or raw[:15].lstrip().startswith((b"<!", b"<html", b"<HTML")):
            continue
        if raw[:2] not in (b"\xff\xd8", b"\x89P") and raw[:4] != b"RIFF":
            continue
        out_path.write_bytes(raw)
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-group", type=int, default=14)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    shard_list = shards(args.max_group)
    if not shard_list:
        raise RuntimeError(f"no shards under {OPENFAKE_ROOT}")

    log(f"OpenFake JPEG extract: {len(shard_list)} shards -> {OUT_ROOT}")
    total = 0
    for j, shard in enumerate(shard_list, 1):
        log(f"  [{j}/{len(shard_list)}] {shard.name}")
        n = extract_shard(shard, args.skip_existing)
        total += n
        log(f"    extracted {n}")
    log(f"done — {total} images")


if __name__ == "__main__":
    main()
