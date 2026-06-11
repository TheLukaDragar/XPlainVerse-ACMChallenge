#!/usr/bin/env python3
"""Re-extract missing/broken OpenFake JPEGs from parquet via PIL (PNG/WebP/JPEG -> JPEG)."""
from __future__ import annotations

import argparse
import re
import time
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

OPENFAKE_ROOT = Path("/home/jakob/luka/data/external/OpenFake")
JPEG_ROOT = Path("/primoz/luka/external/OpenFake_jpeg")


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


def needs_extract(out: Path) -> bool:
    if not out.is_file() or out.stat().st_size < 64:
        return True
    try:
        with Image.open(out) as im:
            im.load()
            im.convert("RGB")
        return False
    except Exception:
        return True


def recover_shard(shard: Path, force: bool) -> tuple[int, int, int, int]:
    rel = shard.relative_to(OPENFAKE_ROOT)
    out_dir = JPEG_ROOT / rel.parent / shard.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(shard, columns=["image"])
    col = table.column("image")
    ok = skip = fail = 0

    for i in range(len(table)):
        out = out_dir / f"{i:06d}.jpg"
        if not force and not needs_extract(out):
            ok += 1
            continue

        raw = col[i].as_py().get("bytes") or b""
        if not raw or raw.lstrip().startswith((b"<!", b"<html", b"<HTML", b"<?xml")):
            skip += 1
            continue

        try:
            im = Image.open(BytesIO(raw)).convert("RGB")
            tmp = out.with_suffix(".jpg.part")
            im.save(tmp, format="JPEG", quality=95)
            tmp.replace(out)
            ok += 1
        except Exception:
            if out.is_file():
                out.unlink(missing_ok=True)
            fail += 1

    return ok, skip, fail, len(table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-group", type=int, default=14)
    parser.add_argument("--force", action="store_true", help="re-extract even if file looks OK")
    args = parser.parse_args()

    shard_list = shards(args.max_group)
    if not shard_list:
        raise RuntimeError(f"no shards under {OPENFAKE_ROOT}")

    log(f"recover missing OpenFake -> {JPEG_ROOT}  shards={len(shard_list)}")
    total_ok = total_skip = total_fail = total_rows = 0

    for n, shard in enumerate(shard_list, 1):
        log(f"  [{n}/{len(shard_list)}] {shard.name}")
        ok, skip, fail, rows = recover_shard(shard, args.force)
        total_ok += ok
        total_skip += skip
        total_fail += fail
        total_rows += rows
        log(f"    ok={ok} skip={skip} fail={fail} rows={rows}")

    log(
        f"done — rows={total_rows} ok={total_ok} skip={total_skip} fail={total_fail} "
        f"jpegs_on_disk={sum(1 for _ in JPEG_ROOT.rglob('*.jpg'))}"
    )


if __name__ == "__main__":
    main()
