#!/usr/bin/env python3
"""Re-extract missing/broken OpenFake JPEGs from parquet via PIL (PNG/WebP/JPEG -> JPEG)."""
from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def indices_to_recover(out_dir: Path, n_rows: int, force: bool) -> list[int] | None:
    """Return indices needing work, or None if shard is fully OK (fast skip, no parquet read)."""
    if force:
        return list(range(n_rows))

    missing: list[int] = []
    broken: list[int] = []
    for i in range(n_rows):
        out = out_dir / f"{i:06d}.jpg"
        if not out.is_file() or out.stat().st_size < 64:
            missing.append(i)
        elif needs_extract(out):
            broken.append(i)

    if not missing and not broken:
        return None
    return sorted(set(missing + broken))


def recover_shard(shard: Path, force: bool) -> tuple[str, int, int, int, int]:
    rel = shard.relative_to(OPENFAKE_ROOT)
    out_dir = JPEG_ROOT / rel.parent / shard.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = pq.read_metadata(shard).num_rows

    todo = indices_to_recover(out_dir, n_rows, force)
    if todo is None:
        return shard.name, n_rows, n_rows, 0, 0

    table = pq.read_table(shard, columns=["image"])
    col = table.column("image")
    skip = fail = 0

    for i in todo:
        out = out_dir / f"{i:06d}.jpg"
        raw = col[i].as_py().get("bytes") or b""
        if not raw or raw.lstrip().startswith((b"<!", b"<html", b"<HTML", b"<?xml")):
            skip += 1
            continue
        try:
            im = Image.open(BytesIO(raw)).convert("RGB")
            tmp = out.with_suffix(".jpg.part")
            im.save(tmp, format="JPEG", quality=95)
            tmp.replace(out)
        except Exception:
            out.unlink(missing_ok=True)
            fail += 1

    ok = n_rows - skip - fail
    return shard.name, n_rows, ok, skip, fail


def recover_shard_worker(args: tuple[str, bool]) -> tuple[str, int, int, int, int]:
    return recover_shard(Path(args[0]), args[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-group", type=int, default=14)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    shard_list = shards(args.max_group)
    if not shard_list:
        raise RuntimeError(f"no shards under {OPENFAKE_ROOT}")

    log(f"recover OpenFake -> {JPEG_ROOT}  shards={len(shard_list)}  workers={args.workers}")
    total_ok = total_skip = total_fail = total_rows = 0
    done = 0

    work = [(str(s), args.force) for s in shard_list]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(recover_shard_worker, w): w[0] for w in work}
        for fut in as_completed(futs):
            name, rows, ok, skip, fail = fut.result()
            done += 1
            total_ok += ok
            total_skip += skip
            total_fail += fail
            total_rows += rows
            log(f"  [{done}/{len(shard_list)}] {name}  ok={ok} skip={skip} fail={fail} rows={rows}")

    log(
        f"done — rows={total_rows} ok={total_ok} skip={total_skip} fail={total_fail} "
        f"jpegs_on_disk={sum(1 for _ in JPEG_ROOT.rglob('*.jpg'))}"
    )


if __name__ == "__main__":
    main()
