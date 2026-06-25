#!/usr/bin/env python3
"""Audit SID_Set parquet on /primoz before extract/manifest."""
from __future__ import annotations

import time
from collections import Counter
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

SID_ROOT = Path("/primoz/luka/external/SID_Set")
DATA = SID_ROOT / "data"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    if not DATA.is_dir():
        raise SystemExit(f"missing {DATA}")

    shards = sorted(DATA.glob("train-*.parquet")) + sorted(DATA.glob("validation-*.parquet"))
    log(f"shards={len(shards)} (expect 283)")

    labels: Counter[int] = Counter()
    generators: Counter[str] = Counter()
    decode_ok = decode_fail = 0
    widths: list[int] = []
    heights: list[int] = []

    for j, shard in enumerate(shards, 1):
        t = pq.read_table(shard, columns=["label", "img_id", "width", "height", "image"])
        for lab, iid, w, h in zip(
            t.column("label").to_pylist(),
            t.column("img_id").to_pylist(),
            t.column("width").to_pylist(),
            t.column("height").to_pylist(),
        ):
            labels[int(lab)] += 1
            if int(lab) == 0:
                generators["real"] += 1
            elif int(lab) == 1:
                generators["full_synthetic"] += 1
            else:
                generators["tampered"] += 1
            if w and h:
                widths.append(int(w))
                heights.append(int(h))

        if j == 1:
            for i in range(min(20, t.num_rows)):
                raw = t.column("image")[i].as_py().get("bytes") or b""
                try:
                    Image.open(BytesIO(raw)).convert("RGB")
                    decode_ok += 1
                except Exception:
                    decode_fail += 1

        if j % 50 == 0:
            log(f"  scanned {j}/{len(shards)} shards")

    total = sum(labels.values())
    log(f"rows={total} (expect 240000)")
    log(f"labels={dict(sorted(labels.items()))}  # 0=real 1=full_synthetic 2=tampered")
    log(f"generators={dict(generators)}")
    if widths:
        log(f"width  min/med/max = {min(widths)}/{sorted(widths)[len(widths)//2]}/{max(widths)}")
        log(f"height min/med/max = {min(heights)}/{sorted(heights)[len(heights)//2]}/{max(heights)}")
    log(f"sample decode check shard0: ok={decode_ok} fail={decode_fail}")
    if total != 240_000 or len(shards) != 283:
        raise SystemExit("audit failed: unexpected row/shard counts")
    if decode_fail:
        raise SystemExit("audit failed: PIL decode errors in sample")
    log("audit OK")


if __name__ == "__main__":
    main()
