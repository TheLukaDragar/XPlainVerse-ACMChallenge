#!/usr/bin/env python3
"""Extract DFBench ZIP archives to DFBench/<source>/ paths expected by img_train.jsonl."""
from __future__ import annotations

import os
import shutil
import time
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get("DFBENCH_ROOT", "/primoz/luka/external/DFBench"))
DEST = ROOT / "DFBench"

# (zip filename, jsonl source folder, mode)
# flat: numbered jpgs at zip root -> DFBench/<source>/
# nested: deep tree -> flatten by basename into DFBench/<source>/
# prefix: strip top folder (edit/, partial_source/) -> DFBench/<source>/
SPECS: list[tuple[str, str, str]] = [
    ("CLIVE.zip", "CLIVE", "nested"),
    ("CSIQ.zip", "CSIQ", "nested"),
    ("Flick8kimg.zip", "Flick8k", "nested"),
    ("LIVE.zip", "LIVE", "nested"),
    ("TID2013.zip", "TID2013", "nested"),
    ("kadid10k.zip", "kadid10k", "nested"),
    ("koniq10k_512x384.zip", "koniq10k", "nested"),
    ("Janus_000001-040000.zip", "Janus", "flat"),
    ("Kandinsky-3_000001-040000.zip", "Kandinsky-3", "flat"),
    ("Kolors_000001-040000.zip", "Kolors", "flat"),
    ("LaVi-Bridge_000001-040000.zip", "LaVi-Bridge", "flat"),
    ("NOVA_000001-040000.zip", "NOVA", "flat"),
    ("PixArt_000001-040000.zip", "PixArt", "flat"),
    ("Playground_000001-040000.zip", "Playground", "flat"),
    ("ali_flux_dev_000001-040000.zip", "ali_flux_dev", "flat"),
    ("ali_flux_schnell_000001-040000.zip", "ali_flux_schnell", "flat"),
    ("edit.zip", "edit", "prefix"),
    ("infinity_000001-040000.zip", "infinity", "flat"),
    ("partial_source.zip", "partial_source", "prefix"),
    ("sd3_5_large_000001-040000.zip", "sd3_5_large", "flat"),
    ("sd3_medium_000001-040000.zip", "sd3_medium", "flat"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def extract_member(zf: zipfile.ZipFile, member: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size > 0:
        return
    tmp = out.with_suffix(out.suffix + ".part")
    with zf.open(member) as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp.replace(out)


def extract_zip(zip_path: Path, source: str, mode: str) -> int:
    dest_dir = DEST / source
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    prefix = f"{source}/"
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            name = Path(member).name
            if mode == "prefix":
                if not member.startswith(prefix):
                    continue
            elif mode == "flat":
                if "/" in member.rstrip("/"):
                    continue
            out = dest_dir / name
            extract_member(zf, member, out)
            n += 1
    return n


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    log(f"DFBench extract -> {DEST}")
    total = 0
    for zip_name, source, mode in SPECS:
        zip_path = ROOT / zip_name
        if not zip_path.is_file():
            raise FileNotFoundError(zip_path)
        log(f"  {zip_name} -> DFBench/{source}/ ({mode})")
        n = extract_zip(zip_path, source, mode)
        total += n
        log(f"    wrote {n} files")
    log(f"done — {total} files under {DEST}")


if __name__ == "__main__":
    main()
