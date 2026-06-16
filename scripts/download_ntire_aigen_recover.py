#!/usr/bin/env python3
"""Reliable resume download for deepfakesMSU/NTIRE-RobustAIGenDetection-train.

6 zip shards (~114 GB), ~277k real/AI in-the-wild images (label 0=real, 1=generated).
After download, unzip each shard_*.zip -> shard_i/{images/*.jpg, labels.csv}.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO = "deepfakesMSU/NTIRE-RobustAIGenDetection-train"
TARGET = Path(os.environ.get("NTIRE_AIGEN_ROOT", "/primoz/luka/external/NTIRE_AIGen"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def local_has_file(rel: str, min_size: int = 1024) -> bool:
    p = TARGET / rel
    return p.is_file() and p.stat().st_size > min_size


def download_missing_one_by_one() -> int:
    api = HfApi()
    files = [
        f
        for f in api.list_repo_files(REPO, repo_type="dataset")
        if f.endswith(".zip")
    ]
    missing = [f for f in files if not local_has_file(f, min_size=1_000_000)]
    log(f"repo zip files={len(files)}  missing={len(missing)}")
    ok = 0
    for i, rel in enumerate(missing, 1):
        log(f"  [{i}/{len(missing)}] {rel}")
        for attempt in range(1, 5):
            try:
                hf_hub_download(
                    repo_id=REPO,
                    filename=rel,
                    repo_type="dataset",
                    local_dir=str(TARGET),
                )
                ok += 1
                break
            except Exception as e:  # noqa: BLE001
                log(f"    attempt {attempt} failed: {e}")
                if attempt == 4:
                    raise
                time.sleep(10 * attempt)
    return ok


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    log(f"NTIRE AIGen reliable download -> {TARGET}")
    try:
        log("snapshot_download (max_workers=2)...")
        snapshot_download(
            repo_id=REPO,
            repo_type="dataset",
            local_dir=str(TARGET),
            max_workers=2,
            allow_patterns=["*.zip", "*.md", "*.csv"],
        )
        log("snapshot_download finished")
    except Exception as e:  # noqa: BLE001
        log(f"snapshot_download error (will try per-file): {e}")

    n = download_missing_one_by_one()
    log(f"done — downloaded/verified {n} missing zip files")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
