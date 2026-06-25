#!/usr/bin/env python3
"""Reliable resume download for saberzl/SID_Set."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO = "saberzl/SID_Set"
TARGET = Path("/primoz/luka/external/SID_Set")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def local_has_file(rel: str) -> bool:
    p = TARGET / rel
    return p.is_file() and p.stat().st_size > 1024


def download_missing_one_by_one() -> int:
    api = HfApi()
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith("data/")]
    missing = [f for f in files if not local_has_file(f)]
    log(f"repo data files={len(files)}  missing={len(missing)}")
    ok = 0
    for i, rel in enumerate(missing, 1):
        log(f"  [{i}/{len(missing)}] {rel}")
        for attempt in range(1, 4):
            try:
                hf_hub_download(
                    repo_id=REPO,
                    filename=rel,
                    repo_type="dataset",
                    local_dir=str(TARGET),
                )
                ok += 1
                break
            except Exception as e:
                log(f"    attempt {attempt} failed: {e}")
                if attempt == 3:
                    raise
                time.sleep(10 * attempt)
    return ok


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    log(f"SID_Set reliable download -> {TARGET}")
    try:
        log("snapshot_download (max_workers=1)...")
        snapshot_download(
            repo_id=REPO,
            repo_type="dataset",
            local_dir=str(TARGET),
            max_workers=1,
        )
        log("snapshot_download finished")
    except Exception as e:
        log(f"snapshot_download error (will try per-file): {e}")

    n = download_missing_one_by_one()
    log(f"done — downloaded/verified {n} missing files")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
