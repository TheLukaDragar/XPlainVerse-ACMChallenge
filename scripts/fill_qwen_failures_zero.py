#!/usr/bin/env python3
"""Set null pred->gt coverage fields to 0.0 after exhausted Qwen retries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ("_pred_to_gt_entity", "_pred_to_gt_evidence")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    filled: list[str] = []
    for shard_dir in sorted(args.out_dir.glob("shard_*")):
        cache_path = shard_dir / "_stage_cache.jsonl"
        if not cache_path.is_file():
            continue
        rows = []
        changed = False
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                if any(rec.get(k) is None for k in FIELDS):
                    for k in FIELDS:
                        if rec.get(k) is None:
                            rec[k] = 0.0
                    filled.append(rec["sample_id"])
                    changed = True
                rows.append(rec)
        if changed:
            tmp = cache_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                for rec in rows:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp.replace(cache_path)

    print(json.dumps({"filled_zero": len(set(filled)), "sample_ids": sorted(set(filled))}, indent=2))


if __name__ == "__main__":
    main()
