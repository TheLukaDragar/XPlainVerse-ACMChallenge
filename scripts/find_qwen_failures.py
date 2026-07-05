#!/usr/bin/env python3
"""Find sample_ids with incomplete Qwen stage cache (for retry)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = (
    "_pred_extraction",
    "_gt_to_pred_entity",
    "_gt_to_pred_evidence",
    "_pred_to_gt_entity",
    "_pred_to_gt_evidence",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    failed: list[dict] = []
    for shard_dir in sorted(args.out_dir.glob("shard_*")):
        cache = shard_dir / "_stage_cache.jsonl"
        if not cache.is_file():
            continue
        shard = shard_dir.name.split("_", 1)[1]
        with cache.open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                missing = [k for k in REQUIRED if rec.get(k) is None]
                if missing:
                    failed.append(
                        {
                            "sample_id": rec["sample_id"],
                            "shard": int(shard),
                            "missing": missing,
                        }
                    )

    out = {
        "count": len(failed),
        "samples": failed,
        "shards_affected": sorted({s["shard"] for s in failed}),
    }
    text = json.dumps(out, indent=2) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
