#!/usr/bin/env python3
"""Merge sharded ms-swift infer JSONL files into one sorted file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict] = []
    for shard in args.shards:
        if not shard.is_file():
            raise FileNotFoundError(shard)
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    def sample_id(row: dict) -> str:
        sid = row.get("sample_id")
        if sid:
            return str(sid)
        images = row.get("images") or []
        if images:
            first = images[0]
            path = first.get("path") if isinstance(first, dict) else first
            if path:
                return Path(str(path)).stem
        return str(row.get("id", ""))

    rows.sort(key=sample_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"merged {len(rows)} rows from {len(args.shards)} shards -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
