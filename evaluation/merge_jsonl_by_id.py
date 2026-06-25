#!/usr/bin/env python3
"""Merge JSONL rows: patch overrides base by sample_id (or id suffix after __)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def row_id(row: dict) -> str | None:
    sid = row.get("sample_id")
    if sid:
        return str(sid)
    rid = row.get("id")
    if isinstance(rid, str) and "__" in rid:
        return rid.split("__", 1)[1]
    if rid:
        return str(rid)
    return None


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--patch", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--sort-by-id", action="store_true", help="sort output by sample_id")
    args = ap.parse_args()

    merged: dict[str, dict] = {}
    order: list[str] = []
    for path in (args.base, args.patch):
        for row in load_jsonl(path):
            sid = row_id(row)
            if not sid:
                continue
            if sid not in merged:
                order.append(sid)
            merged[sid] = row

    if args.sort_by_id:
        order = sorted(merged.keys())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for sid in order:
            fh.write(json.dumps(merged[sid], ensure_ascii=False) + "\n")

    print(f"merged {len(merged)} rows -> {args.output}")
    print(f"  base:  {sum(1 for _ in load_jsonl(args.base))} rows")
    print(f"  patch: {sum(1 for _ in load_jsonl(args.patch))} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
