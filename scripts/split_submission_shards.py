#!/usr/bin/env python3
"""Split a 3-file challenge zip into N shard zips (same ids in all three files)."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

FILES = ("detection.jsonl", "complex.jsonl", "simple.jsonl")


def read_jsonl(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def row_id(row: dict) -> str:
    for key in ("id", "sample_id"):
        if key in row and row[key]:
            return str(row[key]).strip()
    raise KeyError(row)


def write_jsonl(rows: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()

    ref_ids = []
    for line in args.reference.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ref_ids.append(row_id(json.loads(line)))

    with zipfile.ZipFile(args.submission) as zf:
        by_file = {}
        for name in FILES:
            member = next(n for n in zf.namelist() if Path(n).name == name)
            by_file[name] = read_jsonl(zf.read(member).decode("utf-8"))

    id_sets = [set(ref_ids[i :: args.num_shards]) for i in range(args.num_shards)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref_rows = []
    for line in args.reference.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ref_rows.append(json.loads(line))
    ref_by_id = {row_id(r): r for r in ref_rows}

    for shard_idx, id_set in enumerate(id_sets):
        zip_path = args.output_dir / f"shard_{shard_idx}.zip"
        ref_shard = args.output_dir / f"shard_{shard_idx}_reference.jsonl"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as out:
            for fname, rows in by_file.items():
                filtered = [r for r in rows if row_id(r) in id_set]
                out.writestr(fname, write_jsonl(filtered))
        ref_shard.write_text(
            write_jsonl([ref_by_id[i] for i in ref_ids if i in id_set]),
            encoding="utf-8",
        )
        print(f"shard_{shard_idx}: {len(id_set)} ids -> {zip_path}")


if __name__ == "__main__":
    main()
