#!/usr/bin/env python3
"""Merge compressor infer JSONL: patch overrides base by sample_id.

Swift infer drops sample_id from outputs. Re-attach ids from conditioned shards
(fake__{sample_id}) or, for legacy base-only files, zip fake rows from a complex
JSONL with infer responses in file order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.challenge_eval_utils import read_jsonl


def infer_id(row: dict) -> str | None:
    sid = row.get("sample_id")
    if sid:
        return str(sid)
    rid = row.get("id")
    if isinstance(rid, str) and "__" in rid:
        return rid.split("__", 1)[1]
    return None


def load_tagged_infer(
    infer_path: Path,
    *,
    conditioned_glob: str | None = None,
    complex_for_order: Path | None = None,
) -> dict[str, dict]:
    infer_rows = read_jsonl(infer_path)
    tagged: dict[str, dict] = {}

    if conditioned_glob:
        cond_paths = sorted(Path(infer_path.parent).glob(conditioned_glob))
        if cond_paths:
            cond_by_id: dict[str, str] = {}
            for cond_path in cond_paths:
                for row in read_jsonl(cond_path):
                    sid = infer_id(row)
                    if sid:
                        cond_by_id[sid] = sid
            idx = 0
            cond_ids = []
            for cond_path in cond_paths:
                for row in read_jsonl(cond_path):
                    sid = infer_id(row)
                    if sid:
                        cond_ids.append(sid)
            if len(cond_ids) == len(infer_rows):
                for sid, infer_row in zip(cond_ids, infer_rows, strict=True):
                    out = dict(infer_row)
                    out["sample_id"] = sid
                    tagged[sid] = out
                return tagged

    if complex_for_order is not None:
        fake_ids = [
            str(r["sample_id"])
            for r in read_jsonl(complex_for_order)
            if str(r.get("label", "")).lower() == "fake"
        ]
        if len(fake_ids) == len(infer_rows):
            for sid, infer_row in zip(fake_ids, infer_rows, strict=True):
                out = dict(infer_row)
                out["sample_id"] = sid
                tagged[sid] = out
            return tagged

    for row in infer_rows:
        sid = infer_id(row)
        if sid:
            tagged[sid] = row
    return tagged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-infer", required=True, type=Path)
    ap.add_argument("--patch-infer", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--base-complex", type=Path, default=None, help="zip base infer with fake order")
    ap.add_argument(
        "--patch-conditioned-glob",
        default="compressor_infer_shard*.jsonl",
        help="conditioned shards beside patch infer (carry sample_id)",
    )
    args = ap.parse_args()

    merged: dict[str, dict] = {}
    merged.update(
        load_tagged_infer(
            args.base_infer,
            complex_for_order=args.base_complex,
        )
    )
    merged.update(
        load_tagged_infer(
            args.patch_infer,
            conditioned_glob=args.patch_conditioned_glob,
        )
    )

    if not merged:
        raise SystemExit("no tagged compressor rows — check infer/conditioned paths")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for sid in sorted(merged.keys()):
            fh.write(json.dumps(merged[sid], ensure_ascii=False) + "\n")

    print(f"merged {len(merged)} compressor rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
