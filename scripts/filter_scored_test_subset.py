#!/usr/bin/env python3
"""Keep only test rows with score_explanations=true (CodaBench explanation protocol)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_id(row: dict) -> str:
    for key in ("id", "sample_id"):
        if key in row and row[key]:
            return str(row[key]).strip()
    raise KeyError(row)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--gt-entity-facts", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ref_rows = read_jsonl(args.reference)
    scored = [row for row in ref_rows if row.get("score_explanations")]
    scored_ids = {row_id(row) for row in scored}

    out_ref = args.output_dir / "reference_scored.jsonl"
    write_jsonl(out_ref, scored)
    print(f"reference: {len(scored)} scored rows -> {out_ref}")

    if args.gt_entity_facts is not None:
        cache_rows = read_jsonl(args.gt_entity_facts)
        filtered = [row for row in cache_rows if row_id(row) in scored_ids]
        out_cache = args.output_dir / "gt_entity_facts_scored.jsonl"
        write_jsonl(out_cache, filtered)
        print(f"gt cache: {len(filtered)} rows -> {out_cache}")


if __name__ == "__main__":
    main()
