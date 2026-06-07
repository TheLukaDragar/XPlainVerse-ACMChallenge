#!/usr/bin/env python3
"""Merge compressor infer (fake) + complex copy (real) into full submission rows."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from build_submission import strip_verdict_and_tags
from utils.challenge_eval_utils import read_jsonl, write_json

VERDICT_RE = re.compile(
    r"(?:^|\n)\s*Verdict:\s*(real|fake)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_simple(text: str) -> str:
    text = strip_verdict_and_tags(text)
    text = VERDICT_RE.sub("", text).strip()
    return " ".join(text.split())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex", required=True, type=Path)
    parser.add_argument("--compressor-infer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--errors-json", type=Path, default=None)
    args = parser.parse_args()

    complex_rows = read_jsonl(args.complex)
    infer_rows = read_jsonl(args.compressor_infer)

    infer_by_id: dict[str, str] = {}
    infer_queue: list[str] = []
    for row in infer_rows:
        response = row.get("response") or row.get("prediction") or ""
        if not isinstance(response, str) or not response.strip():
            continue
        simple = clean_simple(response)
        sid = row.get("sample_id")
        if not sid:
            rid = row.get("id", "")
            if isinstance(rid, str) and "__" in rid:
                sid = rid.split("__", 1)[1]
        if sid:
            infer_by_id[str(sid)] = simple
        else:
            infer_queue.append(simple)

    fake_complex = [r for r in complex_rows if str(r.get("label", "")).lower() == "fake"]
    if infer_queue and len(infer_queue) == len(fake_complex):
        for row, simple in zip(fake_complex, infer_queue):
            infer_by_id[str(row["sample_id"])] = simple

    records: list[dict[str, str]] = []
    errors: list[dict[str, Any]] = []
    missing_fake = 0

    for index, row in enumerate(complex_rows, start=1):
        sid = str(row["sample_id"])
        label = str(row["label"]).lower()
        complex_text = str(row["complex_explanation"]).strip()
        if label == "real":
            simple_text = complex_text
        elif label == "fake":
            simple_text = infer_by_id.get(sid)
            if not simple_text:
                missing_fake += 1
                errors.append({"row_index": index, "sample_id": sid, "error": "missing compressor output"})
                continue
        else:
            errors.append({"row_index": index, "sample_id": sid, "error": f"unknown label: {label}"})
            continue

        records.append({
            "sample_id": sid,
            "label": label,
            "complex_explanation": complex_text,
            "simple_explanation": simple_text,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.errors_json:
        write_json(
            args.errors_json,
            {
                "complex_path": str(args.complex),
                "compressor_infer": str(args.compressor_infer),
                "missing_fake": missing_fake,
                "error_count": len(errors),
                "errors": errors[:100],
            },
        )

    print(f"wrote {len(records)} submission rows → {args.output}")
    print(f"  real (copy complex): {sum(1 for r in records if r['label']=='real')}")
    print(f"  fake (compressor):   {sum(1 for r in records if r['label']=='fake')}")
    if errors:
        print(f"  skipped/errors:      {len(errors)}", flush=True)
    return 1 if errors and not records else 0


if __name__ == "__main__":
    raise SystemExit(main())
