#!/usr/bin/env python3
"""Build compressor infer JSONL from test complex explanations (fake rows only).

User message: COMPRESSOR_USER_PROMPT + complex_explanation (no image).
Real rows are excluded — simple = complex at merge time.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from utils.challenge_eval_utils import read_jsonl

SECTION_RE = re.compile(r"^===\s*(.+?)\s*===\s*$")


def parse_prompt_file(path: Path) -> str:
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = SECTION_RE.match(line)
        if m:
            name = m.group(1).strip()
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = None if name.upper() == "END" else name
            buf = []
            continue
        if current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    key = "COMPRESSOR_USER_PROMPT"
    if not sections.get(key):
        raise ValueError(f"prompt file missing section: {key}")
    return sections[key]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    prompt = parse_prompt_file(args.prompt_file)
    rows = read_jsonl(args.complex)
    fake_rows = [r for r in rows if str(r.get("label", "")).lower() == "fake"]
    if args.shard_count > 1:
        fake_rows = [r for i, r in enumerate(fake_rows) if i % args.shard_count == args.shard_id]

    out_rows = []
    for row in fake_rows:
        sid = str(row["sample_id"])
        complex_text = str(row["complex_explanation"]).strip()
        out_rows.append({
            "id": f"fake__{sid}",
            "sample_id": sid,
            "label": "fake",
            "messages": [{"role": "user", "content": f"{prompt}\n{complex_text}"}],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for out_row in out_rows:
            fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"wrote {args.out} ({len(out_rows)} fake rows)")
    print(f"  shard {args.shard_id}/{args.shard_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
