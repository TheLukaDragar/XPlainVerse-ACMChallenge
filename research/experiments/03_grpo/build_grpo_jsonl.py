#!/usr/bin/env python3
"""Build GRPO datasets for Pass-2 complex-explanation RL from the v2 SFT data.

GRPO needs prompt-only rows (the model generates the completion) plus the
columns the reward reads. We reuse dataset/train_vlm_v2.jsonl (and val) which
already carry the verdict-conditioned user prompt and the GT assistant target,
and transform each row into:

  {
    "messages": [{"role": "user", "content": "<image>\n<conditioned prompt>"}],
    "images": ["/abs/path.png"],
    "reference_complex": "<GT complex paragraph, verdict stripped>",
    "solution": "real" | "fake"            # gold verdict, for the verdict reward
  }

The user prompt is kept verbatim from the SFT row (so train-time conditioning
matches the two-stage inference). The assistant target is dropped; its complex
text becomes the BERTScore reference.

Usage (inside the lj container):
  python3 research/experiments/03_grpo/build_grpo_jsonl.py \
    --in dataset/train_vlm_v2.jsonl --out dataset/train_grpo.jsonl
  python3 research/experiments/03_grpo/build_grpo_jsonl.py \
    --in dataset/val_vlm_v2.jsonl --out dataset/val_grpo.jsonl --max-rows 2000
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_VERDICT_RE = re.compile(r"(?:^|\n)\s*Verdict:\s*(real|fake)\s*\.?\s*$", re.IGNORECASE | re.MULTILINE)


def strip_verdict(text: str) -> str:
    text = (text or "").strip()
    return " ".join(_VERDICT_RE.sub("", text).strip().split())


def parse_verdict(text: str) -> str | None:
    m = _VERDICT_RE.search(text or "")
    return m.group(1).lower() if m else None


def user_message(row: dict) -> dict | None:
    for msg in row.get("messages", []):
        if msg.get("role") == "user":
            return {"role": "user", "content": msg["content"]}
    return None


def assistant_content(row: dict) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-rows", type=int, default=0, help="cap rows (0 = all)")
    args = ap.parse_args()

    n_in = n_out = n_skip = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.inp.open(encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.max_rows and n_out >= args.max_rows:
                break
            row = json.loads(line)
            user = user_message(row)
            images = row.get("images") or []
            target = assistant_content(row)
            ref_complex = strip_verdict(target)
            label = (row.get("label") or parse_verdict(target) or "").strip().lower()
            if user is None or not images or not ref_complex or label not in ("real", "fake"):
                n_skip += 1
                continue
            fout.write(json.dumps({
                "messages": [user],
                "images": images,
                "reference_complex": ref_complex,
                "solution": label,
            }, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"read {n_in} rows; wrote {n_out}; skipped {n_skip} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
