#!/usr/bin/env python3
"""Build GRPO dataset for the compressor from the SFT compressor JSONL.

GRPO needs a PROMPT-ONLY row plus reward kwargs (the model generates the
completion, the reward scores it). We reuse the exact compressor SFT prompt so
the GRPO policy stays on-distribution with `compressor_vl` SFT.

Input  (dataset/train_compressor.jsonl): messages=[user, assistant], fake only.
Output (dataset/train_compressor_grpo.jsonl):
    {
      "messages": [ {role: user, ...} ],     # prompt only (assistant dropped)
      "reference_simple": <GT simple>,        # reward target for BERTScore
      "solution": <GT simple>,                # alias some swift versions expect
      "label": "fake",
      "sample_id": ...
    }

Reward (research/experiments/03_grpo/compressor_reward.py):
    0.7 * BERT(completion, reference_simple) + 0.3 * SLE_norm(completion)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert(row: dict) -> dict | None:
    messages = row.get("messages") or []
    user = next((m for m in messages if m.get("role") == "user"), None)
    assistant = next((m for m in messages if m.get("role") == "assistant"), None)
    if user is None or assistant is None:
        return None
    ref = str(assistant.get("content", "")).strip()
    if not ref:
        return None
    out = {
        "messages": [{"role": "user", "content": user.get("content", "")}],
        "reference_simple": ref,
        "solution": ref,
        "label": str(row.get("label", "fake")),
    }
    if row.get("sample_id"):
        out["sample_id"] = row["sample_id"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    ap.add_argument("--in-train", type=Path, default=root / "train_compressor.jsonl")
    ap.add_argument("--in-val", type=Path, default=root / "val_compressor.jsonl")
    ap.add_argument("--out-train", type=Path, default=root / "train_compressor_grpo.jsonl")
    ap.add_argument("--out-val", type=Path, default=root / "val_compressor_grpo.jsonl")
    ap.add_argument("--max-train", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-val", type=int, default=2000)
    args = ap.parse_args()

    for src, dst, cap in (
        (args.in_train, args.out_train, args.max_train),
        (args.in_val, args.out_val, args.max_val),
    ):
        if not src.is_file():
            print(f"skip (missing): {src}")
            continue
        n_in = n_out = 0
        with dst.open("w", encoding="utf-8") as fh:
            for row in read_jsonl(src):
                n_in += 1
                conv = convert(row)
                if conv is None:
                    continue
                fh.write(json.dumps(conv, ensure_ascii=False) + "\n")
                n_out += 1
                if cap and n_out >= cap:
                    break
        print(f"{src.name} -> {dst.name}: {n_out} rows (read {n_in})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
