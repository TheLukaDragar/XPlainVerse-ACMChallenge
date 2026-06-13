#!/usr/bin/env python3
"""Build probe submissions to decompose the simple-explanation score.

No model inference here. We construct submission JSONLs directly from the
ground truth to measure:

  * fake ceiling:   pred simple = GT simple  -> BERT ~1.0, SLE(GT simple)
  * real baseline:  pred simple = GT complex -> BERT ~1.0, SLE(GT complex)  [current pipeline]
  * real levers:    shorten the real simple to trade BERT for SLE (no retrain)

Simple overall = 0.7 * BERT_f1 + 0.3 * normalize_sle(SLE)
normalize_sle(x) = (clip(x, -1, 4) + 1) / 5
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GT_DEFAULT = Path(__file__).resolve().parents[3] / "evaluation/data/val_ground_truth.jsonl"

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def read_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_sentence(text: str) -> str:
    parts = _SENT_SPLIT.split(text.strip())
    return parts[0].strip() if parts else text.strip()


def first_n_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n])


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", type=Path, default=GT_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "probe")
    ap.add_argument("--n-per-class", type=int, default=1000)
    args = ap.parse_args()

    rows = read_jsonl(args.ground_truth)
    fakes = [r for r in rows if str(r.get("label")).lower() == "fake"][: args.n_per_class]
    reals = [r for r in rows if str(r.get("label")).lower() == "real"][: args.n_per_class]
    subset = fakes + reals
    print(f"subset: {len(fakes)} fake + {len(reals)} real = {len(subset)}")

    # Shared reference subset (so the evaluator aligns on this subset only).
    ref_path = args.out_dir / "ref_subset.jsonl"
    write_jsonl(ref_path, subset)

    def base_row(r):
        return {
            "sample_id": r["sample_id"],
            "label": r["label"],
            "complex_explanation": r["complex_explanation"],
        }

    policies = {
        # fake: GT simple (compressor ceiling); real: GT complex (current copy policy)
        "oracle": lambda r: r["simple_explanation"],
        # real only: first sentence of complex; fake: GT simple
        "real_firstsent": lambda r: (
            r["simple_explanation"] if str(r["label"]).lower() == "fake"
            else first_sentence(r["complex_explanation"])
        ),
        # real only: first 35 words; fake: GT simple
        "real_w35": lambda r: (
            r["simple_explanation"] if str(r["label"]).lower() == "fake"
            else first_n_words(r["complex_explanation"], 35)
        ),
        # real only: first 25 words; fake: GT simple
        "real_w25": lambda r: (
            r["simple_explanation"] if str(r["label"]).lower() == "fake"
            else first_n_words(r["complex_explanation"], 25)
        ),
    }

    for name, fn in policies.items():
        out_rows = []
        for r in subset:
            row = base_row(r)
            row["simple_explanation"] = fn(r)
            out_rows.append(row)
        out_path = args.out_dir / f"submission_{name}.jsonl"
        write_jsonl(out_path, out_rows)
        print(f"wrote {out_path}")

    print(f"reference subset: {ref_path}")


if __name__ == "__main__":
    main()
