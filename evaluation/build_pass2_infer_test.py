#!/usr/bin/env python3
"""Build verdict-conditioned Pass-2 infer JSONL for the test split.

Uses Pass-1 ensemble predictions (default: p_fake_orig @ full-val threshold).
See research/experiments/02_pass1_classifier/PASS1_STRATEGY.md.
from dataset/prompt.txt. Supports sharding for multi-GPU inference.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SECTION_RE = re.compile(r"^===\s*(.+?)\s*===\s*$")
HYP_FAKE = "VLM_USER_PROMPT_HYPOTHETICAL_FAKE"
HYP_REAL = "VLM_USER_PROMPT_HYPOTHETICAL_REAL"


def parse_prompt_file(path: Path) -> dict[str, str]:
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
    for key in (HYP_FAKE, HYP_REAL):
        if not sections.get(key):
            raise ValueError(f"prompt file missing section: {key}")
    return sections


INT2LABEL = {0: "real", 1: "fake"}


def load_ensemble_verdicts(
    parquet: Path,
    threshold: float,
    *,
    score_col: str | None = None,
) -> dict[str, str]:
    df = pd.read_parquet(parquet)
    if "pred_label" in df.columns:
        return {
            str(sid): INT2LABEL[int(v)]
            for sid, v in zip(df["sample_id"], df["pred_label"])
        }
    if score_col and score_col in df.columns:
        col = score_col
    elif "p_fake" in df.columns:
        col = "p_fake"
    elif "p_fake_orig" in df.columns:
        col = "p_fake_orig"
    elif "p_fake_mean" in df.columns:
        col = "p_fake_mean"
    else:
        raise ValueError(f"no score column in {parquet}; columns={list(df.columns)}")
    return {
        str(sid): ("fake" if float(p) >= threshold else "real")
        for sid, p in zip(df["sample_id"], df[col])
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--ensemble-pred", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.0838903859257698)
    parser.add_argument("--score-col", default="p_fake_orig")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verdicts-json", type=Path, default=None)
    args = parser.parse_args()

    prompts = parse_prompt_file(args.prompt_file)
    df = pd.read_parquet(args.test_manifest).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if args.shard_count > 1:
        df = df[df.index % args.shard_count == args.shard_id].reset_index(drop=True)

    verdicts = load_ensemble_verdicts(args.ensemble_pred, args.threshold, score_col=args.score_col)

    out_rows = []
    summary = {
        "pred": {"real": 0, "fake": 0},
        "total": 0,
        "missing_verdict": 0,
        "shard_id": args.shard_id,
        "shard_count": args.shard_count,
    }
    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        verdict = verdicts.get(sid)
        if verdict is None:
            summary["missing_verdict"] += 1
            continue
        key = HYP_FAKE if verdict == "fake" else HYP_REAL
        out_rows.append({
            "id": f"{verdict}__{sid}",
            "sample_id": sid,
            "pass1_verdict": verdict,
            "messages": [{"role": "user", "content": "<image>\n" + prompts[key]}],
            "images": [str(row["image_path"])],
        })
        summary["pred"][verdict] += 1
        summary["total"] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for out_row in out_rows:
            fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    if args.verdicts_json:
        args.verdicts_json.write_text(json.dumps(summary, indent=2))

    print(f"wrote {args.out} ({len(out_rows)} rows)")
    print(f"  shard {args.shard_id}/{args.shard_count}  pred={summary['pred']}  missing={summary['missing_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
