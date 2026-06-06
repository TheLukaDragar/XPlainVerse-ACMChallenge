#!/usr/bin/env python3
"""Extract complex explanations from Pass-2 infer output (test / val).

Uses Pass-1 ensemble labels for detection (not the VLM Verdict line).
Simple explanations are omitted — fill those in a later stage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from build_submission import (
    extract_verdict,
    resolve_sample_id,
    strip_verdict_and_tags,
)
from utils.challenge_eval_utils import read_jsonl, write_json

INT2LABEL = {0: "real", 1: "fake"}


def load_pass1_labels(parquet_path: Path, *, label_col: str = "pred_label") -> dict[str, str]:
    df = pd.read_parquet(parquet_path)
    for col in (label_col, "pred_label", "pred_label_mean"):
        if col in df.columns:
            return {str(sid): INT2LABEL[int(v)] for sid, v in zip(df["sample_id"], df[col])}
    if "p_fake_orig" in df.columns:
        threshold = 0.0838903859257698
        return {
            str(sid): ("fake" if float(p) >= threshold else "real")
            for sid, p in zip(df["sample_id"], df["p_fake_orig"])
        }
    raise ValueError(f"parquet missing pred_label / p_fake_orig: {parquet_path}")


def parse_complex_row(
    row: dict[str, Any],
    *,
    labels: dict[str, str],
    fallback_verdict: bool,
) -> dict[str, str]:
    response = row.get("response") or row.get("prediction") or ""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Missing or empty response")

    sample_id = resolve_sample_id(row)
    label = labels.get(sample_id)
    if label is None:
        if fallback_verdict:
            verdict = extract_verdict(response)
            if verdict is None:
                raise ValueError("missing Pass-1 label and could not parse VLM Verdict")
            label = verdict
        else:
            raise ValueError(f"missing Pass-1 label for sample_id={sample_id}")

    complex_text = strip_verdict_and_tags(response)
    if not complex_text:
        raise ValueError("Empty complex explanation after stripping verdict")

    return {
        "sample_id": sample_id,
        "label": label,
        "complex_explanation": complex_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build complex-only test explanations JSONL.")
    parser.add_argument("--infer", required=True, type=Path, help="ms-swift infer result JSONL")
    parser.add_argument("--pass1-pred", required=True, type=Path, help="Pass-1 predictions.parquet")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL (sample_id, label, complex_explanation)")
    parser.add_argument(
        "--pass1-label-col",
        default="pred_label",
        help="Column in pass1 parquet for detection label (default pred_label)",
    )
    parser.add_argument(
        "--fallback-vlm-verdict",
        action="store_true",
        help="If Pass-1 label missing, fall back to VLM Verdict line (not recommended)",
    )
    parser.add_argument("--errors-json", type=Path, default=None)
    args = parser.parse_args()

    labels = load_pass1_labels(args.pass1_pred, label_col=args.pass1_label_col)
    infer_rows = read_jsonl(args.infer)

    records: list[dict[str, str]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, row in enumerate(infer_rows, start=1):
        try:
            record = parse_complex_row(
                row,
                labels=labels,
                fallback_verdict=args.fallback_vlm_verdict,
            )
        except Exception as exc:
            sample_id = None
            try:
                sample_id = resolve_sample_id(row)
            except Exception:
                pass
            errors.append({"row_index": index, "sample_id": sample_id, "error": str(exc)})
            continue

        if record["sample_id"] in seen:
            errors.append(
                {"row_index": index, "sample_id": record["sample_id"], "error": "duplicate sample_id"}
            )
            continue
        seen.add(record["sample_id"])
        records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.errors_json is not None:
        write_json(
            args.errors_json,
            {
                "infer_path": str(args.infer),
                "pass1_pred": str(args.pass1_pred),
                "error_count": len(errors),
                "errors": errors,
            },
        )

    print(f"Wrote {len(records)} complex explanation rows → {args.output}")
    if errors:
        print(f"Skipped {len(errors)} rows with parse errors.", file=sys.stderr)
        if args.errors_json:
            print(f"Error details → {args.errors_json}", file=sys.stderr)
        sys.exit(1 if not records else 0)


if __name__ == "__main__":
    main()
