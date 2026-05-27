#!/usr/bin/env python3
"""Convert ms-swift infer JSONL to XPlainVerse challenge submission format.

Input rows (from ``swift infer --result_path``) typically contain:
  - ``response``: model output (complex paragraph + ``Verdict: real|fake``)
  - ``sample_id`` and/or ``images`` with path stem
  - optional ``label`` (only when infer dataset includes metadata)

Output (one JSON line per sample):
  {"sample_id": "...", "label": "fake", "complex_explanation": "...",
   "simple_explanation": "..."}

Simple explanation policy (until compressor stage 2):
  - ``match_verdict`` (default): real → copy complex; fake → first-sentence placeholder
  - ``copy``: duplicate complex into simple (fake SLE will be poor — debug only)
  - ``first_sentence``: always use first sentence heuristic
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from utils.challenge_eval_utils import read_jsonl, write_json


VERDICT_RE = re.compile(
    r"(?:^|\n)\s*Verdict:\s*(real|fake)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
BASELINE_ANSWER_RE = re.compile(
    r"<answer>\s*(real|fake)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)
GENERIC_SIMPLE_PATTERNS = (
    re.compile(r"^this (?:picture|image) looks real because\b", re.I),
    re.compile(r"^the image (?:contains|shows|displays)\b", re.I),
)


def _image_path(row: dict[str, Any]) -> str | None:
    images = row.get("images") or []
    if not images:
        return None
    first = images[0]
    if isinstance(first, dict):
        return first.get("path") or first.get("bytes")
    return first if isinstance(first, str) else None


def resolve_sample_id(row: dict[str, Any]) -> str:
    sample_id = row.get("sample_id")
    if sample_id:
        return str(sample_id)

    record_id = row.get("id")
    if isinstance(record_id, str) and "__" in record_id:
        return record_id.split("__", 1)[1]

    image_path = _image_path(row)
    if image_path and isinstance(image_path, str):
        return Path(image_path).stem

    raise ValueError("Could not resolve sample_id (need sample_id, id, or images[].path)")


def extract_verdict(response: str) -> str | None:
    if not response:
        return None
    match = VERDICT_RE.search(response.strip())
    if match:
        return match.group(1).lower()
    baseline = BASELINE_ANSWER_RE.search(response)
    if baseline:
        return baseline.group(1).lower()
    return None


def strip_verdict_and_tags(response: str) -> str:
    text = response.strip()
    if VERDICT_RE.search(text):
        text = VERDICT_RE.sub("", text).strip()
    baseline = BASELINE_ANSWER_RE.search(text)
    if baseline:
        reasoning_match = re.search(
            r"<reasoning>\s*(.*?)\s*</reasoning>",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if reasoning_match:
            return reasoning_match.group(1).strip()
        text = BASELINE_ANSWER_RE.sub("", text).strip()
    text = re.sub(r"</?reasoning>", "", text, flags=re.I).strip()
    return " ".join(text.split())


def first_sentence(text: str, *, max_chars: int = 220) -> str:
    text = text.strip()
    if not text:
        return text
    for pattern in (r"(?<=[.!?])\s+(?=[A-Z\"'])", r"\.\s+(?=Additionally|Furthermore|Finally|On the )"):
        parts = re.split(pattern, text, maxsplit=1)
        if len(parts) > 1 and len(parts[0]) >= 20:
            text = parts[0].strip()
            break
    if len(text) > max_chars:
        cut = text[:max_chars].rsplit(" ", 1)[0]
        text = cut if cut else text[:max_chars]
        if not text.endswith("."):
            text += "."
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def build_simple_explanation(complex_text: str, label: str, mode: str) -> str:
    if mode == "copy":
        return complex_text
    if mode == "first_sentence":
        return first_sentence(complex_text)
    if mode == "match_verdict":
        if label == "real":
            return complex_text
        return first_sentence(complex_text)
    raise ValueError(f"Unknown simple_mode: {mode}")


def parse_infer_row(row: dict[str, Any], *, simple_mode: str) -> dict[str, str]:
    response = row.get("response") or row.get("prediction") or ""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Missing or empty response")

    sample_id = resolve_sample_id(row)
    verdict = extract_verdict(response)
    if verdict is None:
        raise ValueError("Could not parse Verdict: real|fake (or <answer> tag)")

    complex_text = strip_verdict_and_tags(response)
    if not complex_text:
        raise ValueError("Empty complex explanation after stripping verdict")

    return {
        "sample_id": sample_id,
        "label": verdict,
        "complex_explanation": complex_text,
        "simple_explanation": build_simple_explanation(complex_text, verdict, simple_mode),
    }


def build_submission(
    infer_rows: list[dict[str, Any]],
    *,
    simple_mode: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    submission: list[dict[str, str]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, row in enumerate(infer_rows, start=1):
        try:
            record = parse_infer_row(row, simple_mode=simple_mode)
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
                {
                    "row_index": index,
                    "sample_id": record["sample_id"],
                    "error": "duplicate sample_id",
                }
            )
            continue
        seen.add(record["sample_id"])
        submission.append(record)

    return submission, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build challenge submission JSONL from infer output.")
    parser.add_argument("--infer", required=True, type=Path, help="ms-swift infer result JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Output submission JSONL")
    parser.add_argument(
        "--simple-mode",
        choices=("match_verdict", "copy", "first_sentence"),
        default="match_verdict",
        help="How to fill simple_explanation before compressor stage (default: match_verdict)",
    )
    parser.add_argument(
        "--errors-json",
        type=Path,
        default=None,
        help="Optional path to write parse errors as JSON",
    )
    args = parser.parse_args()

    infer_rows = read_jsonl(args.infer)
    submission, errors = build_submission(infer_rows, simple_mode=args.simple_mode)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in submission:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.errors_json is not None:
        write_json(
            args.errors_json,
            {"infer_path": str(args.infer), "error_count": len(errors), "errors": errors},
        )

    print(f"Wrote {len(submission)} submission rows → {args.output}")
    if errors:
        print(f"Skipped {len(errors)} rows with parse errors.", file=sys.stderr)
        if args.errors_json:
            print(f"Error details → {args.errors_json}", file=sys.stderr)
        sys.exit(1 if not submission else 0)


if __name__ == "__main__":
    main()
