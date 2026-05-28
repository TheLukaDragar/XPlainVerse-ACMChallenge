#!/usr/bin/env python3
"""Convert single-file XPlainVerse submission JSONL to CodaBench 3-file zip."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def label_to_int(label: str) -> int:
    normalized = label.strip().lower()
    if normalized == "real":
        return 0
    if normalized == "fake":
        return 1
    raise ValueError(f"label must be 'real' or 'fake', got: {label!r}")


def convert(input_path: Path, output_path: Path) -> dict[str, int]:
    detection_lines: list[str] = []
    complex_lines: list[str] = []
    simple_lines: list[str] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            sample_id = row.get("sample_id") or row.get("id")
            if not sample_id:
                raise ValueError(f"line {line_number}: missing sample_id/id")

            label = row.get("label")
            if label is None:
                raise ValueError(f"line {line_number}: missing label")
            detection_lines.append(
                json.dumps({"id": str(sample_id), "pred_label": label_to_int(str(label))}, ensure_ascii=False)
            )

            complex_text = row.get("complex_explanation", "")
            if complex_text:
                complex_lines.append(
                    json.dumps(
                        {"id": str(sample_id), "complex_explanation": str(complex_text)},
                        ensure_ascii=False,
                    )
                )

            simple_text = row.get("simple_explanation", "")
            if simple_text:
                simple_lines.append(
                    json.dumps(
                        {"id": str(sample_id), "simple_explanation": str(simple_text)},
                        ensure_ascii=False,
                    )
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("detection.jsonl", "\n".join(detection_lines) + ("\n" if detection_lines else ""))
        archive.writestr("complex.jsonl", "\n".join(complex_lines) + ("\n" if complex_lines else ""))
        archive.writestr("simple.jsonl", "\n".join(simple_lines) + ("\n" if simple_lines else ""))

    return {
        "detection_rows": len(detection_lines),
        "complex_rows": len(complex_lines),
        "simple_rows": len(simple_lines),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CodaBench submission zip from val JSONL")
    parser.add_argument("--input", required=True, help="submission.jsonl (sample_id, label, complex, simple)")
    parser.add_argument("--output", required=True, help="output submission.zip")
    args = parser.parse_args()

    counts = convert(Path(args.input), Path(args.output))
    print(f"wrote {args.output}")
    for key, value in counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
