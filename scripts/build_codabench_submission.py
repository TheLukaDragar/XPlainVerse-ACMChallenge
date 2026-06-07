#!/usr/bin/env python3
"""Convert single-file XPlainVerse submission JSONL to CodaBench 3-file zip."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd


def label_to_int(label: str) -> int:
    normalized = label.strip().lower()
    if normalized == "real":
        return 0
    if normalized == "fake":
        return 1
    raise ValueError(f"label must be 'real' or 'fake', got: {label!r}")


def load_id_map(manifest_path: Path) -> dict[str, str]:
    """Map sample_id (stem) to released test image filename (with extension)."""
    df = pd.read_parquet(manifest_path)
    for column in ("sample_id", "image_path"):
        if column not in df.columns:
            raise ValueError(f"manifest missing required column: {column}")

    id_map: dict[str, str] = {}
    for sample_id, image_path in zip(df["sample_id"], df["image_path"], strict=False):
        stem = str(sample_id).strip()
        filename = Path(str(image_path)).name
        if not filename:
            raise ValueError(f"manifest row sample_id={stem!r} has empty image_path")
        if stem in id_map and id_map[stem] != filename:
            raise ValueError(f"duplicate sample_id with conflicting filenames: {stem!r}")
        id_map[stem] = filename
    return id_map


def resolve_submission_id(raw_id: str, id_map: dict[str, str] | None) -> str:
    if id_map is None:
        return raw_id

    if raw_id in id_map:
        return id_map[raw_id]

    stem = Path(raw_id).stem
    if stem in id_map:
        return id_map[stem]

    if raw_id in id_map.values():
        return raw_id

    raise ValueError(f"unknown sample_id/id not in manifest: {raw_id!r}")


def convert(
    input_path: Path,
    output_path: Path,
    *,
    id_map: dict[str, str] | None = None,
) -> dict[str, int]:
    detection_lines: list[str] = []
    complex_lines: list[str] = []
    simple_lines: list[str] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            raw_id = row.get("sample_id") or row.get("id")
            if not raw_id:
                raise ValueError(f"line {line_number}: missing sample_id/id")

            submission_id = resolve_submission_id(str(raw_id), id_map)

            label = row.get("label")
            if label is None:
                raise ValueError(f"line {line_number}: missing label")
            detection_lines.append(
                json.dumps(
                    {"id": submission_id, "pred_label": label_to_int(str(label))},
                    ensure_ascii=False,
                )
            )

            complex_text = row.get("complex_explanation", "")
            if complex_text:
                complex_lines.append(
                    json.dumps(
                        {"id": submission_id, "complex_explanation": str(complex_text)},
                        ensure_ascii=False,
                    )
                )

            simple_text = row.get("simple_explanation", "")
            if simple_text:
                simple_lines.append(
                    json.dumps(
                        {"id": submission_id, "simple_explanation": str(simple_text)},
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
    repo_root = Path(__file__).resolve().parents[1]
    default_manifest = repo_root / "research/experiments/02_pass1_classifier/manifests/manifest_test.parquet"

    parser = argparse.ArgumentParser(description="Build CodaBench submission zip from val JSONL")
    parser.add_argument("--input", required=True, help="submission.jsonl (sample_id, label, complex, simple)")
    parser.add_argument("--output", required=True, help="output submission.zip")
    parser.add_argument(
        "--manifest",
        default=str(default_manifest),
        help="Parquet manifest with sample_id + image_path (default: manifest_test.parquet).",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Emit bare sample_id as id (val/dev only; not valid for CodaBench test).",
    )
    args = parser.parse_args()

    id_map: dict[str, str] | None = None
    if not args.no_manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            raise SystemExit(f"manifest not found: {manifest_path}")
        id_map = load_id_map(manifest_path)
        print(f"loaded {len(id_map)} ids from {manifest_path}")

    counts = convert(Path(args.input), Path(args.output), id_map=id_map)
    print(f"wrote {args.output}")
    for key, value in counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
