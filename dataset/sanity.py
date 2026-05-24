#!/usr/bin/env python3
"""Verify XPlainVerse manifest rows reference files that exist on disk."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_DATA_ROOT = Path(
    "/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/data/XPlainVerse"
)

PATH_FIELDS = ("image_path", "complex_explanation_path", "simple_explanation_path")


@dataclass
class SplitReport:
    split: str
    max_examples: int = 20
    rows: int = 0
    missing_rows: int = 0
    missing_files: Counter = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)

    def add_missing(self, field_name: str, rel_path: str, row_index: int) -> None:
        self.missing_files[field_name] += 1
        if len(self.examples) < self.max_examples:
            self.examples.append(f"  row {row_index}: missing {field_name}: {rel_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that every path listed in train/val manifest.jsonl exists on disk."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"XPlainVerse dataset root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val"),
        default=("train", "val"),
        help="Which splits to validate (default: train val)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Max missing-file examples to print per split (default: 20)",
    )
    return parser.parse_args()


def resolve_path(data_root: Path, rel_path: str) -> Path:
    path = Path(rel_path)
    if path.is_absolute():
        return path
    return data_root / path


def check_split(data_root: Path, split: str, max_examples: int) -> SplitReport:
    manifest_path = data_root / split / "manifest.jsonl"
    report = SplitReport(split=split, max_examples=max_examples)

    if not manifest_path.is_file():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        report.examples.append(f"  manifest missing: {manifest_path}")
        report.missing_files["manifest"] += 1
        return report

    missing_row_indexes: set[int] = set()

    with manifest_path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            report.rows += 1
            row = json.loads(line)
            row_had_missing = False

            for field_name in PATH_FIELDS:
                rel_path = row.get(field_name)
                if not rel_path:
                    report.add_missing(field_name, "<empty>", row_index)
                    row_had_missing = True
                    continue

                if not resolve_path(data_root, rel_path).is_file():
                    report.add_missing(field_name, rel_path, row_index)
                    row_had_missing = True

            if row_had_missing:
                missing_row_indexes.add(row_index)

    report.missing_rows = len(missing_row_indexes)
    return report


def print_report(report: SplitReport) -> None:
    print(f"\n=== {report.split} ===")
    print(f"manifest rows: {report.rows}")
    print(f"rows with missing files: {report.missing_rows}")

    if report.missing_files:
        print("missing by field:")
        for field_name, count in sorted(report.missing_files.items()):
            print(f"  {field_name}: {count}")

    if report.examples:
        print("examples:")
        for example in report.examples:
            print(example)
    else:
        print("all referenced files present")


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()

    if not data_root.is_dir():
        print(f"error: data root does not exist: {data_root}", file=sys.stderr)
        return 1

    print(f"data root: {data_root}")

    reports = [
        check_split(data_root, split, args.max_examples) for split in args.splits
    ]
    any_missing = False

    for report in reports:
        print_report(report)
        if report.missing_files:
            any_missing = True

    if any_missing:
        print("\nFAIL: one or more manifest paths are missing on disk.", file=sys.stderr)
        return 1

    print("\nOK: all manifest paths exist for requested splits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
