#!/usr/bin/env python3
"""Build ms-swift JSONL datasets for XPlainVerse VLM + compressor training.

Reads prompt templates from dataset/prompt.txt (sectioned by "=== NAME ==="
markers), iterates the train/val manifests, balances classes, mixes in a
fraction of FFAA-style hypothetical prompts, and writes JSONL files next to
this script.

Outputs (all under --output-dir, defaults to the script's own directory):
    train_vlm.jsonl          balanced fake+real, user prompt + assistant target
    train_vlm_infer.jsonl    same rows, user-only (rarely used)
    train_compressor.jsonl   fake-only train, complex -> simple
    val_vlm.jsonl            val rows with assistant target (eval_steps during SFT)
    val_vlm_infer.jsonl      ALL val rows, user-only (for final inference + submission)
    val_compressor.jsonl     fake-only val (offline compressor eval)

VLM assistant target format:
    {complex_explanation}\\n\\nVerdict: {real|fake}

Compressor user format:
    {COMPRESSOR_USER_PROMPT}\\n{complex_explanation}
Compressor assistant target:
    {simple_explanation}
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    "/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/data/XPlainVerse"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "prompt.txt"

SECTION_RE = re.compile(r"^===\s*(.+?)\s*===\s*$")
REQUIRED_PROMPT_KEYS = (
    "VLM_USER_PROMPT",
    "VLM_USER_PROMPT_HYPOTHETICAL_FAKE",
    "VLM_USER_PROMPT_HYPOTHETICAL_REAL",
    "COMPRESSOR_USER_PROMPT",
)


# --------------------------------------------------------------------------- #
# Prompt loading
# --------------------------------------------------------------------------- #

def parse_prompt_file(path: Path) -> dict[str, str]:
    """Parse `=== NAME ===` ... `=== END ===` sections into a dict."""
    if not path.is_file():
        raise FileNotFoundError(f"prompt file not found: {path}")

    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        match = SECTION_RE.match(line)
        if match:
            name = match.group(1).strip()
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            if name.upper() == "END":
                current = None
            else:
                current = name
            buf = []
            continue
        if current is not None:
            buf.append(line)

    if current is not None:
        sections[current] = "\n".join(buf).strip()

    missing = [key for key in REQUIRED_PROMPT_KEYS if key not in sections or not sections[key]]
    if missing:
        raise ValueError(
            f"prompt file {path} is missing or has empty sections: {missing}"
        )
    return sections


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert XPlainVerse manifest rows into ms-swift JSONL files "
            "(VLM SFT + compressor SFT) using prompts from dataset/prompt.txt."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"XPlainVerse dataset root (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Where to write JSONL files (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE,
                        help=f"Path to prompt.txt (default: {DEFAULT_PROMPT_FILE})")
    parser.add_argument("--splits", nargs="+", choices=("train", "val"),
                        default=("train", "val"),
                        help="Which splits to export (default: train val)")
    parser.add_argument("--train-max-per-class", type=int, default=130000,
                        help="Cap VLM train rows per class; matches real-class size "
                             "(default: 130000). 0 = use all rows, no balancing.")
    parser.add_argument("--val-max-per-class", type=int, default=0,
                        help="Cap VLM val rows per class (default: 0 = use all). "
                             "val_vlm_infer.jsonl always contains ALL val images.")
    parser.add_argument("--compressor-max-train", type=int, default=130000,
                        help="Cap compressor train rows (fake-only); 0 = all (default: 130000).")
    parser.add_argument("--compressor-max-val", type=int, default=0,
                        help="Cap compressor val rows (fake-only); 0 = all (default: 0).")
    parser.add_argument("--hypothetical-ratio", type=float, default=0.33,
                        help="Fraction of VLM train rows that use FFAA-style "
                             "hypothetical prompts presuming the label (default: 0.33). "
                             "Set 0.0 to disable.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for shuffling / hypothetical assignment (default: 42).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-split progress output.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Manifest handling
# --------------------------------------------------------------------------- #

@dataclass
class ManifestRow:
    label: str
    sample_id: str
    image_path: Path
    complex_path: Path
    simple_path: Path | None


@dataclass
class SplitStats:
    split: str
    manifest_rows: int = 0
    skipped_missing_files: int = 0
    skipped_missing_text: int = 0
    rows_by_label: Counter = field(default_factory=Counter)
    vlm_rows: int = 0
    vlm_infer_rows: int = 0
    compressor_rows: int = 0
    hypothetical_used: int = 0


def resolve_path(data_root: Path, rel_path: str) -> Path:
    path = Path(rel_path)
    return path if path.is_absolute() else data_root / path


def load_explanation(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    text = payload.get("explanation")
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip()


def iter_manifest(data_root: Path, split: str, stats: SplitStats) -> list[ManifestRow]:
    """Read manifest.jsonl, drop rows whose image/complex files are missing."""
    manifest_path = data_root / split / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    rows: list[ManifestRow] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stats.manifest_rows += 1
            raw = json.loads(line)

            label = raw.get("label")
            image_rel = raw.get("image_path")
            complex_rel = raw.get("complex_explanation_path")
            simple_rel = raw.get("simple_explanation_path")

            if not label or not image_rel or not complex_rel:
                stats.skipped_missing_files += 1
                continue

            image_path = resolve_path(data_root, image_rel)
            complex_path = resolve_path(data_root, complex_rel)
            if not image_path.is_file() or not complex_path.is_file():
                stats.skipped_missing_files += 1
                continue

            simple_path = resolve_path(data_root, simple_rel) if simple_rel else None
            rows.append(
                ManifestRow(
                    label=label,
                    sample_id=Path(image_rel).stem,
                    image_path=image_path,
                    complex_path=complex_path,
                    simple_path=simple_path,
                )
            )
            stats.rows_by_label[label] += 1

    return rows


# --------------------------------------------------------------------------- #
# Balancing / shuffling
# --------------------------------------------------------------------------- #

def balance_per_class(rows: list[ManifestRow], max_per_class: int,
                      rng: random.Random) -> list[ManifestRow]:
    """Shuffle within each label then cap to max_per_class (0 = no cap)."""
    by_label: dict[str, list[ManifestRow]] = {}
    for row in rows:
        by_label.setdefault(row.label, []).append(row)

    out: list[ManifestRow] = []
    for label, group in by_label.items():
        rng.shuffle(group)
        if max_per_class and max_per_class > 0:
            group = group[:max_per_class]
        out.extend(group)

    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- #
# Message builders
# --------------------------------------------------------------------------- #

def vlm_user_content(prompt_text: str) -> str:
    return f"<image>\n{prompt_text}"


def vlm_assistant_content(label: str, complex_text: str) -> str:
    return f"{complex_text}\n\nVerdict: {label}"


def compressor_user_content(prompt_text: str, complex_text: str) -> str:
    return f"{prompt_text}\n{complex_text}"


def pick_vlm_prompt(prompts: dict[str, str], label: str, use_hypothetical: bool) -> str:
    if not use_hypothetical:
        return prompts["VLM_USER_PROMPT"]
    key = ("VLM_USER_PROMPT_HYPOTHETICAL_FAKE" if label == "fake"
           else "VLM_USER_PROMPT_HYPOTHETICAL_REAL")
    return prompts[key]


# --------------------------------------------------------------------------- #
# Build per split
# --------------------------------------------------------------------------- #

def build_vlm_rows(rows: list[ManifestRow], prompts: dict[str, str],
                   hypothetical_ratio: float, stats: SplitStats,
                   rng: random.Random,
                   ) -> tuple[list[dict], list[dict], list[tuple[ManifestRow, str]]]:
    """Build (train_target_rows, infer_rows, (row, complex_text) cache for compressor)."""
    target_rows: list[dict] = []
    infer_rows: list[dict] = []
    complex_cache: list[tuple[ManifestRow, str]] = []

    hyp_threshold = max(0.0, min(1.0, hypothetical_ratio))

    for row in rows:
        complex_text = load_explanation(row.complex_path)
        if complex_text is None:
            stats.skipped_missing_text += 1
            continue

        use_hyp = hyp_threshold > 0 and rng.random() < hyp_threshold
        if use_hyp:
            stats.hypothetical_used += 1

        record_id = f"{row.label}__{row.sample_id}"
        image_str = str(row.image_path)

        user_target = vlm_user_content(pick_vlm_prompt(prompts, row.label, use_hyp))
        user_infer = vlm_user_content(prompts["VLM_USER_PROMPT"])

        target_rows.append({
            "id": record_id,
            "sample_id": row.sample_id,
            "label": row.label,
            "prompt_kind": "hypothetical" if use_hyp else "primary",
            "messages": [
                {"role": "user", "content": user_target},
                {"role": "assistant",
                 "content": vlm_assistant_content(row.label, complex_text)},
            ],
            "images": [image_str],
        })
        infer_rows.append({
            "id": record_id,
            "sample_id": row.sample_id,
            "label": row.label,
            "messages": [{"role": "user", "content": user_infer}],
            "images": [image_str],
        })
        complex_cache.append((row, complex_text))

    stats.vlm_rows = len(target_rows)
    stats.vlm_infer_rows = len(infer_rows)
    return target_rows, infer_rows, complex_cache


def build_compressor_rows(complex_cache: list[tuple[ManifestRow, str]],
                          prompts: dict[str, str], max_n: int,
                          stats: SplitStats, rng: random.Random) -> list[dict]:
    """Compressor trains on fake samples only (real has no separate simple GT)."""
    candidates: list[dict] = []
    for row, complex_text in complex_cache:
        if row.label != "fake":
            continue
        simple_text = load_explanation(row.simple_path)
        if simple_text is None:
            stats.skipped_missing_text += 1
            continue

        candidates.append({
            "id": f"{row.label}__{row.sample_id}",
            "sample_id": row.sample_id,
            "label": row.label,
            "messages": [
                {"role": "user",
                 "content": compressor_user_content(prompts["COMPRESSOR_USER_PROMPT"],
                                                   complex_text)},
                {"role": "assistant", "content": simple_text},
            ],
        })

    rng.shuffle(candidates)
    if max_n and max_n > 0:
        candidates = candidates[:max_n]
    stats.compressor_rows = len(candidates)
    return candidates


# --------------------------------------------------------------------------- #
# Writing / reporting
# --------------------------------------------------------------------------- #

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_split_report(stats: SplitStats, files: dict[str, Path], quiet: bool) -> None:
    if quiet:
        return
    print(f"\n=== {stats.split} ===")
    print(f"  manifest rows:       {stats.manifest_rows}")
    print(f"  skipped (no files):  {stats.skipped_missing_files}")
    print(f"  skipped (no text):   {stats.skipped_missing_text}")
    if stats.rows_by_label:
        kept = ", ".join(f"{k}={v}" for k, v in sorted(stats.rows_by_label.items()))
        print(f"  kept by label:       {kept}")
    print(f"  vlm rows:            {stats.vlm_rows}  (hypothetical: {stats.hypothetical_used})")
    print(f"  vlm infer rows:      {stats.vlm_infer_rows}")
    print(f"  compressor rows:     {stats.compressor_rows}")
    for label, path in files.items():
        print(f"  -> {label:18s} {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    prompt_file = args.prompt_file.resolve()

    if not data_root.is_dir():
        print(f"error: data root does not exist: {data_root}", file=sys.stderr)
        return 1

    try:
        prompts = parse_prompt_file(prompt_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"data root:    {data_root}")
        print(f"output dir:   {output_dir}")
        print(f"prompt file:  {prompt_file}")
        print(f"hypothetical_ratio={args.hypothetical_ratio}  seed={args.seed}")

    rng_master = random.Random(args.seed)

    for split in args.splits:
        stats = SplitStats(split=split)

        # Per-split deterministic RNG (independent of split order).
        rng = random.Random(rng_master.randint(0, 2**32 - 1))

        all_rows = iter_manifest(data_root, split, stats)

        max_per_class = (args.train_max_per_class if split == "train"
                         else args.val_max_per_class)
        balanced_rows = balance_per_class(all_rows, max_per_class, rng)

        vlm_rows, vlm_infer_rows_balanced, complex_cache = build_vlm_rows(
            balanced_rows, prompts, args.hypothetical_ratio, stats, rng,
        )

        # For val, infer file must contain ALL val rows (not just balanced
        # subset) so we can produce a complete submission.
        if split == "val":
            vlm_infer_rows = []
            for row in all_rows:
                vlm_infer_rows.append({
                    "id": f"{row.label}__{row.sample_id}",
                    "sample_id": row.sample_id,
                    "label": row.label,
                    "messages": [{"role": "user",
                                  "content": vlm_user_content(prompts["VLM_USER_PROMPT"])}],
                    "images": [str(row.image_path)],
                })
            stats.vlm_infer_rows = len(vlm_infer_rows)
        else:
            vlm_infer_rows = vlm_infer_rows_balanced

        max_compressor = (args.compressor_max_train if split == "train"
                          else args.compressor_max_val)
        compressor_rows = build_compressor_rows(
            complex_cache, prompts, max_compressor, stats, rng,
        )

        files = {
            "vlm":             output_dir / f"{split}_vlm.jsonl",
            "vlm_infer":       output_dir / f"{split}_vlm_infer.jsonl",
            "compressor":      output_dir / f"{split}_compressor.jsonl",
        }
        write_jsonl(files["vlm"], vlm_rows)
        write_jsonl(files["vlm_infer"], vlm_infer_rows)
        write_jsonl(files["compressor"], compressor_rows)
        print_split_report(stats, files, args.quiet)

    if not args.quiet:
        print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
