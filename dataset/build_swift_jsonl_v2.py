#!/usr/bin/env python3
"""Build v2 ms-swift JSONL training data for XPlainVerse VLM Pass-2 SFT.

Differences vs build_swift_jsonl.py:
  * Fake GTs go through a quality filter (sentences >= 3 AND words >= 80
    AND >= 2 connectives). Justified in research/06_training_data_v2.md.
  * Default --hypothetical-ratio bumped from 0.33 to 0.50 (FFAA 2025 recipe).
  * Class ratio enforced: --fake-real-ratio (default 4.0). Real rows are
    unfiltered; fakes are downsampled (or upsampled by repetition if the
    ratio requires it, though we expect ample fakes survive the filter).
  * Always prints filter survival statistics before writing JSONL.
  * --measure-only mode runs the filter and prints stats without writing.

Outputs (under --output-dir, defaults to dataset/):
  train_vlm_v2.jsonl          balanced + filtered + 50%-hypothetical training
  train_vlm_infer_v2.jsonl    same rows, user-only
  val_vlm_v2.jsonl            val rows (always primary prompt, no filter,
                              for eval_steps during SFT)
  val_vlm_infer_v2.jsonl      ALL val rows, user-only (for submission)

Everything else (compressor data, prompt file format, assistant template)
is identical to v1. The compressor data files from v1 still apply.
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
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "prompt_v2.txt"

SECTION_RE = re.compile(r"^===\s*(.+?)\s*===\s*$")
REQUIRED_PROMPT_KEYS = (
    "VLM_USER_PROMPT",
    "VLM_USER_PROMPT_HYPOTHETICAL_FAKE",
    "VLM_USER_PROMPT_HYPOTHETICAL_REAL",
)

# Connectives we count toward the "structural multi-region" filter signal.
# Matches the wording our own prompt asks the model to use.
CONNECTIVE_RE = re.compile(
    r"\b(additionally|furthermore|moreover|also|notably|specifically|"
    r"in addition|besides)\b",
    re.IGNORECASE,
)
SENTENCE_RE = re.compile(r"[.!?]+(?:\s+|$)")


# --------------------------------------------------------------------------- #
# Prompt loading (unchanged from v1)
# --------------------------------------------------------------------------- #

def parse_prompt_file(path: Path) -> dict[str, str]:
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
    missing = [k for k in REQUIRED_PROMPT_KEYS if k not in sections or not sections[k]]
    if missing:
        raise ValueError(f"prompt file {path} missing sections: {missing}")
    return sections


# --------------------------------------------------------------------------- #
# Filter (new in v2)
# --------------------------------------------------------------------------- #

@dataclass
class FilterStats:
    seen: int = 0
    kept: int = 0
    dropped_sentences: int = 0
    dropped_words: int = 0
    dropped_connectives: int = 0
    # All dropped reasons recorded independently (a sample may fail multiple).


def count_sentences(text: str) -> int:
    return sum(1 for _ in SENTENCE_RE.finditer(text))


def count_connectives(text: str) -> int:
    return sum(1 for _ in CONNECTIVE_RE.finditer(text))


def fake_gt_passes(text: str, *, min_sentences: int, min_words: int,
                   min_connectives: int, stats: FilterStats) -> bool:
    """Return True iff the fake GT explanation meets quality thresholds.

    Side effect: increments stats counters.
    """
    stats.seen += 1
    n_sentences = count_sentences(text)
    n_words = len(text.split())
    n_conn = count_connectives(text)

    ok_sent = n_sentences >= min_sentences
    ok_word = n_words >= min_words
    ok_conn = n_conn >= min_connectives

    if not ok_sent:
        stats.dropped_sentences += 1
    if not ok_word:
        stats.dropped_words += 1
    if not ok_conn:
        stats.dropped_connectives += 1

    if ok_sent and ok_word and ok_conn:
        stats.kept += 1
        return True
    return False


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v2 ms-swift training JSONL with fake-GT filter."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"),
                        default=("train", "val"))

    parser.add_argument("--hypothetical-ratio", type=float, default=0.50,
                        help="Fraction of TRAIN rows that use hypothetical "
                             "(label-presuming) prompts. Default 0.50 matches "
                             "FFAA 2025 MMTD-Set. v1 used 0.33.")
    parser.add_argument("--fake-real-ratio", type=float, default=4.0,
                        help="Target ratio fake:real for TRAIN rows. v2 default "
                             "is 4.0 because the metric pain is on fakes. "
                             "Reals are unfiltered. Set 0 to keep raw counts.")

    parser.add_argument("--filter-min-sentences", type=int, default=3)
    parser.add_argument("--filter-min-words", type=int, default=80)
    parser.add_argument("--filter-min-connectives", type=int, default=0,
                        help="0 (default) disables the connective check. The "
                             "first 500-row dry-run showed it dropped 63%% of "
                             "fakes including obviously multi-region GTs that "
                             "just use varied transitions instead of the "
                             "'Additionally/Furthermore' words.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable fake-GT filter (debug only).")

    parser.add_argument("--val-suffix", default="_v2",
                        help="Filename suffix for v2 outputs (default _v2). "
                             "Set '' to overwrite v1 files (NOT recommended).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measure-only", action="store_true",
                        help="Compute filter stats on TRAIN fake rows only, "
                             "print, then exit without writing JSONL.")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Cap manifest rows read per split (debug). 0 = all.")
    parser.add_argument("--sample-examples", type=int, default=0,
                        help="Print N kept and N dropped fake GT examples to "
                             "sanity-check the filter (debug).")
    parser.add_argument("--quiet", action="store_true")
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
    complex_text: str | None = None


def resolve_path(data_root: Path, rel_path: str) -> Path:
    p = Path(rel_path)
    return p if p.is_absolute() else data_root / p


def load_explanation(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    with path.open(encoding="utf-8") as h:
        payload = json.load(h)
    text = payload.get("explanation")
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip()


def iter_manifest_with_text(data_root: Path, split: str,
                            max_rows: int = 0,
                            quiet: bool = False) -> list[ManifestRow]:
    """Read manifest, drop missing rows, and load complex_explanation text up-front.

    Loading the text up-front lets us filter before deciding whether a row
    is worth keeping at all, which means we don't waste filter compute on
    rows whose explanation file is corrupt or missing.

    If max_rows > 0, stops after reading that many MANIFEST lines (not kept
    rows) — useful for debug runs on the first N samples.
    """
    manifest_path = data_root / split / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    rows: list[ManifestRow] = []
    seen = 0
    with manifest_path.open(encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            seen += 1
            if max_rows and seen > max_rows:
                break
            if not quiet and seen % 50000 == 0:
                print(f"  ... read {seen} manifest rows ({len(rows)} kept so far)",
                      file=sys.stderr, flush=True)
            raw = json.loads(line)
            label = raw.get("label")
            image_rel = raw.get("image_path")
            complex_rel = raw.get("complex_explanation_path")
            simple_rel = raw.get("simple_explanation_path")
            if not label or not image_rel or not complex_rel:
                continue
            image_path = resolve_path(data_root, image_rel)
            complex_path = resolve_path(data_root, complex_rel)
            if not image_path.is_file() or not complex_path.is_file():
                continue
            text = load_explanation(complex_path)
            if text is None:
                continue
            simple_path = resolve_path(data_root, simple_rel) if simple_rel else None
            rows.append(ManifestRow(
                label=label,
                sample_id=Path(image_rel).stem,
                image_path=image_path,
                complex_path=complex_path,
                simple_path=simple_path,
                complex_text=text,
            ))
    return rows


# --------------------------------------------------------------------------- #
# Sampling / balancing
# --------------------------------------------------------------------------- #

def apply_fake_filter(rows: list[ManifestRow], *, min_sentences: int,
                      min_words: int, min_connectives: int,
                      ) -> tuple[list[ManifestRow], FilterStats,
                                 list[ManifestRow], list[ManifestRow]]:
    """Apply filter to fake rows. Reals pass through unchanged.

    Returns (kept_rows, stats, sample_kept_fakes, sample_dropped_fakes)
    where the sample lists are the first few of each (capped at 20) for
    qualitative inspection.
    """
    stats = FilterStats()
    kept: list[ManifestRow] = []
    sample_kept: list[ManifestRow] = []
    sample_dropped: list[ManifestRow] = []
    SAMPLE_CAP = 20
    for row in rows:
        if row.label != "fake":
            kept.append(row)
            continue
        if fake_gt_passes(row.complex_text or "",
                          min_sentences=min_sentences,
                          min_words=min_words,
                          min_connectives=min_connectives,
                          stats=stats):
            kept.append(row)
            if len(sample_kept) < SAMPLE_CAP:
                sample_kept.append(row)
        else:
            if len(sample_dropped) < SAMPLE_CAP:
                sample_dropped.append(row)
    return kept, stats, sample_kept, sample_dropped


def print_examples(label: str, rows: list[ManifestRow], n: int) -> None:
    print(f"\n--- {label} (showing {min(n, len(rows))} of {len(rows)}) ---")
    for row in rows[:n]:
        text = (row.complex_text or "").strip()
        n_sent = count_sentences(text)
        n_words = len(text.split())
        n_conn = count_connectives(text)
        snippet = text if len(text) <= 240 else text[:240] + "..."
        print(f"  [{row.sample_id}] sentences={n_sent} words={n_words} "
              f"connectives={n_conn}")
        print(f"    {snippet}")


def enforce_ratio(rows: list[ManifestRow], fake_per_real: float,
                  rng: random.Random) -> list[ManifestRow]:
    """Down/up-sample so fake_count = ratio * real_count.

    We never duplicate fakes (they're plentiful). If reals exceed
    fakes/ratio we downsample reals. Otherwise we downsample fakes.
    """
    fakes = [r for r in rows if r.label == "fake"]
    reals = [r for r in rows if r.label == "real"]
    if fake_per_real <= 0:
        return rows

    target_fakes = int(len(reals) * fake_per_real)
    target_reals = len(reals)
    if target_fakes > len(fakes):
        # Not enough fakes — downsample reals instead so the ratio holds.
        target_reals = int(len(fakes) / fake_per_real)
        target_fakes = len(fakes)

    rng.shuffle(fakes)
    rng.shuffle(reals)
    return fakes[:target_fakes] + reals[:target_reals]


def vlm_user_content(prompt_text: str) -> str:
    return f"<image>\n{prompt_text}"


def vlm_assistant_content(label: str, complex_text: str) -> str:
    return f"{complex_text}\n\nVerdict: {label}"


def pick_vlm_prompt(prompts: dict[str, str], label: str, use_hyp: bool) -> str:
    if not use_hyp:
        return prompts["VLM_USER_PROMPT"]
    key = ("VLM_USER_PROMPT_HYPOTHETICAL_FAKE" if label == "fake"
           else "VLM_USER_PROMPT_HYPOTHETICAL_REAL")
    return prompts[key]


@dataclass
class SplitStats:
    split: str
    raw_rows: int = 0
    raw_by_label: Counter = field(default_factory=Counter)
    filter: FilterStats | None = None
    after_filter_by_label: Counter = field(default_factory=Counter)
    after_ratio_by_label: Counter = field(default_factory=Counter)
    hyp_used: int = 0
    vlm_target_rows: int = 0
    vlm_infer_rows: int = 0


def build_vlm_rows(rows: list[ManifestRow], prompts: dict[str, str],
                   hyp_ratio: float, stats: SplitStats,
                   rng: random.Random,
                   ) -> tuple[list[dict], list[dict]]:
    target_rows: list[dict] = []
    infer_rows: list[dict] = []
    hyp_threshold = max(0.0, min(1.0, hyp_ratio))
    for row in rows:
        complex_text = row.complex_text
        assert complex_text is not None, "complex_text must be loaded"
        use_hyp = hyp_threshold > 0 and rng.random() < hyp_threshold
        if use_hyp:
            stats.hyp_used += 1
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
    stats.vlm_target_rows = len(target_rows)
    stats.vlm_infer_rows = len(infer_rows)
    return target_rows, infer_rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def fmt_pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def print_split_report(stats: SplitStats, files: dict[str, Path] | None,
                       quiet: bool) -> None:
    if quiet:
        return
    print(f"\n=== {stats.split} ===")
    print(f"  raw manifest rows:       {stats.raw_rows}")
    if stats.raw_by_label:
        kept = ", ".join(f"{k}={v}" for k, v in sorted(stats.raw_by_label.items()))
        print(f"  raw by label:            {kept}")
    if stats.filter is not None:
        f = stats.filter
        print(f"  fake-GT filter:          seen={f.seen}  kept={f.kept} ({fmt_pct(f.kept, f.seen)})")
        print(f"    dropped_sentences:     {f.dropped_sentences} ({fmt_pct(f.dropped_sentences, f.seen)})")
        print(f"    dropped_words:         {f.dropped_words} ({fmt_pct(f.dropped_words, f.seen)})")
        print(f"    dropped_connectives:   {f.dropped_connectives} ({fmt_pct(f.dropped_connectives, f.seen)})")
        kept = ", ".join(f"{k}={v}" for k, v in sorted(stats.after_filter_by_label.items()))
        print(f"  after filter by label:   {kept}")
    if stats.after_ratio_by_label:
        kept = ", ".join(f"{k}={v}" for k, v in sorted(stats.after_ratio_by_label.items()))
        print(f"  after ratio by label:    {kept}")
    print(f"  vlm target rows:         {stats.vlm_target_rows}  (hypothetical: {stats.hyp_used})")
    print(f"  vlm infer rows:          {stats.vlm_infer_rows}")
    if files:
        for k, path in files.items():
            print(f"  -> {k:18s} {path}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    prompt_file = args.prompt_file.resolve()

    if not data_root.is_dir():
        print(f"error: data root missing: {data_root}", file=sys.stderr)
        return 1
    try:
        prompts = parse_prompt_file(prompt_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"data root:           {data_root}")
        print(f"output dir:          {output_dir}")
        print(f"prompt file:         {prompt_file}")
        print(f"hypothetical_ratio:  {args.hypothetical_ratio}")
        print(f"fake_real_ratio:     {args.fake_real_ratio}")
        print(f"filter (fake-only):  sentences>={args.filter_min_sentences} "
              f"words>={args.filter_min_words} "
              f"connectives>={args.filter_min_connectives}")
        if args.no_filter:
            print("  (disabled via --no-filter)")
        if args.measure_only:
            print("MODE: measure-only (no files written)")
        print()

    rng_master = random.Random(args.seed)

    for split in args.splits:
        rng = random.Random(rng_master.randint(0, 2**32 - 1))
        stats = SplitStats(split=split)

        all_rows = iter_manifest_with_text(data_root, split,
                                           max_rows=args.max_rows,
                                           quiet=args.quiet)
        stats.raw_rows = len(all_rows)
        for r in all_rows:
            stats.raw_by_label[r.label] += 1

        if split == "train":
            sample_kept: list[ManifestRow] = []
            sample_dropped: list[ManifestRow] = []
            if args.no_filter:
                filtered = all_rows
                stats.filter = None
            else:
                filtered, fstats, sample_kept, sample_dropped = apply_fake_filter(
                    all_rows,
                    min_sentences=args.filter_min_sentences,
                    min_words=args.filter_min_words,
                    min_connectives=args.filter_min_connectives,
                )
                stats.filter = fstats
            for r in filtered:
                stats.after_filter_by_label[r.label] += 1

            if args.sample_examples > 0:
                print_examples("KEPT fake GTs", sample_kept, args.sample_examples)
                print_examples("DROPPED fake GTs", sample_dropped, args.sample_examples)

            if args.measure_only:
                print_split_report(stats, None, args.quiet)
                continue

            balanced = enforce_ratio(filtered, args.fake_real_ratio, rng)
            for r in balanced:
                stats.after_ratio_by_label[r.label] += 1
            rng.shuffle(balanced)
            vlm_rows, vlm_infer_rows = build_vlm_rows(
                balanced, prompts, args.hypothetical_ratio, stats, rng,
            )

            files = {
                "train_vlm":       output_dir / f"train_vlm{args.val_suffix}.jsonl",
                "train_vlm_infer": output_dir / f"train_vlm_infer{args.val_suffix}.jsonl",
            }
            write_jsonl(files["train_vlm"], vlm_rows)
            write_jsonl(files["train_vlm_infer"], vlm_infer_rows)
            print_split_report(stats, files, args.quiet)

        else:  # val
            if args.measure_only:
                continue
            # val gets no filter, no ratio enforcement, no hypothetical
            stats.after_filter_by_label = Counter(stats.raw_by_label)
            stats.after_ratio_by_label = Counter(stats.raw_by_label)
            vlm_rows, _ = build_vlm_rows(all_rows, prompts, 0.0, stats, rng)
            vlm_infer_rows = []
            for r in all_rows:
                vlm_infer_rows.append({
                    "id": f"{r.label}__{r.sample_id}",
                    "sample_id": r.sample_id,
                    "label": r.label,
                    "messages": [{"role": "user",
                                  "content": vlm_user_content(prompts["VLM_USER_PROMPT"])}],
                    "images": [str(r.image_path)],
                })
            stats.vlm_infer_rows = len(vlm_infer_rows)
            files = {
                "val_vlm":       output_dir / f"val_vlm{args.val_suffix}.jsonl",
                "val_vlm_infer": output_dir / f"val_vlm_infer{args.val_suffix}.jsonl",
            }
            write_jsonl(files["val_vlm"], vlm_rows)
            write_jsonl(files["val_vlm_infer"], vlm_infer_rows)
            print_split_report(stats, files, args.quiet)

    if not args.quiet:
        print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
