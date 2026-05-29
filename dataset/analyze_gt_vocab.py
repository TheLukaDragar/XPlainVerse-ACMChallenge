#!/usr/bin/env python3
"""Mine GT vocabulary and structural patterns from train_vlm.jsonl.

Goal: surface the words/phrases the metric is already rewarding so we can
write prompts that mirror them. Reads only the assistant content (already
loaded into train_vlm.jsonl during the v1 build).

Usage:
  python3 dataset/analyze_gt_vocab.py --jsonl dataset/train_vlm.jsonl \
      --per-class 1000

Prints:
  * Common opening phrases (first 6 words) per class
  * Common transition phrases (word patterns following sentence break)
  * Top artifact / authenticity vocab (adjectives + verbs from fixed seeds)
  * Sentence/word stats
  * Specific high-IDF distinctive words (TF-IDF style: fake vs real)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


STOPWORDS = set("""
the a an and or but if so as is are was were be been being have has had do
does did doing of in on at to for from with by about into onto over under
through during this that these those it its their there here where when
which while because while not no nor very also too quite rather just only
even still already yet much many more most less few some any both each
all every other another some such same different similar like unlike very
its his her our your my their them us we they she he you i one two three
""".split())

ARTIFACT_SEEDS = set("""
distorted warped smudged merged blurred garbled smeared melted fused
unnatural inconsistent illogical impossible synthetic artificial fake
generated AI-generated computer-generated rendered glitched pixelated
oversharp oversmoothed plasticky waxy artificial soft fuzzy halo
floating misaligned overlapping deformed elongated truncated repeated
duplicated mirrored asymmetric asymmetrical incoherent illegible
nonsensical jumbled scrambled broken malformed warped distorted
overexposed underexposed mismatched contradictory implausible
hallucinated rendered cartoony stylized over-saturated
""".lower().split())

AUTHENTIC_SEEDS = set("""
natural realistic plausible sharp crisp detailed authentic genuine real
photographic candid spontaneous unforced organic believable convincing
coherent consistent symmetrical aligned focused defined textured
weathered worn used soft hazy bright sunlit shaded illuminated
photographed documentary candid posed casual
""".lower().split())

OPENER_NGRAM_LEN = 6
NGRAM_OPENERS = 20
TOP_TRANSITIONS = 25
TOP_DISTINCTIVE = 30


def load_complex(jsonl: Path, per_class: int) -> dict[str, list[str]]:
    """Return {'fake': [...], 'real': [...]} of assistant complex texts."""
    by_label: dict[str, list[str]] = {"fake": [], "real": []}
    with jsonl.open(encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("label")
            if label not in by_label:
                continue
            if len(by_label[label]) >= per_class:
                if all(len(v) >= per_class for v in by_label.values()):
                    break
                continue
            # Assistant content = "{complex_explanation}\n\nVerdict: {label}".
            messages = row.get("messages") or []
            asst = None
            for m in messages:
                if m.get("role") == "assistant":
                    asst = m.get("content")
                    break
            if not isinstance(asst, str):
                continue
            text = asst.split("\nVerdict:", 1)[0].strip()
            if text:
                by_label[label].append(text)
    return by_label


def normalize_words(text: str) -> list[str]:
    return [w.lower().strip(",.!?;:'\"()[]{}—–-") for w in text.split()]


def words(text: str) -> list[str]:
    return [w for w in normalize_words(text) if w]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def first_ngram(text: str, n: int) -> str:
    ws = words(text)[:n]
    return " ".join(ws)


def transitions(text: str) -> list[str]:
    """Word patterns at the START of sentences 2..N (the transitions)."""
    out: list[str] = []
    for s in sentences(text)[1:]:
        leading = words(s)[:4]
        if leading:
            out.append(" ".join(leading[:3]))
    return out


def count_seeds(text: str, seeds: set[str]) -> int:
    return sum(1 for w in words(text) if w in seeds)


def top_distinctive_words(texts_a: list[str], texts_b: list[str],
                          min_count: int = 20) -> list[tuple[str, float, int, int]]:
    """Words that appear *much* more often in A than in B.

    Score = (a_count / a_total) / (b_count / b_total + epsilon), with
    additive smoothing. Excludes stopwords.
    """
    ca = Counter()
    cb = Counter()
    for t in texts_a:
        for w in words(t):
            if w in STOPWORDS or not w.isalpha() or len(w) < 4:
                continue
            ca[w] += 1
    for t in texts_b:
        for w in words(t):
            if w in STOPWORDS or not w.isalpha() or len(w) < 4:
                continue
            cb[w] += 1
    ta = max(sum(ca.values()), 1)
    tb = max(sum(cb.values()), 1)
    eps = 1e-6
    scored: list[tuple[str, float, int, int]] = []
    for w, c in ca.items():
        if c < min_count:
            continue
        rate_a = c / ta
        rate_b = (cb.get(w, 0) + 0.5) / tb
        score = rate_a / (rate_b + eps)
        scored.append((w, score, c, cb.get(w, 0)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def report(label: str, texts: list[str]) -> None:
    n = len(texts)
    if n == 0:
        return
    n_sentences = [len(sentences(t)) for t in texts]
    n_words = [len(words(t)) for t in texts]
    n_artifact = [count_seeds(t, ARTIFACT_SEEDS) for t in texts]
    n_authentic = [count_seeds(t, AUTHENTIC_SEEDS) for t in texts]

    def median(xs):
        s = sorted(xs)
        return s[len(s) // 2] if s else 0

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\n=== {label.upper()} ({n} samples) ===")
    print(f"  sentences:  median={median(n_sentences)}  mean={mean(n_sentences):.1f}")
    print(f"  words:      median={median(n_words)}      mean={mean(n_words):.1f}")
    print(f"  artifact-seed hits/sample:   mean={mean(n_artifact):.2f}")
    print(f"  authenticity-seed hits/sample: mean={mean(n_authentic):.2f}")

    print(f"\n  Top {NGRAM_OPENERS} opening {OPENER_NGRAM_LEN}-grams:")
    openers = Counter(first_ngram(t, OPENER_NGRAM_LEN) for t in texts)
    for ng, c in openers.most_common(NGRAM_OPENERS):
        print(f"    {c:4d}  {ng}")

    print(f"\n  Top {TOP_TRANSITIONS} transition phrases (first 3 words of sentences 2..N):")
    trans = Counter()
    for t in texts:
        trans.update(transitions(t))
    for ph, c in trans.most_common(TOP_TRANSITIONS):
        print(f"    {c:5d}  {ph}")


def report_distinctive(by_label: dict[str, list[str]]) -> None:
    fake_distinctive = top_distinctive_words(by_label["fake"], by_label["real"])
    real_distinctive = top_distinctive_words(by_label["real"], by_label["fake"])
    print(f"\n=== TOP {TOP_DISTINCTIVE} words distinctive of FAKE GTs ===")
    print("  word               score        fake#   real#")
    for w, sc, ca, cb in fake_distinctive[:TOP_DISTINCTIVE]:
        print(f"  {w:18s} {sc:10.1f}   {ca:6d}  {cb:6d}")
    print(f"\n=== TOP {TOP_DISTINCTIVE} words distinctive of REAL GTs ===")
    print("  word               score        real#   fake#")
    for w, sc, ca, cb in real_distinctive[:TOP_DISTINCTIVE]:
        print(f"  {w:18s} {sc:10.1f}   {ca:6d}  {cb:6d}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path,
                        default=Path("/shared/workspace/lrv/luka/"
                                     "XPlainVerse-ACMChallenge/code/"
                                     "XPlainVerse-ACMChallenge/dataset/"
                                     "train_vlm.jsonl"))
    parser.add_argument("--per-class", type=int, default=1000)
    args = parser.parse_args()

    if not args.jsonl.is_file():
        print(f"error: {args.jsonl} missing")
        return 1
    by_label = load_complex(args.jsonl, args.per_class)
    for label in ("fake", "real"):
        report(label, by_label[label])
    report_distinctive(by_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
