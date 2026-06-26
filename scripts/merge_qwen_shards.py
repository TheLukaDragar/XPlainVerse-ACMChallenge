#!/usr/bin/env python3
"""Merge per-shard evaluate_val outputs (Qwen entity/facts + BERT/SLE)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 6)


def harmonic_mean(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    denom = float(a) + float(b)
    if denom == 0.0:
        return 0.0
    return round((2.0 * float(a) * float(b)) / denom, 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sample-output", type=Path, default=None)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for shard_dir in args.shard_dirs:
        path = shard_dir / "per_sample_scores.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        all_rows.extend(load_jsonl(path))

    n = len(all_rows)
    complex_bert = mean([r.get("complex_bert_f1") for r in all_rows])
    complex_entity = mean([r.get("complex_entity_f1") for r in all_rows])
    complex_facts = mean([r.get("complex_facts_f1") or r.get("complex_evidence_f1") for r in all_rows])
    simple_bert = mean([r.get("simple_bert_f1") for r in all_rows])
    simple_sle = mean([r.get("simple_sle_score") for r in all_rows])
    simple_sle_norm = mean([r.get("simple_sle_normalized") for r in all_rows])

    complex_overall = None
    if complex_bert is not None and complex_entity is not None and complex_facts is not None:
        complex_overall = round(
            0.3 * complex_bert + 0.4 * complex_entity + 0.3 * complex_facts, 6
        )

    simple_overall = None
    if simple_bert is not None and simple_sle_norm is not None:
        simple_overall = round(0.7 * simple_bert + 0.3 * simple_sle_norm, 6)
    elif simple_bert is not None and simple_sle is not None:
        clipped = max(-1.0, min(4.0, simple_sle))
        simple_overall = round(0.7 * simple_bert + 0.3 * ((clipped + 1.0) / 5.0), 6)

    reference_explanation = None
    if complex_bert is not None and simple_overall is not None:
        reference_explanation = round((complex_bert + simple_overall) / 2, 6)

    grounding = None
    if complex_entity is not None and complex_facts is not None:
        grounding = harmonic_mean(complex_entity, complex_facts)

    explanation = None
    if reference_explanation is not None and grounding is not None:
        explanation = round(0.4 * reference_explanation + 0.6 * grounding, 6)

    out = {
        "samples_merged": n,
        "complex_bert_f1": complex_bert,
        "complex_entity_f1": complex_entity,
        "complex_facts_f1": complex_facts,
        "complex_evidence_f1": complex_facts,
        "complex_overall_score": complex_overall,
        "simple_bert_f1": simple_bert,
        "simple_sle_score": simple_sle,
        "simple_sle_normalized": simple_sle_norm,
        "simple_overall_score": simple_overall,
        "reference_explanation_score": reference_explanation,
        "grounding_score": grounding,
        "explanation_score": explanation,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if args.per_sample_output is not None:
        args.per_sample_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_sample_output.open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
