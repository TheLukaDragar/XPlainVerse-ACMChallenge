#!/usr/bin/env python3
"""Merge per-shard evaluate_val outputs into full paper metrics (200k)."""
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


def round_float(value: float) -> float:
    return round(float(value), 6)


def zero_filled_mean(values, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    total = sum(float(v) for v in values if v is not None)
    return round_float(total / denominator)


def bool_mean(values, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round_float(sum(1.0 for v in values if v is True) / denominator)


def binary_f1(rows: list[dict], *, positive_label: str) -> float | None:
    true_positive = false_positive = false_negative = 0
    seen = False
    for row in rows:
        gt = row.get("ground_truth_label")
        pred = row.get("predicted_label")
        if gt is None:
            continue
        seen = True
        if pred == positive_label and gt == positive_label:
            true_positive += 1
        elif pred == positive_label and gt != positive_label:
            false_positive += 1
        elif gt == positive_label:
            false_negative += 1
    if not seen:
        return None
    denom = (2 * true_positive) + false_positive + false_negative
    if denom == 0:
        return 0.0
    return round_float((2 * true_positive) / denom)


def compute_detection_metrics(rows: list[dict]) -> dict:
    labeled = [r for r in rows if r.get("ground_truth_label") is not None]
    if not labeled:
        return {
            "detection_macro_f1": None,
            "detection_fake_f1": None,
            "detection_real_f1": None,
            "detection_accuracy": None,
        }
    fake_f1 = binary_f1(labeled, positive_label="fake")
    real_f1 = binary_f1(labeled, positive_label="real")
    macro = None
    if fake_f1 is not None and real_f1 is not None:
        macro = round_float((fake_f1 + real_f1) / 2.0)
    return {
        "detection_macro_f1": macro,
        "detection_fake_f1": fake_f1,
        "detection_real_f1": real_f1,
        "detection_accuracy": bool_mean(
            (r.get("label_correct") for r in labeled), len(labeled)
        ),
    }


def build_final_scores(rows: list[dict]) -> dict:
    n = len(rows)
    complex_bert = zero_filled_mean((r.get("complex_bert_f1") for r in rows), n)
    simple_bert = zero_filled_mean((r.get("simple_bert_f1") for r in rows), n)
    simple_sle = zero_filled_mean((r.get("simple_sle_score") for r in rows), n)
    simple_sle_norm = zero_filled_mean((r.get("simple_sle_normalized") for r in rows), n)
    complex_entity = zero_filled_mean((r.get("complex_entity_f1") for r in rows), n)
    complex_evidence = zero_filled_mean(
        (r.get("complex_evidence_f1") or r.get("complex_facts_f1") for r in rows), n
    )
    complex_overall = zero_filled_mean((r.get("complex_overall_score") for r in rows), n)

    simple_overall = None
    if simple_bert is not None and simple_sle_norm is not None:
        simple_overall = round_float(0.7 * simple_bert + 0.3 * simple_sle_norm)

    reference_explanation = None
    if complex_bert is not None and simple_overall is not None:
        reference_explanation = round_float((complex_bert + simple_overall) / 2.0)

    grounding = None
    if complex_entity is not None and complex_evidence is not None:
        grounding = round_float((complex_entity + complex_evidence) / 2.0)

    explanation = None
    if reference_explanation is not None and grounding is not None:
        explanation = round_float(0.4 * reference_explanation + 0.6 * grounding)

    detection = compute_detection_metrics(rows)
    detection_f1 = detection["detection_macro_f1"]
    overall = None
    if detection_f1 is not None and explanation is not None:
        overall = round_float((detection_f1 + explanation) / 2.0)

    submitted = [
        r
        for r in rows
        if (
            r.get("predicted_label") is not None
            or r.get("complex_bert_f1") is not None
            or r.get("simple_bert_f1") is not None
        )
    ]

    return {
        "metric_version": "acm_mm_2026_paper",
        "samples_merged": n,
        "samples_expected": n,
        "samples_completed": n,
        "submission_rows_with_any_scored_field": len(submitted),
        "detection_macro_f1": detection_f1,
        "detection_f1": detection_f1,
        "detection_fake_f1": detection["detection_fake_f1"],
        "detection_real_f1": detection["detection_real_f1"],
        "detection_accuracy": detection["detection_accuracy"],
        "accuracy": detection["detection_accuracy"],
        "complex_bert_f1": complex_bert,
        "complex_explanation_score": complex_bert,
        "complex_entity_f1": complex_entity,
        "entity_score": complex_entity,
        "complex_facts_f1": complex_evidence,
        "complex_evidence_f1": complex_evidence,
        "evidence_score": complex_evidence,
        "complex_overall_score": complex_overall,
        "simple_bert_f1": simple_bert,
        "simple_sle_score": simple_sle,
        "simple_sle_normalized": simple_sle_norm,
        "simple_overall_score": simple_overall,
        "simple_explanation_score": simple_overall,
        "reference_explanation_score": reference_explanation,
        "grounding_score": grounding,
        "explanation_score": explanation,
        "final_score": overall,
        "overall_score": overall,
        "score_formula": {
            "detection_macro_f1": "(F1_fake + F1_real) / 2",
            "simple_explanation_score": "0.7 * simple_bert_f1 + 0.3 * simple_sle_normalized",
            "reference_explanation_score": "(complex_explanation_score + simple_explanation_score) / 2",
            "grounding_score": "(entity_score + evidence_score) / 2",
            "explanation_score": "0.4 * reference_explanation_score + 0.6 * grounding_score",
            "final_score": "(detection_macro_f1 + explanation_score) / 2",
        },
    }


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

    out = build_final_scores(all_rows)

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
