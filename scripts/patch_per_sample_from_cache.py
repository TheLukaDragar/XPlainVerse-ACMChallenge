#!/usr/bin/env python3
"""Patch per_sample_scores.jsonl from _stage_cache after Qwen retry (no BERT rerun)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def round_float(v: float) -> float:
    return round(float(v), 6)


def harmonic(a, b):
    if a is None or b is None:
        return None
    d = float(a) + float(b)
    if d == 0.0:
        return 0.0
    return round_float((2.0 * float(a) * float(b)) / d)


def complex_overall(bert, entity, evidence):
    if any(x is None for x in (bert, entity, evidence)):
        return None
    return round_float(0.3 * bert + 0.4 * entity + 0.3 * evidence)


def simple_overall(bert, sle):
    if bert is None or sle is None:
        return None
    clipped = max(-1.0, min(4.0, float(sle)))
    norm = (clipped + 1.0) / 5.0
    return round_float(0.7 * bert + 0.3 * norm)


def explanation_score(ref_exp, grounding):
    if ref_exp is None or grounding is None:
        return None
    return round_float(0.4 * ref_exp + 0.6 * grounding)


def finalize_from_cache(rec: dict, row: dict) -> dict:
    entity = harmonic(rec.get("_pred_to_gt_entity"), rec.get("_gt_to_pred_entity"))
    evidence = harmonic(rec.get("_pred_to_gt_evidence"), rec.get("_gt_to_pred_evidence"))
    bert = row.get("complex_bert_f1")
    sle = row.get("simple_sle_score")
    simple_bert = row.get("simple_bert_f1")
    simple_sle_norm = row.get("simple_sle_normalized")
    if sle is not None and simple_sle_norm is None:
        clipped = max(-1.0, min(4.0, float(sle)))
        simple_sle_norm = round_float((clipped + 1.0) / 5.0)

    comp_over = complex_overall(bert, entity, evidence)
    simp_over = row.get("simple_overall_score")
    if simp_over is None:
        simp_over = simple_overall(simple_bert, sle if simple_sle_norm is not None else sle)

    ref_exp = None
    if bert is not None and simp_over is not None:
        ref_exp = round_float((bert + simp_over) / 2.0)

    grounding = None
    if entity is not None and evidence is not None:
        grounding = round_float((entity + evidence) / 2.0)

    expl = explanation_score(ref_exp, grounding)

    row.update(
        {
            "complex_entity_f1": entity,
            "complex_evidence_f1": evidence,
            "complex_overall_score": comp_over,
            "simple_sle_normalized": simple_sle_norm,
            "simple_overall_score": simp_over,
            "simple_explanation_score": simp_over,
            "reference_explanation_score": ref_exp,
            "grounding_score": grounding,
            "explanation_score": expl,
        }
    )
    return row


def load_cache(path: Path) -> dict[str, dict]:
    by_id = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            by_id[rec["sample_id"]] = rec
    return by_id


def patch_shard(shard_dir: Path, sample_ids: set[str] | None = None) -> int:
    cache = load_cache(shard_dir / "_stage_cache.jsonl")
    per_path = shard_dir / "per_sample_scores.jsonl"
    rows = []
    updated = 0
    with per_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sid = row["sample_id"]
            if sample_ids is not None and sid not in sample_ids:
                rows.append(row)
                continue
            if sid not in cache:
                rows.append(row)
                continue
            rec = cache[sid]
            before = (row.get("complex_entity_f1"), row.get("complex_evidence_f1"))
            row = finalize_from_cache(rec, row)
            after = (row.get("complex_entity_f1"), row.get("complex_evidence_f1"))
            if before != after:
                updated += 1
            rows.append(row)
    with per_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return updated


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sample-ids", nargs="*", default=None)
    args = p.parse_args()

    ids = set(args.sample_ids) if args.sample_ids else None
    total = 0
    for shard_dir in sorted(args.out_dir.glob("shard_*")):
        if not (shard_dir / "per_sample_scores.jsonl").is_file():
            continue
        n = patch_shard(shard_dir, ids)
        if n:
            print(f"{shard_dir.name}: updated {n} rows")
        total += n
    print(f"total updated: {total}")


if __name__ == "__main__":
    main()
