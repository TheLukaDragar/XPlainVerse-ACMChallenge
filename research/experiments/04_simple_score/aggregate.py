#!/usr/bin/env python3
"""Aggregate simple-eval reports into per-label simple_overall scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def norm_sle(x):
    return (clip(float(x), -1.0, 4.0) + 1.0) / 5.0


def simple_overall(bert, sle):
    return 0.7 * float(bert) + 0.3 * norm_sle(sle)


def summarize(report_path: Path):
    rep = json.loads(report_path.read_text())
    per = [s for s in rep["per_sample"] if s.get("status") == "scored"]
    by = {"fake": [], "real": []}
    for s in per:
        lab = str(s.get("label", "")).lower()
        if lab not in by:
            continue
        by[lab].append((s["bertscore_f1"], s["simplicity_score"]))

    out = {}
    for lab, vals in by.items():
        if not vals:
            continue
        n = len(vals)
        bert = sum(v[0] for v in vals) / n
        sle = sum(v[1] for v in vals) / n
        sle_n = sum(norm_sle(v[1]) for v in vals) / n
        ov = sum(simple_overall(b, s) for b, s in vals) / n
        out[lab] = {"n": n, "bert_f1": round(bert, 4), "sle_raw": round(sle, 4),
                    "sle_norm": round(sle_n, 4), "simple_overall": round(ov, 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=Path)
    # Real val class balance: 50k real / 60k fake.
    ap.add_argument("--w-fake", type=float, default=60000 / 110000)
    ap.add_argument("--w-real", type=float, default=50000 / 110000)
    args = ap.parse_args()

    print(f"{'policy':<18} {'class':<6} {'n':>5} {'BERT':>7} {'SLEraw':>8} {'SLEnorm':>8} {'simple':>8}")
    print("-" * 70)
    for rp in args.reports:
        name = rp.stem.replace("simple_eval_", "").replace("report_", "")
        res = summarize(rp)
        weighted = 0.0
        wsum = 0.0
        for lab in ("fake", "real"):
            if lab not in res:
                continue
            d = res[lab]
            print(f"{name:<18} {lab:<6} {d['n']:>5} {d['bert_f1']:>7} {d['sle_raw']:>8} {d['sle_norm']:>8} {d['simple_overall']:>8}")
            w = args.w_fake if lab == "fake" else args.w_real
            weighted += w * d["simple_overall"]
            wsum += w
        if wsum:
            print(f"{name:<18} {'WTD':<6} {'':>5} {'':>7} {'':>8} {'':>8} {round(weighted / wsum, 4):>8}")
        print()


if __name__ == "__main__":
    main()
