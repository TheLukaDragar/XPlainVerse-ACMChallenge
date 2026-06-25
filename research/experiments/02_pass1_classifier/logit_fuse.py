#!/usr/bin/env python3
"""Logit-space fusion of multiple Pass-1 detectors + macro-F1 threshold calibration.

NTIRE 2026 winners (LOGER/HEDGE) fuse models in LOGIT space, not probability space:
    p_fused = sigmoid( sum_m w_m * logit(p_m) ),   logit(p)=log(p/(1-p))
which preserves each model's confidence range (a confident minority can flip the vote),
beating probability averaging and majority voting.

Usage (holdout = has label_int; calibrate weights + threshold here):
  python3 logit_fuse.py \
    --holdout name=v3:/path/holdout_v3.parquet name=warm:/path/holdout_warm.parquet \
    --test    name=v3:/path/test_v3.parquet    name=warm:/path/test_warm.parquet \
    --score-col p_fake_mean \
    --out /path/fused_test.parquet

It sweeps a weight simplex on the holdout, picks the (weights, threshold) that maximize
macro F1, then writes fused + recut test predictions. Probability fusion is reported as a
baseline for comparison.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6


def _f1(y: np.ndarray, pred: np.ndarray, pos: int) -> float:
    tp = float(((pred == pos) & (y == pos)).sum())
    fp = float(((pred == pos) & (y != pos)).sum())
    fn = float(((pred != pos) & (y == pos)).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def roc_auc_score(y: np.ndarray, score: np.ndarray) -> float:
    # rank-based AUC (Mann-Whitney U)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    s = score[order]
    ranks_sorted = np.arange(1, len(score) + 1, dtype=float)
    # average ranks for ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks_sorted[i : j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks[order] = ranks_sorted
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = ranks[y == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def parse_named(specs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"bad spec (need name=path): {s}")
        name, path = s.split("=", 1)
        name = name.replace("name=", "").strip()
        out[name] = Path(path)
    return out


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def macro_f1_at_best(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.01, 0.99, 197)
    best_thr, best = 0.5, -1.0
    for thr in grid:
        pred = (score >= thr).astype(int)
        m = 0.5 * (_f1(y, pred, 1) + _f1(y, pred, 0))
        if m > best:
            best, best_thr = m, float(thr)
    return best_thr, best


def load_aligned(specs: dict[str, Path], score_col: str, need_label: bool):
    frames = {}
    label = None
    ref_ids = None
    for name, path in specs.items():
        df = pd.read_parquet(path)
        df = df[["sample_id", score_col] + (["label_int"] if need_label and "label_int" in df.columns else [])]
        df = df.drop_duplicates("sample_id").set_index("sample_id")
        frames[name] = df[score_col]
        if need_label and "label_int" in df.columns and label is None:
            label = df["label_int"]
        ref_ids = df.index if ref_ids is None else ref_ids.intersection(df.index)
    mat = pd.DataFrame({k: v for k, v in frames.items()}).loc[ref_ids]
    lab = label.loc[ref_ids] if label is not None else None
    return mat, lab


def weight_simplex(n: int, step: float = 0.1):
    """All weight vectors on the n-simplex with given step (sum=1)."""
    ticks = int(round(1.0 / step))
    for combo in itertools.product(range(ticks + 1), repeat=n):
        if sum(combo) == ticks:
            yield np.array(combo, dtype=float) / ticks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", nargs="+", required=True, help="name=path parquet(s) with label_int")
    ap.add_argument("--test", nargs="+", help="name=path parquet(s) to fuse + recut (optional)")
    ap.add_argument("--score-col", default="p_fake_mean")
    ap.add_argument("--out", type=Path, help="output fused test parquet")
    ap.add_argument("--step", type=float, default=0.1, help="weight simplex granularity")
    args = ap.parse_args()

    hold_specs = parse_named(args.holdout)
    names = list(hold_specs.keys())
    hmat, y = load_aligned(hold_specs, args.score_col, need_label=True)
    if y is None:
        raise SystemExit("no label_int found in holdout parquets")
    y = y.to_numpy().astype(int)
    hlogit = to_logit(hmat.to_numpy())  # (N, M)
    print(f"holdout: {hmat.shape[0]} rows, {len(names)} models: {names}")

    # Single-model baselines.
    print("\n=== single-model holdout macro-F1 ===")
    for j, nm in enumerate(names):
        thr, mf1 = macro_f1_at_best(y, hmat.to_numpy()[:, j])
        auc = roc_auc_score(y, hmat.to_numpy()[:, j])
        print(f"  {nm:12s} macroF1={mf1:.4f} thr={thr:.3f} auc={auc:.4f}")

    # Probability-average baseline.
    pavg = hmat.to_numpy().mean(axis=1)
    thr_p, mf1_p = macro_f1_at_best(y, pavg)
    print(f"\n  prob-avg     macroF1={mf1_p:.4f} thr={thr_p:.3f}")

    # Logit-space weight sweep.
    best = {"mf1": -1.0}
    for w in weight_simplex(len(names), args.step):
        fused = 1.0 / (1.0 + np.exp(-(hlogit * w).sum(axis=1)))
        thr, mf1 = macro_f1_at_best(y, fused)
        if mf1 > best["mf1"]:
            best = {"mf1": mf1, "w": w, "thr": thr}
    print(
        f"\n=== best LOGIT fusion ===\n  weights={dict(zip(names, np.round(best['w'],3)))}"
        f"\n  macroF1={best['mf1']:.4f} thr={best['thr']:.3f}"
    )
    fused_hold = 1.0 / (1.0 + np.exp(-(hlogit * best["w"]).sum(axis=1)))
    print(f"  holdout pred-fake @thr = {(fused_hold >= best['thr']).mean()*100:.1f}%")

    if args.test and args.out:
        test_specs = parse_named(args.test)
        missing = set(names) - set(test_specs)
        if missing:
            raise SystemExit(f"test missing models present in holdout: {missing}")
        ordered = {nm: test_specs[nm] for nm in names}
        tmat, _ = load_aligned(ordered, args.score_col, need_label=False)
        tlogit = to_logit(tmat.to_numpy())
        fused_test = 1.0 / (1.0 + np.exp(-(tlogit * best["w"]).sum(axis=1)))
        out = pd.DataFrame(
            {
                "sample_id": tmat.index,
                "p_fake": fused_test,
                "pred_label": (fused_test >= best["thr"]).astype(int),
            }
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.out)
        print(
            f"\nwrote {len(out)} fused test rows -> {args.out}"
            f"\n  test pred-fake @{best['thr']:.3f} = {out.pred_label.mean()*100:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
