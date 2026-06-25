#!/usr/bin/env python3
"""Build a verdict-conditioned Pass-2 infer set for the two-stage pipeline.

Stage 1 (verdict) comes from either:
  * ``gt``       — ground-truth label (upper bound on explanation quality), or
  * ``ensemble`` — our Pass-1 SigLIP2+DINOv2 ensemble predictions parquet
                   (sample_id, p_fake); label = fake if p_fake >= threshold.

Stage 2 (explanation) is the VLM conditioned on that verdict using the v2
HYPOTHETICAL_{FAKE,REAL} prompts from dataset/prompt_v2.txt — verbatim the
prompts the model was trained on, so train/infer match.

Two subcommands share the SAME deterministic sample selection (balanced by GT,
sorted by sample_id), so they never drift:

  manifest  -> write a Pass-1 classifier manifest parquet (image_path,
               label_int, sample_id) to run eval_ensemble.py on.
  infer     -> write the conditioned ms-swift infer JSONL (and a verdicts.json
               summary), reading verdicts from gt or an ensemble parquet.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SECTION_RE = re.compile(r"^===\s*(.+?)\s*===\s*$")
HYP_FAKE = "VLM_USER_PROMPT_HYPOTHETICAL_FAKE"
HYP_REAL = "VLM_USER_PROMPT_HYPOTHETICAL_REAL"
PRIMARY = "VLM_USER_PROMPT"
LABEL2INT = {"real": 0, "fake": 1}


def load_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_prompt_file(path: Path) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = SECTION_RE.match(line)
        if m:
            name = m.group(1).strip()
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = None if name.upper() == "END" else name
            buf = []
            continue
        if current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    for key in (PRIMARY, HYP_FAKE, HYP_REAL):
        if not sections.get(key):
            raise ValueError(f"prompt file missing section: {key}")
    return sections


def image_path_of(row: dict) -> str | None:
    imgs = row.get("images") or []
    if not imgs:
        return None
    first = imgs[0]
    if isinstance(first, dict):
        return first.get("path")
    return first if isinstance(first, str) else None


def select_rows(val_infer: Path, gt_path: Path, n: int, balanced: bool) -> list[dict]:
    """Deterministic: balance by GT label, take first-N per class by sorted sample_id."""
    gt = {r["sample_id"]: r["label"] for r in load_jsonl(gt_path)}
    rows = []
    for r in load_jsonl(val_infer):
        sid = r.get("sample_id")
        if sid in gt and gt[sid] in LABEL2INT and image_path_of(r):
            r["_gt"] = gt[sid]
            rows.append(r)
    if balanced:
        reals = sorted((r for r in rows if r["_gt"] == "real"), key=lambda x: x["sample_id"])
        fakes = sorted((r for r in rows if r["_gt"] == "fake"), key=lambda x: x["sample_id"])
        half = n // 2
        sel = reals[:half] + fakes[:half]
    else:
        sel = sorted(rows, key=lambda x: x["sample_id"])[:n]
    sel.sort(key=lambda x: x["sample_id"])
    return sel


def cmd_manifest(args: argparse.Namespace) -> int:
    import pandas as pd

    sel = select_rows(args.val_infer, args.gt, args.n, not args.no_balanced)
    df = pd.DataFrame(
        {
            "sample_id": [r["sample_id"] for r in sel],
            "image_path": [image_path_of(r) for r in sel],
            "label": [r["_gt"] for r in sel],
            "label_int": [LABEL2INT[r["_gt"]] for r in sel],
        }
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    n_real = int((df.label == "real").sum())
    n_fake = int((df.label == "fake").sum())
    print(f"wrote Pass-1 manifest: {args.out}  ({len(df)} rows: {n_real} real + {n_fake} fake)")
    return 0


def load_ensemble_verdicts(parquet: Path, threshold: float) -> dict[str, str]:
    import pandas as pd

    df = pd.read_parquet(parquet)
    col = "p_fake" if "p_fake" in df.columns else df.columns[-1]
    return {
        str(sid): ("fake" if float(p) >= threshold else "real")
        for sid, p in zip(df["sample_id"], df[col])
    }


def cmd_infer(args: argparse.Namespace) -> int:
    prompts = parse_prompt_file(args.prompt_file)
    sel = select_rows(args.val_infer, args.gt, args.n, not args.no_balanced)

    if args.verdict_source == "ensemble":
        verdicts = load_ensemble_verdicts(Path(args.ensemble_pred), args.threshold)
    else:
        verdicts = {r["sample_id"]: r["_gt"] for r in sel}

    out_rows = []
    summary = {"gt": {"real": 0, "fake": 0}, "pred": {"real": 0, "fake": 0},
               "correct": 0, "total": 0, "missing_verdict": 0}
    for r in sel:
        sid = r["sample_id"]
        gt_label = r["_gt"]
        verdict = verdicts.get(sid)
        if verdict is None:
            summary["missing_verdict"] += 1
            verdict = gt_label  # fall back so the row is still scored
        key = HYP_FAKE if verdict == "fake" else HYP_REAL
        user_content = "<image>\n" + prompts[key]
        out_rows.append({
            "id": r.get("id", f"{gt_label}__{sid}"),
            "sample_id": sid,
            "label": gt_label,          # GT kept for reference only
            "pass1_verdict": verdict,    # what we conditioned on
            "messages": [{"role": "user", "content": user_content}],
            "images": [image_path_of(r)],
        })
        summary["gt"][gt_label] += 1
        summary["pred"][verdict] += 1
        summary["total"] += 1
        summary["correct"] += int(verdict == gt_label)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.verdicts_json:
        summary["pass1_accuracy"] = (summary["correct"] / summary["total"]) if summary["total"] else None
        Path(args.verdicts_json).write_text(json.dumps(summary, indent=2))

    acc = summary["correct"] / summary["total"] if summary["total"] else 0.0
    print(f"wrote conditioned infer: {args.out}  ({len(out_rows)} rows)")
    print(f"  verdict source: {args.verdict_source}  Pass-1 accuracy vs GT: {acc:.3f} "
          f"({summary['correct']}/{summary['total']})")
    print(f"  gt={summary['gt']}  pred={summary['pred']}  missing={summary['missing_verdict']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--val-infer", required=True, type=Path,
                        help="dataset/val_vlm_infer_v2.jsonl")
    common.add_argument("--gt", required=True, type=Path,
                        help="evaluation/data/val_ground_truth.jsonl")
    common.add_argument("--n", type=int, default=1000)
    common.add_argument("--no-balanced", action="store_true",
                        help="take first-N by sample_id instead of balanced per class")

    pm = sub.add_parser("manifest", parents=[common], help="write Pass-1 classifier manifest parquet")
    pm.add_argument("--out", required=True, type=Path)
    pm.set_defaults(func=cmd_manifest)

    pi = sub.add_parser("infer", parents=[common], help="write conditioned ms-swift infer JSONL")
    pi.add_argument("--prompt-file", required=True, type=Path)
    pi.add_argument("--verdict-source", choices=("gt", "ensemble"), default="gt")
    pi.add_argument("--ensemble-pred", type=Path, help="predictions parquet (verdict-source=ensemble)")
    pi.add_argument("--threshold", type=float, default=0.5)
    pi.add_argument("--out", required=True, type=Path)
    pi.add_argument("--verdicts-json", type=Path, default=None)
    pi.set_defaults(func=cmd_infer)

    args = p.parse_args()
    if args.cmd == "infer" and args.verdict_source == "ensemble" and not args.ensemble_pred:
        p.error("--verdict-source ensemble requires --ensemble-pred")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
