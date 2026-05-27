#!/usr/bin/env python3
"""Build 3 val-infer JSONL subsets for the gold-verdict-conditioning experiment.

Variant A — baseline:    unmodified prompt (control)
Variant B — conditioned: original prompt + "FORENSIC ANALYSIS: this image is {label}"
Variant C — structured:  conditioned + "list exactly 5 regions, one per line"

All three variants use the SAME 200 sample_ids = 100 real + 100 fake, sorted by
sample_id for determinism. The label injected in B and C is the GOLD label
(taken from val_ground_truth.jsonl), not from the VLM.

Outputs: subsets/{baseline,conditioned,structured}.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/code/XPlainVerse-ACMChallenge")
GT_PATH = ROOT / "evaluation/data/val_ground_truth.jsonl"
INFER_PATH = ROOT / "dataset/val_vlm_infer.jsonl"
OUT_DIR = ROOT / "research/experiments/01_gold_verdict/subsets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_CLASS = 100


def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_baseline_messages(orig_messages):
    return orig_messages


def build_conditioned_messages(orig_messages, label: str):
    """Prepend a system-style forensic-analysis preamble to the user message,
    and pre-fix the verdict line."""
    msgs = [dict(m) for m in orig_messages]
    user = msgs[0]
    assert user["role"] == "user"
    content = user["content"]
    preamble = (
        f"FORENSIC ANALYSIS — a separate AI-generated-content detector has determined "
        f"this image is {label.upper()}. Provide the forensic explanation supporting "
        f"that determination.\n\n"
    )
    user["content"] = preamble + content
    return msgs


def build_structured_messages(orig_messages, label: str):
    """Replace the task with a structured 5-region template, conditioned on label."""
    msgs = [dict(m) for m in orig_messages]
    user = msgs[0]
    assert user["role"] == "user"
    if label == "fake":
        body = (
            f"FORENSIC ANALYSIS — a separate AI-generated-content detector has "
            f"determined this image is FAKE.\n"
            f"<image>\n"
            f"TASK: This image is AI-generated. Identify exactly 5 specific objects "
            f"or regions in the image and, in one coherent paragraph, describe the "
            f"visible synthesis artifact in each — distorted text, warped geometry, "
            f"smudged or merged textures, anatomical errors, inconsistent lighting "
            f"or shadows, unnatural object boundaries. Cover all 5 regions. Use "
            f"connectives such as \"Additionally\" and \"Furthermore\" to chain the "
            f"observations together.\n\n"
            f"End your response with a new line containing exactly:\nVerdict: fake"
        )
    else:
        body = (
            f"FORENSIC ANALYSIS — a separate AI-generated-content detector has "
            f"determined this image is REAL.\n"
            f"<image>\n"
            f"TASK: This image is a real photograph. Identify several authentic "
            f"photographic cues — natural lighting and shadows, fine surface "
            f"texture, lens-consistent depth of field, unforced facial expressions, "
            f"plausible object boundaries — and describe them in one coherent "
            f"paragraph that supports the real verdict.\n\n"
            f"End your response with a new line containing exactly:\nVerdict: real"
        )
    user["content"] = body
    return msgs


def main():
    print("[1/3] loading ground truth...")
    gt = {row["sample_id"]: row["label"] for row in load_jsonl(GT_PATH)}
    print(f"      gt rows: {len(gt)}")

    print("[2/3] loading val_vlm_infer.jsonl & filtering...")
    real_rows: list[dict] = []
    fake_rows: list[dict] = []
    n_total = 0
    for row in load_jsonl(INFER_PATH):
        n_total += 1
        sid = row["sample_id"]
        gt_label = gt.get(sid)
        if gt_label is None:
            continue
        if gt_label == "real" and len(real_rows) < PER_CLASS:
            real_rows.append((row, "real"))
        elif gt_label == "fake" and len(fake_rows) < PER_CLASS:
            fake_rows.append((row, "fake"))
        if len(real_rows) >= PER_CLASS and len(fake_rows) >= PER_CLASS:
            break
    print(f"      scanned {n_total} infer rows; got {len(real_rows)} real + {len(fake_rows)} fake")
    assert len(real_rows) == PER_CLASS and len(fake_rows) == PER_CLASS, (
        "did not find enough samples — check ordering of val_vlm_infer.jsonl"
    )

    selected = real_rows + fake_rows
    selected.sort(key=lambda x: x[0]["sample_id"])

    print("[3/3] writing 3 variant files...")
    paths = {
        "baseline": OUT_DIR / "baseline.jsonl",
        "conditioned": OUT_DIR / "conditioned.jsonl",
        "structured": OUT_DIR / "structured.jsonl",
    }

    f_base = paths["baseline"].open("w")
    f_cond = paths["conditioned"].open("w")
    f_struct = paths["structured"].open("w")

    for row, label in selected:
        sid = row["sample_id"]
        images = row["images"]
        orig_messages = row["messages"]

        base = {
            "id": row["id"],
            "sample_id": sid,
            "label": label,
            "messages": build_baseline_messages(orig_messages),
            "images": images,
        }
        cond = {
            "id": row["id"],
            "sample_id": sid,
            "label": label,
            "messages": build_conditioned_messages(orig_messages, label),
            "images": images,
        }
        struct = {
            "id": row["id"],
            "sample_id": sid,
            "label": label,
            "messages": build_structured_messages(orig_messages, label),
            "images": images,
        }
        f_base.write(json.dumps(base) + "\n")
        f_cond.write(json.dumps(cond) + "\n")
        f_struct.write(json.dumps(struct) + "\n")

    f_base.close()
    f_cond.close()
    f_struct.close()

    for name, p in paths.items():
        nlines = sum(1 for _ in p.open())
        print(f"      {name}: {p}  ({nlines} rows)")


if __name__ == "__main__":
    main()
