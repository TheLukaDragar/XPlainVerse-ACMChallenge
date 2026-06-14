#!/usr/bin/env python3
"""Build a CodaBench submission zip that swaps ONLY the simple field.

Reuses the prior submission's detection.jsonl + complex.jsonl verbatim (so the
test ids/filenames and detection labels are identical to the scored #6 run),
and replaces simple_explanation for FAKE rows with the GRPO compressor output.
Real rows keep simple = complex.

Inputs:
  --prior-zip       prior submission.zip (detection.jsonl, complex.jsonl, simple.jsonl)
  --grpo-submission compressor GRPO submission.jsonl (sample_id,label,complex,simple)
                    OR --grpo-infer compressor_infer.jsonl (id/sample_id + response)
  --out             output submission.zip
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

VERDICT_RE = re.compile(r"(?:^|\n)\s*Verdict:\s*(real|fake)\s*$", re.IGNORECASE | re.MULTILINE)


def stem(s: str) -> str:
    return Path(str(s)).stem


def clean_simple(text: str) -> str:
    text = VERDICT_RE.sub("", str(text or "")).strip()
    return " ".join(text.split())


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_grpo_simple(args) -> dict[str, str]:
    out: dict[str, str] = {}
    if args.grpo_submission:
        for r in read_jsonl(args.grpo_submission):
            if str(r.get("label", "")).lower() != "fake":
                continue
            out[stem(r["sample_id"])] = clean_simple(r.get("simple_explanation", ""))
    elif args.grpo_infer:
        for r in read_jsonl(args.grpo_infer):
            resp = r.get("response") or r.get("prediction") or ""
            sid = r.get("sample_id")
            if not sid:
                rid = r.get("id", "")
                sid = rid.split("__", 1)[1] if "__" in str(rid) else rid
            if sid and resp:
                out[stem(sid)] = clean_simple(resp)
    else:
        raise SystemExit("provide --grpo-submission or --grpo-infer")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prior-zip", required=True, type=Path)
    ap.add_argument("--grpo-submission", type=Path)
    ap.add_argument("--grpo-infer", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    work = args.out.parent / "_prior_unzip"
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.prior_zip) as zf:
        zf.extractall(work)

    det_path = work / "detection.jsonl"
    cx_path = work / "complex.jsonl"
    for p in (det_path, cx_path):
        if not p.is_file():
            raise SystemExit(f"prior zip missing {p.name}")

    label_by_id = {r["id"]: int(r["pred_label"]) for r in read_jsonl(det_path)}
    complex_by_id = {r["id"]: r["complex_explanation"] for r in read_jsonl(cx_path)}

    grpo = load_grpo_simple(args)
    print(f"prior rows: det={len(label_by_id)} complex={len(complex_by_id)}  grpo fake simples={len(grpo)}")

    simple_lines = []
    n_fake_grpo = n_fake_missing = n_real = 0
    for cid, cx in complex_by_id.items():
        lab = label_by_id.get(cid, 0)
        if lab == 1:  # fake
            s = grpo.get(stem(cid))
            if s:
                n_fake_grpo += 1
            else:
                s = clean_simple(cx)  # fallback: copy complex if a fake is missing
                n_fake_missing += 1
        else:
            s = cx  # real: copy complex (unchanged policy)
            n_real += 1
        simple_lines.append(json.dumps({"id": cid, "simple_explanation": s}, ensure_ascii=False))

    print(f"simple.jsonl: fake_grpo={n_fake_grpo} fake_missing_fallback={n_fake_missing} real_copy={n_real}")
    if n_fake_missing:
        print(f"WARNING: {n_fake_missing} fake rows had no GRPO output (copied complex)")

    simple_path = work / "simple_grpo.jsonl"
    simple_path.write_text("\n".join(simple_lines) + "\n", encoding="utf-8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(det_path, "detection.jsonl")
        zf.write(cx_path, "complex.jsonl")
        zf.write(simple_path, "simple.jsonl")
    print(f"wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
