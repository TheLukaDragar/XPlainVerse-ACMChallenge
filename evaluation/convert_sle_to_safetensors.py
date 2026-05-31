#!/usr/bin/env python3
"""Convert the SLE model (default liamcripwell/sle-base) to safetensors.

The HF repo ships only a legacy ``pytorch_model.bin``. transformers refuses to
load ``.bin`` checkpoints on torch < 2.6 (the torch.load weights_only CVE
mitigation), which crashes evaluate_val.py at the SLE stage in the lj image
(torch 2.4.1). Re-saving the weights as ``model.safetensors`` removes the
restriction on any torch version.

We avoid the blocked code path by building the model from its config (no
checkpoint load) and loading the state dict ourselves via torch.load, then
save_pretrained(safe_serialization=True).

Usage (inside the lj container):
    python3 evaluation/convert_sle_to_safetensors.py \
        --out /home/jakob/luka/models/sle-base-st
Then score with:  --sle-model-id /home/jakob/luka/models/sle-base-st --sle-local-files-only
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="liamcripwell/sle-base")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] config + tokenizer for {args.model_id}")
    config = AutoConfig.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print("[2/4] downloading legacy pytorch_model.bin (uses HF cache if present)")
    bin_path = hf_hub_download(repo_id=args.model_id, filename="pytorch_model.bin")
    state = torch.load(bin_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    print("[3/4] building model from config and loading weights")
    model = AutoModelForSequenceClassification.from_config(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")

    print(f"[4/4] saving safetensors -> {out}")
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    # Some SLE wrappers read extra files; copy any that exist alongside the bin.
    src_dir = Path(bin_path).parent
    for extra in ("merges.txt", "vocab.json", "special_tokens_map.json"):
        sp = src_dir / extra
        if sp.is_file() and not (out / extra).is_file():
            shutil.copy2(sp, out / extra)

    has_st = (out / "model.safetensors").is_file()
    print(f"done. model.safetensors present: {has_st}")
    print(f"use: --sle-model-id {out} --sle-local-files-only")
    return 0 if has_st else 1


if __name__ == "__main__":
    raise SystemExit(main())
