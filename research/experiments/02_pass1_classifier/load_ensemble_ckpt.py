#!/usr/bin/env python3
"""Load Pass-1 ensemble checkpoints (ours or Bombek1 HF weights)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ensemble_model import EnsembleAIDetector, create_model_with_lora

BOMBek_HF_REPO = "Bombek1/ai-image-detector-siglip-dinov2"
BOMBek_HF_FILENAME = "pytorch_model.pt"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_bombek_weights_path() -> Path:
    return repo_root() / "baseline_models/pass1/bombek_pytorch_model.pt"


def download_bombek_weights(dest: Path | None = None) -> Path:
    dest = Path(dest or default_bombek_weights_path())
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub required to download Bombek weights") from exc
    hf_hub_download(
        repo_id=BOMBek_HF_REPO,
        filename=BOMBek_HF_FILENAME,
        local_dir=str(dest.parent),
    )
    src = dest.parent / BOMBek_HF_FILENAME
    if not src.is_file():
        raise FileNotFoundError(f"failed to download Bombek weights to {src}")
    if src != dest:
        src.replace(dest)
    return dest


def resolve_checkpoint_path(ckpt: str | Path) -> Path:
    ckpt = Path(ckpt)
    if ckpt.is_file():
        return ckpt
    token = str(ckpt).strip().lower()
    if token in {"bombek", "bombek1", "hf", "pretrained"}:
        return download_bombek_weights()
    raise FileNotFoundError(f"checkpoint not found: {ckpt}")


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(raw)
    cfg["siglip_model"] = (
        cfg.get("siglip_model")
        or cfg.get("siglip")
        or cfg.get("backbone_siglip")
        or "google/siglip2-so400m-patch14-384"
    )
    cfg["dinov2_model"] = (
        cfg.get("dinov2_model")
        or cfg.get("dinov2")
        or cfg.get("backbone_dinov2")
        or "vit_large_patch14_dinov2.lvd142m"
    )
    lora = cfg.get("lora") or {}
    cfg["lora_rank"] = int(cfg.get("lora_rank", cfg.get("lora_r", lora.get("rank", 32))))
    cfg["lora_alpha"] = int(cfg.get("lora_alpha", lora.get("alpha", 64)))
    cfg["lora_dropout"] = float(cfg.get("lora_dropout", lora.get("dropout", 0.1)))
    cfg["image_size"] = int(cfg.get("image_size", 392))
    cfg["head_hidden"] = int(cfg.get("head_hidden", cfg.get("classifier_hidden_dim", 512)))
    cfg["head_dropout"] = float(cfg.get("head_dropout", cfg.get("classifier_dropout", 0.3)))
    cfg["weight_source"] = cfg.get("weight_source", "unknown")
    return cfg


def build_model_from_config(cfg: dict[str, Any]) -> EnsembleAIDetector:
    return create_model_with_lora(
        cfg["siglip_model"],
        cfg["dinov2_model"],
        image_size=int(cfg["image_size"]),
        lora_rank=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        hidden_dim=int(cfg.get("head_hidden", 512)),
        head_dropout=float(cfg.get("head_dropout", 0.3)),
    )


def load_ensemble_checkpoint(ckpt_path: str | Path, device: str = "cpu") -> tuple[EnsembleAIDetector, dict[str, Any]]:
    path = resolve_checkpoint_path(ckpt_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("config") or ckpt.get("args") or {}
    cfg = normalize_config(raw_cfg)
    cfg["weight_source"] = "bombek_hf" if "bombek" in str(path).lower() else str(path)
    cfg["checkpoint_path"] = str(path)

    model = build_model_from_config(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        raise RuntimeError(f"checkpoint missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"checkpoint unexpected keys ({len(unexpected)}): {unexpected[:8]}")

    model.to(device)
    model.eval()
    return model, cfg


def init_model_from_bombek(model: EnsembleAIDetector, bombek_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_checkpoint_path(bombek_path or "bombek")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        raise RuntimeError(f"Bombek init missing keys ({len(missing)}): {missing[:8]}")
    info = {
        "init_from": str(path),
        "unexpected_keys": len(unexpected),
    }
    return info


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download or verify Bombek ensemble weights.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--verify", type=Path, default=None, help="checkpoint path or 'bombek'")
    args = parser.parse_args()

    if args.download:
        path = download_bombek_weights(args.dest)
        print(f"downloaded {path}")
    if args.verify:
        model, cfg = load_ensemble_checkpoint(args.verify, device="cpu")
        n = sum(p.numel() for p in model.parameters())
        print(json.dumps({"checkpoint": cfg["checkpoint_path"], "config": cfg, "params": n}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
