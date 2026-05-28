"""SLE metric helpers for simple explanation scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


DEFAULT_SLE_MODEL_PATH = Path("/app/data/xdd/sle")
DEFAULT_SLE_BATCH_SIZE = 16
DEFAULT_SLE_MAX_LENGTH = 512
_SLE_COMPONENT_CACHE: dict[tuple[str, bool, str], dict[str, Any]] = {}


def normalize_sle(raw: float, sle_min: float = -1.0, sle_max: float = 4.0) -> float:
    """Normalize raw SLE from ``[sle_min, sle_max]`` to ``[0, 1]`` with clipping."""
    if sle_max <= sle_min:
        raise ValueError(f"sle_max must be greater than sle_min; got {sle_min} and {sle_max}.")
    clipped = min(max(float(raw), sle_min), sle_max)
    return float((clipped - sle_min) / (sle_max - sle_min))


def _config_value(config: Optional[Mapping[str, Any]], keys: Sequence[str], default: Any) -> Any:
    if config is None:
        return default
    for key in keys:
        if key in config:
            return config[key]
    return default


def _candidate_texts(candidates: Sequence[str]) -> list[str]:
    texts = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, str):
            raise ValueError(f"SLE candidate {index} must be a string.")
        texts.append(candidate)
    return texts


def _mock_sle_score(text: str) -> float:
    """Deterministic local stand-in used only for tests."""
    token_count = len(text.split())
    if token_count == 0:
        return -1.0
    return min(4.0, token_count / 10.0)


def _extract_sle_values(logits: Any) -> list[float]:
    logits = logits.squeeze(-1).detach().cpu()
    if getattr(logits, "ndim", None) == 0:
        return [float(logits.item())]
    return [float(value) for value in logits.tolist()]


def _resolve_sle_model_source(config: Optional[Mapping[str, Any]]) -> tuple[str, bool]:
    source = _config_value(
        config,
        ("sle_model_path", "model_path", "sle_model_id", "model_id"),
        str(DEFAULT_SLE_MODEL_PATH),
    )
    source_text = str(source)
    source_path = Path(source_text)
    local_files_only = bool(_config_value(config, ("sle_local_files_only", "local_files_only"), True))

    if source_text == str(DEFAULT_SLE_MODEL_PATH) and not source_path.exists():
        raise RuntimeError(
            "Actual SLE checkpoint is unavailable. Copy the liamcripwell/sle-base "
            f"checkpoint/tokenizer files to {DEFAULT_SLE_MODEL_PATH} on the worker "
            "host mount. Do not rely on the external metric_automation folder."
        )
    if source_path.exists():
        return str(source_path), True
    return source_text, local_files_only


def _load_sle_components(model_source: str, local_files_only: bool, device: str) -> dict[str, Any]:
    cache_key = (model_source, bool(local_files_only), device)
    if cache_key in _SLE_COMPONENT_CACHE:
        return _SLE_COMPONENT_CACHE[cache_key]

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Actual SLE dependencies are unavailable: install torch>=2.6.0 and "
            "transformers in the scoring image."
        ) from exc

    torch_device = torch.device(device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            local_files_only=local_files_only,
        )
    except ValueError as exc:
        error_text = str(exc)
        if "Due to a serious vulnerability issue in `torch.load`" in error_text:
            raise RuntimeError(
                "Failed to load SLE model. The checkpoint uses a legacy "
                "pytorch_model.bin file and Transformers requires torch>=2.6.0 "
                "for safe loading. Prefer model.safetensors at "
                f"{DEFAULT_SLE_MODEL_PATH}, or install torch>=2.6.0."
            ) from exc
        raise RuntimeError(
            "Actual SLE model could not be loaded from "
            f"{model_source}. Ensure the checkpoint/tokenizer files exist under "
            f"{DEFAULT_SLE_MODEL_PATH} or provide config['sle_model_path']."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Actual SLE model could not be loaded from "
            f"{model_source}. Ensure the checkpoint/tokenizer files exist under "
            f"{DEFAULT_SLE_MODEL_PATH} or provide config['sle_model_path']. "
            f"Original error: {exc}"
        ) from exc

    model.to(torch_device)
    model.eval()
    loaded = {"tokenizer": tokenizer, "model": model, "device": torch_device}
    _SLE_COMPONENT_CACHE[cache_key] = loaded
    return loaded


def compute_sle_raw(
    candidates: Sequence[str],
    references: Optional[Sequence[str]] = None,
    device: str = "cpu",
    config: Optional[Mapping[str, Any]] = None,
    mock: bool = False,
) -> float:
    """Compute the mean raw SLE score for candidate simple explanations.

    The upstream SLE implementation is reference-free, so ``references`` is
    accepted for scorer API symmetry and ignored. Real mode loads a local
    Hugging Face sequence-classification checkpoint directly and never falls
    back to mock scores.
    """
    del references

    texts = _candidate_texts(candidates)
    if not texts:
        return 0.0

    if mock:
        values = [_mock_sle_score(text) for text in texts]
        return float(sum(values) / len(values))

    batch_size = int(_config_value(config, ("sle_batch_size", "batch_size"), DEFAULT_SLE_BATCH_SIZE))
    max_length = int(_config_value(config, ("sle_max_length", "max_length"), DEFAULT_SLE_MAX_LENGTH))

    if batch_size <= 0:
        raise ValueError(f"SLE batch_size must be positive; got {batch_size}.")
    if max_length <= 0:
        raise ValueError(f"SLE max_length must be positive; got {max_length}.")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Actual SLE dependencies are unavailable: missing torch. Install the "
            "scoring program requirements in the worker image."
        ) from exc

    model_source, local_files_only = _resolve_sle_model_source(config)
    loaded = _load_sle_components(model_source, local_files_only, device)
    tokenizer = loaded["tokenizer"]
    model = loaded["model"]
    torch_device = loaded["device"]

    scores: list[float] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(torch_device) for key, value in encoded.items()}
                logits = model(**encoded).logits
                scores.extend(_extract_sle_values(logits))
    except Exception as exc:
        raise RuntimeError(
            "Actual SLE scoring failed. Check that the SLE checkpoint, tokenizer, "
            f"and device '{device}' are usable. Original error: {exc}"
        ) from exc

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))
