"""BERTScore metric helpers for explanation scoring."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Optional, Sequence


_BERT_SCORER_CACHE: dict[tuple[Any, ...], Any] = {}


def _token_overlap_f1(candidate: str, reference: str) -> float:
    """Small deterministic stand-in used only by tests."""
    candidate_tokens = candidate.lower().split()
    reference_tokens = reference.lower().split()
    if not candidate_tokens or not reference_tokens:
        return 0.0

    overlap = sum((Counter(candidate_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return float(2 * precision * recall / (precision + recall))


def _validate_inputs(candidates: Sequence[str], references: Sequence[str], batch_size: int) -> None:
    if len(candidates) != len(references):
        raise ValueError(
            f"BERTScore requires the same number of candidates and references; "
            f"got {len(candidates)} candidates and {len(references)} references."
        )
    if batch_size <= 0:
        raise ValueError(f"BERTScore batch_size must be positive; got {batch_size}.")


def _missing_safetensors_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    return "safetensors" in error_text and (
        "cannot be loaded" in error_text
        or "no file named" in error_text
        or "not found" in error_text
    )


def _from_pretrained_prefer_safetensors(model_loader: Any, model_source: str) -> Any:
    try:
        return model_loader.from_pretrained(model_source, use_safetensors=True)
    except (OSError, ValueError) as exc:
        if _missing_safetensors_error(exc):
            return model_loader.from_pretrained(model_source)
        raise


def _get_bert_score_model_with_safetensors(
    model_type: str,
    num_layers: int,
    all_layers: Optional[bool] = None,
) -> Any:
    import torch
    import bert_score.utils as bert_score_utils
    from transformers import AutoModel

    if model_type.startswith("scibert"):
        model = _from_pretrained_prefer_safetensors(
            AutoModel,
            bert_score_utils.cache_scibert(model_type),
        )
    elif "t5" in model_type:
        from transformers import T5EncoderModel

        model = _from_pretrained_prefer_safetensors(T5EncoderModel, model_type)
    else:
        model = _from_pretrained_prefer_safetensors(AutoModel, model_type)
    model.eval()

    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        model = model.encoder

    if not all_layers:
        if hasattr(model, "n_layers"):
            assert 0 <= num_layers <= model.n_layers
            model.n_layers = num_layers
        elif hasattr(model, "layer"):
            assert 0 <= num_layers <= len(model.layer)
            model.layer = torch.nn.ModuleList([layer for layer in model.layer[:num_layers]])
        elif hasattr(model, "encoder"):
            if hasattr(model.encoder, "albert_layer_groups"):
                assert 0 <= num_layers <= model.encoder.config.num_hidden_layers
                model.encoder.config.num_hidden_layers = num_layers
            elif hasattr(model.encoder, "block"):
                assert 0 <= num_layers <= len(model.encoder.block)
                model.encoder.block = torch.nn.ModuleList([layer for layer in model.encoder.block[:num_layers]])
            else:
                assert 0 <= num_layers <= len(model.encoder.layer)
                model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])
        elif hasattr(model, "transformer"):
            assert 0 <= num_layers <= len(model.transformer.layer)
            model.transformer.layer = torch.nn.ModuleList([layer for layer in model.transformer.layer[:num_layers]])
        elif hasattr(model, "layers"):
            assert 0 <= num_layers <= len(model.layers)
            model.layers = torch.nn.ModuleList([layer for layer in model.layers[:num_layers]])
        else:
            raise ValueError("Unsupported BERTScore model architecture.")
    else:
        if hasattr(model, "output_hidden_states"):
            model.output_hidden_states = True
        elif hasattr(model, "encoder"):
            model.encoder.output_hidden_states = True
        elif hasattr(model, "transformer"):
            model.transformer.output_hidden_states = True

    return model


def _normalize_tokenizer_model_max_length(tokenizer: Any, model: Any) -> None:
    config = getattr(model, "config", None)
    max_positions = None
    for attr_name in (
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
        "max_sequence_length",
    ):
        value = getattr(config, attr_name, None)
        if isinstance(value, int) and value > 0:
            max_positions = value
            break

    if max_positions is None:
        return

    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(tokenizer_max_length, int) or tokenizer_max_length <= 0 or tokenizer_max_length > 1000000:
        normalized_length = int(max_positions)
    else:
        normalized_length = int(min(tokenizer_max_length, max_positions))

    tokenizer.model_max_length = normalized_length
    if hasattr(tokenizer, "init_kwargs") and isinstance(tokenizer.init_kwargs, dict):
        tokenizer.init_kwargs["model_max_length"] = normalized_length


def get_bert_scorer(
    *,
    model_type: str,
    lang: str = "en",
    rescale_with_baseline: bool = False,
    device: str = "cpu",
) -> Any:
    """Build a BERTScorer matching the original metric_automation helper."""
    cache_key = (model_type, lang, bool(rescale_with_baseline), device)
    if cache_key in _BERT_SCORER_CACHE:
        return _BERT_SCORER_CACHE[cache_key]

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    try:
        from bert_score import BERTScorer
        import bert_score.scorer as bert_score_scorer
        import bert_score.utils as bert_score_utils
    except ImportError as exc:
        raise RuntimeError(
            "The bert-score package is required for real BERTScore computation. "
            "Install scoring_program/requirements.txt or call with mock=True in tests."
        ) from exc

    original_scorer_get_model = bert_score_scorer.get_model
    original_utils_get_model = bert_score_utils.get_model
    bert_score_scorer.get_model = _get_bert_score_model_with_safetensors
    bert_score_utils.get_model = _get_bert_score_model_with_safetensors
    try:
        scorer = BERTScorer(
            model_type=model_type,
            lang=lang,
            rescale_with_baseline=rescale_with_baseline,
            device=device,
        )
    finally:
        bert_score_scorer.get_model = original_scorer_get_model
        bert_score_utils.get_model = original_utils_get_model
    _normalize_tokenizer_model_max_length(scorer._tokenizer, scorer._model)
    _BERT_SCORER_CACHE[cache_key] = scorer
    return scorer


def compute_bertscore_f1(
    candidates: Sequence[str],
    references: Sequence[str],
    model_type: str,
    device: str,
    batch_size: int,
    *,
    rescale_with_baseline: bool = False,
    mock: bool = False,
) -> float:
    """Compute mean BERTScore F1 for candidate/reference explanation text.

    The real scoring path uses the ``bert-score`` package with English language
    settings. Tests can pass ``mock=True`` to avoid importing bert-score or
    downloading model weights.
    """
    _validate_inputs(candidates, references, batch_size)
    if not candidates or not references:
        return 0.0

    if mock:
        scores = [
            _token_overlap_f1(candidate, reference)
            for candidate, reference in zip(candidates, references)
        ]
        return float(sum(scores) / len(scores))

    scorer = get_bert_scorer(
        model_type=model_type,
        lang="en",
        rescale_with_baseline=rescale_with_baseline,
        device=device,
    )
    _, _, f1_scores = scorer.score(
        list(candidates),
        list(references),
        batch_size=max(1, min(batch_size, len(candidates))),
        verbose=False,
    )
    mean_f1 = f1_scores.mean()
    if hasattr(mean_f1, "item"):
        return float(mean_f1.item())
    return float(mean_f1)
