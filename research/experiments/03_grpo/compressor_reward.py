"""GRPO reward for the XPlainVerse compressor (complex -> simple).

Optimises the official simple-explanation metric directly:

    simple_overall = 0.70 * BERTScore_F1(pred_simple, gt_simple)
                   + 0.30 * SLE_norm(pred_simple)
    SLE_norm(x) = (clip(x, -1, 4) + 1) / 5

Register two reward functions, combine with --reward_weights 0.7 0.3:

    --external_plugins research/experiments/03_grpo/compressor_reward.py
    --reward_funcs xpv_simple_bert xpv_simple_sle
    --reward_weights 0.7 0.3

An optional anti-hacking gate (xpv_simple_gate) penalises degenerate outputs
(empty / 1-2 words / banned technical jargon) so GRPO cannot farm SLE with a
content-free sentence. Keep its weight small (e.g. 0.0-0.1).

The GT simple text is read from dataset columns `reference_simple` (preferred)
or `solution`, passed through by ms-swift aligned with completions
(see dataset/build_compressor_grpo_jsonl.py).
"""
from __future__ import annotations

import os
import re
from typing import Any

try:  # ms-swift exposes the reward registry at different paths across versions
    from swift.plugin import ORM, orms
except ImportError:  # pragma: no cover
    from swift.rewards import ORM, orms


_BANNED = (
    "artifact", "synthesis", "inconsistency", "generation",
    "synthetic", "anatomical",
)
_WS_RE = re.compile(r"\s+")


def _as_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion or "")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def _refs_from_kwargs(kwargs: dict, n: int) -> list[str]:
    refs = kwargs.get("reference_simple")
    if refs is None:
        refs = kwargs.get("solution")
    if refs is None:
        refs = [""] * n
    return [_clean(_as_text(r)) for r in refs]


def _norm_sle(x: float) -> float:
    x = max(-1.0, min(4.0, float(x)))
    return (x + 1.0) / 5.0


class _BertScorer:
    """Lazy BERTScorer matching evaluate_val.py's leaderboard config."""

    _scorer = None

    @classmethod
    def get(cls):
        if cls._scorer is None:
            import torch
            from bert_score import BERTScorer

            cls._scorer = BERTScorer(
                model_type=os.getenv("XPV_BERT_MODEL", "microsoft/deberta-xlarge-mnli"),
                lang="en",
                rescale_with_baseline=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            try:  # DeBERTa's sentinel max_length overflows the Rust tokenizer.
                tok = getattr(cls._scorer, "_tokenizer", None)
                if tok is not None and int(getattr(tok, "model_max_length", 0)) > 100000:
                    tok.model_max_length = 512
            except Exception:
                pass
        return cls._scorer


class _SleScorer:
    """Lazy SLE regression model (liamcripwell/sle-base), reference-free."""

    _tok = None
    _model = None
    _device = None

    @classmethod
    def get(cls):
        if cls._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_id = os.getenv("XPV_SLE_MODEL", "liamcripwell/sle-base")
            cls._tok = AutoTokenizer.from_pretrained(model_id)
            cls._model = AutoModelForSequenceClassification.from_pretrained(model_id)
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model.to(cls._device)
            cls._model.eval()
        return cls._tok, cls._model, cls._device

    @classmethod
    def score(cls, texts: list[str]) -> list[float]:
        import torch

        tok, model, device = cls.get()
        bs = int(os.getenv("XPV_SLE_BATCH", "16"))
        out: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(texts), bs):
                batch = [t if t else " " for t in texts[start:start + bs]]
                enc = tok(batch, padding=True, truncation=True, max_length=512,
                          return_tensors="pt").to(device)
                logits = model(**enc).logits.squeeze(-1).detach().cpu()
                if logits.ndim == 0:
                    out.append(float(logits.item()))
                else:
                    out.extend(float(v) for v in logits.tolist())
        return out


class SimpleBertReward(ORM):
    """0..1 reward = BERTScore F1 of the completion vs the GT simple text."""

    def __call__(self, completions, **kwargs):
        cands = [_clean(_as_text(c)) for c in completions]
        refs = _refs_from_kwargs(kwargs, len(completions))
        safe_c = [c if c else " " for c in cands]
        safe_r = [r if r else " " for r in refs]
        batch = int(os.getenv("XPV_BERT_BATCH", "16"))
        _, _, f1 = _BertScorer.get().score(safe_c, safe_r, batch_size=batch)
        return [0.0 if not c else float(s) for c, s in zip(cands, f1.tolist())]


class SimpleSleReward(ORM):
    """0..1 reward = normalised SLE simplicity of the completion (reference-free)."""

    def __call__(self, completions, **kwargs):
        cands = [_clean(_as_text(c)) for c in completions]
        raw = _SleScorer.score(cands)
        return [0.0 if not c else _norm_sle(s) for c, s in zip(cands, raw)]


class SimpleGateReward(ORM):
    """Anti-hacking gate in [-1, 0]: penalise degenerate / jargon-laden outputs.

    Use with a small positive weight; it only ever subtracts.
    """

    def __call__(self, completions, **kwargs):
        min_words = int(os.getenv("XPV_GATE_MIN_WORDS", "5"))
        max_words = int(os.getenv("XPV_GATE_MAX_WORDS", "60"))
        rewards = []
        for c in completions:
            text = _clean(_as_text(c))
            low = text.lower()
            pen = 0.0
            n = len(text.split())
            if n < min_words:
                pen -= 1.0
            elif n > max_words:
                pen -= 0.3
            if any(b in low for b in _BANNED):
                pen -= 0.5
            rewards.append(pen)
        return rewards


orms["xpv_simple_bert"] = SimpleBertReward
orms["xpv_simple_sle"] = SimpleSleReward
orms["xpv_simple_gate"] = SimpleGateReward
