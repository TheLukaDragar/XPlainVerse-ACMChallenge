"""GRPO reward functions for XPlainVerse Pass-2 (complex explanation).

Registered into ms-swift's ``orms`` registry and selected at train time via
``--external_plugins research/experiments/03_grpo/grpo_reward.py
  --reward_funcs xpv_complex_bert xpv_verdict_format``.

Design (see research notes): the reward must be CHEAP — the official Qwen
entity/facts scorer takes hours, so it cannot run per rollout. The public
leaderboard's complex metric is **BERTScore F1**, which is fast (one DeBERTa
forward, batched), so we optimise exactly that, plus a light verdict/format
term. GRPO's group-relative advantage handles BERTScore's low variance (the
informative signal is the spread among the N rollouts of the same image).

Reward functions (combine via --reward_weights):
  xpv_complex_bert     BERTScore-F1(candidate_complex, reference_complex),
                       using the SAME config as evaluation/evaluate_val.py
                       (microsoft/deberta-xlarge-mnli, lang=en, no rescale).
  xpv_verdict_format   +0.5 if a well-formed `Verdict: real|fake` line exists,
                       +0.5 if it matches the ground-truth label.

The reference complex text and the gold label are read from the dataset
columns ``reference_complex`` and ``solution`` (added by build_grpo_jsonl.py),
which ms-swift passes through to the reward as kwargs aligned with completions.
"""
from __future__ import annotations

import os
import re
from typing import Any

try:  # ms-swift exposes the reward registry at different paths across versions
    from swift.rewards import ORM, orms
except ImportError:  # pragma: no cover - fallback for older/newer layouts
    from swift.plugin import ORM, orms


_VERDICT_RE = re.compile(r"(?:^|\n)\s*Verdict:\s*(real|fake)\s*\.?\s*$", re.IGNORECASE | re.MULTILINE)


def _as_text(completion: Any) -> str:
    """ms-swift may pass a completion as a string or a chat message list."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion or "")


def _parse_verdict(text: str) -> str | None:
    m = _VERDICT_RE.search(text or "")
    return m.group(1).lower() if m else None


def _strip_verdict(text: str) -> str:
    """Return the complex paragraph with the trailing 'Verdict: ...' removed."""
    text = (text or "").strip()
    text = _VERDICT_RE.sub("", text).strip()
    return " ".join(text.split())


def _ref_text(value: Any) -> str:
    if value is None:
        return ""
    return _strip_verdict(_as_text(value))


class _BertScorer:
    """Lazy singleton BERTScorer matching evaluate_val.py's leaderboard config."""

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
            # bert_score passes tokenizer.model_max_length to enable_truncation;
            # DeBERTa's sentinel (~1e30) overflows the newer Rust tokenizer. Clamp
            # to the model's real max positions (512).
            try:
                tok = getattr(cls._scorer, "_tokenizer", None)
                if tok is not None and int(getattr(tok, "model_max_length", 0)) > 100000:
                    tok.model_max_length = 512
            except Exception:
                pass
        return cls._scorer


class ComplexBertScoreReward(ORM):
    """Primary reward = BERTScore F1 of the candidate complex vs the reference."""

    def __call__(self, completions, **kwargs):
        refs_raw = kwargs.get("reference_complex")
        if refs_raw is None:
            refs_raw = kwargs.get("solution") or [""] * len(completions)
        cands = [_strip_verdict(_as_text(c)) for c in completions]
        refs = [_ref_text(r) for r in refs_raw]

        # BERTScore needs non-empty strings; guard degenerate rollouts.
        safe_cands = [c if c else " " for c in cands]
        safe_refs = [r if r else " " for r in refs]
        batch = int(os.getenv("XPV_BERT_BATCH", "16"))
        _, _, f1 = _BertScorer.get().score(safe_cands, safe_refs, batch_size=batch)
        rewards = []
        for cand, score in zip(cands, f1.tolist()):
            rewards.append(0.0 if not cand else float(score))
        return rewards


class VerdictFormatReward(ORM):
    """Cheap term: well-formed verdict line (+0.5) and correct label (+0.5)."""

    def __call__(self, completions, **kwargs):
        labels = kwargs.get("solution")
        if labels is None:
            labels = kwargs.get("label") or [None] * len(completions)
        rewards = []
        for completion, label in zip(completions, labels):
            verdict = _parse_verdict(_as_text(completion))
            reward = 0.0
            if verdict is not None:
                reward += 0.5
                gold = str(label).strip().lower() if label is not None else None
                if gold in ("real", "fake") and verdict == gold:
                    reward += 0.5
            rewards.append(reward)
        return rewards


orms["xpv_complex_bert"] = ComplexBertScoreReward
orms["xpv_verdict_format"] = VerdictFormatReward
