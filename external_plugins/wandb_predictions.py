"""ms-swift custom callback: log eval predictions (image + GT + pred) to W&B.

Hook
----
Activate via the standard ms-swift plumbing:

    --external_plugins external_plugins/wandb_predictions.py
    --callbacks wandb_predictions
    --predict_with_generate true
    --report_to wandb

How it works
------------
ms-swift's `Seq2SeqTrainer.prediction_step` writes one row per generated val
sample to ``{output_dir}/predict.jsonl`` when ``--predict_with_generate true``.
This callback tails that file after each evaluation and pushes the new rows as
a ``wandb.Table`` keyed on the global training step, so you can scrub through
how the model's outputs evolve in the W&B UI.

Tradeoff
--------
``--predict_with_generate true`` switches eval from loss-based to
generate-based, so ``eval_loss`` is no longer logged. ``eval_token_acc`` and
``eval_rouge*`` will appear instead. Keep ``VAL_SLICE`` large for metrics;
``WANDB_SAMPLE_N`` controls how many rows go to the W&B table only.

Env vars
--------
WANDB_SAMPLE_N      Max rows uploaded to W&B per eval (default: 16). Independent
                    of VAL_SLICE — eval still runs on the full val slice.
WANDB_PREDICT_PATH  Override path to predict.jsonl (default: output_dir/predict.jsonl)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from swift.callbacks.base import TrainerCallback
from swift.callbacks.mapping import callbacks_map
from swift.utils import get_logger

logger = get_logger()


def _extract_verdict(text: str | None) -> str | None:
    """Pull `real`/`fake` from a trailing `Verdict: ...` line; None if absent."""
    if not text:
        return None
    parts = text.rsplit('Verdict:', 1)
    if len(parts) < 2:
        return None
    tail = parts[-1].strip().split()
    if not tail:
        return None
    word = tail[0].lower().strip('.,!?:')
    if word in ('real', 'fake'):
        return word
    return None


def _row_image_path(row: dict[str, Any]) -> str | None:
    """Best-effort extraction of the first image path from a predict.jsonl row."""
    images = row.get('images') or []
    if not images:
        return None
    first = images[0]
    if isinstance(first, dict):
        return first.get('path') or first.get('bytes')
    return first if isinstance(first, str) else None


class WandbPredictionsCallback(TrainerCallback):
    """Log a (step, image, label, gt, pred, verdict_match) Table on every eval."""

    def __init__(self, args, trainer):
        super().__init__(args, trainer)

        self.sample_n = int(os.environ.get('WANDB_SAMPLE_N', '8'))
        override = os.environ.get('WANDB_PREDICT_PATH')
        self.predict_path = Path(
            override or os.path.join(args.output_dir, 'predict.jsonl')
        )

        self._byte_offset = 0
        self._eval_idx = 0
        self._enabled = True

        if not getattr(args, 'predict_with_generate', False):
            logger.warning(
                'wandb_predictions: --predict_with_generate is False, '
                'no predict.jsonl will be produced. Disabling callback.'
            )
            self._enabled = False

        report_to = list(getattr(args, 'report_to', []) or [])
        if 'wandb' not in report_to and 'all' not in report_to:
            logger.warning(
                'wandb_predictions: --report_to has no wandb, disabling callback.'
            )
            self._enabled = False

        if self._enabled:
            logger.info(
                f'wandb_predictions: enabled. predict.jsonl={self.predict_path} '
                f'sample_n={self.sample_n}'
            )

    def on_evaluate(self, args, state, control, **kwargs):
        if not self._enabled or not state.is_world_process_zero:
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        if not self.predict_path.is_file():
            logger.warning(
                f'wandb_predictions: predict.jsonl not found at {self.predict_path}. '
                'Skipping this eval (it will appear after the next save).'
            )
            return

        # Detect rotation: at save time ms-swift moves predict.jsonl into the
        # checkpoint dir, then a fresh one starts. If the current file is
        # shorter than our cached offset, reset to 0.
        try:
            current_size = self.predict_path.stat().st_size
        except OSError:
            current_size = 0
        if current_size < self._byte_offset:
            logger.info(
                'wandb_predictions: predict.jsonl rotated (size '
                f'{current_size} < offset {self._byte_offset}); resetting offset.'
            )
            self._byte_offset = 0

        new_rows: list[dict[str, Any]] = []
        with self.predict_path.open('r', encoding='utf-8') as f:
            f.seek(self._byte_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    new_rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(f'wandb_predictions: bad json line: {exc}')
                    continue
            self._byte_offset = f.tell()

        if not new_rows:
            return

        rows = new_rows[: self.sample_n]
        cols = ['step', 'eval_idx', 'image', 'label', 'verdict_gt',
                'verdict_pred', 'verdict_match', 'gt', 'pred']
        table = wandb.Table(columns=cols)

        match_count = 0
        for row in rows:
            img_path = _row_image_path(row)
            wb_img = None
            if img_path and os.path.isfile(img_path):
                try:
                    wb_img = wandb.Image(img_path)
                except Exception as exc:
                    logger.warning(
                        f'wandb_predictions: cannot load image {img_path}: {exc}'
                    )

            gt = (row.get('labels') or '').strip()
            pred = (row.get('response') or '').strip()
            label = row.get('label', '')
            v_gt = _extract_verdict(gt) or label or ''
            v_pred = _extract_verdict(pred) or ''
            verdict_match = bool(v_gt and v_pred and v_gt == v_pred)
            if verdict_match:
                match_count += 1

            table.add_data(
                int(state.global_step),
                self._eval_idx,
                wb_img,
                label,
                v_gt,
                v_pred,
                verdict_match,
                gt,
                pred,
            )

        wandb.log(
            {
                'eval_predictions': table,
                'eval/verdict_match_rate_smoke': match_count / max(len(rows), 1),
                'eval/predict_samples': len(rows),
            },
            step=int(state.global_step),
        )
        logger.info(
            f'wandb_predictions: logged {len(rows)} samples to W&B at step '
            f'{state.global_step} (eval_idx={self._eval_idx}, '
            f'verdict_match={match_count}/{len(rows)})'
        )
        self._eval_idx += 1


callbacks_map['wandb_predictions'] = WandbPredictionsCallback
