"""Detection metrics for the final XDD CodaBench task."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from sklearn.metrics import accuracy_score, f1_score


ZERO_DETECTION_METRICS = {
    "detection_macro_f1": 0.0,
    "detection_accuracy": 0.0,
    "detection_fake_f1": 0.0,
    "detection_real_f1": 0.0,
}


def _prediction_map(predictions: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    if isinstance(predictions, Mapping):
        return dict(predictions)
    return {str(row["id"]): row for row in predictions}


def compute_detection_metrics(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    """Compute detection scores from reference rows and validated predictions.

    Missing detection rows contribute zero by computing metrics over submitted
    rows and multiplying by submitted-row coverage over the full reference set.
    """
    pred_by_id = _prediction_map(predictions)
    if not pred_by_id or not references:
        return dict(ZERO_DETECTION_METRICS)

    y_true = []
    y_pred = []
    for row in references:
        sample_id = str(row["id"])
        if sample_id not in pred_by_id:
            continue
        y_true.append(int(row["label"]))
        y_pred.append(int(pred_by_id[sample_id]["pred_label"]))

    if not y_true:
        return dict(ZERO_DETECTION_METRICS)

    coverage = float(len(y_true) / len(references))
    return {
        "detection_macro_f1": float(
            coverage * f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
        ),
        "detection_accuracy": float(coverage * accuracy_score(y_true, y_pred)),
        "detection_fake_f1": float(
            coverage * f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        ),
        "detection_real_f1": float(
            coverage * f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)
        ),
    }
