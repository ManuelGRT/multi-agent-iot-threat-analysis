"""Metricas compartidas por los entrenadores predictivos vigentes."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    classes: Sequence[Any],
    positive_index: int | None = None,
) -> dict[str, Any]:
    """Devuelve metricas agregadas, por clase y matriz de confusion."""

    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if len(truth) != len(predicted):
        raise ValueError("y_true and y_pred must have equal lengths")
    if not len(truth):
        raise ValueError("metrics require at least one sample")
    if len(classes) < 2:
        raise ValueError("metrics require at least two classes")

    labels = np.arange(len(classes), dtype=np.int64)
    if np.any(truth < 0) or np.any(truth >= len(classes)):
        raise ValueError("y_true contains an encoded label outside classes")
    if np.any(predicted < 0) or np.any(predicted >= len(classes)):
        raise ValueError("y_pred contains an encoded label outside classes")

    weighted = precision_recall_fscore_support(
        truth,
        predicted,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        truth,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    per_class = precision_recall_fscore_support(
        truth,
        predicted,
        labels=labels,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(truth, predicted, labels=labels)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "macro": {
            "precision": float(macro[0]),
            "recall": float(macro[1]),
            "f1": float(macro[2]),
        },
        "per_class": [
            {
                "label": _python_scalar(classes[index]),
                "precision": float(per_class[0][index]),
                "recall": float(per_class[1][index]),
                "f1": float(per_class[2][index]),
                "support": int(per_class[3][index]),
            }
            for index in range(len(classes))
        ],
        "confusion_matrix": {
            "labels": [_python_scalar(label) for label in classes],
            "matrix": matrix.astype(int).tolist(),
        },
    }

    if len(classes) == 2:
        resolved_positive = 1 if positive_index is None else positive_index
        if resolved_positive not in (0, 1):
            raise ValueError("positive_index must identify one of the two classes")
        negative_index = 1 - resolved_positive
        tp = int(np.sum((truth == resolved_positive) & (predicted == resolved_positive)))
        tn = int(np.sum((truth == negative_index) & (predicted == negative_index)))
        fp = int(np.sum((truth == negative_index) & (predicted == resolved_positive)))
        fn = int(np.sum((truth == resolved_positive) & (predicted == negative_index)))
        binary_precision = _safe_div(tp, tp + fp)
        binary_recall = _safe_div(tp, tp + fn)
        metrics["binary"] = {
            "positive_label": _python_scalar(classes[resolved_positive]),
            "negative_label": _python_scalar(classes[negative_index]),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": _safe_div(tp + tn, len(truth)),
            "precision": binary_precision,
            "recall": binary_recall,
            "f1": _safe_div(
                2 * binary_precision * binary_recall,
                binary_precision + binary_recall,
            ),
            "specificity": _safe_div(tn, tn + fp),
        }

    return metrics


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


__all__ = ["compute_classification_metrics"]
