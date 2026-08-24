# src/eval/metrics.py
from __future__ import annotations

from collections import Counter
from typing import Iterable


def binary_classification_metrics(y_true: Iterable[bool], y_pred: Iterable[bool]) -> dict[str, float | int]:
    pairs = list(zip(y_true, y_pred))
    counts = Counter()
    for truth, pred in pairs:
        if truth and pred:
            counts["tp"] += 1
        elif not truth and not pred:
            counts["tn"] += 1
        elif not truth and pred:
            counts["fp"] += 1
        else:
            counts["fn"] += 1

    total = len(pairs)
    precision = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
    recall = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "n": total,
        "tp": counts["tp"],
        "tn": counts["tn"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "accuracy": _safe_div(counts["tp"] + counts["tn"], total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
