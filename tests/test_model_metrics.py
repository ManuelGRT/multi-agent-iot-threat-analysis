from __future__ import annotations

import pytest

from src.eval.model_metrics import compute_classification_metrics


def test_metrics_include_weighted_macro_per_class_confusion_and_binary_values():
    metrics = compute_classification_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        classes=["normal", "attack"],
        positive_index=1,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision_weighted"] == pytest.approx(5 / 6)
    assert metrics["recall_weighted"] == pytest.approx(0.75)
    assert metrics["f1_weighted"] == pytest.approx(11 / 15)
    assert metrics["macro"]["f1"] == pytest.approx(11 / 15)
    assert metrics["per_class"] == [
        {
            "label": "normal",
            "precision": 1.0,
            "recall": 0.5,
            "f1": pytest.approx(2 / 3),
            "support": 2,
        },
        {
            "label": "attack",
            "precision": pytest.approx(2 / 3),
            "recall": 1.0,
            "f1": 0.8,
            "support": 2,
        },
    ]
    assert metrics["confusion_matrix"] == {
        "labels": ["normal", "attack"],
        "matrix": [[1, 1], [0, 2]],
    }
    assert metrics["binary"] == {
        "positive_label": "attack",
        "negative_label": "normal",
        "tp": 2,
        "tn": 1,
        "fp": 1,
        "fn": 0,
        "accuracy": 0.75,
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "f1": 0.8,
        "specificity": 0.5,
    }


@pytest.mark.parametrize(
    ("truth", "predicted", "classes", "message"),
    [
        ([], [], ["a", "b"], "at least one sample"),
        ([0], [0, 1], ["a", "b"], "equal lengths"),
        ([2], [0], ["a", "b"], "outside classes"),
    ],
)
def test_metrics_reject_invalid_inputs(truth, predicted, classes, message):
    with pytest.raises(ValueError, match=message):
        compute_classification_metrics(truth, predicted, classes=classes)
