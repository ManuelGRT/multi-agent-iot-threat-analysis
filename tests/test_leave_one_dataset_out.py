from src.eval.datasets import leave_one_dataset_out
from src.eval.metrics import binary_classification_metrics


def test_leave_one_dataset_out_creates_one_fold_per_dataset():
    records = [
        {"dataset": "iot23", "id": 1},
        {"dataset": "iot23", "id": 2},
        {"dataset": "ton_iot", "id": 3},
    ]

    folds = leave_one_dataset_out(records)

    assert {fold["held_out_dataset"] for fold in folds} == {"iot23", "ton_iot"}
    assert all(fold["test"] for fold in folds)


def test_binary_metrics_are_computed():
    metrics = binary_classification_metrics([True, True, False, False], [True, False, True, False])

    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5
