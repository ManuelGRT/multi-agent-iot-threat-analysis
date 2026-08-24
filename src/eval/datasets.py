# src/eval/datasets.py
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def group_by_dataset(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["dataset"])].append(record)
    return dict(grouped)


def leave_one_dataset_out(records: Iterable[dict[str, Any]]) -> list[dict[str, list[dict[str, Any]] | str]]:
    grouped = group_by_dataset(records)
    folds = []
    for dataset, test_records in grouped.items():
        train_records = [
            record
            for other_dataset, values in grouped.items()
            if other_dataset != dataset
            for record in values
        ]
        folds.append({"held_out_dataset": dataset, "train": train_records, "test": test_records})
    return folds
