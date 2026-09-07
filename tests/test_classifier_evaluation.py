from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.classifier_balancing import MappedClassifierRecord, RejectedClassifierRecord
from src.eval.classifier_evaluation import (
    evaluate_classifier_view,
    evaluate_origin_only_baseline,
    evaluate_rejected_attack_confidence,
    select_confidence_threshold,
)
from src.eval.validation_campaign import PreparedRecord


class _Model:
    classes = ["A", "B", "C", "D"]

    def predict_proba(self, rows):
        return np.asarray([row["probabilities"] for row in rows], dtype=float)

    def predict(self, rows):
        probabilities = self.predict_proba(rows)
        return [self.classes[index] for index in np.argmax(probabilities, axis=1)]


def _mapped(
    manifest_id: str,
    label: str,
    probabilities: list[float],
    *,
    split: str = "test",
    origin: str = "edge_iiotset",
) -> MappedClassifierRecord:
    record = PreparedRecord(
        manifest_id=manifest_id,
        dataset=origin,
        source_file=f"{origin}.csv",
        row_id=manifest_id,
        content_hash=f"c-{manifest_id}",
        split=split,
        tasks=("multiclass",),
        target_class=label,
        target_is_attack=True,
        event=None,
        features={"probabilities": probabilities},
        feature_fingerprint=f"fp-{manifest_id}",
        provider="mistral",
        model="mistral-small-2603",
        mapping_confidence=0.95,
        latency_s=1.0,
        result_source_path=Path("results.jsonl"),
        result_line_number=1,
    )
    return MappedClassifierRecord(record, label, origin, origin, "test")


def test_view_reports_top3_selective_and_outside_present_classes():
    rows = [
        _mapped("a", "A", [0.7, 0.1, 0.1, 0.1]),
        _mapped("b", "B", [0.4, 0.3, 0.2, 0.1]),
        _mapped("c", "A", [0.1, 0.2, 0.3, 0.4]),
    ]
    report = evaluate_classifier_view(_Model(), rows, confidence_threshold=0.65)
    assert report["accuracy"] == 1 / 3
    assert report["top3_accuracy"] == 2 / 3
    assert report["predictions_outside_present_classes"] == 1
    assert report["metric_label_space"] == ["A", "B", "D"]
    assert report["selective"]["coverage"] == 1 / 3
    assert report["selective"]["risk"] == 0.0


def test_rejected_rows_measure_confidence_without_claiming_accuracy():
    mapped = _mapped("ood", "A", [0.1, 0.1, 0.1, 0.7])
    rejected = RejectedClassifierRecord(
        record=mapped.record,
        status="out_of_taxonomy",
        reason="not_represented",
    )
    report = evaluate_rejected_attack_confidence(_Model(), [rejected])
    assert report["accepted_rate"] == 1.0
    assert report["top_predictions"] == {"D": 1}
    assert "No mide exactitud" in report["interpretation"]


def test_origin_only_baseline_learns_using_train_only():
    rows = [
        _mapped("train-a1", "A", [1, 0, 0, 0], split="train", origin="one"),
        _mapped("train-a2", "A", [1, 0, 0, 0], split="train", origin="one"),
        _mapped("train-b", "B", [0, 1, 0, 0], split="train", origin="two"),
        _mapped("test-a", "A", [1, 0, 0, 0], origin="one"),
        _mapped("test-b", "B", [0, 1, 0, 0], origin="two"),
    ]
    report = evaluate_origin_only_baseline(rows)
    assert report["accuracy"] == 1.0
    assert report["learned_majority_by_origin"] == {"one": "A", "two": "B"}


def test_threshold_is_selected_on_validation_without_test_rows():
    rows = [
        _mapped("v1", "A", [0.9, 0.05, 0.03, 0.02], split="val"),
        _mapped("v2", "B", [0.45, 0.4, 0.1, 0.05], split="val"),
    ]
    report = select_confidence_threshold(
        _Model(), rows, maximum_risk=0.05, minimum_coverage=0.5
    )
    assert report["selection_split"] == "val"
    assert report["selected"]["coverage"] == 0.5
    assert report["selected"]["risk"] == 0.0
    assert report["selected"]["threshold"] == 0.46
