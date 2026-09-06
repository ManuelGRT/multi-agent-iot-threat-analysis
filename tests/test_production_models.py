from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from src.eval.production_models import (
    BASE_XGBOOST_PARAMETERS,
    FAMILY_MAPPING_VERSION,
    ProductionXGBoostModel,
    UnmappedAttackLabelError,
    deduplicate_labelled_records,
    family_mapping_report,
    map_attack_family,
    persist_candidate_model,
    train_labelled_attack_subtype_classifier,
    train_global_detector,
    train_production_candidates,
)
from src.eval.validation_campaign import PreparedRecord


def _record(
    manifest_id: str,
    *,
    dataset: str = "edge_iiotset",
    split: str = "train",
    target_class: str = "Normal",
    is_attack: bool = False,
    signal: float = 0.0,
    fingerprint: str | None = None,
    extra_features: Mapping[str, Any] | None = None,
) -> PreparedRecord:
    features = {"a_signal": signal}
    features.update(extra_features or {})
    return PreparedRecord(
        manifest_id=manifest_id,
        dataset=dataset,
        source_file=f"{dataset}.csv",
        row_id=manifest_id,
        content_hash=f"content-{manifest_id}",
        split=split,
        tasks=("binary", "multiclass"),
        target_class=target_class,
        target_is_attack=is_attack,
        event=None,  # Production training consumes PreparedRecord.features only.
        features=features,
        feature_fingerprint=fingerprint or f"fp-{manifest_id}",
        provider="mistral",
        model="mistral-small-2603",
        mapping_confidence=0.95,
        latency_s=1.0,
        result_source_path=Path("results.jsonl"),
        result_line_number=1,
    )


def _candidate_records(*, repetitions: int = 2) -> list[PreparedRecord]:
    rows: list[PreparedRecord] = []
    labels = (
        ("Normal", False, 0.0),
        ("DDoS_UDP", True, 0.0),
        ("Port_Scanning", True, 1.0),
    )
    for split in ("train", "val", "test"):
        for class_index, (target_class, is_attack, signal) in enumerate(labels):
            for repetition in range(repetitions):
                manifest_id = f"{split}-{class_index}-{repetition}"
                extras = {"unseen_val_feature": 99.0} if split == "val" else {}
                rows.append(
                    _record(
                        manifest_id,
                        split=split,
                        target_class=target_class,
                        is_attack=is_attack,
                        signal=signal,
                        extra_features=extras,
                    )
                )
    return rows


class _FakeEncodedEstimator:
    def __init__(self, *, task: str, n_classes: int):
        self.task = task
        self.n_classes = n_classes
        self.fit_rows = 0
        self.fit_targets: list[int] = []

    def fit(self, x, y):
        self.fit_rows = int(x.shape[0])
        self.fit_targets = np.asarray(y, dtype=int).tolist()
        return self

    def predict(self, x):
        signal = np.asarray(x[:, 0].todense()).reshape(-1)
        if self.task == "binary_detection":
            return (signal >= 0.5).astype(int)
        return np.clip(np.rint(signal), 0, self.n_classes - 1).astype(int)

    def predict_proba(self, x):
        prediction = self.predict(x)
        probabilities = np.zeros((len(prediction), self.n_classes), dtype=float)
        probabilities[np.arange(len(prediction)), prediction] = 1.0
        return probabilities


class _CapturingFactory:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, task, n_classes, parameters):
        self.calls.append(
            {
                "task": task,
                "n_classes": n_classes,
                "parameters": dict(parameters),
            }
        )
        return _FakeEncodedEstimator(task=task, n_classes=n_classes)


def test_versioned_mapping_covers_every_declared_native_family():
    expected = {
        ("edge_iiotset", "DDoS_HTTP"): "ddos",
        ("edge_iiotset", "SQL_injection"): "injection",
        ("edge_iiotset", "Uploading"): "injection",
        ("edge_iiotset", "Password"): "bruteforce",
        ("edge_iiotset", "Fingerprinting"): "scanning",
        ("edge_iiotset", "Backdoor"): "malware",
        ("edge_iiotset", "MITM"): "mitm",
        ("ton_iot", "dos"): "ddos",
        ("ton_iot", "ddos"): "ddos",
        ("ton_iot", "scanning"): "scanning",
        ("ton_iot", "password"): "bruteforce",
        ("ton_iot", "injection"): "injection",
        ("ton_iot", "xss"): "injection",
        ("ton_iot", "backdoor"): "malware",
        ("ton_iot", "ransomware"): "malware",
        ("ton_iot", "mitm"): "mitm",
        ("bot_iot", "DoS"): "ddos",
        ("bot_iot", "DDoS"): "ddos",
        ("bot_iot", "Reconnaissance"): "scanning",
        ("bot_iot", "Theft"): "exfiltration",
        ("iot23", "C&C"): "botnet",
        ("iot23", "C&C-FileDownload"): "botnet",
        ("iot23", "C&C-Torii"): "botnet",
        ("iot23", "PartOfAHorizontalPortScan"): "scanning",
        ("iot23", "DDoS"): "ddos",
        ("iot23", "FileDownload"): "unknown_attack",
    }
    for (dataset, native_label), family in expected.items():
        assert map_attack_family(dataset, native_label) == family

    snapshot = family_mapping_report()
    assert snapshot["version"] == FAMILY_MAPPING_VERSION
    assert len(snapshot["sha256"]) == 64
    with pytest.raises(UnmappedAttackLabelError, match="sin mapear"):
        map_attack_family("ton_iot", "future_attack")


def test_global_task_dedup_removes_cross_split_and_conflicts_and_keeps_minimum_id():
    labelled = [
        (_record("z-safe", fingerprint="safe"), True),
        (_record("a-safe", fingerprint="safe"), True),
        (_record("cross-train", split="train", fingerprint="cross"), False),
        (_record("cross-test", split="test", fingerprint="cross"), False),
        (_record("label-true", fingerprint="conflict"), True),
        (_record("label-false", fingerprint="conflict"), False),
        (_record("unique", split="val", fingerprint="unique"), True),
    ]

    kept, report = deduplicate_labelled_records(labelled)

    assert [record.manifest_id for record, _label in kept] == ["a-safe", "unique"]
    assert report["same_split_same_label_groups"] == 1
    assert report["same_split_rows_removed"] == 1
    assert report["cross_split_groups"] == 1
    assert report["conflicting_label_groups"] == 1
    assert report["unsafe_rows_removed"] == 4
    assert report["final_cross_split_overlap_count"] == 0


def test_fake_factory_preserves_splits_fits_vectorizer_on_train_and_reports_metrics():
    factory = _CapturingFactory()
    result = train_production_candidates(
        _candidate_records(), estimator_factory=factory
    )

    assert [call["task"] for call in factory.calls] == [
        "binary_detection",
        "attack_family",
    ]
    for call in factory.calls:
        parameters = call["parameters"]
        assert parameters["n_estimators"] == 300
        assert parameters["max_depth"] == 6
        assert parameters["learning_rate"] == 0.1
        assert parameters["subsample"] == 0.8
        assert parameters["colsample_bytree"] == 0.8
        assert parameters["random_state"] == 42
        assert parameters["n_jobs"] == 1

    detector = result.detector
    classifier = result.family_classifier
    assert detector.artifact_path is None
    assert classifier.artifact_path is None
    assert detector.report["artifact"] is None
    assert classifier.report["artifact"] is None
    assert detector.report["supports"]["train"]["rows"] == 6
    assert classifier.report["supports"]["train"]["rows"] == 4
    assert "unseen_val_feature" not in detector.model.vectorizer.vocabulary_
    assert "unseen_val_feature" not in classifier.model.vectorizer.vocabulary_
    assert set(detector.report["metrics"]) == {"val", "test"}
    assert set(detector.report["metrics"]["test"]["by_dataset"]) == {
        "edge_iiotset"
    }
    assert len(detector.report["data_sha256"]["train"]) == 64
    assert classifier.report["family_mapping"]["version"] == FAMILY_MAPPING_VERSION


def test_explicit_path_is_atomic_joblib_candidate_compatible_with_registry(tmp_path):
    joblib = pytest.importorskip("joblib")
    factory = _CapturingFactory()
    target = tmp_path / "nested" / "detector-candidate.joblib"

    outcome = train_global_detector(
        _candidate_records(), model_path=target, estimator_factory=factory
    )

    assert outcome.artifact_path == target.resolve()
    assert outcome.report["artifact"]["path"] == str(target.resolve())
    assert len(outcome.report["artifact"]["sha256"]) == 64
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))
    loaded = joblib.load(target)
    assert isinstance(loaded, ProductionXGBoostModel)
    assert loaded.classes == [False, True]
    assert loaded.predict([{"a_signal": 0.0}, {"a_signal": 1.0}]) == [False, True]
    assert loaded.predict_proba([{"a_signal": 0.0}]).shape == (1, 2)


def test_subtype_training_validates_declared_classes_and_persists_after_review(tmp_path):
    labelled = []
    for split in ("train", "val", "test"):
        labelled.extend(
            [
                (_record(f"{split}-a", split=split, signal=0.0), "A"),
                (_record(f"{split}-b", split=split, signal=1.0), "B"),
            ]
        )
    factory = _CapturingFactory()
    result = train_labelled_attack_subtype_classifier(
        labelled,
        taxonomy_mapping={"version": "test-v1", "training_classes": ["A", "B"]},
        estimator_factory=factory,
        seed=17,
    )
    assert result.model.task == "attack_subtype"
    assert result.report["parameters"]["random_state"] == 17
    assert factory.calls[-1]["parameters"]["random_state"] == 17
    assert result.artifact_path is None
    target = tmp_path / "reviewed.joblib"
    persist_candidate_model(
        result,
        target,
        confidence_threshold=0.81,
        model_name="xgboost_attack_subtype_jorge14_balanced_20260906",
    )
    assert result.artifact_path == target.resolve()
    assert result.model.confidence_threshold == pytest.approx(0.81)
    assert (
        result.model.model_name
        == "xgboost_attack_subtype_jorge14_balanced_20260906"
    )
    assert result.report["operational_contract"] == {
        "confidence_threshold": pytest.approx(0.81),
        "model_name": "xgboost_attack_subtype_jorge14_balanced_20260906",
        "task": "attack_subtype",
        "taxonomy_version": "test-v1",
    }
    with pytest.raises(ValueError, match="ya fue persistido"):
        persist_candidate_model(result, tmp_path / "second.joblib")

    with pytest.raises(ValueError, match="no coinciden"):
        train_labelled_attack_subtype_classifier(
            labelled,
            taxonomy_mapping={"version": "test-v1", "training_classes": ["A"]},
            estimator_factory=factory,
        )


def test_small_real_xgboost_smoke_trains_and_serializes_both_candidates(tmp_path):
    pytest.importorskip("xgboost")
    joblib = pytest.importorskip("joblib")
    detector_path = tmp_path / "detector.joblib"
    family_path = tmp_path / "family.joblib"

    result = train_production_candidates(
        _candidate_records(repetitions=2),
        detector_path=detector_path,
        family_path=family_path,
        parameter_overrides={"n_estimators": 2, "max_depth": 2},
    )

    assert result.detector.report["parameters"]["n_estimators"] == 2
    assert result.family_classifier.report["parameters"]["n_estimators"] == 2
    assert result.detector.model.predict_proba([{"a_signal": 0.0}]).shape == (1, 2)
    assert result.family_classifier.model.predict_proba([{"a_signal": 1.0}]).shape == (
        1,
        2,
    )
    assert joblib.load(detector_path).classes == [False, True]
    assert joblib.load(family_path).classes == ["ddos", "scanning"]
    assert BASE_XGBOOST_PARAMETERS["n_estimators"] == 300
