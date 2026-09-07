from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.mcp import model_registry


class _DetectorModel:
    task = "binary_detection"
    classes = [True, False]

    def predict_proba(self, rows):
        assert rows == [{"safe": 1.0}]
        return np.asarray([[0.8, 0.2]])


def test_detector_resolves_positive_probability_by_class_and_reports_path_name(
    monkeypatch,
):
    monkeypatch.setattr(model_registry, "load_model", lambda _key: _DetectorModel())
    monkeypatch.setattr(
        model_registry, "event_features", lambda _event: {"safe": 1.0}
    )
    monkeypatch.setattr(
        model_registry,
        "resolve_path",
        lambda _key: Path("custom-balanced-detector.joblib"),
    )

    result = model_registry.detect({})

    assert result["probability"] == pytest.approx(0.8)
    assert result["is_malicious"] is True
    assert result["model_name"] == "custom-balanced-detector"


def test_detector_prefers_explicit_model_name(monkeypatch):
    model = _DetectorModel()
    model.model_name = "detector-reviewed-release"
    monkeypatch.setattr(model_registry, "load_model", lambda _key: model)
    monkeypatch.setattr(
        model_registry, "event_features", lambda _event: {"safe": 1.0}
    )

    assert model_registry.detect({})["model_name"] == "detector-reviewed-release"


@pytest.mark.parametrize(
    ("task", "classes", "message"),
    [
        ("attack_subtype", [False, True], "binary_detection"),
        ("binary_detection", [False], "exactamente las clases"),
        ("binary_detection", [False, True, "other"], "exactamente las clases"),
    ],
)
def test_detector_rejects_incompatible_contract(task, classes, message, monkeypatch):
    model = _DetectorModel()
    model.task = task
    model.classes = classes
    monkeypatch.setattr(model_registry, "load_model", lambda _key: model)

    with pytest.raises(ValueError, match=message):
        model_registry.detect({})
