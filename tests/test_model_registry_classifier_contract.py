from __future__ import annotations

import numpy as np
import pytest

from src.contracts.attack_taxonomy import (
    JORGE_ATTACK_CLASSES,
    JORGE_TAXONOMY_VERSION,
    MULTIDATASET_ATTACK_CLASSES,
    MULTIDATASET_TAXONOMY_VERSION,
)
from src.mcp import inference_server, model_registry


class _AttackTypeModel:
    # Metadatos internos del joblib existente. La API los traduce a
    # ``model_task=attack_type`` y no expone familias.
    task = "attack_subtype"
    classes = list(MULTIDATASET_ATTACK_CLASSES)
    family_mapping_version = MULTIDATASET_TAXONOMY_VERSION

    def predict_proba(self, rows):
        assert rows == [{"safe": 1.0}]
        scores = {label: 0.0 for label in self.classes}
        scores.update(
            {"Command_and_Control": 0.70, "DoS": 0.20, "DDoS_TCP": 0.10}
        )
        return np.asarray([[scores[label] for label in self.classes]])


class _LegacyFamilyModel:
    task = "attack_family"
    classes = ["ddos", "scanning"]

    def predict_proba(self, rows):
        return np.asarray([[0.8, 0.2]])


def test_attack_type_model_returns_type_and_top_scores_only(monkeypatch):
    monkeypatch.setattr(model_registry, "load_model", lambda _key: _AttackTypeModel())
    monkeypatch.setattr(model_registry, "event_features", lambda _event: {"safe": 1.0})

    result = model_registry.classify({}, top_k=3)

    assert result["attack_type"] == "Command_and_Control"
    assert result["confidence"] == pytest.approx(0.70)
    assert result["top_scores"] == {
        "Command_and_Control": pytest.approx(0.70),
        "DoS": pytest.approx(0.20),
        "DDoS_TCP": pytest.approx(0.10),
    }
    assert result["model_task"] == "attack_type"
    assert result["taxonomy_version"] == MULTIDATASET_TAXONOMY_VERSION
    assert result["decision_threshold"] == pytest.approx(0.65)
    assert result["model_name"] == (
        "xgboost_attack_subtype_multidataset16_balanced500_20260906"
    )
    assert not {"attack_family", "family_confidence", "family_scores"} & result.keys()


def test_legacy_family_model_is_rejected(monkeypatch):
    monkeypatch.setattr(
        model_registry, "load_model", lambda _key: _LegacyFamilyModel()
    )
    with pytest.raises(ValueError, match="debe predecir tipos de ataque"):
        model_registry.classify({})


def test_model_metadata_overrides_runtime_fallbacks(monkeypatch):
    model = _AttackTypeModel()
    model.confidence_threshold = 0.72
    model.model_name = "classifier-attack-type-release-7"
    monkeypatch.setattr(model_registry, "load_model", lambda _key: model)
    monkeypatch.setattr(model_registry, "event_features", lambda _event: {"safe": 1.0})

    result = model_registry.classify({})

    assert result["decision_threshold"] == pytest.approx(0.72)
    assert result["model_name"] == "classifier-attack-type-release-7"


@pytest.mark.parametrize(
    ("classes", "version", "message"),
    [
        (
            list(MULTIDATASET_ATTACK_CLASSES[:-1]),
            MULTIDATASET_TAXONOMY_VERSION,
            "exactamente las 16 clases",
        ),
        (
            [*MULTIDATASET_ATTACK_CLASSES[:-1], MULTIDATASET_ATTACK_CLASSES[0]],
            MULTIDATASET_TAXONOMY_VERSION,
            "clases duplicadas",
        ),
        (
            [*MULTIDATASET_ATTACK_CLASSES[:-1], "Novel_Attack"],
            MULTIDATASET_TAXONOMY_VERSION,
            "exactamente las 16 clases",
        ),
        (list(MULTIDATASET_ATTACK_CLASSES), None, "Version de taxonomia incompatible"),
        (
            list(MULTIDATASET_ATTACK_CLASSES),
            JORGE_TAXONOMY_VERSION,
            "Version de taxonomia incompatible",
        ),
        (
            list(JORGE_ATTACK_CLASSES),
            MULTIDATASET_TAXONOMY_VERSION,
            "exactamente las 16 clases",
        ),
    ],
)
def test_model_requires_exact_16_unique_classes_and_taxonomy_version(
    classes, version, message, monkeypatch
):
    model = _AttackTypeModel()
    model.classes = classes
    model.family_mapping_version = version
    monkeypatch.setattr(model_registry, "load_model", lambda _key: model)
    monkeypatch.setattr(model_registry, "event_features", lambda _event: {"safe": 1.0})

    with pytest.raises(ValueError, match=message):
        model_registry.classify({})


def test_classify_event_exposes_attack_type_contract(monkeypatch):
    received = {}

    def fake_classify(event, top_k):
        received.update({"event": event, "top_k": top_k})
        return {
            "attack_type": "DDoS_TCP",
            "confidence": 0.7,
            "top_scores": {
                "DDoS_TCP": 0.7,
                "DDoS_UDP": 0.2,
                "XSS": 0.1,
            },
            "model_task": "attack_type",
        }

    monkeypatch.setattr(
        inference_server, "_validated_canonical_payload", lambda event: event
    )
    monkeypatch.setattr(inference_server.model_registry, "classify", fake_classify)

    result = inference_server.classify_event({"event_id": "evt"}, top_k=3)

    assert result["ok"] is True
    assert received == {"event": {"event_id": "evt"}, "top_k": 3}
    assert result["attack_type"] == "DDoS_TCP"
    assert "attack_family" not in result


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_top_k_must_be_positive_integer(top_k, monkeypatch):
    monkeypatch.setattr(model_registry, "load_model", lambda _key: _AttackTypeModel())
    with pytest.raises(ValueError, match="entero positivo"):
        model_registry.classify({}, top_k=top_k)
