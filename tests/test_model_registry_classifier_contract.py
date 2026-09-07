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


class _SubtypeModel:
    task = "attack_subtype"
    classes = list(JORGE_ATTACK_CLASSES)
    family_mapping_version = JORGE_TAXONOMY_VERSION

    def predict_proba(self, rows):
        assert rows == [{"safe": 1.0}]
        scores = {label: 0.0 for label in self.classes}
        scores.update({"DDoS_TCP": 0.7, "DDoS_UDP": 0.2, "XSS": 0.1})
        return np.asarray([[scores[label] for label in self.classes]])


class _FamilyModel:
    task = "attack_family"
    classes = ["ddos", "scanning"]

    def predict_proba(self, rows):
        return np.asarray([[0.8, 0.2]])


class _MultidatasetSubtypeModel:
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


def test_subtype_model_returns_distinct_subtype_and_broad_family(monkeypatch):
    monkeypatch.setattr(model_registry, "load_model", lambda _key: _SubtypeModel())
    monkeypatch.setattr(model_registry, "event_features", lambda _event: {"safe": 1.0})
    result = model_registry.classify({}, top_k=3)
    assert result["attack_subtype"] == "DDoS_TCP"
    assert result["attack_family"] == "ddos"
    assert result["confidence"] == pytest.approx(0.7)
    assert result["top_scores"] == {
        "DDoS_TCP": pytest.approx(0.7),
        "DDoS_UDP": pytest.approx(0.2),
        "XSS": pytest.approx(0.1),
    }
    assert result["family_scores"]["ddos"] == pytest.approx(0.9)
    assert result["family_confidence"] == pytest.approx(0.9)
    assert result["score_type"] == "attack_subtype"
    assert result["model_task"] == "attack_subtype"
    assert result["taxonomy_version"] == JORGE_TAXONOMY_VERSION
    assert result["decision_threshold"] == pytest.approx(0.65)
    assert result["model_name"].endswith("_20260906")


def test_multidataset_subtype_model_is_accepted_with_its_exact_contract(monkeypatch):
    monkeypatch.setattr(
        model_registry, "load_model", lambda _key: _MultidatasetSubtypeModel()
    )
    monkeypatch.setattr(model_registry, "event_features", lambda _event: {"safe": 1.0})

    result = model_registry.classify({}, top_k=3)

    assert result["attack_subtype"] == "Command_and_Control"
    assert result["attack_family"] == "botnet"
    assert result["family_confidence"] == pytest.approx(0.70)
    assert result["family_scores"] == {
        "ddos": pytest.approx(0.30),
        "botnet": pytest.approx(0.70),
        "malware": pytest.approx(0.0),
        "scanning": pytest.approx(0.0),
        "mitm": pytest.approx(0.0),
        "bruteforce": pytest.approx(0.0),
        "injection": pytest.approx(0.0),
    }
    assert result["taxonomy_version"] == MULTIDATASET_TAXONOMY_VERSION
    assert result["model_name"] == (
        "xgboost_attack_subtype_multidataset16_balanced500_20260906"
    )


def test_legacy_family_model_remains_compatible(monkeypatch):
    monkeypatch.setattr(model_registry, "load_model", lambda _key: _FamilyModel())
    monkeypatch.setattr(model_registry, "event_features", lambda _event: {})
    result = model_registry.classify({})
    assert result["attack_subtype"] is None
    assert result["attack_family"] == "ddos"
    assert result["score_type"] == "attack_family"
    assert result["model_task"] == "attack_family"
    assert result["decision_threshold"] == pytest.approx(0.65)
    assert result["model_name"] == "xgboost_attack_family_validation_2026_20260822"


def test_model_metadata_overrides_runtime_fallbacks(monkeypatch):
    model = _SubtypeModel()
    model.confidence_threshold = 0.72
    model.model_name = "classifier-subtype-release-7"
    monkeypatch.setattr(model_registry, "load_model", lambda _key: model)
    monkeypatch.setattr(model_registry, "event_features", lambda _event: {"safe": 1.0})

    result = model_registry.classify({})

    assert result["decision_threshold"] == pytest.approx(0.72)
    assert result["model_name"] == "classifier-subtype-release-7"


@pytest.mark.parametrize(
    ("classes", "version", "message"),
    [
        (
            list(JORGE_ATTACK_CLASSES[:-1]),
            JORGE_TAXONOMY_VERSION,
            "exactamente las 14 clases",
        ),
        (
            [*JORGE_ATTACK_CLASSES[:-1], JORGE_ATTACK_CLASSES[0]],
            JORGE_TAXONOMY_VERSION,
            "clases duplicadas",
        ),
        (
            [*JORGE_ATTACK_CLASSES[:-1], "Novel_Attack"],
            JORGE_TAXONOMY_VERSION,
            "exactamente las 14 clases",
        ),
        (
            list(JORGE_ATTACK_CLASSES),
            None,
            "Version de taxonomia incompatible",
        ),
        (
            list(JORGE_ATTACK_CLASSES),
            "jorge_taxonomy_obsoleta",
            "Version de taxonomia incompatible",
        ),
        (
            list(JORGE_ATTACK_CLASSES),
            MULTIDATASET_TAXONOMY_VERSION,
            "exactamente las 16 clases",
        ),
    ],
)
def test_subtype_model_requires_exact_unique_classes_and_taxonomy_version(
    classes, version, message, monkeypatch
):
    model = _SubtypeModel()
    model.classes = classes
    model.family_mapping_version = version
    monkeypatch.setattr(model_registry, "load_model", lambda _key: model)
    monkeypatch.setattr(model_registry, "event_features", lambda _event: {"safe": 1.0})

    with pytest.raises(ValueError, match=message):
        model_registry.classify({})


def test_get_family_scores_aggregates_subtype_model_scores(monkeypatch):
    received = {}

    def fake_classify(event, top_k):
        received.update({"event": event, "top_k": top_k})
        return {
            "attack_subtype": "DDoS_TCP",
            "attack_family": "ddos",
            "confidence": 0.7,
            "top_scores": {
                "DDoS_TCP": 0.7,
                "DDoS_UDP": 0.2,
                "XSS": 0.1,
            },
            "family_scores": {"ddos": 0.9, "injection": 0.1},
            "score_type": "attack_subtype",
        }

    monkeypatch.setattr(
        inference_server, "_validated_canonical_payload", lambda event: event
    )
    monkeypatch.setattr(inference_server.model_registry, "classify", fake_classify)

    result = inference_server.get_family_scores({"event_id": "evt"})

    assert result["ok"] is True
    assert received == {"event": {"event_id": "evt"}, "top_k": 100}
    assert result["attack_subtype"] is None
    assert result["attack_family"] == "ddos"
    assert result["confidence"] == pytest.approx(0.9)
    assert result["top_scores"] == {"ddos": 0.9, "injection": 0.1}
    assert result["family_scores"] == result["top_scores"]
    assert result["score_type"] == "attack_family"
    assert "DDoS_TCP" not in result["top_scores"]


def test_get_family_scores_keeps_legacy_family_scores(monkeypatch):
    monkeypatch.setattr(
        inference_server, "_validated_canonical_payload", lambda event: event
    )
    monkeypatch.setattr(
        inference_server.model_registry,
        "classify",
        lambda _event, top_k: {
            "attack_subtype": None,
            "attack_family": "ddos",
            "confidence": 0.8,
            "top_scores": {"ddos": 0.8, "scanning": 0.2},
            "score_type": "attack_family",
        },
    )

    result = inference_server.get_family_scores({"event_id": "legacy"})

    assert result["ok"] is True
    assert result["attack_subtype"] is None
    assert result["attack_family"] == "ddos"
    assert result["confidence"] == pytest.approx(0.8)
    assert result["top_scores"] == {"ddos": 0.8, "scanning": 0.2}
    assert result["family_scores"] == result["top_scores"]
    assert result["score_type"] == "attack_family"


def test_get_family_scores_uses_aggregate_argmax_not_top_subtype_parent(monkeypatch):
    monkeypatch.setattr(
        inference_server, "_validated_canonical_payload", lambda event: event
    )
    monkeypatch.setattr(
        inference_server.model_registry,
        "classify",
        lambda _event, top_k: {
            "attack_subtype": "XSS",
            "attack_family": "injection",
            "confidence": 0.4,
            "top_scores": {
                "XSS": 0.4,
                "DDoS_TCP": 0.35,
                "DDoS_UDP": 0.25,
            },
            "family_scores": {"ddos": 0.6, "injection": 0.4},
            "score_type": "attack_subtype",
        },
    )

    result = inference_server.get_family_scores({"event_id": "aggregate"})

    assert result["attack_subtype"] is None
    assert result["attack_family"] == "ddos"
    assert result["confidence"] == pytest.approx(0.6)
    assert result["top_scores"] == {"ddos": 0.6, "injection": 0.4}


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_top_k_must_be_positive_integer(top_k, monkeypatch):
    monkeypatch.setattr(model_registry, "load_model", lambda _key: _FamilyModel())
    with pytest.raises(ValueError, match="entero positivo"):
        model_registry.classify({}, top_k=top_k)
