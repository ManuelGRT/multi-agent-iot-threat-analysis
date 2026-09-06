# src/mcp/features.py
"""Featurizacion de eventos canonicos para los modelos preparados.

Reproduce EXACTAMENTE ``event_features`` de
``scripts/train_xgboost_detection_standardized_datasets.py`` (usada tambien por
el entrenamiento del clasificador multiclase). Cualquier cambio aqui rompe la
paridad entrenamiento/inferencia; el test ``test_mcp_servers.py::
test_feature_parity_with_training_script`` la vigila.
"""
from __future__ import annotations

from typing import Any

from src.contracts.canonical import CanonicalEvent
from src.mcp.feature_standardizer import GeneralizedFeatureStandardizer

_STANDARDIZER = GeneralizedFeatureStandardizer(include_origin=False, feature_set="full")
_FORBIDDEN_PREFIXES = (
    "origin.",
    "attack_indicator.",
    "semantic_hint.",
    "behavior.",
    "uncertainty.",
    "asset_context.",
)
_FORBIDDEN_KEYS = {
    "attack_indicator_count",
    "behavior_tag_count",
    "uncertainty_count",
}


def event_features(event_data: dict[str, Any] | CanonicalEvent) -> dict[str, Any]:
    # Proyeccion fija compartida con el entrenamiento. La entrada operacional
    # ya debe estar libre de targets; aqui no se elimina ni reescribe ningun
    # campo del evento.
    event = (
        event_data
        if isinstance(event_data, CanonicalEvent)
        else CanonicalEvent(**event_data)
    )
    features = _STANDARDIZER.event_to_features(event)
    clean: dict[str, Any] = {}
    for key, value in features.items():
        key_text = str(key)
        if key_text.startswith(_FORBIDDEN_PREFIXES):
            continue
        if key_text in _FORBIDDEN_KEYS:
            continue
        clean[key_text] = value
    return clean
