# src/mcp/features.py
"""Featurizacion de eventos canonicos para los modelos preparados.

Reproduce EXACTAMENTE ``event_features`` de
``scripts/train_xgboost_detection_standardized_datasets.py`` (usada tambien por
el entrenamiento del clasificador de familia). Cualquier cambio aqui rompe la
paridad entrenamiento/inferencia; el test ``test_mcp_servers.py::
test_feature_parity_with_training_script`` la vigila.
"""
from __future__ import annotations

from typing import Any

from src.agents.predictive_sanitization import is_predictive_target_field
from src.agents.supervised_general import GeneralizedFeatureStandardizer
from src.contracts.canonical import CanonicalEvent

_STANDARDIZER = GeneralizedFeatureStandardizer(include_origin=False, feature_set="full")


def event_features(event_data: dict[str, Any] | CanonicalEvent) -> dict[str, Any]:
    event = event_data if isinstance(event_data, CanonicalEvent) else CanonicalEvent(**event_data)
    features = _STANDARDIZER.event_to_features(event)
    clean: dict[str, Any] = {}
    for key, value in features.items():
        key_text = str(key)
        if is_predictive_target_field(key_text):
            continue
        if key_text.startswith(("origin.", "attack_indicator.", "semantic_hint.")):
            continue
        if key_text in {"attack_indicator_count"}:
            continue
        clean[key_text] = value
    return clean
