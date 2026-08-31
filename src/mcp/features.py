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
from src.mcp.standardization_guard import sanitize_canonical_event

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
    # Defensa final para llamadas MCP directas que no hayan atravesado el
    # grafo. La politica vive fuera de los agentes y falla cerrado al validar.
    event = CanonicalEvent(**sanitize_canonical_event(event_data))
    features = _STANDARDIZER.event_to_features(event)
    clean: dict[str, Any] = {}
    for key, value in features.items():
        key_text = str(key)
        if is_predictive_target_field(key_text):
            continue
        if key_text.startswith(_FORBIDDEN_PREFIXES):
            continue
        if key_text in _FORBIDDEN_KEYS:
            continue
        clean[key_text] = value
    return clean
