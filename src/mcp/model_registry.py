# src/mcp/model_registry.py
"""Carga perezosa (lazy) y cacheada de los modelos .joblib preparados."""
from __future__ import annotations

from typing import Any

from src.mcp.common import resolve_path
from src.mcp.features import event_features


_CACHE: dict[str, Any] = {}


def load_model(key: str) -> Any:
    """Carga y cachea un modelo por clave de artefacto ('detection_model'...)."""
    if key not in _CACHE:
        import joblib

        path = resolve_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {path}")
        _CACHE[key] = joblib.load(path)
    return _CACHE[key]


def clear_cache() -> None:
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Inferencia sobre eventos canonicos
# ---------------------------------------------------------------------------

def detect(event_data: dict[str, Any]) -> dict[str, Any]:
    """Deteccion binaria con el candidato global de validation_2026."""
    model = load_model("detection_model")
    features = event_features(event_data)
    probability = float(model.predict_proba([features])[0][1])
    return {
        "is_malicious": probability >= 0.5,
        "probability": probability,
        "model_name": "xgboost_detection_validation_2026_20260822",
        "feature_count": len(features),
    }


def classify(event_data: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    """Clasificacion de familia con el candidato global de validation_2026."""
    model = load_model("family_model")
    features = event_features(event_data)
    proba = model.predict_proba([features])[0]
    classes = model.classes
    ranked = sorted(zip(classes, proba), key=lambda item: item[1], reverse=True)
    family, confidence = ranked[0]
    return {
        "attack_family": str(family),
        "confidence": float(confidence),
        "top_scores": {str(name): float(score) for name, score in ranked[:top_k]},
        "model_name": "xgboost_attack_family_validation_2026_20260822",
        "feature_count": len(features),
    }
