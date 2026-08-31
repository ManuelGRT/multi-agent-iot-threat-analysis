# src/mcp/model_registry.py
"""Carga perezosa (lazy) y cacheada de los modelos .joblib preparados.

Incluye el shim ``EncodedClassifier``: el clasificador de familia se serializo
desde el script de entrenamiento (``__main__.EncodedClassifier``), por lo que
al deserializar desde otro proceso hay que ofrecer una clase compatible en las
rutas de modulo que pickle pueda buscar.
"""
from __future__ import annotations

import sys
from typing import Any

from src.mcp.common import resolve_path
from src.mcp.features import event_features


class EncodedClassifier:
    """Shim compatible con la clase del script de entrenamiento.

    Solo necesita los atributos que pickle restaura (``pipeline`` y
    ``encoder``) y los metodos usados en inferencia.
    """

    def __init__(self, pipeline: Any = None):
        self.pipeline = pipeline
        self.encoder = None

    def predict(self, x: list[dict[str, Any]]) -> list[str]:
        encoded = self.pipeline.predict(x)
        return [str(item) for item in self.encoder.inverse_transform(encoded)]

    def predict_proba(self, x: list[dict[str, Any]]):
        return self.pipeline.predict_proba(x)

    @property
    def classes(self) -> list[str]:
        return [str(item) for item in self.encoder.classes_]


def _register_shims() -> None:
    """Expone EncodedClassifier alli donde el pickle pueda buscarlo."""
    for module_name in ("__main__", "train_xgboost_attack_family_balanced_standardized"):
        module = sys.modules.get(module_name)
        if module is None:
            import types

            module = types.ModuleType(module_name)
            sys.modules[module_name] = module
        if not hasattr(module, "EncodedClassifier"):
            module.EncodedClassifier = EncodedClassifier


_CACHE: dict[str, Any] = {}


def load_model(key: str) -> Any:
    """Carga y cachea un modelo por clave de artefacto ('detection_model'...)."""
    if key not in _CACHE:
        import joblib

        _register_shims()
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
