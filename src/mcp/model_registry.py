# src/mcp/model_registry.py
"""Carga perezosa (lazy) y cacheada de los modelos .joblib preparados."""
from __future__ import annotations

from typing import Any

from src.contracts.attack_taxonomy import (
    MULTIDATASET_TAXONOMY_VERSION,
    attack_classes_for_taxonomy,
)
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
    """Deteccion binaria con el artefacto balanceado por origen."""
    model = load_model("detection_model")
    if str(getattr(model, "task", "")) != "binary_detection":
        raise ValueError(
            "El detector desplegado debe declarar task='binary_detection'"
        )
    classes = tuple(getattr(model, "classes", ()))
    if (
        len(classes) != 2
        or not all(type(value) is bool for value in classes)
        or set(classes) != {False, True}
    ):
        raise ValueError(
            "El detector desplegado debe contener exactamente las clases "
            "[False, True]"
        )
    features = event_features(event_data)
    probabilities = model.predict_proba([features])[0]
    probability = float(probabilities[classes.index(True)])
    return {
        "is_malicious": probability >= 0.5,
        "probability": probability,
        "model_name": _model_name(model, resolve_path("detection_model").stem),
        "feature_count": len(features),
    }


def classify(event_data: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    """Predice directamente uno de los 16 tipos de ataque desplegados."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k debe ser un entero positivo")
    model = load_model("attack_type_model")
    classes = tuple(str(value) for value in model.classes)
    artifact_task = str(getattr(model, "task", ""))
    if artifact_task != "attack_subtype":
        raise ValueError(
            "El clasificador desplegado debe predecir tipos de ataque: "
            f"task={artifact_task!r}"
        )
    taxonomy_version = _validate_attack_type_model(model, classes)

    threshold_raw = getattr(model, "confidence_threshold", None)
    decision_threshold = 0.65 if threshold_raw is None else float(threshold_raw)
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError(
            f"Umbral de confianza invalido en el modelo: {threshold_raw!r}"
        )

    features = event_features(event_data)
    proba = model.predict_proba([features])[0]
    ranked = sorted(zip(classes, proba), key=lambda item: item[1], reverse=True)
    predicted_label, confidence = ranked[0]
    return {
        "attack_type": str(predicted_label),
        "confidence": float(confidence),
        "top_scores": {str(name): float(score) for name, score in ranked[:top_k]},
        "feature_count": len(features),
        "model_task": "attack_type",
        "decision_threshold": decision_threshold,
        "taxonomy_version": taxonomy_version,
        "model_name": _model_name(
            model, "xgboost_attack_subtype_multidataset16_balanced500_20260906"
        ),
    }


def _validate_attack_type_model(model: Any, classes: tuple[str, ...]) -> str:
    """Falla cerrado si el artefacto no implementa los 16 tipos desplegados."""

    if len(classes) != len(set(classes)):
        raise ValueError("El modelo attack_type contiene clases duplicadas")

    mapping_version = getattr(model, "family_mapping_version", None)
    if mapping_version != MULTIDATASET_TAXONOMY_VERSION:
        raise ValueError(
            "Version de taxonomia incompatible para attack_type: "
            f"recibida={mapping_version!r}, "
            f"esperada={MULTIDATASET_TAXONOMY_VERSION!r}"
        )
    expected_classes = attack_classes_for_taxonomy(mapping_version)

    expected = set(expected_classes)
    observed = set(classes)
    if len(classes) != len(expected_classes) or observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "El modelo attack_type no contiene exactamente las "
            f"{len(expected_classes)} clases de la taxonomia {mapping_version!r}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return str(mapping_version)


def _model_name(model: Any, fallback: str) -> str:
    configured = getattr(model, "model_name", None)
    if configured is None:
        return fallback
    value = str(configured).strip()
    return value or fallback
