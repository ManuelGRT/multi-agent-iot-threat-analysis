# src/mcp/model_registry.py
"""Carga perezosa (lazy) y cacheada de los modelos .joblib preparados."""
from __future__ import annotations

from typing import Any

from src.contracts.attack_taxonomy import (
    MULTIDATASET_TAXONOMY_VERSION,
    SUPPORTED_ATTACK_TAXONOMY_VERSIONS,
    attack_classes_for_taxonomy,
    broad_family_for_attack_type,
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
    """Clasifica con el artefacto activo, compatible con familia o subtipo."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k debe ser un entero positivo")
    model = load_model("family_model")
    classes = tuple(str(value) for value in model.classes)
    task = str(getattr(model, "task", "attack_family"))
    taxonomy_version: str | None = None
    if task == "attack_subtype":
        taxonomy_version = _validate_attack_subtype_model(model, classes)
    elif task != "attack_family":
        raise ValueError(f"Tarea de clasificacion no soportada: {task!r}")

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
    response = {
        "confidence": float(confidence),
        "top_scores": {str(name): float(score) for name, score in ranked[:top_k]},
        "feature_count": len(features),
        "model_task": task,
        "decision_threshold": decision_threshold,
    }
    if task == "attack_subtype":
        subtype = str(predicted_label)
        family = broad_family_for_attack_type(subtype)
        family_scores: dict[str, float] = {}
        for name, score in ranked:
            broad = broad_family_for_attack_type(str(name))
            family_scores[broad] = family_scores.get(broad, 0.0) + float(score)
        return {
            **response,
            "attack_subtype": subtype,
            "attack_family": family,
            "family_confidence": family_scores[family],
            "score_type": "attack_subtype",
            "taxonomy_version": taxonomy_version,
            "family_scores": dict(
                sorted(family_scores.items(), key=lambda item: item[1], reverse=True)
            ),
            "model_name": _model_name(
                model,
                (
                    "xgboost_attack_subtype_multidataset16_balanced500_20260906"
                    if taxonomy_version == MULTIDATASET_TAXONOMY_VERSION
                    else "xgboost_attack_subtype_jorge14_balanced_20260906"
                ),
            ),
        }
    return {
        **response,
        "attack_subtype": None,
        "attack_family": str(predicted_label),
        "family_confidence": float(confidence),
        "score_type": "attack_family",
        "taxonomy_version": getattr(model, "family_mapping_version", None),
        "model_name": _model_name(
            model, "xgboost_attack_family_validation_2026_20260822"
        ),
    }


def _validate_attack_subtype_model(model: Any, classes: tuple[str, ...]) -> str:
    """Falla cerrado si el artefacto no implementa la taxonomia desplegada."""

    if len(classes) != len(set(classes)):
        raise ValueError("El modelo attack_subtype contiene clases duplicadas")

    mapping_version = getattr(model, "family_mapping_version", None)
    try:
        expected_classes = attack_classes_for_taxonomy(mapping_version)
    except ValueError as exc:
        raise ValueError(
            "Version de taxonomia incompatible para attack_subtype: "
            f"recibida={mapping_version!r}, "
            f"soportadas={list(SUPPORTED_ATTACK_TAXONOMY_VERSIONS)!r}"
        ) from exc

    expected = set(expected_classes)
    observed = set(classes)
    if len(classes) != len(expected_classes) or observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "El modelo attack_subtype no contiene exactamente las "
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
