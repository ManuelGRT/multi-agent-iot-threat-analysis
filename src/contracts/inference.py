"""Contrato ligero y estable de los modelos desplegados en inferencia.

Este modulo no depende de los pipelines de entrenamiento ni de evaluacion. Su
ruta forma parte del formato persistido de los artefactos ``joblib`` activos,
por lo que debe mantenerse estable mientras esos artefactos sigan desplegados.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np


ProductionTask = Literal["binary_detection", "attack_family", "attack_subtype"]


class ProductionModelError(ValueError):
    """El modelo desplegado o sus entradas incumplen el contrato de inferencia."""


@dataclass(slots=True)
class ProductionXGBoostModel:
    """Envoltorio estable que decodifica las predicciones del estimador.

    ``vectorizer`` y ``estimator`` se mantienen deliberadamente como objetos
    opacos. Asi, cargar el contrato no arrastra modulos de entrenamiento y solo
    requiere las dependencias que ya utiliza la inferencia.
    """

    vectorizer: Any
    estimator: Any
    encoded_classes: tuple[bool | str, ...]
    task: ProductionTask
    feature_schema_sha256: str
    family_mapping_version: str | None = None
    confidence_threshold: float | None = None
    model_name: str | None = None

    @property
    def classes(self) -> list[bool | str]:
        return list(self.encoded_classes)

    @property
    def classes_(self) -> np.ndarray:
        """Alias compatible con sklearn; el registro usa ``classes``."""

        return np.asarray(self.encoded_classes, dtype=object)

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> list[bool | str]:
        materialised = _materialise_inference_rows(rows)
        encoded = _validated_encoded_predictions(
            self.estimator.predict(self.vectorizer.transform(materialised)),
            expected_size=len(materialised),
            n_classes=len(self.encoded_classes),
            context="inference",
        )
        return [self.encoded_classes[index] for index in encoded]

    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        materialised = _materialise_inference_rows(rows)
        method = getattr(self.estimator, "predict_proba", None)
        if not callable(method):
            raise TypeError("El estimador serializado no implementa predict_proba")
        probabilities = np.asarray(
            method(self.vectorizer.transform(materialised)), dtype=float
        )
        expected = (len(materialised), len(self.encoded_classes))
        if probabilities.shape != expected:
            raise ProductionModelError(
                f"predict_proba devolvio shape={probabilities.shape}; esperado={expected}"
            )
        if not np.all(np.isfinite(probabilities)):
            raise ProductionModelError("predict_proba devolvio NaN/Inf")
        return probabilities


def _materialise_inference_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    materialised = [dict(row) for row in rows]
    if not materialised:
        raise ValueError("La inferencia requiere al menos una fila")
    return materialised


def _validated_encoded_predictions(
    values: Any,
    *,
    expected_size: int,
    n_classes: int,
    context: str,
) -> np.ndarray:
    predictions = np.asarray(values)
    if predictions.ndim != 1 or len(predictions) != expected_size:
        raise ProductionModelError(
            f"{context}: predict devolvio shape={predictions.shape}; "
            f"esperado=({expected_size},)"
        )
    try:
        numeric = predictions.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ProductionModelError(
            f"{context}: predict no devolvio indices enteros"
        ) from exc
    if not np.all(predictions == numeric):
        raise ProductionModelError(f"{context}: predict no devolvio indices enteros")
    if np.any(numeric < 0) or np.any(numeric >= n_classes):
        raise ProductionModelError(f"{context}: predict devolvio una clase fuera de rango")
    return numeric


__all__ = [
    "ProductionModelError",
    "ProductionTask",
    "ProductionXGBoostModel",
]
