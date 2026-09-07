"""Metricas especificas del clasificador de tipos de ataque."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.contracts.inference import ProductionXGBoostModel
from src.eval.classifier_balancing import (
    MappedClassifierRecord,
    RejectedClassifierRecord,
)


def evaluate_classifier_view(
    model: ProductionXGBoostModel,
    records: Sequence[MappedClassifierRecord] | Iterable[MappedClassifierRecord],
    *,
    confidence_threshold: float = 0.65,
    complete_label_space: bool = False,
) -> dict[str, Any]:
    """Evalua una vista cerrada, incluyendo top-3 y riesgo selectivo."""

    values = tuple(records)
    if not values:
        raise ValueError("La evaluacion requiere al menos una fila")
    _validate_threshold(confidence_threshold)
    model_classes = tuple(str(value) for value in model.classes)
    truth = np.asarray([item.label for item in values], dtype=object)
    unknown_truth = sorted(set(truth.tolist()) - set(model_classes))
    if unknown_truth:
        raise ValueError(f"Etiquetas ausentes del modelo: {unknown_truth}")

    features = [dict(item.record.features) for item in values]
    predicted = np.asarray(model.predict(features), dtype=object)
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.shape != (len(values), len(model_classes)):
        raise ValueError("Matriz de probabilidades incompatible con la vista")
    confidence = probabilities.max(axis=1)
    decided = confidence >= confidence_threshold
    top_k = min(3, len(model_classes))
    top_indices = np.argsort(probabilities, axis=1)[:, -top_k:]
    class_to_index = {label: index for index, label in enumerate(model_classes)}
    truth_indices = np.asarray([class_to_index[str(label)] for label in truth])
    top_hits = np.asarray(
        [truth_indices[index] in top_indices[index] for index in range(len(values))],
        dtype=bool,
    )

    present_truth = tuple(sorted(set(str(label) for label in truth)))
    metric_labels = (
        model_classes
        if complete_label_space
        else tuple(sorted(set(truth.tolist()).union(predicted.tolist())))
    )
    metrics = _label_metrics(truth, predicted, labels=metric_labels)
    confusion_labels = model_classes
    matrix = confusion_matrix(truth, predicted, labels=confusion_labels)
    outside = np.asarray([str(label) not in present_truth for label in predicted])

    selective: dict[str, Any] = {
        "threshold": confidence_threshold,
        "decided_rows": int(decided.sum()),
        "abstained_rows": int((~decided).sum()),
        "coverage": float(decided.mean()),
        "risk": (
            float(np.mean(predicted[decided] != truth[decided]))
            if np.any(decided)
            else None
        ),
    }
    if np.any(decided):
        selective["accuracy"] = float(
            accuracy_score(truth[decided], predicted[decided])
        )
        decided_labels = (
            model_classes
            if complete_label_space
            else tuple(sorted(set(str(label) for label in truth[decided])))
        )
        selective["macro_f1"] = _label_metrics(
            truth[decided], predicted[decided], labels=decided_labels
        )["macro"]["f1"]
    else:
        selective.update({"accuracy": None, "macro_f1": None})

    return {
        "rows": len(values),
        "classes_present": list(present_truth),
        "metric_label_space": list(metric_labels),
        "class_support": dict(sorted(Counter(map(str, truth)).items())),
        "balanced_within_present_classes": len(set(Counter(truth).values())) == 1,
        **metrics,
        "top3_accuracy": float(top_hits.mean()),
        "predictions_outside_present_classes": int(outside.sum()),
        "predictions_outside_present_classes_rate": float(outside.mean()),
        "confidence": _confidence_summary(confidence),
        "selective": selective,
        "confusion_matrix": {
            "labels": list(confusion_labels),
            "matrix": matrix.astype(int).tolist(),
        },
    }


def evaluate_rejected_attack_confidence(
    model: ProductionXGBoostModel,
    records: Sequence[RejectedClassifierRecord]
    | Iterable[RejectedClassifierRecord],
    *,
    confidence_threshold: float = 0.65,
) -> dict[str, Any]:
    """Mide aceptación indebida de ataques sin clase válida en la taxonomía."""

    values = tuple(records)
    _validate_threshold(confidence_threshold)
    if not values:
        return {
            "rows": 0,
            "threshold": confidence_threshold,
            "accepted_rows": 0,
            "accepted_rate": 0.0,
            "note": "No hay ataques rechazados en esta vista",
        }
    features = [dict(item.record.features) for item in values]
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    model_classes = tuple(str(value) for value in model.classes)
    confidence = probabilities.max(axis=1)
    predicted_indices = np.argmax(probabilities, axis=1)
    predicted = [model_classes[int(index)] for index in predicted_indices]
    accepted = confidence >= confidence_threshold

    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(values):
        key = "{dataset}|{native}|{status}|{reason}".format(
            dataset=item.record.dataset,
            native=item.record.target_class,
            status=item.status,
            reason=item.reason,
        )
        groups[key].append(index)
    by_group: dict[str, Any] = {}
    for key, positions in sorted(groups.items()):
        group_confidence = confidence[positions]
        group_accepted = accepted[positions]
        by_group[key] = {
            "rows": len(positions),
            "accepted_rows": int(group_accepted.sum()),
            "accepted_rate": float(group_accepted.mean()),
            "confidence": _confidence_summary(group_confidence),
            "top_predictions": dict(
                Counter(predicted[position] for position in positions).most_common()
            ),
        }
    return {
        "rows": len(values),
        "threshold": confidence_threshold,
        "accepted_rows": int(accepted.sum()),
        "accepted_rate": float(accepted.mean()),
        "confidence": _confidence_summary(confidence),
        "top_predictions": dict(Counter(predicted).most_common()),
        "by_native_group": by_group,
        "interpretation": (
            "No mide exactitud ni implica que todas las filas sean OOD: sus targets "
            "no permiten verificar una clase del espacio cerrado evaluado. La tasa "
            "aceptada indica "
            "cuántas decisiones emitiría el modelo sin una abstención adicional."
        ),
    }


def evaluate_origin_only_baseline(
    records: Sequence[MappedClassifierRecord] | Iterable[MappedClassifierRecord],
    *,
    detailed: bool = False,
) -> dict[str, Any]:
    """Baseline diagnóstico que solo conoce el origen y aprende en train."""

    values = tuple(records)
    train = [item for item in values if item.record.split == "train"]
    test = [item for item in values if item.record.split == "test"]
    if not train or not test:
        raise ValueError("El baseline de origen requiere train y test")
    origin = (
        (lambda item: item.detailed_origin)
        if detailed
        else (lambda item: item.dataset_origin)
    )
    global_label = _majority_label(item.label for item in train)
    by_origin: dict[str, str] = {}
    for name in sorted({origin(item) for item in train}):
        by_origin[name] = _majority_label(
            item.label for item in train if origin(item) == name
        )
    truth = np.asarray([item.label for item in test], dtype=object)
    predicted = np.asarray(
        [by_origin.get(origin(item), global_label) for item in test], dtype=object
    )
    labels = tuple(sorted(set(item.label for item in values)))
    return {
        "rows": len(test),
        "granularity": "detailed_origin" if detailed else "dataset_origin",
        "learned_majority_by_origin": by_origin,
        "fallback_global_majority": global_label,
        **_label_metrics(truth, predicted, labels=labels),
        "interpretation": (
            "Diagnóstico de confusión clase-origen; no utiliza las características "
            "del evento ni es un modelo candidato."
        ),
    }


def select_confidence_threshold(
    model: ProductionXGBoostModel,
    records: Sequence[MappedClassifierRecord] | Iterable[MappedClassifierRecord],
    *,
    maximum_risk: float = 0.05,
    minimum_coverage: float = 0.50,
) -> dict[str, Any]:
    """Selecciona umbral solo en validación mediante una rejilla predefinida.

    Prioriza la mayor cobertura que satisface el riesgo máximo. Si ningún
    umbral lo logra con la cobertura mínima, elige el menor riesgo y después
    la mayor cobertura. Test no interviene en la elección.
    """

    values = tuple(records)
    if not values or any(item.record.split != "val" for item in values):
        raise ValueError("La selección de umbral requiere exclusivamente filas val")
    _validate_threshold(maximum_risk)
    _validate_threshold(minimum_coverage)
    features = [dict(item.record.features) for item in values]
    truth = np.asarray([item.label for item in values], dtype=object)
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = tuple(str(value) for value in model.classes)
    predicted = np.asarray(
        [classes[int(index)] for index in np.argmax(probabilities, axis=1)],
        dtype=object,
    )
    confidence = probabilities.max(axis=1)
    candidates: list[dict[str, Any]] = []
    for integer in range(0, 100):
        threshold = integer / 100.0
        decided = confidence >= threshold
        coverage = float(decided.mean())
        risk = (
            float(np.mean(predicted[decided] != truth[decided]))
            if np.any(decided)
            else None
        )
        candidates.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "risk": risk,
                "decided_rows": int(decided.sum()),
            }
        )
    eligible = [
        row
        for row in candidates
        if row["coverage"] >= minimum_coverage and row["risk"] is not None
    ]
    feasible = [row for row in eligible if row["risk"] <= maximum_risk]
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (-row["coverage"], row["risk"], row["threshold"]),
        )
        outcome = "risk_target_met"
    elif eligible:
        selected = min(
            eligible,
            key=lambda row: (row["risk"], -row["coverage"], row["threshold"]),
        )
        outcome = "risk_target_not_met_best_available"
    else:
        raise ValueError("Ningún umbral alcanza la cobertura mínima solicitada")
    return {
        "selection_split": "val",
        "grid": "0.00_to_0.99_step_0.01",
        "maximum_risk_target": maximum_risk,
        "minimum_coverage": minimum_coverage,
        "outcome": outcome,
        "selected": selected,
        "candidates": candidates,
    }


def _label_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    labels: Sequence[str],
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=list(labels),
        average=None,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        truth,
        predicted,
        labels=list(labels),
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        truth,
        predicted,
        labels=list(labels),
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro": {
            "precision": float(macro[0]),
            "recall": float(macro[1]),
            "f1": float(macro[2]),
        },
        "weighted": {
            "precision": float(weighted[0]),
            "recall": float(weighted[1]),
            "f1": float(weighted[2]),
        },
        "per_class": [
            {
                "label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        ],
    }


def _confidence_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _majority_label(labels: Iterable[str]) -> str:
    counts = Counter(labels)
    if not counts:
        raise ValueError("No hay etiquetas para calcular la mayoría")
    maximum = max(counts.values())
    return min(label for label, count in counts.items() if count == maximum)


def _validate_threshold(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("confidence_threshold debe ser numerico")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("confidence_threshold debe estar entre 0 y 1")


__all__ = [
    "evaluate_classifier_view",
    "evaluate_origin_only_baseline",
    "evaluate_rejected_attack_confidence",
    "select_confidence_threshold",
]
