"""Training of candidate models used by the deployed predictive agents.

The validation campaign already owns ingestion, target authority and feature
sanitisation.  This module deliberately accepts only :class:`PreparedRecord`
objects and preserves their frozen ``train``/``val``/``test`` split.  It trains
two global candidates:

* a binary attack detector over every dataset that declares the ``binary`` task;
* an attack-only multiclass classifier over an explicit, versioned broad-family
  taxonomy.

Both candidates fit their :class:`~sklearn.feature_extraction.DictVectorizer`
on train only.  Saving is opt-in and atomic: no default deployment artefact is
ever overwritten merely by calling a training function.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, Sequence
import unicodedata

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from src.agents.supervised_edge import EDGE_TO_FAMILY
from src.eval.jorge_models import compute_classification_metrics
from src.eval.validation_campaign import PreparedRecord


ProductionTask = Literal["binary_detection", "attack_family"]
EstimatorFactory = Callable[..., Any]

FAMILY_MAPPING_VERSION = "broad_attack_family_v1_2026-08-06"
SPLITS = ("train", "val", "test")

BASE_XGBOOST_PARAMETERS: Mapping[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": 1,
    "tree_method": "hist",
    "verbosity": 0,
}


class ProductionModelError(ValueError):
    """The candidate dataset or estimator violates the production contract."""


class UnmappedAttackLabelError(ProductionModelError):
    """An attack label is not covered by the explicit family mapping."""


class EstimatorFactoryProtocol(Protocol):
    def __call__(
        self,
        *,
        task: ProductionTask,
        n_classes: int,
        parameters: Mapping[str, Any],
    ) -> Any: ...


@dataclass(slots=True)
class ProductionXGBoostModel:
    """Stable joblib-serialisable adapter consumed by ``model_registry``.

    The wrapped estimator operates on integer-encoded targets.  The public
    methods accept the same feature dictionaries as the deployed registry and
    expose decoded predictions plus a ``classes`` property aligned with the
    columns returned by ``predict_proba``.
    """

    vectorizer: DictVectorizer
    estimator: Any
    encoded_classes: tuple[bool | str, ...]
    task: ProductionTask
    feature_schema_sha256: str
    family_mapping_version: str | None = None

    @property
    def classes(self) -> list[bool | str]:
        return list(self.encoded_classes)

    @property
    def classes_(self) -> np.ndarray:
        """Sklearn-compatible alias; ``model_registry`` uses ``classes``."""

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


@dataclass(slots=True)
class CandidateTrainingResult:
    model: ProductionXGBoostModel
    report: dict[str, Any]
    artifact_path: Path | None = None


@dataclass(slots=True)
class ProductionTrainingResult:
    detector: CandidateTrainingResult
    family_classifier: CandidateTrainingResult

    def report(self) -> dict[str, Any]:
        return {
            "detector": self.detector.report,
            "family_classifier": self.family_classifier.report,
        }


def train_production_candidates(
    records: Sequence[PreparedRecord] | Iterable[PreparedRecord],
    *,
    detector_path: str | Path | None = None,
    family_path: str | Path | None = None,
    estimator_factory: EstimatorFactoryProtocol | None = None,
    parameter_overrides: Mapping[str, Any] | None = None,
) -> ProductionTrainingResult:
    """Train both global candidates, then optionally persist explicit paths.

    Training of both models completes before either path is touched.  This
    avoids publishing a detector when the family candidate cannot be built.
    Each individual replacement is an atomic ``os.replace`` in the target
    directory.
    """

    if detector_path is not None and family_path is not None:
        detector_target = Path(detector_path).expanduser().resolve()
        family_target = Path(family_path).expanduser().resolve()
        if detector_target == family_target:
            raise ProductionModelError(
                "detector_path y family_path deben ser artefactos distintos"
            )
    values = tuple(records)
    detector = train_global_detector(
        values,
        estimator_factory=estimator_factory,
        parameter_overrides=parameter_overrides,
    )
    classifier = train_global_family_classifier(
        values,
        estimator_factory=estimator_factory,
        parameter_overrides=parameter_overrides,
    )
    if detector_path is not None:
        _persist_result(detector, detector_path)
    if family_path is not None:
        _persist_result(classifier, family_path)
    return ProductionTrainingResult(detector=detector, family_classifier=classifier)


def train_global_detector(
    records: Sequence[PreparedRecord] | Iterable[PreparedRecord],
    *,
    model_path: str | Path | None = None,
    estimator_factory: EstimatorFactoryProtocol | None = None,
    parameter_overrides: Mapping[str, Any] | None = None,
) -> CandidateTrainingResult:
    """Train the global binary detector from frozen-split binary records."""

    values = tuple(records)
    labelled: list[tuple[PreparedRecord, bool]] = []
    skipped = Counter()
    for record in values:
        if "binary" not in record.tasks:
            skipped["task_not_declared"] += 1
        elif not isinstance(record.target_is_attack, bool):
            skipped["missing_or_invalid_target"] += 1
        else:
            labelled.append((record, record.target_is_attack))

    result = _train_candidate(
        labelled,
        task="binary_detection",
        skipped=dict(sorted(skipped.items())),
        estimator_factory=estimator_factory,
        parameter_overrides=parameter_overrides,
        family_mapping=None,
    )
    if model_path is not None:
        _persist_result(result, model_path)
    return result


def train_global_family_classifier(
    records: Sequence[PreparedRecord] | Iterable[PreparedRecord],
    *,
    model_path: str | Path | None = None,
    estimator_factory: EstimatorFactoryProtocol | None = None,
    parameter_overrides: Mapping[str, Any] | None = None,
) -> CandidateTrainingResult:
    """Train the global attack-only broad-family classifier."""

    values = tuple(records)
    labelled: list[tuple[PreparedRecord, str]] = []
    skipped = Counter()
    unmapped: list[tuple[str, str, str]] = []
    for record in values:
        if "multiclass" not in record.tasks:
            skipped["task_not_declared"] += 1
            continue
        if record.target_is_attack is not True:
            skipped["non_attack"] += 1
            continue
        if not isinstance(record.target_class, str) or not record.target_class.strip():
            skipped["missing_or_invalid_target"] += 1
            continue
        try:
            family = map_attack_family(record.dataset, record.target_class)
        except UnmappedAttackLabelError:
            unmapped.append((record.manifest_id, record.dataset, record.target_class))
            continue
        labelled.append((record, family))

    if unmapped:
        sample = ", ".join(
            f"{manifest_id}:{dataset}/{label}"
            for manifest_id, dataset, label in sorted(unmapped)[:5]
        )
        raise UnmappedAttackLabelError(
            f"{len(unmapped)} filas de ataque no tienen familia explicita en "
            f"{FAMILY_MAPPING_VERSION} (ejemplos: {sample})"
        )

    result = _train_candidate(
        labelled,
        task="attack_family",
        skipped=dict(sorted(skipped.items())),
        estimator_factory=estimator_factory,
        parameter_overrides=parameter_overrides,
        family_mapping=family_mapping_report(),
    )
    if model_path is not None:
        _persist_result(result, model_path)
    return result


def deduplicate_labelled_records(
    labelled_records: Sequence[tuple[PreparedRecord, bool | str]]
    | Iterable[tuple[PreparedRecord, bool | str]],
) -> tuple[tuple[tuple[PreparedRecord, bool | str], ...], dict[str, Any]]:
    """Global task-aware deduplication safe for frozen splits and targets.

    A fingerprint spanning splits or carrying conflicting labels is removed in
    full.  A same-split/same-label duplicate keeps the lexicographically lowest
    manifest id, irrespective of source dataset.
    """

    values = tuple(labelled_records)
    groups: dict[str, list[tuple[PreparedRecord, bool | str]]] = defaultdict(list)
    for record, label in values:
        if not isinstance(record.feature_fingerprint, str) or not record.feature_fingerprint:
            raise ProductionModelError(
                f"feature_fingerprint invalido en {record.manifest_id!r}"
            )
        groups[record.feature_fingerprint].append((record, label))

    kept: list[tuple[PreparedRecord, bool | str]] = []
    same_split_removed: list[str] = []
    unsafe_removed: list[str] = []
    removed_groups: list[dict[str, Any]] = []
    same_groups = 0
    cross_groups = 0
    conflict_groups = 0

    for fingerprint in sorted(groups):
        group = sorted(groups[fingerprint], key=lambda item: item[0].manifest_id)
        splits = {record.split for record, _label in group}
        labels = {_label_key(label) for _record, label in group}
        reasons: list[str] = []
        if len(splits) > 1:
            cross_groups += 1
            reasons.append("cross_split")
        if len(labels) > 1:
            conflict_groups += 1
            reasons.append("conflicting_labels")
        if reasons:
            identifiers = [record.manifest_id for record, _label in group]
            unsafe_removed.extend(identifiers)
            removed_groups.append(
                {
                    "feature_fingerprint": fingerprint,
                    "reasons": reasons,
                    "manifest_ids": identifiers,
                    "splits": sorted(splits),
                    "labels": sorted(labels),
                }
            )
            continue

        kept.append(group[0])
        if len(group) > 1:
            same_groups += 1
            identifiers = [record.manifest_id for record, _label in group[1:]]
            same_split_removed.extend(identifiers)
            removed_groups.append(
                {
                    "feature_fingerprint": fingerprint,
                    "reasons": ["same_split_same_label"],
                    "manifest_ids": [record.manifest_id for record, _label in group],
                    "kept_manifest_id": group[0][0].manifest_id,
                }
            )

    split_order = {name: index for index, name in enumerate(SPLITS)}
    kept.sort(
        key=lambda item: (
            split_order.get(item[0].split, len(split_order)),
            item[0].dataset,
            item[0].manifest_id,
        )
    )
    report = {
        "scope": "task_global",
        "input_rows": len(values),
        "output_rows": len(kept),
        "same_split_same_label_groups": same_groups,
        "same_split_rows_removed": len(same_split_removed),
        "same_split_removed_ids": sorted(same_split_removed),
        "cross_split_groups": cross_groups,
        "conflicting_label_groups": conflict_groups,
        "unsafe_rows_removed": len(unsafe_removed),
        "unsafe_removed_ids": sorted(unsafe_removed),
        "removed_groups": removed_groups,
        "final_cross_split_overlap_count": _cross_split_overlap_count(kept),
    }
    return tuple(kept), report


def map_attack_family(dataset: str, target_class: str) -> str:
    """Map one native malicious label using the versioned explicit taxonomy."""

    dataset_key = _normalise_dataset(dataset)
    label_key = _normalise_label(target_class)
    mapping = _FAMILY_MAPPINGS.get(dataset_key)
    if mapping is None or label_key not in mapping:
        raise UnmappedAttackLabelError(
            f"Etiqueta de ataque sin mapear: dataset={dataset!r}, class={target_class!r}, "
            f"version={FAMILY_MAPPING_VERSION}"
        )
    return mapping[label_key]


def family_mapping_report() -> dict[str, Any]:
    """Return a JSON-compatible snapshot and hash of the active mapping."""

    datasets = {
        dataset: dict(sorted(mapping.items()))
        for dataset, mapping in sorted(_NATIVE_FAMILY_MAPPINGS.items())
    }
    core = {"version": FAMILY_MAPPING_VERSION, "datasets": datasets}
    return {**core, "sha256": _json_sha256(core)}


def _train_candidate(
    labelled: Sequence[tuple[PreparedRecord, bool | str]],
    *,
    task: ProductionTask,
    skipped: Mapping[str, int],
    estimator_factory: EstimatorFactoryProtocol | None,
    parameter_overrides: Mapping[str, Any] | None,
    family_mapping: Mapping[str, Any] | None,
) -> CandidateTrainingResult:
    deduplicated, dedup_report = deduplicate_labelled_records(labelled)
    by_split: dict[str, list[tuple[PreparedRecord, bool | str]]] = {
        split: [] for split in SPLITS
    }
    for item in deduplicated:
        split = item[0].split
        if split not in by_split:
            raise ProductionModelError(
                f"Split no soportado en {item[0].manifest_id!r}: {split!r}"
            )
        by_split[split].append(item)

    missing = [split for split in SPLITS if not by_split[split]]
    if missing:
        raise ProductionModelError(
            f"{task} no tiene filas tras deduplicar en splits: {', '.join(missing)}"
        )

    train_labels = [label for _record, label in by_split["train"]]
    if task == "binary_detection":
        classes: tuple[bool | str, ...] = (False, True)
        if set(train_labels) != {False, True}:
            raise ProductionModelError(
                "binary_detection requiere False y True en train tras deduplicar"
            )
    else:
        classes = tuple(sorted({str(label) for label in train_labels}))
        if len(classes) < 2:
            raise ProductionModelError(
                "attack_family requiere al menos dos familias en train tras deduplicar"
            )

    class_to_index = {label: index for index, label in enumerate(classes)}
    for split in SPLITS:
        unknown = sorted(
            {
                str(label)
                for _record, label in by_split[split]
                if label not in class_to_index
            }
        )
        if unknown:
            raise ProductionModelError(
                f"{task}: {split} contiene clases ausentes de train: {unknown}"
            )

    vectorizer = DictVectorizer(sparse=True, sort=True)
    train_features = [dict(record.features) for record, _label in by_split["train"]]
    x_train = vectorizer.fit_transform(train_features)
    if x_train.shape[1] == 0:
        raise ProductionModelError(f"{task}: DictVectorizer no encontro features en train")
    y_train = np.asarray(
        [class_to_index[label] for _record, label in by_split["train"]],
        dtype=np.int64,
    )

    parameters = _effective_parameters(
        task, len(classes), parameter_overrides=parameter_overrides
    )
    factory = estimator_factory or _default_estimator_factory
    estimator = factory(task=task, n_classes=len(classes), parameters=dict(parameters))
    if not callable(getattr(estimator, "fit", None)):
        raise TypeError("estimator_factory debe devolver un objeto con fit")
    if not callable(getattr(estimator, "predict", None)):
        raise TypeError("estimator_factory debe devolver un objeto con predict")
    if not callable(getattr(estimator, "predict_proba", None)):
        raise TypeError("estimator_factory debe devolver un objeto con predict_proba")
    estimator.fit(x_train, y_train)

    feature_names = [str(value) for value in vectorizer.get_feature_names_out()]
    feature_schema_sha256 = _json_sha256(feature_names)
    model = ProductionXGBoostModel(
        vectorizer=vectorizer,
        estimator=estimator,
        encoded_classes=classes,
        task=task,
        feature_schema_sha256=feature_schema_sha256,
        family_mapping_version=(
            FAMILY_MAPPING_VERSION if task == "attack_family" else None
        ),
    )

    metrics: dict[str, Any] = {}
    for split in ("val", "test"):
        records_in_split = by_split[split]
        features = [dict(record.features) for record, _label in records_in_split]
        truth = np.asarray(
            [class_to_index[label] for _record, label in records_in_split],
            dtype=np.int64,
        )
        predictions = _validated_encoded_predictions(
            estimator.predict(vectorizer.transform(features)),
            expected_size=len(records_in_split),
            n_classes=len(classes),
            context=f"{task}/{split}",
        )
        metrics[split] = {
            "global": compute_classification_metrics(
                truth,
                predictions,
                classes=classes,
                positive_index=(classes.index(True) if task == "binary_detection" else None),
            ),
            "by_dataset": _metrics_by_dataset(
                records_in_split,
                predictions,
                classes=classes,
                class_to_index=class_to_index,
                positive_index=(classes.index(True) if task == "binary_detection" else None),
            ),
        }

    report: dict[str, Any] = {
        "task": task,
        "model": "xgboost",
        "parameters": dict(parameters),
        "classes": list(classes),
        "input_rows": len(labelled),
        "skipped_rows": dict(skipped),
        "deduplication": dedup_report,
        "supports": {
            split: _support_report(items) for split, items in by_split.items()
        },
        "data_sha256": {
            "eligible": _labelled_records_sha256(labelled),
            **{
                split: _labelled_records_sha256(items)
                for split, items in by_split.items()
            },
        },
        "features": {
            "count": len(feature_names),
            "names_sha256": feature_schema_sha256,
        },
        "metrics": metrics,
        "artifact": None,
    }
    if family_mapping is not None:
        report["family_mapping"] = dict(family_mapping)
    return CandidateTrainingResult(model=model, report=report)


def _metrics_by_dataset(
    records: Sequence[tuple[PreparedRecord, bool | str]],
    predictions: np.ndarray,
    *,
    classes: Sequence[bool | str],
    class_to_index: Mapping[bool | str, int],
    positive_index: int | None,
) -> dict[str, Any]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, (record, _label) in enumerate(records):
        indices[record.dataset].append(index)
    output: dict[str, Any] = {}
    for dataset, positions in sorted(indices.items()):
        truth = np.asarray(
            [class_to_index[records[index][1]] for index in positions],
            dtype=np.int64,
        )
        predicted = np.asarray([predictions[index] for index in positions], dtype=np.int64)
        output[dataset] = compute_classification_metrics(
            truth,
            predicted,
            classes=classes,
            positive_index=positive_index,
        )
    return output


def _support_report(
    records: Sequence[tuple[PreparedRecord, bool | str]],
) -> dict[str, Any]:
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    class_counts: Counter[str] = Counter()
    for record, label in records:
        label_text = _report_label(label)
        class_counts[label_text] += 1
        by_dataset[record.dataset][label_text] += 1
    return {
        "rows": len(records),
        "class_counts": dict(sorted(class_counts.items())),
        "by_dataset": {
            dataset: {
                "rows": sum(counts.values()),
                "class_counts": dict(sorted(counts.items())),
            }
            for dataset, counts in sorted(by_dataset.items())
        },
    }


def _effective_parameters(
    task: ProductionTask,
    n_classes: int,
    *,
    parameter_overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    parameters = dict(BASE_XGBOOST_PARAMETERS)
    uses_binary_objective = n_classes == 2
    parameters.update(
        {
            "objective": "binary:logistic" if uses_binary_objective else "multi:softprob",
            "eval_metric": "logloss" if uses_binary_objective else "mlogloss",
        }
    )
    if n_classes > 2:
        parameters["num_class"] = n_classes
    overrides = dict(parameter_overrides or {})
    forbidden = {"objective", "eval_metric", "num_class", "random_state", "n_jobs"}
    invalid = sorted(forbidden.intersection(overrides))
    if invalid:
        raise ProductionModelError(
            f"No se pueden sobrescribir parametros de contrato: {invalid}"
        )
    parameters.update(overrides)
    return parameters


def _default_estimator_factory(
    *, task: ProductionTask, n_classes: int, parameters: Mapping[str, Any]
) -> Any:
    del task, n_classes
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - exercised only without evaluation extra
        raise RuntimeError(
            "xgboost no esta instalado; instala el extra 'evaluation'"
        ) from exc
    return XGBClassifier(**dict(parameters))


def _persist_result(result: CandidateTrainingResult, path: str | Path) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_joblib_dump(result.model, target)
    result.artifact_path = target
    result.report["artifact"] = {
        "path": str(target),
        "sha256": _file_sha256(target),
        "bytes": target.stat().st_size,
    }


def _atomic_joblib_dump(model: ProductionXGBoostModel, target: Path) -> None:
    import joblib

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(model, temporary, compress=3)
        # Windows rejects fsync on a read-only descriptor (EBADF), hence r+b.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


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
        raise ProductionModelError(f"{context}: predict no devolvio indices enteros") from exc
    if not np.all(predictions == numeric):
        raise ProductionModelError(f"{context}: predict no devolvio indices enteros")
    if np.any(numeric < 0) or np.any(numeric >= n_classes):
        raise ProductionModelError(f"{context}: predict devolvio una clase fuera de rango")
    return numeric


def _labelled_records_sha256(
    values: Sequence[tuple[PreparedRecord, bool | str]],
) -> str:
    payload = [
        {
            "manifest_id": record.manifest_id,
            "dataset": record.dataset,
            "split": record.split,
            "content_hash": record.content_hash,
            "feature_fingerprint": record.feature_fingerprint,
            "label": label,
        }
        for record, label in sorted(values, key=lambda item: item[0].manifest_id)
    ]
    return _json_sha256(payload)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cross_split_overlap_count(
    values: Sequence[tuple[PreparedRecord, bool | str]],
) -> int:
    splits_by_fingerprint: dict[str, set[str]] = defaultdict(set)
    for record, _label in values:
        splits_by_fingerprint[record.feature_fingerprint].add(record.split)
    return sum(len(splits) > 1 for splits in splits_by_fingerprint.values())


def _label_key(label: bool | str) -> str:
    if isinstance(label, bool):
        return "attack" if label else "benign"
    return str(label)


def _report_label(label: bool | str) -> str:
    return _label_key(label)


def _normalise_dataset(value: str) -> str:
    key = _normalise_label(value)
    aliases = {
        "edge_iiotset": "edge_iiotset",
        "edgeiiotset": "edge_iiotset",
        "ton_iot": "ton_iot",
        "toniot": "ton_iot",
        "bot_iot": "bot_iot",
        "botiot": "bot_iot",
        "iot23": "iot23",
        "iot_23": "iot23",
    }
    return aliases.get(key, key)


def _normalise_label(value: str) -> str:
    normalised = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"[^a-z0-9]+", "_", normalised).strip("_")


_NATIVE_FAMILY_MAPPINGS: Mapping[str, Mapping[str, str]] = {
    "edge_iiotset": {
        label: family for label, family in EDGE_TO_FAMILY.items() if family is not None
    },
    "ton_iot": {
        "dos": "ddos",
        "ddos": "ddos",
        "scanning": "scanning",
        "password": "bruteforce",
        "injection": "injection",
        "xss": "injection",
        "backdoor": "malware",
        "ransomware": "malware",
        "mitm": "mitm",
    },
    "bot_iot": {
        "dos": "ddos",
        "ddos": "ddos",
        "reconnaissance": "scanning",
        "theft": "exfiltration",
    },
    "iot23": {
        "C&C": "botnet",
        "C&C-FileDownload": "botnet",
        "C&C-Torii": "botnet",
        "PartOfAHorizontalPortScan": "scanning",
        "DDoS": "ddos",
        "FileDownload": "unknown_attack",
    },
}

_FAMILY_MAPPINGS: Mapping[str, Mapping[str, str]] = {
    dataset: {
        _normalise_label(native_label): family
        for native_label, family in mapping.items()
    }
    for dataset, mapping in _NATIVE_FAMILY_MAPPINGS.items()
}


__all__ = [
    "BASE_XGBOOST_PARAMETERS",
    "CandidateTrainingResult",
    "FAMILY_MAPPING_VERSION",
    "ProductionModelError",
    "ProductionTrainingResult",
    "ProductionXGBoostModel",
    "UnmappedAttackLabelError",
    "deduplicate_labelled_records",
    "family_mapping_report",
    "map_attack_family",
    "train_global_detector",
    "train_global_family_classifier",
    "train_production_candidates",
]
