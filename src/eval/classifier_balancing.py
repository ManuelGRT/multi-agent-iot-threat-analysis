"""Preparacion reproducible de clasificadores multidataset de ataques.

El modulo no entrena modelos. Resuelve la taxonomia, elimina duplicados y
selecciona corpus balanceados. Conserva el protocolo historico basado en las
particiones de ``validation_2026`` y ofrece, por separado, un protocolo que
selecciona primero el corpus y crea despues un split especifico del
clasificador, estratificado por clase y procedencia.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
import hashlib
from itertools import combinations
import json
import random
from typing import Any, Iterable, Mapping, Sequence

from src.contracts.attack_taxonomy import (
    JORGE_ATTACK_CLASSES,
    JORGE_TAXONOMY_VERSION,
    MULTIDATASET_ATTACK_CLASSES,
    MULTIDATASET_TAXONOMY_VERSION,
    normalise_dataset,
    normalise_taxonomy_token,
    resolve_jorge_class,
    resolve_multidataset_class,
)
from src.eval.classifier_targets import MappingContext
from src.eval.validation_campaign import PreparedRecord


SPLIT_UNITS: Mapping[str, int] = {"train": 14, "val": 3, "test": 3}
TOTAL_UNITS = sum(SPLIT_UNITS.values())


@dataclass(frozen=True, slots=True)
class MappedClassifierRecord:
    record: PreparedRecord
    label: str
    dataset_origin: str
    detailed_origin: str
    mapping_reason: str
    mapping_status: str = "exact"


@dataclass(frozen=True, slots=True)
class RejectedClassifierRecord:
    record: PreparedRecord
    status: str
    reason: str
    native_subclass: str | None = None


def classifier_origins(dataset: str, source_file: str) -> tuple[str, str]:
    """Devuelve origen principal y suborigen auditable de una fila."""

    dataset_origin = normalise_dataset(dataset)
    if dataset_origin != "ton_iot":
        return dataset_origin, dataset_origin

    path = str(source_file or "").replace("\\", "/").casefold()
    if "train_test_network_dataset" in path:
        detail = "ton_iot_network"
    elif "train_test_linux_dataset" in path:
        detail = "ton_iot_linux"
    elif "train_test_windows_dataset" in path:
        detail = "ton_iot_windows"
    elif "train_test_iot_dataset" in path:
        detail = "ton_iot_telemetry"
    else:
        raise ValueError(f"Fichero TON-IoT de modalidad desconocida: {source_file!r}")
    return dataset_origin, detail


def map_classifier_records(
    records: Iterable[PreparedRecord],
    *,
    native_subclasses: Mapping[str, str] | None = None,
) -> tuple[
    tuple[MappedClassifierRecord, ...],
    tuple[RejectedClassifierRecord, ...],
    dict[str, Any],
]:
    """Mapea ataques a las 14 clases sin convertir ambiguedades en etiquetas."""

    subclasses = native_subclasses or {}
    mapped: list[MappedClassifierRecord] = []
    rejected: list[RejectedClassifierRecord] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    mapped_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        if "multiclass" not in record.tasks:
            status_counts["task_not_declared"] += 1
            continue
        if not isinstance(record.target_is_attack, bool):
            raise ValueError(
                f"{record.manifest_id}: target_is_attack invalido para multiclass"
            )
        resolution = resolve_jorge_class(
            record.dataset,
            record.target_class,
            subclasses.get(record.manifest_id),
        )
        if record.target_is_attack is False:
            if resolution.status != "benign":
                raise ValueError(
                    f"{record.manifest_id}: target benigno incoherente con "
                    f"target_class={record.target_class!r}"
                )
            status_counts["non_attack"] += 1
            continue
        native_subclass = subclasses.get(record.manifest_id)
        if resolution.status == "benign":
            raise ValueError(
                f"{record.manifest_id}: target de ataque incoherente con clase benigna"
            )
        status_counts[resolution.status] += 1
        reason_counts[resolution.reason] += 1
        if resolution.status != "exact" or resolution.attack_type is None:
            rejected.append(
                RejectedClassifierRecord(
                    record=record,
                    status=resolution.status,
                    reason=resolution.reason,
                    native_subclass=native_subclass,
                )
            )
            continue
        dataset_origin, detailed_origin = classifier_origins(
            record.dataset, record.source_file
        )
        mapped.append(
            MappedClassifierRecord(
                record=record,
                label=resolution.attack_type,
                dataset_origin=dataset_origin,
                detailed_origin=detailed_origin,
                mapping_reason=resolution.reason,
                mapping_status=resolution.status,
            )
        )
        mapped_by_dataset[dataset_origin][resolution.attack_type] += 1

    mapped.sort(key=lambda item: item.record.manifest_id)
    rejected.sort(key=lambda item: item.record.manifest_id)
    report = {
        "taxonomy_version": JORGE_TAXONOMY_VERSION,
        "input_rows": sum(status_counts.values()),
        "mapped_rows": len(mapped),
        "rejected_attack_rows": len(rejected),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "mapped_by_dataset": {
            dataset: dict(sorted(counts.items()))
            for dataset, counts in sorted(mapped_by_dataset.items())
        },
    }
    return tuple(mapped), tuple(rejected), report


def map_multidataset_classifier_records(
    records: Iterable[PreparedRecord],
    *,
    native_subclasses: Mapping[str, str] | None = None,
    mapping_contexts: Mapping[str, MappingContext] | None = None,
) -> tuple[
    tuple[MappedClassifierRecord, ...],
    tuple[RejectedClassifierRecord, ...],
    dict[str, Any],
]:
    """Mapea ataques a las 16 clases, conservando calidad y procedencia.

    ``forced`` no significa target nativo exacto: identifica una regla
    versionada de armonizacion. Los contextos de protocolo deben proceder de
    los manifiestos crudos, nunca del evento generado por el LLM.
    """

    subclasses = native_subclasses or {}
    contexts = mapping_contexts or {}
    mapped: list[MappedClassifierRecord] = []
    rejected: list[RejectedClassifierRecord] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    mapped_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    mapped_by_status: dict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        if "multiclass" not in record.tasks:
            status_counts["task_not_declared"] += 1
            continue
        if not isinstance(record.target_is_attack, bool):
            raise ValueError(
                f"{record.manifest_id}: target_is_attack invalido para multiclass"
            )
        context = contexts.get(record.manifest_id)
        resolution = resolve_multidataset_class(
            record.dataset,
            record.target_class,
            subclasses.get(record.manifest_id),
            transport_proto=(context.transport_proto if context else None),
            app_proto=(context.app_proto if context else None),
        )
        if record.target_is_attack is False:
            if resolution.status != "benign":
                raise ValueError(
                    f"{record.manifest_id}: target benigno incoherente con "
                    f"target_class={record.target_class!r}"
                )
            status_counts["non_attack"] += 1
            continue
        native_subclass = subclasses.get(record.manifest_id)
        if resolution.status == "benign":
            raise ValueError(
                f"{record.manifest_id}: target de ataque incoherente con clase benigna"
            )
        status_counts[resolution.status] += 1
        reason_counts[resolution.reason] += 1
        if resolution.status not in {"exact", "forced"} or resolution.attack_type is None:
            rejected.append(
                RejectedClassifierRecord(
                    record=record,
                    status=resolution.status,
                    reason=resolution.reason,
                    native_subclass=native_subclass,
                )
            )
            continue
        if resolution.attack_type not in MULTIDATASET_ATTACK_CLASSES:
            raise ValueError(
                f"{record.manifest_id}: resolver devolvio clase inesperada "
                f"{resolution.attack_type!r}"
            )
        dataset_origin, detailed_origin = classifier_origins(
            record.dataset, record.source_file
        )
        mapped.append(
            MappedClassifierRecord(
                record=record,
                label=resolution.attack_type,
                dataset_origin=dataset_origin,
                detailed_origin=detailed_origin,
                mapping_reason=resolution.reason,
                mapping_status=resolution.status,
            )
        )
        mapped_by_dataset[dataset_origin][resolution.attack_type] += 1
        mapped_by_status[resolution.status][resolution.attack_type] += 1

    mapped.sort(key=lambda item: item.record.manifest_id)
    rejected.sort(key=lambda item: item.record.manifest_id)
    report = {
        "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        "classes": list(MULTIDATASET_ATTACK_CLASSES),
        "input_rows": sum(status_counts.values()),
        "mapped_rows": len(mapped),
        "rejected_attack_rows": len(rejected),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "mapped_by_status": {
            status: dict(sorted(counts.items()))
            for status, counts in sorted(mapped_by_status.items())
        },
        "mapped_by_dataset": {
            dataset: dict(sorted(counts.items()))
            for dataset, counts in sorted(mapped_by_dataset.items())
        },
    }
    return tuple(mapped), tuple(rejected), report


def deduplicate_mapped_records(
    records: Sequence[MappedClassifierRecord] | Iterable[MappedClassifierRecord],
) -> tuple[tuple[MappedClassifierRecord, ...], dict[str, Any]]:
    """Deduplica globalmente antes del balanceo y falla cerrado entre splits."""

    mapped, rejected, report = deduplicate_classifier_universe(records, ())
    if rejected:
        raise AssertionError("El wrapper de deduplicacion mapeada genero rechazados")
    return mapped, report


def deduplicate_classifier_universe(
    mapped_records: Sequence[MappedClassifierRecord] | Iterable[MappedClassifierRecord],
    rejected_records: Sequence[RejectedClassifierRecord]
    | Iterable[RejectedClassifierRecord],
) -> tuple[
    tuple[MappedClassifierRecord, ...],
    tuple[RejectedClassifierRecord, ...],
    dict[str, Any],
]:
    """Deduplica conjuntamente el conjunto cerrado y el conjunto OOD.

    Así, una huella de entrenamiento exacta nunca puede reaparecer en la
    evaluación de ataques rechazados por la taxonomía.
    """

    mapped_values = tuple(mapped_records)
    rejected_values = tuple(rejected_records)
    values: tuple[MappedClassifierRecord | RejectedClassifierRecord, ...] = (
        *mapped_values,
        *rejected_values,
    )
    groups: dict[
        str, list[MappedClassifierRecord | RejectedClassifierRecord]
    ] = defaultdict(list)
    for item in values:
        fingerprint = item.record.feature_fingerprint
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(
                f"feature_fingerprint invalido en {item.record.manifest_id!r}"
            )
        groups[fingerprint].append(item)

    kept: list[MappedClassifierRecord | RejectedClassifierRecord] = []
    removed_groups: list[dict[str, Any]] = []
    same_split_removed = 0
    unsafe_removed = 0
    cross_split_groups = 0
    conflicting_label_groups = 0

    for fingerprint in sorted(groups):
        group = sorted(groups[fingerprint], key=lambda item: item.record.manifest_id)
        splits = {item.record.split for item in group}
        labels = {_supervision_key(item) for item in group}
        reasons: list[str] = []
        if len(splits) > 1:
            reasons.append("cross_split")
            cross_split_groups += 1
        if len(labels) > 1:
            reasons.append("conflicting_labels")
            conflicting_label_groups += 1
        if reasons:
            unsafe_removed += len(group)
            removed_groups.append(
                {
                    "feature_fingerprint": fingerprint,
                    "reasons": reasons,
                    "manifest_ids": [item.record.manifest_id for item in group],
                    "splits": sorted(splits),
                    "labels": sorted(labels),
                    "kept_manifest_id": None,
                }
            )
            continue
        kept.append(group[0])
        if len(group) > 1:
            same_split_removed += len(group) - 1
            removed_groups.append(
                {
                    "feature_fingerprint": fingerprint,
                    "reasons": ["same_split_same_label"],
                    "manifest_ids": [item.record.manifest_id for item in group],
                    "splits": sorted(splits),
                    "labels": sorted(labels),
                    "kept_manifest_id": group[0].record.manifest_id,
                }
            )

    split_order = {name: index for index, name in enumerate(SPLIT_UNITS)}
    kept.sort(
        key=lambda item: (
            split_order.get(item.record.split, len(split_order)),
            _supervision_key(item),
            item.record.manifest_id,
        )
    )
    report = {
        "scope": "global_feature_fingerprint_across_exact_and_rejected_attacks",
        "input_rows": len(values),
        "output_rows": len(kept),
        "same_split_rows_removed": same_split_removed,
        "unsafe_rows_removed": unsafe_removed,
        "cross_split_groups": cross_split_groups,
        "conflicting_label_groups": conflicting_label_groups,
        "final_cross_split_overlap_count": _cross_split_overlap_count(kept),
        "removed_groups": removed_groups,
    }
    if report["final_cross_split_overlap_count"]:
        raise ValueError("La deduplicacion dejo huellas compartidas entre splits")
    kept_mapped = tuple(
        item for item in kept if isinstance(item, MappedClassifierRecord)
    )
    kept_rejected = tuple(
        item for item in kept if isinstance(item, RejectedClassifierRecord)
    )
    return kept_mapped, kept_rejected, report


def balance_classifier_splits(
    records: Sequence[MappedClassifierRecord] | Iterable[MappedClassifierRecord],
    *,
    per_class_total: int | None = None,
    seed: int = 42,
    classes: Sequence[str] = JORGE_ATTACK_CLASSES,
) -> tuple[tuple[MappedClassifierRecord, ...], dict[str, Any]]:
    """Balancea las clases indicadas por split y diversifica su procedencia."""

    values = tuple(records)
    expected_classes = tuple(str(label) for label in classes)
    if not expected_classes or len(expected_classes) != len(set(expected_classes)):
        raise ValueError("classes debe contener etiquetas unicas y no estar vacio")
    available = _support_report(values)
    present = set(available["class_counts"])
    expected = set(expected_classes)
    if present != expected:
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        raise ValueError(
            f"Taxonomia incompleta para balancear; missing={missing}, unexpected={unexpected}"
        )

    cycles = min(
        int(available["splits"][split]["class_counts"].get(label, 0)) // units
        for split, units in SPLIT_UNITS.items()
        for label in expected_classes
    )
    if per_class_total is not None:
        if (
            isinstance(per_class_total, bool)
            or not isinstance(per_class_total, int)
            or per_class_total <= 0
        ):
            raise ValueError("per_class_total debe ser un entero positivo")
        if per_class_total % TOTAL_UNITS:
            raise ValueError(
                f"per_class_total debe ser multiplo de {TOTAL_UNITS} para 70/15/15"
            )
        requested_cycles = per_class_total // TOTAL_UNITS
        if requested_cycles > cycles:
            raise ValueError(
                f"Cuota solicitada {per_class_total} superior al maximo {cycles * TOTAL_UNITS}"
            )
        cycles = requested_cycles
    if cycles < 1:
        raise ValueError("No hay soporte para una unidad completa 70/15/15 por clase")

    quotas = {split: units * cycles for split, units in SPLIT_UNITS.items()}
    selected: list[MappedClassifierRecord] = []
    for split in SPLIT_UNITS:
        for label in expected_classes:
            candidates = [
                item
                for item in values
                if item.record.split == split and item.label == label
            ]
            selected.extend(
                _fair_select_by_dataset_and_origin(
                    candidates,
                    quotas[split],
                    seed=f"{seed}:{split}:{label}",
                )
            )

    selected.sort(
        key=lambda item: (
            list(SPLIT_UNITS).index(item.record.split),
            item.label,
            item.record.manifest_id,
        )
    )
    selected_support = _support_report(selected)
    for split, quota in quotas.items():
        counts = selected_support["splits"][split]["class_counts"]
        if any(counts.get(label) != quota for label in expected_classes):
            raise AssertionError(f"Balance incompleto en {split}: {counts}")
    identifiers = [item.record.manifest_id for item in selected]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("El balanceo selecciono una fila mas de una vez")

    report = {
        "policy": "global_class_balance_then_max_min_dataset_origin_diversification",
        "seed": seed,
        "ratio_units": dict(SPLIT_UNITS),
        "classes": list(expected_classes),
        "cycles": cycles,
        "per_class_total": cycles * TOTAL_UNITS,
        "quotas_per_class": quotas,
        "available": available,
        "selected": selected_support,
        "selected_rows": len(selected),
        "selection_sha256": _selection_sha256(selected),
        "split_membership_preserved": True,
        "sampling_with_replacement": False,
    }
    return tuple(selected), report


def build_balanced_stratified_classifier_corpus(
    records: Sequence[MappedClassifierRecord] | Iterable[MappedClassifierRecord],
    *,
    per_class_total: int | None = None,
    seed: int = 42,
    classes: Sequence[str] = JORGE_ATTACK_CLASSES,
) -> tuple[tuple[MappedClassifierRecord, ...], dict[str, Any]]:
    """Selecciona primero un corpus balanceado y despues crea su split.

    La cuota automatica es el menor soporte global entre clases. El split de
    la campana fuente no interviene en la seleccion ni en la asignacion nueva.
    Tanto el muestreo como el reparto 70/15/15 conservan proporcionalmente los
    dataset de origen y, dentro de TON-IoT, sus cuatro modalidades. La calidad
    ``exact``/``forced`` se conserva como tercer nivel de estratificacion.

    Se exige una cuota multiplo de veinte porque solo asi 70/15/15 puede ser
    exacto con numeros enteros para todas las clases.
    """

    values = tuple(records)
    expected_classes = tuple(str(label) for label in classes)
    if not expected_classes or len(expected_classes) != len(set(expected_classes)):
        raise ValueError("classes debe contener etiquetas unicas y no estar vacio")

    available = _support_report(values)
    present = set(available["class_counts"])
    expected = set(expected_classes)
    if present != expected:
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        raise ValueError(
            f"Taxonomia incompleta para balancear; missing={missing}, unexpected={unexpected}"
        )

    maximum = min(int(available["class_counts"][label]) for label in expected_classes)
    requested = maximum if per_class_total is None else per_class_total
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise ValueError("per_class_total debe ser un entero positivo")
    if requested > maximum:
        raise ValueError(
            f"Cuota solicitada {requested} superior al minimo global disponible {maximum}"
        )
    if requested % TOTAL_UNITS:
        raise ValueError(
            f"per_class_total debe ser multiplo de {TOTAL_UNITS} para 70/15/15 exacto"
        )

    selected_before_split: list[MappedClassifierRecord] = []
    selection_by_class: dict[str, Any] = {}
    for label in expected_classes:
        candidates = [item for item in values if item.label == label]
        chosen, selection_report = _proportional_select_by_origin(
            candidates,
            requested,
            seed=f"{seed}:corpus:{label}",
        )
        selected_before_split.extend(chosen)
        selection_by_class[label] = selection_report

    source_splits = {
        item.record.manifest_id: item.record.split for item in selected_before_split
    }
    quotas = {
        split: requested * units // TOTAL_UNITS
        for split, units in SPLIT_UNITS.items()
    }
    assigned, stratification = _stratified_reassign_splits(
        selected_before_split,
        per_class_quotas=quotas,
        classes=expected_classes,
        seed=seed,
    )

    selected_support = _support_report(assigned)
    for split, quota in quotas.items():
        counts = selected_support["splits"][split]["class_counts"]
        if any(counts.get(label) != quota for label in expected_classes):
            raise AssertionError(f"Balance incompleto en {split}: {counts}")
    identifiers = [item.record.manifest_id for item in assigned]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("El balanceo selecciono una fila mas de una vez")
    overlap = _cross_split_overlap_count(assigned)
    if overlap:
        raise AssertionError(
            f"El nuevo split comparte {overlap} huellas predictivas entre particiones"
        )

    movement: dict[str, Counter[str]] = defaultdict(Counter)
    for item in assigned:
        movement[source_splits[item.record.manifest_id]][item.record.split] += 1
    limiting_classes = sorted(
        label
        for label in expected_classes
        if int(available["class_counts"][label]) == maximum
    )
    report = {
        "policy": (
            "minimum_global_class_support_then_stratified_70_15_15_by_"
            "class_dataset_detailed_origin_and_mapping_status"
        ),
        "seed": seed,
        "ratio_units": dict(SPLIT_UNITS),
        "classes": list(expected_classes),
        "minimum_global_class_support": maximum,
        "limiting_classes": limiting_classes,
        "per_class_total": requested,
        "quotas_per_class": quotas,
        "available": available,
        "corpus_selection": {
            "method": "proportional_hamilton_without_replacement",
            "classes": selection_by_class,
        },
        "selected": selected_support,
        "selected_rows": len(assigned),
        "selection_sha256": _selection_sha256(assigned),
        "split_assignment": stratification,
        "source_to_classifier_split": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(movement.items())
        },
        "split_membership_preserved": False,
        "source_split_used_for_corpus_selection": False,
        "source_split_used_for_assignment": False,
        "sampling_with_replacement": False,
        "final_cross_split_overlap_count": overlap,
    }
    return assigned, report


def _proportional_select_by_origin(
    records: Sequence[MappedClassifierRecord],
    count: int,
    *,
    seed: str,
) -> tuple[list[MappedClassifierRecord], dict[str, Any]]:
    """Muestrea sin reemplazo preservando origen, modalidad y calidad."""

    if count > len(records):
        raise ValueError(f"Se solicitaron {count} filas de un pool de {len(records)}")
    by_dataset: dict[str, list[MappedClassifierRecord]] = defaultdict(list)
    for item in records:
        by_dataset[item.dataset_origin].append(item)
    dataset_quotas = _hamilton_quotas(
        {name: len(rows) for name, rows in by_dataset.items()},
        count,
        seed=f"{seed}:dataset",
    )

    selected: list[MappedClassifierRecord] = []
    report: dict[str, Any] = {}
    for dataset_origin, dataset_rows in sorted(by_dataset.items()):
        by_detail: dict[str, list[MappedClassifierRecord]] = defaultdict(list)
        for item in dataset_rows:
            by_detail[item.detailed_origin].append(item)
        detail_quotas = _hamilton_quotas(
            {name: len(rows) for name, rows in by_detail.items()},
            dataset_quotas[dataset_origin],
            seed=f"{seed}:detail:{dataset_origin}",
        )
        detail_report: dict[str, Any] = {}
        for detailed_origin, detail_rows in sorted(by_detail.items()):
            by_status: dict[str, list[MappedClassifierRecord]] = defaultdict(list)
            for item in detail_rows:
                by_status[item.mapping_status].append(item)
            status_quotas = _hamilton_quotas(
                {name: len(rows) for name, rows in by_status.items()},
                detail_quotas[detailed_origin],
                seed=f"{seed}:status:{dataset_origin}:{detailed_origin}",
            )
            status_report: dict[str, Any] = {}
            for status, status_rows in sorted(by_status.items()):
                quota = status_quotas[status]
                ordered = _deterministic_shuffle(
                    status_rows,
                    f"{seed}:rows:{dataset_origin}:{detailed_origin}:{status}",
                )
                selected.extend(ordered[:quota])
                status_report[status] = {
                    "available": len(status_rows),
                    "selected": quota,
                }
            detail_report[detailed_origin] = {
                "available": len(detail_rows),
                "selected": detail_quotas[detailed_origin],
                "mapping_statuses": status_report,
            }
        report[dataset_origin] = {
            "available": len(dataset_rows),
            "selected": dataset_quotas[dataset_origin],
            "detailed_origins": detail_report,
        }
    if len(selected) != count:
        raise AssertionError("El muestreo proporcional no alcanzo la cuota solicitada")
    return selected, report


def _hamilton_quotas(
    group_counts: Mapping[str, int],
    target: int,
    *,
    seed: str,
) -> dict[str, int]:
    """Aplica mayores restos con desempate SHA-256 reproducible."""

    counts = {str(name): int(value) for name, value in group_counts.items()}
    total = sum(counts.values())
    if not counts or total <= 0 or any(value <= 0 for value in counts.values()):
        raise ValueError("group_counts debe contener grupos positivos")
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target <= total:
        raise ValueError("target debe ser un entero entre cero y el soporte disponible")
    quotas = {name: counts[name] * target // total for name in counts}
    missing = target - sum(quotas.values())
    ranking = sorted(
        counts,
        key=lambda name: (
            -((counts[name] * target) % total),
            hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest(),
            name,
        ),
    )
    for name in ranking[:missing]:
        quotas[name] += 1
    if sum(quotas.values()) != target:
        raise AssertionError("Hamilton no alcanzo el total solicitado")
    if any(quotas[name] > counts[name] for name in quotas):
        raise AssertionError("Hamilton excedio el soporte de un origen")
    return dict(sorted(quotas.items()))


def _stratified_reassign_splits(
    records: Sequence[MappedClassifierRecord],
    *,
    per_class_quotas: Mapping[str, int],
    classes: Sequence[str],
    seed: int,
) -> tuple[tuple[MappedClassifierRecord, ...], dict[str, Any]]:
    split_names = tuple(SPLIT_UNITS)
    if set(per_class_quotas) != set(split_names):
        raise ValueError("per_class_quotas debe declarar train, val y test")
    expected_total = sum(int(per_class_quotas[name]) for name in split_names)
    assigned: list[MappedClassifierRecord] = []
    class_reports: dict[str, Any] = {}

    for label in classes:
        class_rows = [item for item in records if item.label == label]
        if len(class_rows) != expected_total:
            raise ValueError(
                f"La clase {label!r} contiene {len(class_rows)} filas; "
                f"se esperaban {expected_total}"
            )
        by_dataset: dict[str, list[MappedClassifierRecord]] = defaultdict(list)
        for item in class_rows:
            by_dataset[item.dataset_origin].append(item)
        dataset_quotas = _apportion_split_matrix(
            {origin: len(rows) for origin, rows in sorted(by_dataset.items())},
            per_class_quotas,
            seed=f"{seed}:{label}:dataset",
        )

        dataset_reports: dict[str, Any] = {}
        for dataset_origin, dataset_rows in sorted(by_dataset.items()):
            by_detail: dict[str, list[MappedClassifierRecord]] = defaultdict(list)
            for item in dataset_rows:
                by_detail[item.detailed_origin].append(item)
            detail_quotas = _apportion_split_matrix(
                {origin: len(rows) for origin, rows in sorted(by_detail.items())},
                dataset_quotas[dataset_origin],
                seed=f"{seed}:{label}:detail:{dataset_origin}",
            )

            detail_reports: dict[str, Any] = {}
            for detailed_origin, detail_rows in sorted(by_detail.items()):
                by_status: dict[str, list[MappedClassifierRecord]] = defaultdict(list)
                for item in detail_rows:
                    by_status[item.mapping_status].append(item)
                status_quotas = _apportion_split_matrix(
                    {status: len(rows) for status, rows in sorted(by_status.items())},
                    detail_quotas[detailed_origin],
                    seed=(
                        f"{seed}:{label}:status:{dataset_origin}:"
                        f"{detailed_origin}"
                    ),
                )
                status_reports: dict[str, Any] = {}
                for status, status_rows in sorted(by_status.items()):
                    ordered = _deterministic_shuffle(
                        status_rows,
                        (
                            f"{seed}:split-rows:{label}:{dataset_origin}:"
                            f"{detailed_origin}:{status}"
                        ),
                    )
                    offset = 0
                    split_counts: dict[str, int] = {}
                    for split in split_names:
                        split_count = int(status_quotas[status][split])
                        split_counts[split] = split_count
                        for item in ordered[offset : offset + split_count]:
                            assigned.append(
                                replace(item, record=replace(item.record, split=split))
                            )
                        offset += split_count
                    if offset != len(ordered):
                        raise AssertionError(
                            f"Asignacion incompleta para {label}/{detailed_origin}/{status}"
                        )
                    status_reports[status] = {
                        "rows": len(status_rows),
                        "split_counts": split_counts,
                    }

                detail_split_counts = {
                    split: sum(
                        int(status_quotas[status][split])
                        for status in status_quotas
                    )
                    for split in split_names
                }
                if detail_split_counts != detail_quotas[detailed_origin]:
                    raise AssertionError(
                        f"Cuotas incoherentes para {label}/{detailed_origin}"
                    )
                detail_reports[detailed_origin] = {
                    "rows": len(detail_rows),
                    "split_counts": detail_split_counts,
                    "split_ratios": {
                        split: split_count / len(detail_rows)
                        for split, split_count in detail_split_counts.items()
                    },
                    "mapping_statuses": status_reports,
                }

            dataset_split_counts = {
                split: sum(
                    int(detail_quotas[origin][split]) for origin in detail_quotas
                )
                for split in split_names
            }
            if dataset_split_counts != dataset_quotas[dataset_origin]:
                raise AssertionError(
                    f"Cuotas incoherentes para {label}/{dataset_origin}"
                )
            dataset_reports[dataset_origin] = {
                "rows": len(dataset_rows),
                "split_counts": dataset_split_counts,
                "split_ratios": {
                    split: split_count / len(dataset_rows)
                    for split, split_count in dataset_split_counts.items()
                },
                "detailed_origins": detail_reports,
            }

        observed_class_counts = {
            split: sum(
                int(dataset_quotas[origin][split]) for origin in dataset_quotas
            )
            for split in split_names
        }
        expected_class_counts = {
            split: int(per_class_quotas[split]) for split in split_names
        }
        if observed_class_counts != expected_class_counts:
            raise AssertionError(f"Split de clase incoherente para {label}")
        class_reports[label] = {
            "rows": len(class_rows),
            "split_counts": observed_class_counts,
            "dataset_origins": dataset_reports,
        }

    split_order = {name: index for index, name in enumerate(split_names)}
    assigned.sort(
        key=lambda item: (
            split_order[item.record.split],
            item.label,
            item.dataset_origin,
            item.detailed_origin,
            item.record.manifest_id,
        )
    )
    return tuple(assigned), {
        "method": (
            "hierarchical_integer_apportionment_by_class_dataset_origin_"
            "detailed_origin_and_mapping_status"
        ),
        "tie_breaker": "sha256(seed, scope, origin, split)",
        "class_totals_exact": True,
        "origin_counts_use_floor_or_ceil_of_ideal": True,
        "classes": class_reports,
    }


def _apportion_split_matrix(
    group_counts: Mapping[str, int],
    split_targets: Mapping[str, int],
    *,
    seed: str,
) -> dict[str, dict[str, int]]:
    """Redondea una matriz proporcional respetando filas y columnas exactas."""

    groups = tuple(sorted(str(name) for name in group_counts))
    splits = tuple(SPLIT_UNITS)
    counts = {name: int(group_counts[name]) for name in groups}
    targets = {split: int(split_targets[split]) for split in splits}
    total = sum(counts.values())
    if not groups or total <= 0 or any(value <= 0 for value in counts.values()):
        raise ValueError("group_counts debe contener grupos positivos")
    if any(value < 0 for value in targets.values()) or sum(targets.values()) != total:
        raise ValueError("split_targets debe ser no negativo y sumar group_counts")

    base: dict[str, dict[str, int]] = {}
    remainders: dict[str, dict[str, int]] = {}
    row_deficits: dict[str, int] = {}
    for group in groups:
        base[group] = {
            split: counts[group] * targets[split] // total for split in splits
        }
        remainders[group] = {
            split: (counts[group] * targets[split]) % total for split in splits
        }
        row_deficits[group] = counts[group] - sum(base[group].values())

    column_deficits = {
        split: targets[split] - sum(base[group][split] for group in groups)
        for split in splits
    }
    if any(value < 0 for value in column_deficits.values()):
        raise AssertionError("El suelo proporcional excedio una cuota de split")

    states: dict[
        tuple[int, int, int],
        tuple[int, int, tuple[tuple[str, ...], ...]],
    ] = {(0, 0, 0): (0, 0, ())}
    for group in groups:
        options = tuple(combinations(splits, row_deficits[group]))
        updated: dict[
            tuple[int, int, int],
            tuple[int, int, tuple[tuple[str, ...], ...]],
        ] = {}
        for state, (score, tie_score, path) in states.items():
            for option in options:
                increment = tuple(1 if split in option else 0 for split in splits)
                candidate_state = tuple(
                    state[index] + increment[index] for index in range(len(splits))
                )
                if any(
                    candidate_state[index] > column_deficits[split]
                    for index, split in enumerate(splits)
                ):
                    continue
                candidate_score = score + sum(
                    remainders[group][split] for split in option
                )
                tie_payload = f"{seed}:{group}:{','.join(option)}".encode("utf-8")
                candidate_tie = tie_score + int.from_bytes(
                    hashlib.sha256(tie_payload).digest()[:8], "big"
                )
                candidate = (candidate_score, candidate_tie, (*path, option))
                previous = updated.get(candidate_state)
                if previous is None or candidate[:2] > previous[:2]:
                    updated[candidate_state] = candidate
        states = updated

    goal = tuple(column_deficits[split] for split in splits)
    if goal not in states:
        raise AssertionError(
            f"No se pudo redondear la matriz: filas={counts}, columnas={targets}"
        )
    path = states[goal][2]
    output: dict[str, dict[str, int]] = {}
    for group, option in zip(groups, path, strict=True):
        output[group] = {
            split: base[group][split] + (1 if split in option else 0)
            for split in splits
        }
        if sum(output[group].values()) != counts[group]:
            raise AssertionError(f"Cuota de origen incoherente para {group}")
    for split in splits:
        if sum(output[group][split] for group in groups) != targets[split]:
            raise AssertionError(f"Cuota de split incoherente para {split}")
    return output


def balanced_origin_test_views(
    records: Sequence[MappedClassifierRecord] | Iterable[MappedClassifierRecord],
    *,
    detailed: bool = True,
    seed: int = 42,
    minimum_class_support: int = 1,
) -> dict[str, tuple[MappedClassifierRecord, ...]]:
    """Crea tests balanceados por origen usando clases con soporte suficiente."""

    if (
        isinstance(minimum_class_support, bool)
        or not isinstance(minimum_class_support, int)
        or minimum_class_support < 1
    ):
        raise ValueError("minimum_class_support debe ser un entero positivo")

    values = [item for item in records if item.record.split == "test"]
    key = (lambda item: item.detailed_origin) if detailed else (lambda item: item.dataset_origin)
    grouped: dict[str, list[MappedClassifierRecord]] = defaultdict(list)
    for item in values:
        grouped[key(item)].append(item)

    views: dict[str, tuple[MappedClassifierRecord, ...]] = {}
    for origin, rows in sorted(grouped.items()):
        by_label: dict[str, list[MappedClassifierRecord]] = defaultdict(list)
        for item in rows:
            by_label[item.label].append(item)
        by_label = {
            label: items
            for label, items in by_label.items()
            if len(items) >= minimum_class_support
        }
        if not by_label:
            continue
        quota = min(len(items) for items in by_label.values())
        selected: list[MappedClassifierRecord] = []
        for label, items in sorted(by_label.items()):
            ordered = _deterministic_shuffle(items, f"{seed}:origin:{origin}:{label}")
            selected.extend(ordered[:quota])
        selected.sort(key=lambda item: (item.label, item.record.manifest_id))
        views[origin] = tuple(selected)
    return views


def shared_dataset_test_views(
    records: Sequence[MappedClassifierRecord] | Iterable[MappedClassifierRecord],
    *,
    seed: int = 42,
    detailed: bool = False,
) -> dict[str, dict[str, tuple[MappedClassifierRecord, ...]]]:
    """Construye pares con igual clase y soporte para comparar datasets."""

    test = [item for item in records if item.record.split == "test"]
    origin_key = (
        (lambda item: item.detailed_origin)
        if detailed
        else (lambda item: item.dataset_origin)
    )
    origins = sorted({origin_key(item) for item in test})
    output: dict[str, dict[str, tuple[MappedClassifierRecord, ...]]] = {}
    for left_index, left in enumerate(origins):
        for right in origins[left_index + 1 :]:
            left_rows = [item for item in test if origin_key(item) == left]
            right_rows = [item for item in test if origin_key(item) == right]
            left_counts = Counter(item.label for item in left_rows)
            right_counts = Counter(item.label for item in right_rows)
            shared = sorted(set(left_counts).intersection(right_counts))
            if not shared:
                continue
            quota = min(
                min(left_counts[label], right_counts[label]) for label in shared
            )
            pair: dict[str, tuple[MappedClassifierRecord, ...]] = {}
            for origin, rows in ((left, left_rows), (right, right_rows)):
                selected: list[MappedClassifierRecord] = []
                for label in shared:
                    candidates = [item for item in rows if item.label == label]
                    ordered = _deterministic_shuffle(
                        candidates, f"{seed}:shared:{left}:{right}:{origin}:{label}"
                    )
                    selected.extend(ordered[:quota])
                selected.sort(key=lambda item: (item.label, item.record.manifest_id))
                pair[origin] = tuple(selected)
            output[f"{left}__vs__{right}"] = pair
    return output


def _fair_select_by_dataset_and_origin(
    records: Sequence[MappedClassifierRecord],
    count: int,
    *,
    seed: str,
) -> list[MappedClassifierRecord]:
    if count > len(records):
        raise ValueError(f"Se solicitaron {count} filas de un pool de {len(records)}")

    by_dataset: dict[str, list[MappedClassifierRecord]] = defaultdict(list)
    for item in records:
        by_dataset[item.dataset_origin].append(item)

    dataset_queues: dict[str, deque[MappedClassifierRecord]] = {}
    for dataset, rows in sorted(by_dataset.items()):
        origin_groups: dict[str, list[MappedClassifierRecord]] = defaultdict(list)
        for item in rows:
            origin_groups[item.detailed_origin].append(item)
        interleaved = _round_robin_groups(
            origin_groups,
            seed=f"{seed}:dataset:{dataset}",
        )
        dataset_queues[dataset] = deque(interleaved)

    dataset_order = sorted(dataset_queues)
    random.Random(f"{seed}:dataset-order").shuffle(dataset_order)
    selected: list[MappedClassifierRecord] = []
    while len(selected) < count:
        progressed = False
        for dataset in dataset_order:
            queue = dataset_queues[dataset]
            if queue:
                selected.append(queue.popleft())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise AssertionError("El reparto por origen agoto el pool prematuramente")
    return selected


def _round_robin_groups(
    groups: Mapping[str, Sequence[MappedClassifierRecord]],
    *,
    seed: str,
) -> list[MappedClassifierRecord]:
    queues: dict[str, deque[MappedClassifierRecord]] = {}
    for name, values in sorted(groups.items()):
        queues[name] = deque(_deterministic_shuffle(values, f"{seed}:{name}"))
    order = sorted(queues)
    random.Random(f"{seed}:order").shuffle(order)
    output: list[MappedClassifierRecord] = []
    while any(queues.values()):
        for name in order:
            if queues[name]:
                output.append(queues[name].popleft())
    return output


def _deterministic_shuffle(
    records: Sequence[MappedClassifierRecord] | Iterable[MappedClassifierRecord],
    seed: str,
) -> list[MappedClassifierRecord]:
    values = sorted(records, key=lambda item: item.record.manifest_id)
    random.Random(seed).shuffle(values)
    return values


def _support_report(records: Iterable[MappedClassifierRecord]) -> dict[str, Any]:
    values = tuple(records)
    class_counts = Counter(item.label for item in values)
    split_counts: dict[str, Any] = {}
    for split in SPLIT_UNITS:
        subset = [item for item in values if item.record.split == split]
        split_counts[split] = {
            "rows": len(subset),
            "class_counts": dict(sorted(Counter(item.label for item in subset).items())),
            "by_dataset": _nested_support(subset, "dataset_origin"),
            "by_origin": _nested_support(subset, "detailed_origin"),
        }
    return {
        "rows": len(values),
        "class_counts": dict(sorted(class_counts.items())),
        "mapping_status_counts": dict(
            sorted(Counter(item.mapping_status for item in values).items())
        ),
        "by_dataset": _nested_support(values, "dataset_origin"),
        "by_origin": _nested_support(values, "detailed_origin"),
        "splits": split_counts,
    }


def _nested_support(
    records: Iterable[MappedClassifierRecord], attribute: str
) -> dict[str, Any]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for item in records:
        grouped[str(getattr(item, attribute))][item.label] += 1
    return {
        name: {
            "rows": sum(counts.values()),
            "class_counts": dict(sorted(counts.items())),
        }
        for name, counts in sorted(grouped.items())
    }


def _cross_split_overlap_count(
    records: Iterable[MappedClassifierRecord | RejectedClassifierRecord],
) -> int:
    splits: dict[str, set[str]] = defaultdict(set)
    for item in records:
        splits[item.record.feature_fingerprint].add(item.record.split)
    return sum(len(values) > 1 for values in splits.values())


def _supervision_key(
    item: MappedClassifierRecord | RejectedClassifierRecord,
) -> str:
    if isinstance(item, MappedClassifierRecord):
        return f"mapped:{item.label}"
    return "rejected:{dataset}:{native}:{subclass}:{status}".format(
        dataset=normalise_dataset(item.record.dataset),
        native=normalise_taxonomy_token(item.record.target_class),
        subclass=normalise_taxonomy_token(item.native_subclass),
        status=item.status,
    )


def _selection_sha256(records: Iterable[MappedClassifierRecord]) -> str:
    payload = [
        {
            "manifest_id": item.record.manifest_id,
            "split": item.record.split,
            "label": item.label,
            "dataset_origin": item.dataset_origin,
            "detailed_origin": item.detailed_origin,
            "mapping_status": item.mapping_status,
            "mapping_reason": item.mapping_reason,
        }
        for item in records
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "MappedClassifierRecord",
    "RejectedClassifierRecord",
    "SPLIT_UNITS",
    "balance_classifier_splits",
    "balanced_origin_test_views",
    "build_balanced_stratified_classifier_corpus",
    "classifier_origins",
    "deduplicate_classifier_universe",
    "deduplicate_mapped_records",
    "map_classifier_records",
    "map_multidataset_classifier_records",
    "shared_dataset_test_views",
]
