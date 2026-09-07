"""Deterministic class balancing for the binary detector.

The validation manifests freeze targets and splits before standardisation.  A
binary detector must therefore be balanced by selecting a subset inside every
``split x logical origin`` cell; moving rows between splits would invalidate
the frozen protocol.  TON-IoT is deliberately expanded into its four source
modalities so that an aggregate 50/50 count cannot hide an imbalanced network,
telemetry, Linux or Windows subset.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from src.eval.validation_campaign import PreparedRecord


BINARY_BALANCE_POLICY_VERSION = "binary_split_origin_1to1_v1_2026-09-05"
SPLITS = ("train", "val", "test")
_SPLIT_ORDER = {split: index for index, split in enumerate(SPLITS)}
_TON_DATASET_KEYS = frozenset({"toniot"})
_TON_SOURCE_GROUPS = (
    ("train_test_network_dataset", "ton_iot_network"),
    ("train_test_iot_dataset", "ton_iot_telemetry"),
    ("train_test_linux_dataset", "ton_iot_linux"),
    ("train_test_windows_dataset", "ton_iot_windows"),
)


class BinaryBalanceError(ValueError):
    """The validated corpus cannot satisfy the strict 1:1 policy."""


def logical_binary_origin(record: PreparedRecord) -> str:
    """Return the auditable origin used for binary balancing and metrics."""

    dataset_key = re.sub(r"[^a-z0-9]", "", record.dataset.casefold())
    if dataset_key not in _TON_DATASET_KEYS:
        return record.dataset.casefold().strip()

    source = record.source_file.replace("\\", "/").casefold()
    matches = [name for marker, name in _TON_SOURCE_GROUPS if marker in source]
    if len(matches) != 1:
        raise BinaryBalanceError(
            "No se puede asignar exactamente una modalidad TON-IoT a "
            f"{record.manifest_id!r}: source_file={record.source_file!r}"
        )
    return matches[0]


def balance_binary_records(
    records: Sequence[PreparedRecord] | Iterable[PreparedRecord],
    *,
    seed: int = 42,
) -> tuple[tuple[PreparedRecord, ...], dict[str, Any]]:
    """Select equal normal/attack support in each split and logical origin.

    Selection is deterministic and independent of input order.  Rows are
    ranked by a seeded SHA-256 value instead of taking the first records from a
    source file, which avoids temporal or file-order bias.
    """

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed debe ser un entero")
    values = tuple(records)
    if not values:
        raise BinaryBalanceError("No hay registros binarios para balancear")

    grouped: dict[tuple[str, str, bool], list[PreparedRecord]] = defaultdict(list)
    for record in values:
        if "binary" not in record.tasks:
            raise BinaryBalanceError(
                f"{record.manifest_id!r} no declara la tarea binary"
            )
        if not isinstance(record.target_is_attack, bool):
            raise BinaryBalanceError(
                f"{record.manifest_id!r} no tiene target binario valido"
            )
        if record.split not in _SPLIT_ORDER:
            raise BinaryBalanceError(
                f"{record.manifest_id!r} usa un split no soportado: {record.split!r}"
            )
        grouped[(record.split, logical_binary_origin(record), record.target_is_attack)].append(
            record
        )

    origins = sorted({origin for _split, origin, _label in grouped})
    missing: list[str] = []
    for split in SPLITS:
        for origin in origins:
            absent = [
                _label_name(label)
                for label in (False, True)
                if not grouped.get((split, origin, label))
            ]
            if absent:
                missing.append(f"{split}/{origin}: {','.join(absent)}")
    if missing:
        raise BinaryBalanceError(
            "No se puede aplicar balance 1:1; faltan clases en: " + "; ".join(missing)
        )

    selected: list[PreparedRecord] = []
    quotas: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for split in SPLITS:
        for origin in origins:
            normal = grouped[(split, origin, False)]
            attack = grouped[(split, origin, True)]
            quota = min(len(normal), len(attack))
            quotas[split][origin] = quota
            for label, candidates in ((False, normal), (True, attack)):
                ranked = sorted(
                    candidates,
                    key=lambda record: (
                        _selection_rank(
                            record,
                            seed=seed,
                            split=split,
                            origin=origin,
                            label=label,
                        ),
                        record.manifest_id,
                    ),
                )
                selected.extend(ranked[:quota])

    selected.sort(
        key=lambda record: (
            _SPLIT_ORDER[record.split],
            logical_binary_origin(record),
            int(bool(record.target_is_attack)),
            record.manifest_id,
        )
    )
    report = {
        "policy_version": BINARY_BALANCE_POLICY_VERSION,
        "seed": seed,
        "grouping": "split x logical_origin x binary_label",
        "selection": "seeded_sha256_without_replacement",
        "input_rows": len(values),
        "output_rows": len(selected),
        "removed_rows": len(values) - len(selected),
        "origins": origins,
        "quota_per_label": quotas,
        "before": _support_report(values),
        "after": _support_report(selected),
        "selected_manifest_ids_sha256": _manifest_ids_sha256(selected),
    }
    _assert_strict_balance(report["after"])
    return tuple(selected), report


def _selection_rank(
    record: PreparedRecord,
    *,
    seed: int,
    split: str,
    origin: str,
    label: bool,
) -> bytes:
    payload = "\x1f".join(
        (
            BINARY_BALANCE_POLICY_VERSION,
            str(seed),
            split,
            origin,
            "attack" if label else "normal",
            record.manifest_id,
            record.feature_fingerprint,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _support_report(records: Sequence[PreparedRecord]) -> dict[str, Any]:
    global_counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    origin_counts: dict[str, Counter[str]] = defaultdict(Counter)
    origin_split_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for record in records:
        label = _label_name(bool(record.target_is_attack))
        origin = logical_binary_origin(record)
        global_counts[label] += 1
        split_counts[record.split][label] += 1
        origin_counts[origin][label] += 1
        origin_split_counts[origin][record.split][label] += 1

    def payload(counts: Mapping[str, int]) -> dict[str, Any]:
        normal = int(counts.get("normal", 0))
        attack = int(counts.get("attack", 0))
        return {
            "rows": normal + attack,
            "class_counts": {"normal": normal, "attack": attack},
            "balanced": normal == attack and normal > 0,
        }

    return {
        "global": payload(global_counts),
        "by_split": {
            split: payload(split_counts[split])
            for split in SPLITS
        },
        "by_origin": {
            origin: {
                **payload(origin_counts[origin]),
                "by_split": {
                    split: payload(origin_split_counts[origin][split])
                    for split in SPLITS
                },
            }
            for origin in sorted(origin_counts)
        },
    }


def _assert_strict_balance(support: Mapping[str, Any]) -> None:
    failures: list[str] = []
    if not support["global"]["balanced"]:
        failures.append("global")
    failures.extend(
        f"split:{split}"
        for split, values in support["by_split"].items()
        if not values["balanced"]
    )
    for origin, values in support["by_origin"].items():
        if not values["balanced"]:
            failures.append(f"origin:{origin}")
        failures.extend(
            f"origin_split:{origin}/{split}"
            for split, split_values in values["by_split"].items()
            if not split_values["balanced"]
        )
    if failures:
        raise BinaryBalanceError(
            "El resultado no cumple el balance binario estricto: " + ", ".join(failures)
        )


def _manifest_ids_sha256(records: Sequence[PreparedRecord]) -> str:
    payload = json.dumps(
        sorted(record.manifest_id for record in records),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _label_name(label: bool) -> str:
    return "attack" if label else "normal"
