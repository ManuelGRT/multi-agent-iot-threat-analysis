"""Contrato puro y reutilizable del catalogo operativo de amenazas.

Este modulo no registra herramientas MCP. Centraliza la version desplegada,
la validacion del fichero y la comprobacion de que una salida del mitigador
conserva exactamente la base catalogada para el tipo de ataque observado.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import date
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from src.contracts.attack_taxonomy import (
    MULTIDATASET_ATTACK_CLASSES,
    MULTIDATASET_TAXONOMY_VERSION,
    SUPPORTED_ATTACK_TAXONOMY_VERSIONS,
)


THREAT_INTEL_CATALOG_VERSION = "3.0"
CATALOG_PATH = Path(__file__).resolve().parent / "data" / "threat_intel_catalog.json"

REFERENCE_ID_FIELDS = {
    "attack_techniques": "attack_technique_ids",
    "capec_patterns": "capec_pattern_ids",
    "attack_mitigation_refs": "attack_mitigation_ref_ids",
}
MITIGATION_PHASES = ("containment", "eradication", "prevention")
CATALOG_TOP_LEVEL_FIELDS = {
    "version",
    "generated_at",
    "description",
    "attack_taxonomy",
    "reference_catalog",
    "attack_types",
    "schema_profile_actions",
}


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_reference_url(value: Any) -> bool:
    if not _nonempty_text(value):
        return False
    parsed = urlparse(str(value))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_catalog(value: dict[str, Any]) -> None:
    """Falla cerrado cuando el fichero no satisface el contrato de catalogo v3."""

    if not isinstance(value, dict):
        raise ValueError("El catalogo debe ser un objeto JSON")
    if set(value) != CATALOG_TOP_LEVEL_FIELDS:
        raise ValueError(
            "Campos de primer nivel del catalogo incompatibles: "
            f"missing={sorted(CATALOG_TOP_LEVEL_FIELDS - set(value))} "
            f"extra={sorted(set(value) - CATALOG_TOP_LEVEL_FIELDS)}"
        )
    if value.get("version") != THREAT_INTEL_CATALOG_VERSION:
        raise ValueError(
            "Version del catalogo incompatible: "
            f"{value.get('version')!r}; esperada={THREAT_INTEL_CATALOG_VERSION!r}"
        )
    if not _nonempty_text(value.get("description")):
        raise ValueError("El catalogo requiere una descripcion no vacia")
    try:
        date.fromisoformat(str(value.get("generated_at")))
    except (TypeError, ValueError) as exc:
        raise ValueError("generated_at debe ser una fecha ISO valida") from exc
    if "families" in value:
        raise ValueError("El catalogo operativo no debe contener familias amplias")

    taxonomy = value.get("attack_taxonomy") or {}
    if not isinstance(taxonomy, dict):
        raise ValueError("attack_taxonomy debe ser un objeto")
    if taxonomy.get("version") != MULTIDATASET_TAXONOMY_VERSION:
        raise ValueError(
            "Version de taxonomia del catalogo incompatible: "
            f"{taxonomy.get('version')!r}"
        )
    raw_compatible_versions = taxonomy.get("compatible_versions")
    if not isinstance(raw_compatible_versions, list) or any(
        not _nonempty_text(version) for version in raw_compatible_versions
    ):
        raise ValueError("compatible_versions debe ser una lista de versiones")
    compatible_versions = tuple(raw_compatible_versions)
    if set(compatible_versions) != set(SUPPORTED_ATTACK_TAXONOMY_VERSIONS):
        raise ValueError(
            "Versiones de taxonomia compatibles incompletas: "
            f"{compatible_versions!r}"
        )
    declared = tuple(taxonomy.get("classes") or ())
    if declared != MULTIDATASET_ATTACK_CLASSES:
        raise ValueError("El catalogo no declara exactamente las 16 clases operativas")

    reference_catalog = value.get("reference_catalog") or {}
    if (
        not isinstance(reference_catalog, dict)
        or set(reference_catalog) != set(REFERENCE_ID_FIELDS)
    ):
        raise ValueError("El catalogo global de referencias esta incompleto")
    for group, entries in reference_catalog.items():
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Registro global {group} invalido")
        for reference_id, reference in entries.items():
            if (
                not isinstance(reference, dict)
                or reference.get("id") != reference_id
                or not _nonempty_text(reference.get("name"))
                or not _valid_reference_url(reference.get("url"))
            ):
                raise ValueError(
                    f"Referencia invalida en {group}: {reference_id!r}"
                )

    attack_types = value.get("attack_types") or {}
    if not isinstance(attack_types, dict):
        raise ValueError("attack_types debe ser un objeto")
    if set(attack_types) != set(MULTIDATASET_ATTACK_CLASSES):
        missing = sorted(set(MULTIDATASET_ATTACK_CLASSES) - set(attack_types))
        extra = sorted(set(attack_types) - set(MULTIDATASET_ATTACK_CLASSES))
        raise ValueError(
            f"Cobertura de tipos incompleta: missing={missing} extra={extra}"
        )
    for attack_type in MULTIDATASET_ATTACK_CLASSES:
        entry = attack_types[attack_type]
        if not isinstance(entry, dict):
            raise ValueError(f"Entrada de catalogo invalida para {attack_type}")
        if "family" in entry:
            raise ValueError(f"{attack_type} no debe mapear a una familia amplia")

        mitigations = entry.get("mitigations") or {}
        if set(mitigations) != set(MITIGATION_PHASES):
            raise ValueError(f"Fases de mitigacion incompletas para {attack_type}")
        ordered_actions: list[str] = []
        for phase in MITIGATION_PHASES:
            actions = mitigations.get(phase)
            if (
                not isinstance(actions, list)
                or not actions
                or any(not _nonempty_text(action) for action in actions)
            ):
                raise ValueError(
                    f"Acciones de mitigacion invalidas para {attack_type}/{phase}"
                )
            ordered_actions.extend(str(action) for action in actions)
        if len(ordered_actions) < 5 or len(set(ordered_actions)) != len(ordered_actions):
            raise ValueError(
                f"{attack_type} requiere al menos cinco mitigaciones unicas"
            )

        for group, id_field in REFERENCE_ID_FIELDS.items():
            requested_ids = entry.get(id_field)
            available_ids = set(reference_catalog[group])
            if (
                not isinstance(requested_ids, list)
                or not requested_ids
                or any(
                    not isinstance(reference_id, str)
                    or reference_id not in available_ids
                    for reference_id in requested_ids
                )
                or len(set(requested_ids)) != len(requested_ids)
            ):
                raise ValueError(
                    f"Referencias {group} invalidas para {attack_type}: "
                    f"{requested_ids!r}"
                )

        reference_quality = entry.get("reference_quality") or {}
        if not isinstance(reference_quality, dict) or any(
            not _nonempty_text(key) or not _nonempty_text(item)
            for key, item in reference_quality.items()
        ):
            raise ValueError(f"reference_quality invalido para {attack_type}")
        if entry.get("note") is not None and not _nonempty_text(entry.get("note")):
            raise ValueError(f"note invalida para {attack_type}")

    for group, id_field in REFERENCE_ID_FIELDS.items():
        used_ids = {
            reference_id
            for entry in attack_types.values()
            for reference_id in entry[id_field]
        }
        registered_ids = set(reference_catalog[group])
        if registered_ids != used_ids:
            raise ValueError(
                f"Registro global {group} no minimal: "
                f"unused={sorted(registered_ids - used_ids)} "
                f"missing={sorted(used_ids - registered_ids)}"
            )

    profile_actions = value.get("schema_profile_actions")
    if not isinstance(profile_actions, dict) or "unknown" not in profile_actions:
        raise ValueError("schema_profile_actions requiere el perfil unknown")
    for profile, actions in profile_actions.items():
        if (
            not _nonempty_text(profile)
            or not isinstance(actions, list)
            or not actions
            or any(not _nonempty_text(action) for action in actions)
            or len(set(actions)) != len(actions)
        ):
            raise ValueError(f"Acciones de perfil invalidas para {profile!r}")


@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, Any]:
    with CATALOG_PATH.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    validate_catalog(value)
    return value


def catalog() -> dict[str, Any]:
    """Devuelve una copia para que ningun consumidor altere la cache validada."""

    return deepcopy(_load_catalog())


# Conserva la interfaz de invalidacion usada por la auditoria ejecutable sin
# exponer el objeto mutable almacenado en cache.
catalog.cache_clear = _load_catalog.cache_clear  # type: ignore[attr-defined]


def normalize_attack_type(attack_type: str | None) -> str | None:
    if attack_type is None:
        return None
    value = str(attack_type)
    return value if value in MULTIDATASET_ATTACK_CLASSES else None


def _select_references(
    reference_catalog: dict[str, Any],
    attack_entry: dict[str, Any],
    group: str,
) -> list[dict[str, Any]]:
    return [
        dict(reference_catalog[group][reference_id])
        for reference_id in attack_entry[REFERENCE_ID_FIELDS[group]]
    ]


def resolve_catalog_entry(attack_type: str | None) -> dict[str, Any]:
    """Materializa una entrada exacta, sin aliases ni fallback de familia."""

    value = catalog()
    if attack_type is None or not str(attack_type).strip():
        raise ValueError("attack_type es obligatorio para consultar el catalogo")
    canonical_attack_type = normalize_attack_type(attack_type)
    if canonical_attack_type is None:
        raise ValueError(
            f"Tipo de ataque fuera de la taxonomia operativa: {attack_type!r}"
        )
    attack_entry = value["attack_types"][canonical_attack_type]
    materialized = {
        "mitigations": attack_entry["mitigations"],
        "note": attack_entry.get("note"),
        "reference_quality": attack_entry.get("reference_quality") or {},
    }
    for group in REFERENCE_ID_FIELDS:
        materialized[group] = _select_references(
            value["reference_catalog"], attack_entry, group
        )
    return {
        "attack_type": canonical_attack_type,
        "catalog_scope": "attack_type",
        "entry": materialized,
    }


def expected_catalog_contract(
    attack_type: str,
    schema_profile: str | None,
) -> dict[str, Any]:
    """Salida catalogada exacta que debe conservar un caso malicioso."""

    value = catalog()
    resolved = resolve_catalog_entry(attack_type)
    entry = resolved["entry"]
    profile = str(schema_profile or "unknown").strip().lower()
    profile_actions = value["schema_profile_actions"].get(
        profile, value["schema_profile_actions"]["unknown"]
    )
    items = [
        {"text": text, "phase": phase}
        for phase in MITIGATION_PHASES
        for text in entry["mitigations"][phase]
    ]
    items.extend({"text": text, "phase": "profile"} for text in profile_actions)

    references: list[dict[str, Any]] = []
    for group in ("attack_techniques", "capec_patterns", "attack_mitigation_refs"):
        for reference in entry[group]:
            references.append(
                {
                    "attack_id": None if group == "capec_patterns" else reference["id"],
                    "capec_id": reference["id"] if group == "capec_patterns" else None,
                    "name": reference["name"],
                    "url": reference["url"],
                    "source": "catalog",
                }
            )
    return {
        "attack_type": resolved["attack_type"],
        "catalog_scope": resolved["catalog_scope"],
        "catalog_version": value["version"],
        "taxonomy_version": value["attack_taxonomy"]["version"],
        "compatible_taxonomy_versions": tuple(
            value["attack_taxonomy"]["compatible_versions"]
        ),
        "reference_quality": dict(entry.get("reference_quality") or {}),
        "items": items,
        "references": references,
    }


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json"))
    return {}


def _reference_key(reference: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        reference.get("attack_id"),
        reference.get("capec_id"),
        reference.get("name"),
        reference.get("url"),
        reference.get("source"),
    )


def catalog_output_issues(
    explanation: Any,
    *,
    attack_type: str,
    schema_profile: str | None,
) -> list[str]:
    """Contrasta metadatos, acciones y referencias con la entrada v3 real."""

    payload = _as_mapping(explanation)
    try:
        expected = expected_catalog_contract(attack_type, schema_profile)
    except (OSError, ValueError, json.JSONDecodeError):
        return ["catalog_unavailable_or_invalid"]

    issues: list[str] = []
    metadata = {
        "attack_type": attack_type,
        "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        "catalog_scope": "attack_type",
        "catalog_version": THREAT_INTEL_CATALOG_VERSION,
        "catalog_taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
    }
    for field, expected_value in metadata.items():
        if payload.get(field) != expected_value:
            issues.append(f"{field}_mismatch")
    if tuple(payload.get("catalog_compatible_taxonomy_versions") or ()) != tuple(
        expected["compatible_taxonomy_versions"]
    ):
        issues.append("catalog_compatible_taxonomy_versions_mismatch")
    raw_reference_quality = payload.get("reference_quality") or {}
    if not isinstance(raw_reference_quality, Mapping):
        issues.append("reference_quality_invalid")
        actual_reference_quality: dict[str, Any] = {}
    else:
        actual_reference_quality = dict(raw_reference_quality)
    if actual_reference_quality != expected["reference_quality"]:
        issues.append("reference_quality_mismatch")

    raw_items = payload.get("mitigation_items")
    items = [_as_mapping(item) for item in raw_items] if isinstance(raw_items, list) else []
    if not items:
        issues.append("mitigation_items_missing")
    expected_items = Counter(
        (item["text"], item["phase"]) for item in expected["items"]
    )
    anchored_items: Counter[tuple[Any, Any]] = Counter()
    for item in items:
        source = item.get("source")
        if source in {"catalog", "llm"}:
            text = item.get("text")
            phase = item.get("phase")
            if not _nonempty_text(text) or phase not in {
                *MITIGATION_PHASES,
                "profile",
            }:
                issues.append("anchored_mitigation_item_invalid")
                continue
            anchored_items[(text, phase)] += 1
            if source == "catalog" and item.get("base") not in (None, ""):
                issues.append("catalog_item_has_base")
            if source == "llm":
                if item.get("base") != item.get("text"):
                    issues.append("llm_item_base_mismatch")
                if not _nonempty_text(item.get("context")):
                    issues.append("llm_item_context_missing")
            if bool(item.get("context_trusted", False)):
                issues.append("mitigation_context_marked_trusted")
        elif source == "llm_suggested":
            if not _nonempty_text(item.get("text")):
                issues.append("llm_suggested_text_invalid")
            if item.get("phase") is not None:
                issues.append("llm_suggested_phase_invalid")
        else:
            issues.append("mitigation_item_source_invalid")
    if anchored_items != expected_items:
        issues.append("catalog_mitigation_set_mismatch")

    mitigation_texts = payload.get("mitigations")
    if not isinstance(mitigation_texts, list) or mitigation_texts != [
        item.get("text") for item in items
    ]:
        issues.append("mitigations_flattened_mismatch")

    raw_references = payload.get("references")
    if not isinstance(raw_references, list):
        issues.append("references_invalid")
        raw_references = []
    references = [_as_mapping(item) for item in raw_references]
    valid_references: list[dict[str, Any]] = []
    for reference in references:
        if any(
            value is not None and not isinstance(value, str)
            for value in (
                reference.get("attack_id"),
                reference.get("capec_id"),
                reference.get("name"),
                reference.get("url"),
                reference.get("source"),
            )
        ):
            issues.append("reference_entry_invalid")
            continue
        valid_references.append(reference)
    catalog_references = Counter(
        _reference_key(item)
        for item in valid_references
        if item.get("source") == "catalog"
    )
    expected_references = Counter(
        _reference_key(item) for item in expected["references"]
    )
    if catalog_references != expected_references:
        issues.append("catalog_reference_set_mismatch")

    has_suggested = any(
        item.get("source") == "llm_suggested" for item in items
    ) or any(
        item.get("source") == "llm_suggested" for item in valid_references
    )
    if bool(payload.get("has_llm_suggested", False)) != has_suggested:
        issues.append("has_llm_suggested_mismatch")
    if payload.get("source") not in {"catalog", "hybrid"}:
        issues.append("explanation_source_invalid")
    if payload.get("source") == "catalog" and any(
        item.get("source") != "catalog" for item in items
    ):
        issues.append("catalog_source_contains_non_catalog_item")

    first_five = items[:5]
    actually_anchored = len(first_five) == 5 and all(
        item.get("source") == "llm"
        and item.get("base") == item.get("text")
        and _nonempty_text(item.get("context"))
        for item in first_five
    )
    if bool(payload.get("first_five_catalog_anchored", False)) != actually_anchored:
        issues.append("first_five_catalog_anchored_mismatch")

    return list(dict.fromkeys(issues))


__all__ = [
    "CATALOG_PATH",
    "CATALOG_TOP_LEVEL_FIELDS",
    "REFERENCE_ID_FIELDS",
    "THREAT_INTEL_CATALOG_VERSION",
    "catalog",
    "catalog_output_issues",
    "expected_catalog_contract",
    "normalize_attack_type",
    "resolve_catalog_entry",
    "validate_catalog",
]
