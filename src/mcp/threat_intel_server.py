# src/mcp/threat_intel_server.py
"""Servidor MCP de threat intel: tipo de ataque -> ATT&CK, CAPEC y mitigaciones.

El clasificador operativo puede devolver uno de los dieciseis tipos de la
taxonomia multidataset. Todas las consultas operativas exigen ese tipo y se
resuelven exclusivamente contra su entrada especifica. Un tipo ausente o fuera
de la taxonomia produce un error controlado; nunca se degrada silenciosamente
a una familia generica.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

try:  # SDK MCP oficial; opcional para el modo in-process
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - sin SDK solo se pierde el modo stdio
    FastMCP = None

from src.mcp.common import register_tools, tool_result
from src.contracts.attack_taxonomy import (
    MULTIDATASET_ATTACK_CLASSES,
    MULTIDATASET_TAXONOMY_VERSION,
    SUPPORTED_ATTACK_TAXONOMY_VERSIONS,
)

mcp_app = FastMCP("mcp-threat-intel") if FastMCP is not None else None

CATALOG_PATH = Path(__file__).resolve().parent / "data" / "threat_intel_catalog.json"

REFERENCE_ID_FIELDS = {
    "attack_techniques": "attack_technique_ids",
    "capec_patterns": "capec_pattern_ids",
    "attack_mitigation_refs": "attack_mitigation_ref_ids",
}

def _validate_catalog(catalog: dict[str, Any]) -> None:
    """Falla de forma cerrada si el catalogo no cubre el contrato de 16 tipos."""

    if "families" in catalog:
        raise ValueError("El catalogo operativo no debe contener familias amplias")

    taxonomy = catalog.get("attack_taxonomy") or {}
    if taxonomy.get("version") != MULTIDATASET_TAXONOMY_VERSION:
        raise ValueError(
            "Version de taxonomia del catalogo incompatible: "
            f"{taxonomy.get('version')!r}"
        )
    compatible_versions = tuple(taxonomy.get("compatible_versions") or ())
    if set(compatible_versions) != set(SUPPORTED_ATTACK_TAXONOMY_VERSIONS):
        raise ValueError(
            "Versiones de taxonomia compatibles incompletas: "
            f"{compatible_versions!r}"
        )
    declared = tuple(taxonomy.get("classes") or ())
    if declared != MULTIDATASET_ATTACK_CLASSES:
        raise ValueError("El catalogo no declara exactamente las 16 clases operativas")

    attack_types = catalog.get("attack_types") or {}
    if set(attack_types) != set(MULTIDATASET_ATTACK_CLASSES):
        missing = sorted(set(MULTIDATASET_ATTACK_CLASSES) - set(attack_types))
        extra = sorted(set(attack_types) - set(MULTIDATASET_ATTACK_CLASSES))
        raise ValueError(
            f"Cobertura de tipos incompleta: missing={missing} extra={extra}"
        )

    reference_catalog = catalog.get("reference_catalog") or {}
    if set(reference_catalog) != set(REFERENCE_ID_FIELDS):
        raise ValueError("El catalogo global de referencias esta incompleto")
    for attack_type in MULTIDATASET_ATTACK_CLASSES:
        entry = attack_types[attack_type]
        if "family" in entry:
            raise ValueError(
                f"{attack_type} no debe contener un mapeo a familia amplia"
            )

        mitigations = entry.get("mitigations") or {}
        phases = ("containment", "eradication", "prevention")
        if any(not isinstance(mitigations.get(phase), list) for phase in phases):
            raise ValueError(f"Fases de mitigacion incompletas para {attack_type}")
        if sum(len(mitigations[phase]) for phase in phases) < 5:
            raise ValueError(
                f"{attack_type} debe aportar al menos cinco mitigaciones catalogadas"
            )

        for group, id_field in REFERENCE_ID_FIELDS.items():
            requested_ids = entry.get(id_field) or []
            registry = reference_catalog.get(group) or {}
            available_ids = set(registry)
            if any(
                not isinstance(reference, dict)
                or reference.get("id") != reference_id
                for reference_id, reference in registry.items()
            ):
                raise ValueError(f"Registro global {group} invalido")
            if not requested_ids or not set(requested_ids) <= available_ids:
                raise ValueError(
                    f"Referencias {group} invalidas para {attack_type}: "
                    f"{requested_ids!r}"
                )

    for group, id_field in REFERENCE_ID_FIELDS.items():
        used_ids = {
            reference_id
            for entry in attack_types.values()
            for reference_id in (entry.get(id_field) or [])
        }
        registered_ids = set(reference_catalog[group])
        if registered_ids != used_ids:
            raise ValueError(
                f"Registro global {group} no minimal: "
                f"unused={sorted(registered_ids - used_ids)} "
                f"missing={sorted(used_ids - registered_ids)}"
            )


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    with CATALOG_PATH.open("r", encoding="utf-8") as fh:
        catalog = json.load(fh)
    _validate_catalog(catalog)
    return catalog


def normalize_attack_type(attack_type: str | None) -> str | None:
    """Acepta exclusivamente una de las 16 etiquetas contractuales exactas."""

    if attack_type is None:
        return None
    value = str(attack_type)
    return value if value in MULTIDATASET_ATTACK_CLASSES else None


def _select_references(
    reference_catalog: dict[str, Any],
    attack_entry: dict[str, Any],
    group: str,
) -> list[dict[str, Any]]:
    requested = attack_entry.get(REFERENCE_ID_FIELDS[group]) or []
    by_id = reference_catalog[group]
    return [dict(by_id[reference_id]) for reference_id in requested]


def _resolve_catalog_entry(attack_type: str | None) -> dict[str, Any]:
    """Resuelve estrictamente una entrada de las 16 clases operativas."""

    catalog = _catalog()
    if attack_type is None or not str(attack_type).strip():
        raise ValueError("attack_type es obligatorio para consultar el catalogo")
    canonical_attack_type = normalize_attack_type(attack_type)
    if canonical_attack_type is None:
        raise ValueError(
            f"Tipo de ataque fuera de la taxonomia operativa: {attack_type!r}"
        )

    attack_entry = catalog["attack_types"][canonical_attack_type]
    materialized = {
        "mitigations": attack_entry["mitigations"],
        "note": attack_entry.get("note"),
        "reference_quality": attack_entry.get("reference_quality") or {},
    }
    for group in REFERENCE_ID_FIELDS:
        materialized[group] = _select_references(
            catalog["reference_catalog"], attack_entry, group
        )
    return {
        "attack_type": canonical_attack_type,
        "catalog_scope": "attack_type",
        "entry": materialized,
    }


@tool_result
def list_attack_types() -> dict[str, Any]:
    """Tipos de ataque cubiertos por el mitigador operativo."""

    catalog = _catalog()
    return {
        "attack_types": list(MULTIDATASET_ATTACK_CLASSES),
        "taxonomy_version": catalog["attack_taxonomy"]["version"],
        "compatible_taxonomy_versions": list(
            catalog["attack_taxonomy"]["compatible_versions"]
        ),
        "version": catalog["version"],
    }


@tool_result
def map_attack_type_to_attack(attack_type: str) -> dict[str, Any]:
    """Tecnicas ATT&CK asociadas estrictamente a un tipo operativo."""

    resolved = _resolve_catalog_entry(attack_type)
    entry = resolved["entry"]
    return {
        "attack_type": resolved["attack_type"],
        "catalog_scope": resolved["catalog_scope"],
        "requested_attack_type": attack_type,
        "attack_techniques": entry.get("attack_techniques", []),
        "attack_mitigation_refs": entry.get("attack_mitigation_refs", []),
    }


@tool_result
def map_attack_type_to_capec(attack_type: str) -> dict[str, Any]:
    """Patrones CAPEC asociados estrictamente a un tipo operativo."""

    resolved = _resolve_catalog_entry(attack_type)
    entry = resolved["entry"]
    return {
        "attack_type": resolved["attack_type"],
        "catalog_scope": resolved["catalog_scope"],
        "requested_attack_type": attack_type,
        "capec_patterns": entry.get("capec_patterns", []),
    }


@tool_result
def suggest_mitigations(
    attack_type: str,
    schema_profile: str | None = None,
) -> dict[str, Any]:
    """Mitigaciones del tipo predicho y acciones complementarias por perfil."""

    catalog = _catalog()
    resolved = _resolve_catalog_entry(attack_type)
    entry = resolved["entry"]
    mitigations = entry.get("mitigations", {})
    profile_key = (schema_profile or "unknown").strip().lower()
    profile_actions = catalog["schema_profile_actions"].get(
        profile_key, catalog["schema_profile_actions"]["unknown"]
    )
    ordered: list[str] = [
        *mitigations.get("containment", []),
        *mitigations.get("eradication", []),
        *mitigations.get("prevention", []),
    ]
    return {
        "attack_type": resolved["attack_type"],
        "catalog_scope": resolved["catalog_scope"],
        "requested_attack_type": attack_type,
        "catalog_version": catalog["version"],
        "taxonomy_version": catalog["attack_taxonomy"]["version"],
        "compatible_taxonomy_versions": list(
            catalog["attack_taxonomy"]["compatible_versions"]
        ),
        "schema_profile": profile_key,
        "mitigations_by_phase": mitigations,
        "mitigations_ordered": ordered,
        "profile_actions": profile_actions,
        "references": {
            "attack_techniques": entry.get("attack_techniques", []),
            "capec_patterns": entry.get("capec_patterns", []),
            "attack_mitigation_refs": entry.get("attack_mitigation_refs", []),
        },
        "reference_quality": entry.get("reference_quality") or {},
        "note": entry.get("note"),
    }


@tool_result
def get_multidataset_attack_type_coverage() -> dict[str, Any]:
    """Cobertura auditable del catalogo para las 16 clases operativas."""

    catalog = _catalog()
    rows: list[dict[str, Any]] = []
    for attack_type in MULTIDATASET_ATTACK_CLASSES:
        resolved = _resolve_catalog_entry(attack_type)
        entry = resolved["entry"]
        rows.append(
            {
                "attack_type": attack_type,
                "catalog_scope": resolved["catalog_scope"],
                "mitigations": sum(
                    len(items)
                    for items in (entry.get("mitigations") or {}).values()
                ),
                "attack_techniques": len(entry.get("attack_techniques") or []),
                "capec_patterns": len(entry.get("capec_patterns") or []),
                "attack_mitigation_refs": len(
                    entry.get("attack_mitigation_refs") or []
                ),
            }
        )
    return {
        "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        "total": len(rows),
        "covered": sum(
            1
            for row in rows
            if row["catalog_scope"] == "attack_type"
            and row["mitigations"] >= 5
            and row["attack_techniques"] > 0
            and row["capec_patterns"] > 0
            and row["attack_mitigation_refs"] > 0
        ),
        "mappings": rows,
    }


@tool_result
def get_jorge_capec_coverage() -> dict[str, Any]:
    """Cobertura del mapeo CAPEC del TFM de Jorge por este catalogo.

    Para cada tipo de Edge-IIoTset que Jorge mapeo manualmente a CAPEC
    (Tabla 3.4 de su TFM), comprueba directamente su entrada tipada. Lo
    consumen el auditor (F5) y la memoria.
    """
    catalog = _catalog()
    table = catalog.get("jorge_tfm_capec_table", {})
    rows: list[dict[str, Any]] = []
    for mapping in table.get("mappings", []):
        attack_type = mapping.get("attack_type")
        entry = catalog["attack_types"].get(attack_type, {})
        catalog_capec_ids = set(entry.get("capec_pattern_ids") or [])
        public_mapping = {
            key: value for key, value in mapping.items() if key != "family"
        }
        rows.append(
            {
                **public_mapping,
                "covered": mapping.get("capec_id") in catalog_capec_ids,
                "extra_attack_techniques": list(
                    entry.get("attack_technique_ids") or []
                ),
                "extra_attack_mitigations": list(
                    entry.get("attack_mitigation_ref_ids") or []
                ),
            }
        )
    return {
        "source": table.get("source"),
        "description": table.get("description"),
        "total": len(rows),
        "covered": sum(1 for row in rows if row["covered"]),
        "mappings": rows,
    }


TOOLS = {
    "list_attack_types": list_attack_types,
    "map_attack_type_to_attack": map_attack_type_to_attack,
    "map_attack_type_to_capec": map_attack_type_to_capec,
    "suggest_mitigations": suggest_mitigations,
    "get_jorge_capec_coverage": get_jorge_capec_coverage,
    "get_multidataset_attack_type_coverage": get_multidataset_attack_type_coverage,
}
register_tools(mcp_app, TOOLS)


if __name__ == "__main__":
    if mcp_app is None:
        raise SystemExit("SDK MCP no disponible: instala el proyecto con 'pip install .'.")
    mcp_app.run()
