# src/mcp/threat_intel_server.py
"""Servidor MCP de threat intel: tipo/familia -> ATT&CK, CAPEC y mitigaciones.

El clasificador operativo puede devolver uno de los dieciseis tipos de la
taxonomia multidataset. El catalogo conserva las familias amplias para
compatibilidad con el modelo historico, pero una prediccion tipada se resuelve
siempre contra su entrada especifica; nunca se degrada silenciosamente a una
familia generica.
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
    broad_family_for_attack_type,
    normalise_taxonomy_token,
)

mcp_app = FastMCP("mcp-threat-intel") if FastMCP is not None else None

CATALOG_PATH = Path(__file__).resolve().parent / "data" / "threat_intel_catalog.json"

# Alias de familias que pueden llegar del clasificador o de labels crudas.
# Incluye los 14 attack types de Edge-IIoTset que usaba el TFM de Jorge.
FAMILY_ALIASES = {
    "normal": "benign",
    "no_attack": "benign",
    "dos": "ddos",
    "ddos_attack": "ddos",
    "ddos_udp": "ddos",
    "ddos_icmp": "ddos",
    "ddos_tcp": "ddos",
    "ddos_http": "ddos",
    "brute_force": "bruteforce",
    "password": "bruteforce",
    "port_scanning": "scanning",
    "reconnaissance": "scanning",
    "recon": "scanning",
    "vulnerability_scanner": "scanning",
    "fingerprinting": "scanning",
    "mirai": "botnet",
    "man_in_the_middle": "mitm",
    "data_exfiltration": "exfiltration",
    "sql_injection": "injection",
    "xss": "injection",
    "uploading": "injection",
    "backdoor": "malware",
    "ransomware": "malware",
    "command_and_control": "botnet",
    "c_c": "botnet",
    "unknown": "unknown_attack",
}

REFERENCE_ID_FIELDS = {
    "attack_techniques": "attack_technique_ids",
    "capec_patterns": "capec_pattern_ids",
    "attack_mitigation_refs": "attack_mitigation_ref_ids",
}

ATTACK_TYPE_ALIASES = {
    "c_c": "Command_and_Control",
    "c2": "Command_and_Control",
    "command_control": "Command_and_Control",
}


def _validate_catalog(catalog: dict[str, Any]) -> None:
    """Falla de forma cerrada si el catalogo no cubre el contrato de 16 tipos."""

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

    families = catalog.get("families") or {}
    for attack_type in MULTIDATASET_ATTACK_CLASSES:
        entry = attack_types[attack_type]
        family = entry.get("family")
        expected_family = broad_family_for_attack_type(attack_type)
        if family != expected_family or family not in families:
            raise ValueError(
                f"Familia invalida para {attack_type}: {family!r}; "
                f"esperada={expected_family!r}"
            )

        mitigations = entry.get("mitigations") or {}
        phases = ("containment", "eradication", "prevention")
        if any(not isinstance(mitigations.get(phase), list) for phase in phases):
            raise ValueError(f"Fases de mitigacion incompletas para {attack_type}")
        if sum(len(mitigations[phase]) for phase in phases) < 5:
            raise ValueError(
                f"{attack_type} debe aportar al menos cinco mitigaciones catalogadas"
            )

        parent = families[family]
        for group, id_field in REFERENCE_ID_FIELDS.items():
            requested_ids = entry.get(id_field) or []
            available_ids = {
                str(reference.get("id"))
                for reference in (parent.get(group) or [])
                if reference.get("id")
            }
            if not requested_ids or not set(requested_ids) <= available_ids:
                raise ValueError(
                    f"Referencias {group} invalidas para {attack_type}: "
                    f"{requested_ids!r}"
                )


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    with CATALOG_PATH.open("r", encoding="utf-8") as fh:
        catalog = json.load(fh)
    _validate_catalog(catalog)
    return catalog


def normalize_family(family: str | None) -> str:
    key = normalise_taxonomy_token(family or "unknown_attack")
    key = FAMILY_ALIASES.get(key, key)
    if key not in _catalog()["families"]:
        return "unknown_attack"
    return key


def normalize_attack_type(attack_type: str | None) -> str | None:
    """Normaliza solo alias ortograficos de las 16 clases, sin agruparlas."""

    key = normalise_taxonomy_token(attack_type)
    if not key:
        return None
    index = {
        normalise_taxonomy_token(label): label
        for label in MULTIDATASET_ATTACK_CLASSES
    }
    return ATTACK_TYPE_ALIASES.get(key) or index.get(key)


def _select_references(
    parent: dict[str, Any],
    attack_entry: dict[str, Any],
    group: str,
) -> list[dict[str, Any]]:
    requested = attack_entry.get(REFERENCE_ID_FIELDS[group]) or []
    by_id = {
        str(item["id"]): item
        for item in (parent.get(group) or [])
        if item.get("id")
    }
    return [dict(by_id[reference_id]) for reference_id in requested]


def _resolve_catalog_entry(
    family: str | None,
    attack_type: str | None = None,
) -> dict[str, Any]:
    """Resuelve una entrada especifica o el fallback familiar compatible."""

    catalog = _catalog()
    explicit_attack_type = attack_type is not None
    # ``family`` pertenece al contrato historico y puede colisionar con el
    # nombre de un tipo (por ejemplo ``mitm``). Solo el argumento explicito
    # ``attack_type`` activa el catalogo de las 16 clases.
    canonical_attack_type = (
        normalize_attack_type(attack_type) if explicit_attack_type else None
    )

    if explicit_attack_type and canonical_attack_type is None:
        unknown = catalog["families"]["unknown_attack"]
        return {
            "family": "unknown_attack",
            "attack_type": None,
            "catalog_scope": "family",
            "family_consistent": False,
            "entry": unknown,
        }

    if canonical_attack_type is not None:
        attack_entry = catalog["attack_types"][canonical_attack_type]
        canonical_family = str(attack_entry["family"])
        requested_family = normalize_family(family)
        parent = catalog["families"][canonical_family]
        materialized = {
            "mitigations": attack_entry["mitigations"],
            "note": attack_entry.get("note"),
            "reference_quality": attack_entry.get("reference_quality") or {},
        }
        for group in REFERENCE_ID_FIELDS:
            materialized[group] = _select_references(parent, attack_entry, group)
        return {
            "family": canonical_family,
            "attack_type": canonical_attack_type,
            "catalog_scope": "attack_type",
            "family_consistent": requested_family == canonical_family,
            "entry": materialized,
        }

    canonical_family = normalize_family(family)
    return {
        "family": canonical_family,
        "attack_type": None,
        "catalog_scope": "family",
        "family_consistent": canonical_family != "unknown_attack",
        "entry": catalog["families"][canonical_family],
    }


@tool_result
def list_families() -> dict[str, Any]:
    """Familias cubiertas por el catalogo."""
    catalog = _catalog()
    return {
        "families": sorted(catalog["families"].keys()),
        "version": catalog["version"],
    }


@tool_result
def list_attack_types() -> dict[str, Any]:
    """Tipos de ataque cubiertos por el mitigador operativo."""

    catalog = _catalog()
    return {
        "attack_types": list(MULTIDATASET_ATTACK_CLASSES),
        "families_by_attack_type": {
            attack_type: catalog["attack_types"][attack_type]["family"]
            for attack_type in MULTIDATASET_ATTACK_CLASSES
        },
        "taxonomy_version": catalog["attack_taxonomy"]["version"],
        "compatible_taxonomy_versions": list(
            catalog["attack_taxonomy"]["compatible_versions"]
        ),
        "version": catalog["version"],
    }


@tool_result
def map_family_to_attack(
    family: str, attack_type: str | None = None
) -> dict[str, Any]:
    """Tecnicas ATT&CK asociadas al tipo; usa familia solo como fallback."""

    resolved = _resolve_catalog_entry(family, attack_type)
    entry = resolved["entry"]
    return {
        "family": resolved["family"],
        "attack_type": resolved["attack_type"],
        "catalog_scope": resolved["catalog_scope"],
        "requested_family": family,
        "requested_attack_type": attack_type,
        "attack_techniques": entry.get("attack_techniques", []),
        "attack_mitigation_refs": entry.get("attack_mitigation_refs", []),
    }


@tool_result
def map_family_to_capec(
    family: str, attack_type: str | None = None
) -> dict[str, Any]:
    """Patrones CAPEC asociados al tipo; usa familia solo como fallback."""

    resolved = _resolve_catalog_entry(family, attack_type)
    entry = resolved["entry"]
    return {
        "family": resolved["family"],
        "attack_type": resolved["attack_type"],
        "catalog_scope": resolved["catalog_scope"],
        "requested_family": family,
        "requested_attack_type": attack_type,
        "capec_patterns": entry.get("capec_patterns", []),
    }


@tool_result
def suggest_mitigations(
    family: str,
    schema_profile: str | None = None,
    attack_type: str | None = None,
) -> dict[str, Any]:
    """Mitigaciones del tipo predicho y acciones complementarias por perfil."""

    catalog = _catalog()
    resolved = _resolve_catalog_entry(family, attack_type)
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
        "family": resolved["family"],
        "attack_type": resolved["attack_type"],
        "catalog_scope": resolved["catalog_scope"],
        "family_consistent": resolved["family_consistent"],
        "requested_family": family,
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
        resolved = _resolve_catalog_entry(
            broad_family_for_attack_type(attack_type), attack_type
        )
        entry = resolved["entry"]
        rows.append(
            {
                "attack_type": attack_type,
                "family": resolved["family"],
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

    Para cada attack type de Edge-IIoTset que Jorge mapeo manualmente a CAPEC
    (Tabla 3.4 de su TFM), indica que familia del catalogo lo cubre y si el
    patron CAPEC esta presente. Lo consumen el auditor (F5) y la memoria.
    """
    catalog = _catalog()
    table = catalog.get("jorge_tfm_capec_table", {})
    families = catalog["families"]
    rows: list[dict[str, Any]] = []
    for mapping in table.get("mappings", []):
        family = mapping.get("family")
        entry = families.get(family, {})
        catalog_capec_ids = {p["id"] for p in entry.get("capec_patterns", [])}
        rows.append(
            {
                **mapping,
                "covered": mapping.get("capec_id") in catalog_capec_ids,
                "extra_attack_techniques": [t["id"] for t in entry.get("attack_techniques", [])],
                "extra_attack_mitigations": [m["id"] for m in entry.get("attack_mitigation_refs", [])],
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
    "list_families": list_families,
    "list_attack_types": list_attack_types,
    "map_family_to_attack": map_family_to_attack,
    "map_family_to_capec": map_family_to_capec,
    "suggest_mitigations": suggest_mitigations,
    "get_jorge_capec_coverage": get_jorge_capec_coverage,
    "get_multidataset_attack_type_coverage": get_multidataset_attack_type_coverage,
}
register_tools(mcp_app, TOOLS)


if __name__ == "__main__":
    if mcp_app is None:
        raise SystemExit("SDK MCP no disponible: instala mcp[cli]>=1.2 (extra [mcp]).")
    mcp_app.run()
