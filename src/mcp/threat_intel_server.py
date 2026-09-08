# src/mcp/threat_intel_server.py
"""Servidor MCP de threat intel: tipo de ataque -> ATT&CK, CAPEC y mitigaciones.

El clasificador operativo puede devolver uno de los dieciseis tipos de la
taxonomia multidataset. Todas las consultas operativas exigen ese tipo y se
resuelven exclusivamente contra su entrada especifica. Un tipo ausente o fuera
de la taxonomia produce un error controlado; nunca se degrada silenciosamente
a una familia generica.
"""
from __future__ import annotations

from typing import Any

try:  # SDK MCP oficial; opcional para el modo in-process
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - sin SDK solo se pierde el modo stdio
    FastMCP = None

from src.mcp.common import register_tools, tool_result
from src.contracts.attack_taxonomy import (
    MULTIDATASET_ATTACK_CLASSES,
    MULTIDATASET_TAXONOMY_VERSION,
)
from src.mcp.threat_catalog import (
    THREAT_INTEL_CATALOG_VERSION,
    catalog as _catalog,
    normalize_attack_type,
    resolve_catalog_entry as _resolve_catalog_entry,
)

mcp_app = FastMCP("mcp-threat-intel") if FastMCP is not None else None

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
        "version": THREAT_INTEL_CATALOG_VERSION,
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


TOOLS = {
    "list_attack_types": list_attack_types,
    "map_attack_type_to_attack": map_attack_type_to_attack,
    "map_attack_type_to_capec": map_attack_type_to_capec,
    "suggest_mitigations": suggest_mitigations,
    "get_multidataset_attack_type_coverage": get_multidataset_attack_type_coverage,
}
register_tools(mcp_app, TOOLS)


if __name__ == "__main__":
    if mcp_app is None:
        raise SystemExit("SDK MCP no disponible: instala el proyecto con 'pip install .'.")
    mcp_app.run()
