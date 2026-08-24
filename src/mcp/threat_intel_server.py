# src/mcp/threat_intel_server.py
"""Servidor MCP de threat intel: familia -> ATT&CK / CAPEC / mitigaciones."""
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
    "unknown": "unknown_attack",
}


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    with CATALOG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize_family(family: str | None) -> str:
    key = (family or "unknown_attack").strip().lower().replace("-", "_").replace(" ", "_")
    key = FAMILY_ALIASES.get(key, key)
    if key not in _catalog()["families"]:
        return "unknown_attack"
    return key


@tool_result
def list_families() -> dict[str, Any]:
    """Familias cubiertas por el catalogo."""
    return {"families": sorted(_catalog()["families"].keys()), "version": _catalog()["version"]}


@tool_result
def map_family_to_attack(family: str) -> dict[str, Any]:
    """Tecnicas MITRE ATT&CK asociadas a la familia."""
    key = normalize_family(family)
    entry = _catalog()["families"][key]
    return {
        "family": key,
        "requested_family": family,
        "attack_techniques": entry.get("attack_techniques", []),
        "attack_mitigation_refs": entry.get("attack_mitigation_refs", []),
    }


@tool_result
def map_family_to_capec(family: str) -> dict[str, Any]:
    """Patrones CAPEC asociados a la familia."""
    key = normalize_family(family)
    entry = _catalog()["families"][key]
    return {"family": key, "requested_family": family, "capec_patterns": entry.get("capec_patterns", [])}


@tool_result
def suggest_mitigations(family: str, schema_profile: str | None = None) -> dict[str, Any]:
    """Mitigaciones priorizadas (contencion/erradicacion/prevencion) + acciones por perfil."""
    key = normalize_family(family)
    catalog = _catalog()
    entry = catalog["families"][key]
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
        "family": key,
        "requested_family": family,
        "schema_profile": profile_key,
        "mitigations_by_phase": mitigations,
        "mitigations_ordered": ordered,
        "profile_actions": profile_actions,
        "references": {
            "attack_techniques": entry.get("attack_techniques", []),
            "capec_patterns": entry.get("capec_patterns", []),
            "attack_mitigation_refs": entry.get("attack_mitigation_refs", []),
        },
        "note": entry.get("note"),
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
    "map_family_to_attack": map_family_to_attack,
    "map_family_to_capec": map_family_to_capec,
    "suggest_mitigations": suggest_mitigations,
    "get_jorge_capec_coverage": get_jorge_capec_coverage,
}
register_tools(mcp_app, TOOLS)


if __name__ == "__main__":
    if mcp_app is None:
        raise SystemExit("SDK MCP no disponible: instala mcp[cli]>=1.2 (extra [mcp]).")
    mcp_app.run()
