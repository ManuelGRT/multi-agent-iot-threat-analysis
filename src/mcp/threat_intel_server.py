# src/mcp/threat_intel_server.py
"""Servidor MCP de threat intel: catalogo y contextualizacion de mitigaciones.

El clasificador operativo puede devolver uno de los dieciseis tipos de la
taxonomia multidataset. Todas las consultas operativas exigen ese tipo y se
resuelven exclusivamente contra su entrada especifica. Un tipo ausente o fuera
de la taxonomia produce un error controlado; nunca se degrada silenciosamente
a una familia generica. La llamada externa a Mistral para contextualizar las
medidas tambien se encapsula en este servidor; el agente mitigador nunca accede
directamente al proveedor LLM.
"""
from __future__ import annotations

import math
import os
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
DEFAULT_MITIGATOR_LLM_TIMEOUT_SECONDS = 60.0
DEFAULT_MITIGATOR_LLM_TOTAL_TIMEOUT_SECONDS = 105.0
MCP_TOTAL_TIMEOUT_BUDGET_RATIO = 0.875


def _mitigation_catalog_payload(attack_type: str) -> dict[str, Any]:
    """Construye la respuesta autoritativa sin atravesar el decorador MCP."""

    catalog = _catalog()
    resolved = _resolve_catalog_entry(attack_type)
    entry = resolved["entry"]
    mitigations = entry.get("mitigations", {})
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
        "mitigations_by_phase": mitigations,
        "mitigations_ordered": ordered,
        "references": {
            "attack_techniques": entry.get("attack_techniques", []),
            "capec_patterns": entry.get("capec_patterns", []),
            "attack_mitigation_refs": entry.get("attack_mitigation_refs", []),
        },
        "reference_quality": entry.get("reference_quality") or {},
        "note": entry.get("note"),
    }


def _configured_mitigator_model() -> str:
    return (
        os.getenv("MITIGATOR_LLM_MODEL")
        or os.getenv("MISTRAL_AGENT_MODEL")
        or os.getenv("INGEST_LLM_MODEL")
        or "mistral-small-2603"
    )


def _configured_mitigator_timeout() -> float:
    raw = os.getenv("MITIGATOR_LLM_TIMEOUT_SECONDS")
    if raw in (None, ""):
        return DEFAULT_MITIGATOR_LLM_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("MITIGATOR_LLM_TIMEOUT_SECONDS no es numerico") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "MITIGATOR_LLM_TIMEOUT_SECONDS debe ser positivo y finito"
        )
    return value


def _configured_mitigator_total_timeout() -> float:
    raw = os.getenv("MITIGATOR_LLM_TOTAL_TIMEOUT_SECONDS")
    if raw in (None, ""):
        return DEFAULT_MITIGATOR_LLM_TOTAL_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            "MITIGATOR_LLM_TOTAL_TIMEOUT_SECONDS no es numerico"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "MITIGATOR_LLM_TOTAL_TIMEOUT_SECONDS debe ser positivo y finito"
        )
    return value


def _effective_mitigator_total_timeout() -> float:
    """Reserva margen dentro del plazo MCP comunicado por el cliente stdio."""

    configured = _configured_mitigator_total_timeout()
    client_mode = os.getenv("MCP_CLIENT_MODE", "stdio").strip().lower()
    if client_mode != "stdio":
        return configured
    raw_transport_timeout = os.getenv("MCP_STDIO_TIMEOUT_SECONDS", "120")
    try:
        transport_timeout = float(raw_transport_timeout)
    except ValueError as exc:
        raise ValueError("MCP_STDIO_TIMEOUT_SECONDS no es numerico") from exc
    if not math.isfinite(transport_timeout) or transport_timeout <= 0:
        raise ValueError(
            "MCP_STDIO_TIMEOUT_SECONDS debe ser positivo y finito"
        )
    return min(configured, transport_timeout * MCP_TOTAL_TIMEOUT_BUDGET_RATIO)


def _build_mitigation_llm():
    """Construye el cliente Mistral dentro del proceso ``threat_intel``."""

    from src.agents.final.llm_mitigator import LLMMitigationAgent

    return LLMMitigationAgent(
        model=_configured_mitigator_model(),
        timeout_seconds=_configured_mitigator_timeout(),
        provider="mistral",
    )


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
) -> dict[str, Any]:
    """Mitigaciones y referencias del tipo de ataque predicho."""

    return _mitigation_catalog_payload(attack_type)


@tool_result
def contextualize_mitigations(
    attack_type: str,
    canonical_event: dict[str, Any],
    detection: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    """Contextualiza con Mistral la entrada catalogada de un tipo de ataque.

    El cliente solo aporta el tipo predicho y el contexto del caso. La base de
    mitigaciones se resuelve de nuevo dentro del servidor para impedir que una
    entrada catalogada manipulada alcance el prompt. La respuesta LLM se
    devuelve sin confiar para que ``FinalMitigator`` aplique el anclaje final.
    """

    if not isinstance(canonical_event, dict):
        raise TypeError("canonical_event debe ser un objeto")
    if not isinstance(detection, dict):
        raise TypeError("detection debe ser un objeto")
    if not isinstance(classification, dict):
        raise TypeError("classification debe ser un objeto")

    catalog_result = _mitigation_catalog_payload(attack_type)
    normalized_attack_type = str(catalog_result["attack_type"])
    if classification.get("attack_type") != normalized_attack_type:
        raise ValueError(
            "classification.attack_type no coincide con la entrada del catalogo"
        )

    from src.agents.final.llm_mitigator import (
        build_base_items,
        validate_mitigation_payload,
    )

    base_items = build_base_items(catalog_result)
    contextualizer = _build_mitigation_llm()
    raw = contextualizer.contextualize(
        canonical_event=canonical_event,
        detection=detection,
        classification=classification,
        catalog_result=catalog_result,
        base_items=base_items,
        total_timeout_seconds=_effective_mitigator_total_timeout(),
    )
    response_metadata = {
        "attack_type": normalized_attack_type,
        "catalog_scope": catalog_result["catalog_scope"],
        "catalog_version": catalog_result["catalog_version"],
        "taxonomy_version": catalog_result["taxonomy_version"],
        "compatible_taxonomy_versions": list(
            catalog_result["compatible_taxonomy_versions"]
        ),
        "provider": "mistral",
        "model_name": contextualizer.model_name,
        "base_count": len(base_items),
    }
    try:
        validated_contextualization = validate_mitigation_payload(raw)
    except (TypeError, ValueError) as exc:
        # La tool falla de forma cerrada para el runtime, pero conserva la
        # respuesta bruta para que las campañas de evaluación puedan medir el
        # cumplimiento real del contrato anterior al anclaje.
        return {
            **response_metadata,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "contextualization_schema_valid": False,
            "contextualization": raw,
        }
    return {
        **response_metadata,
        "contextualization_schema_valid": True,
        "contextualization": validated_contextualization,
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
    "contextualize_mitigations": contextualize_mitigations,
    "get_multidataset_attack_type_coverage": get_multidataset_attack_type_coverage,
}
register_tools(mcp_app, TOOLS)


if __name__ == "__main__":
    if mcp_app is None:
        raise SystemExit("SDK MCP no disponible: instala el proyecto con 'pip install .'.")
    mcp_app.run()
