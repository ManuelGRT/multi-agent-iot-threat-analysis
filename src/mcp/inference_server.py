# src/mcp/inference_server.py
"""Servidor MCP de inferencia: estandarizacion, deteccion y clasificacion.

Expone los modelos preparados del TFM:
- Estandarizacion: cache Mistral primero (regla del handoff: no repetir
  llamadas masivas); fallback al adapter determinista; LLM en vivo solo si se
  pide explicitamente y hay credenciales.
- Deteccion: ``xgboost_detection_standardized_with_edge_20260705.joblib``.
- Clasificacion: ``xgboost_attack_family_balanced_group_20260705.joblib``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

try:  # SDK MCP oficial; opcional para el modo in-process
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - sin SDK solo se pierde el modo stdio
    FastMCP = None

from src.mcp import model_registry
from src.mcp.common import register_tools, resolve_path, tool_result

mcp_app = FastMCP("mcp-inference") if FastMCP is not None else None


# ---------------------------------------------------------------------------
# Cache de estandarizacion Mistral
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _cache_index() -> dict[str, int]:
    """Indice cache_key -> offset de byte en el jsonl (carga unica, barata)."""
    path = resolve_path("mistral_cache")
    index: dict[str, int] = {}
    if not path.exists():
        return index
    offset = 0
    with path.open("rb") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                try:
                    key = json.loads(stripped).get("cache_key")
                    if key:
                        index[str(key)] = offset
                except json.JSONDecodeError:
                    pass
            offset += len(line)
    return index


def _cache_lookup(cache_key: str) -> dict[str, Any] | None:
    index = _cache_index()
    position = index.get(cache_key)
    if position is None:
        return None
    path = resolve_path("mistral_cache")
    with path.open("rb") as fh:
        fh.seek(position)
        record = json.loads(fh.readline().decode("utf-8"))
    return record


def build_cache_key(dataset: str, source_file: str, row_id: int | str) -> str:
    return f"{dataset}::{source_file}::{row_id}"


@tool_result
def standardize_event(
    dataset: str = "generic",
    row: dict[str, Any] | None = None,
    text: str | None = None,
    source_file: str = "api",
    row_id: int | str = 0,
    cache_key: str | None = None,
    allow_llm: bool = False,
) -> dict[str, Any]:
    """Convierte una entrada cruda en CanonicalEvent.

    Orden de resolucion: 1) cache Mistral, 2) LLM en vivo (solo si
    ``allow_llm=True`` y hay backend disponible), 3) adapter determinista.
    """
    key = cache_key or build_cache_key(dataset, source_file, row_id)
    cached = _cache_lookup(key)
    if cached and cached.get("ok") and cached.get("event"):
        return {
            "canonical_event": cached["event"],
            "from_cache": True,
            "cache_key": key,
            "model": cached.get("model", "mistral-small-latest"),
            "mapping_confidence": float(cached["event"].get("mapping_confidence", 0.0) or 0.0),
        }

    raw_input = {
        "dataset": dataset,
        "row": row,
        "text": text,
        "source_file": source_file,
        "row_id": row_id,
    }

    if allow_llm:
        try:
            from src.orchestration.graph import default_agents
            import asyncio

            agents = default_agents(use_llm=True)
            # use_llm=True fuerza la ruta LLM tambien para datasets con
            # adapter conocido (sin la marca, ingest_async elige el adapter);
            # copia del dict porque el fallback de mas abajo rechaza la clave.
            event, output = asyncio.run(
                agents.ingest.ingest_async({**raw_input, "use_llm": True})
            )
            notes = list(output.notes or [])
            return {
                "canonical_event": event.model_dump(mode="json"),
                "from_cache": False,
                "cache_key": key,
                "model": "llm_ingest_parser" if "parsed_by_llm" in notes else "adapter_fallback_after_llm_error",
                "mapping_confidence": output.mapping_confidence,
                "notes": notes,
            }
        except Exception:
            pass  # cae al adapter determinista

    from src.orchestration.graph import default_agents

    agents = default_agents(use_llm=False)
    event, output = agents.ingest.ingest(raw_input)
    return {
        "canonical_event": event.model_dump(mode="json"),
        "from_cache": False,
        "cache_key": key,
        "model": "deterministic_adapter",
        "mapping_confidence": output.mapping_confidence,
    }


@tool_result
def standardize_batch(items: list[dict[str, Any]], allow_llm: bool = False) -> dict[str, Any]:
    """Estandariza una lista de entradas: [{dataset, row|text, source_file, row_id}]."""
    results = [standardize_event(**item, allow_llm=allow_llm) for item in items]
    hits = sum(1 for item in results if item.get("from_cache"))
    return {"count": len(results), "cache_hits": hits, "results": results}


@tool_result
def get_mapping_confidence(canonical_event: dict[str, Any]) -> dict[str, Any]:
    """Devuelve la confianza de mapeo de un evento canonico."""
    return {"mapping_confidence": float(canonical_event.get("mapping_confidence", 0.0) or 0.0)}


# ---------------------------------------------------------------------------
# Deteccion y clasificacion
# ---------------------------------------------------------------------------

@tool_result
def detect_event(canonical_event: dict[str, Any]) -> dict[str, Any]:
    """Deteccion binaria benigno/malicioso sobre un CanonicalEvent."""
    return model_registry.detect(canonical_event)


@tool_result
def detect_batch(canonical_events: list[dict[str, Any]]) -> dict[str, Any]:
    results = [model_registry.detect(event) for event in canonical_events]
    return {"count": len(results), "results": results}


@tool_result
def classify_event(canonical_event: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    """Clasificacion de familia de ataque sobre un CanonicalEvent."""
    return model_registry.classify(canonical_event, top_k=top_k)


@tool_result
def classify_batch(canonical_events: list[dict[str, Any]], top_k: int = 3) -> dict[str, Any]:
    results = [model_registry.classify(event, top_k=top_k) for event in canonical_events]
    return {"count": len(results), "results": results}


@tool_result
def get_family_scores(canonical_event: dict[str, Any]) -> dict[str, Any]:
    """Puntuaciones completas por familia (todas las clases del modelo)."""
    return model_registry.classify(canonical_event, top_k=100)


TOOLS = {
    "standardize_event": standardize_event,
    "standardize_batch": standardize_batch,
    "get_mapping_confidence": get_mapping_confidence,
    "detect_event": detect_event,
    "detect_batch": detect_batch,
    "classify_event": classify_event,
    "classify_batch": classify_batch,
    "get_family_scores": get_family_scores,
}
register_tools(mcp_app, TOOLS)


if __name__ == "__main__":
    if mcp_app is None:
        raise SystemExit("SDK MCP no disponible: instala mcp[cli]>=1.2 (extra [mcp]).")
    mcp_app.run()
