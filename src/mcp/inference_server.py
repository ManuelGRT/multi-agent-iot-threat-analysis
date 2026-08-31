# src/mcp/inference_server.py
"""Servidor MCP de inferencia: estandarizacion, deteccion y clasificacion.

Expone los modelos preparados del TFM:
- Estandarizacion final: Mistral en vivo obligatorio para toda entrada cruda,
  sin cache ni adaptadores. Un fallo produce una abstencion estructurada para
  que el juez solicite revision humana.
- Deteccion: ``xgboost_detection_validation_2026_20260822.joblib``.
- Clasificacion: ``xgboost_attack_family_validation_2026_20260822.joblib``.
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

try:  # SDK MCP oficial; opcional para el modo in-process
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - sin SDK solo se pierde el modo stdio
    FastMCP = None

from src.mcp import model_registry
from src.mcp.common import register_tools, tool_result
from src.mcp.standardization_guard import sanitize_canonical_event, sanitize_llm_input

mcp_app = FastMCP("mcp-inference") if FastMCP is not None else None


def _configured_ingest_model() -> str:
    return (
        os.getenv("INGEST_LLM_MODEL")
        or os.getenv("MISTRAL_AGENT_MODEL")
        or "mistral-small-2603"
    )


def _build_llm_parser():
    """Construye el parser estricto usado exclusivamente por el flujo final."""
    from src.agents.llm_ingest_parser import LLMIngestParser

    timeout_raw = os.getenv("INGEST_LLM_TIMEOUT_SECONDS")
    timeout = float(timeout_raw) if timeout_raw else None
    return LLMIngestParser(
        model=_configured_ingest_model(),
        provider="mistral",
        timeout_seconds=timeout,
        require_llm_column_selection=True,
        strict_output_validation=True,
    )


def _parse_with_llm(parser: Any, raw_input: dict[str, Any]):
    """Ejecuta el parser async desde tools MCP sincronas, incluso bajo un loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(parser.parse(raw_input))

    # Algunos hosts MCP invocan tools sync desde un event loop. Crear la
    # corrutina dentro del hilo evita ``asyncio.run() cannot be called...`` y
    # tambien evita dejar una corrutina sin esperar al producir la abstencion.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mistral-ingest") as pool:
        return pool.submit(lambda: asyncio.run(parser.parse(raw_input))).result()


def _abstention(
    exc: Exception,
    *,
    failure_code: str = "llm_standardization_failed",
) -> dict[str, Any]:
    reason = f"{type(exc).__name__}: {exc}"
    return {
        "ok": False,
        "abstain": True,
        "requires_human_review": True,
        "failure_code": failure_code,
        "failure_reason": reason[:1000],
        "error": reason[:1000],
        "source": (
            "prestandardized"
            if failure_code == "canonical_event_invalid"
            else "llm"
        ),
        "provider": (
            "mistral" if failure_code == "llm_standardization_failed" else None
        ),
        "model": (
            _configured_ingest_model()
            if failure_code == "llm_standardization_failed"
            else None
        ),
        "from_cache": False,
    }


@tool_result
def standardize_event(
    dataset: str = "generic",
    row: dict[str, Any] | None = None,
    text: str | None = None,
    canonical_event: dict[str, Any] | None = None,
    source_file: str = "api",
    row_id: int | str = 0,
) -> dict[str, Any]:
    """Convierte una entrada cruda en CanonicalEvent.

    Toda entrada cruda se envia a Mistral. No existe una ruta de cache ni un
    adaptador de reserva. ``canonical_event`` es la unica excepcion: ya es un
    artefacto estandarizado y solo atraviesa la frontera tecnica de validacion
    y saneamiento.
    """
    has_raw_input = row is not None or bool(str(text or "").strip())
    if canonical_event is not None and has_raw_input:
        return _abstention(
            ValueError(
                "canonical_event no puede combinarse con una entrada row/text"
            ),
            failure_code="standardization_input_ambiguous",
        )
    if canonical_event is not None:
        try:
            canonical = sanitize_canonical_event(canonical_event)
        except Exception as exc:  # el juez recibe una abstencion controlada
            return _abstention(exc, failure_code="canonical_event_invalid")
        return {
            "canonical_event": canonical,
            "from_cache": False,
            "source": "prestandardized",
            "provider": None,
            "model": "prestandardized_passthrough",
            "mapping_confidence": float(canonical.get("mapping_confidence", 0.0) or 0.0),
            "selected_columns": [],
            "notes": ["security_boundary:sanitized", "source:prestandardized"],
        }

    if row is None and not str(text or "").strip():
        return _abstention(
            ValueError("se requiere row o text para estandarizar"),
            failure_code="standardization_input_missing",
        )

    try:
        raw_input = sanitize_llm_input(
            {
                "dataset": dataset,
                "row": row,
                "text": text,
                "source_file": source_file,
                "row_id": row_id,
            }
        )
        sanitized_row = raw_input.get("row")
        sanitized_text = str(raw_input.get("text") or "").strip()
        if (
            (not isinstance(sanitized_row, dict) or not sanitized_row)
            and not sanitized_text
        ):
            return _abstention(
                ValueError(
                    "la sanitizacion no dejo campos tecnicos para estandarizar"
                ),
                failure_code="standardization_input_without_technical_fields",
            )
        # En registros tabulares, incluso la preseleccion de columnas debe ser
        # una decision del LLM. En modo estricto, su fallo no activa heuristicas.
        if isinstance(raw_input.get("row"), dict) and raw_input["row"]:
            raw_input["use_column_selection"] = True
        parser = _build_llm_parser()
        event = _parse_with_llm(parser, raw_input)
        canonical = sanitize_canonical_event(event)
    except Exception as exc:
        return _abstention(exc)

    selection = parser.last_column_selection or {}
    return {
        "canonical_event": canonical,
        "from_cache": False,
        "source": "llm",
        "provider": "mistral",
        "model": str(getattr(parser.agent, "model", _configured_ingest_model())),
        "mapping_confidence": float(canonical.get("mapping_confidence", 0.0) or 0.0),
        "selected_columns": list(selection.get("selected_columns") or []),
        "notes": ["parsed_by_llm", "security_boundary:sanitized"],
    }


@tool_result
def standardize_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Estandariza una lista de entradas: [{dataset, row|text, source_file, row_id}]."""
    results = [standardize_event(**item) for item in items]
    abstentions = sum(1 for item in results if item.get("abstain"))
    return {"count": len(results), "abstentions": abstentions, "results": results}


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
