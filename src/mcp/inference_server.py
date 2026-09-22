# src/mcp/inference_server.py
"""Servidor MCP de inferencia: estandarizacion, deteccion y clasificacion.

Expone los modelos preparados del TFM:
- Estandarizacion final: ejecuta Mistral en vivo y nunca usa adaptadores. La
  cache persistente pertenece al servidor ``case_memory`` y la coordina el
  agente estandarizador antes de invocar esta tool.
- Deteccion: ``xgboost_detection_final.joblib``.
- Clasificacion:
  ``xgboost_classification_final.joblib``.
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
from src.contracts.canonical import CanonicalEvent
from src.mcp.common import configured_ingest_model, register_tools, tool_result
from src.mcp.standardization_cache import (
    bind_event_identity,
    canonicalize_row,
)
from src.mcp.standardization_contract import validate_target_free_canonical

mcp_app = FastMCP("mcp-inference") if FastMCP is not None else None


def _validated_canonical_payload(
    event_data: dict[str, Any] | CanonicalEvent,
) -> dict[str, Any]:
    if isinstance(event_data, CanonicalEvent):
        payload = event_data.model_dump(mode="json")
    else:
        if not isinstance(event_data, dict):
            raise TypeError("canonical_event debe ser un objeto")
        unknown = sorted(set(event_data).difference(CanonicalEvent.model_fields))
        if unknown:
            raise ValueError(
                "canonical_event contiene campos fuera del contrato: "
                + ", ".join(unknown)
            )
        payload = CanonicalEvent(**event_data).model_dump(mode="json")
    validate_target_free_canonical(payload)
    return payload


def _selected_columns_are_compatible(
    selected_columns: list[str],
    row: dict[str, Any] | None,
) -> bool:
    if row:
        return (
            bool(selected_columns)
            and len(selected_columns) == len(set(selected_columns))
            and all(column in row for column in selected_columns)
        )
    return not selected_columns


def _build_llm_parser():
    """Construye el parser estricto usado exclusivamente por el flujo final."""
    from src.agents.llm_ingest_parser import LLMIngestParser

    timeout_raw = os.getenv("INGEST_LLM_TIMEOUT_SECONDS")
    timeout = float(timeout_raw) if timeout_raw else None
    return LLMIngestParser(
        model=configured_ingest_model(),
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
            configured_ingest_model()
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
    split: str = "stream",
) -> dict[str, Any]:
    """Convierte una entrada cruda en CanonicalEvent.

    Para entradas crudas se llama obligatoriamente a Mistral; nunca se usa un
    adaptador ni se abre estado persistente. La consulta y escritura de cache
    se realizan exclusivamente mediante el servidor MCP ``case_memory``. La
    entrada se consume tal como llega: su sanitizacion corresponde al proceso
    offline que prepara datasets de entrenamiento o evaluacion.
    """
    if split not in {"train", "val", "test", "stream"}:
        return _abstention(
            ValueError(f"split no valido: {split!r}"),
            failure_code="standardization_input_invalid",
        )
    if not isinstance(dataset, str) or not dataset.strip():
        return _abstention(
            ValueError("dataset debe ser una cadena no vacia"),
            failure_code="standardization_input_invalid",
        )
    if source_file is not None and not isinstance(source_file, str):
        return _abstention(
            TypeError("source_file debe ser una cadena o null"),
            failure_code="standardization_input_invalid",
        )
    if isinstance(row_id, bool) or not isinstance(row_id, (str, int, type(None))):
        return _abstention(
            TypeError("row_id debe ser una cadena, entero o null"),
            failure_code="standardization_input_invalid",
        )
    if row is not None and not isinstance(row, dict):
        return _abstention(
            TypeError("row debe ser un objeto JSON"),
            failure_code="standardization_input_invalid",
        )
    if text is not None and not isinstance(text, str):
        return _abstention(
            TypeError("text debe ser una cadena"),
            failure_code="standardization_input_invalid",
        )

    has_row = isinstance(row, dict) and bool(row)
    has_text = bool(str(text or "").strip())
    if row is not None and has_text:
        return _abstention(
            ValueError("row y text son alternativas y no pueden combinarse"),
            failure_code="standardization_input_ambiguous",
        )
    has_raw_input = has_row or has_text
    if canonical_event is not None and has_raw_input:
        return _abstention(
            ValueError(
                "canonical_event no puede combinarse con una entrada row/text"
            ),
            failure_code="standardization_input_ambiguous",
        )
    if canonical_event is not None:
        try:
            canonical = _validated_canonical_payload(canonical_event)
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
            "notes": ["source:prestandardized"],
        }

    if not has_raw_input:
        return _abstention(
            ValueError("se requiere row o text para estandarizar"),
            failure_code="standardization_input_missing",
        )

    dataset_name = dataset.strip()
    source_name = source_file
    try:
        prompt_row = canonicalize_row(row) if row is not None else None
    except (TypeError, ValueError) as exc:
        return _abstention(
            exc,
            failure_code="standardization_input_invalid",
        )
    raw_input = {
        "dataset": dataset_name,
        "row": prompt_row,
        "text": text,
        "source_file": source_name,
        "row_id": row_id,
        "split": split,
    }
    if isinstance(prompt_row, dict) and prompt_row:
        # En modo final incluso la preseleccion pertenece al LLM. Su fallo no
        # activa una seleccion heuristica.
        raw_input["use_column_selection"] = True

    configured_model = configured_ingest_model()
    try:
        parser = _build_llm_parser()
        event = _parse_with_llm(parser, raw_input)
        canonical = _validated_canonical_payload(event)
        selection = parser.last_column_selection or {}
        selected_columns = list(selection.get("selected_columns") or [])
        if not _selected_columns_are_compatible(selected_columns, row):
            raise ValueError(
                "la seleccion LLM no coincide con las columnas recibidas"
            )
        model = str(getattr(parser.agent, "model", configured_model))
        if model != configured_model:
            raise ValueError("el modelo ejecutado no coincide con el configurado")
        parser_version = str(
            (canonical.get("provenance") or {}).get("parser_version")
            or "llm-0.2.0"
        )
        canonical = _validated_canonical_payload(
            bind_event_identity(
                canonical,
                dataset=dataset_name,
                source_file=source_name,
                row_id=row_id,
                split=split,
                parser_version=parser_version,
            )
        )
    except Exception as exc:
        return _abstention(exc)

    return {
        "canonical_event": canonical,
        "from_cache": False,
        "source": "llm",
        "provider": "mistral",
        "model": model,
        "mapping_confidence": float(
            canonical.get("mapping_confidence", 0.0) or 0.0
        ),
        "selected_columns": selected_columns,
        "notes": ["parsed_by_llm", "source:mistral_live"],
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
    return model_registry.detect(_validated_canonical_payload(canonical_event))


@tool_result
def detect_batch(canonical_events: list[dict[str, Any]]) -> dict[str, Any]:
    results = [
        model_registry.detect(_validated_canonical_payload(event))
        for event in canonical_events
    ]
    return {"count": len(results), "results": results}


@tool_result
def classify_event(canonical_event: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    """Clasificacion directa de uno de los 16 tipos de ataque."""
    return model_registry.classify(
        _validated_canonical_payload(canonical_event),
        top_k=top_k,
    )


@tool_result
def classify_batch(canonical_events: list[dict[str, Any]], top_k: int = 3) -> dict[str, Any]:
    results = [
        model_registry.classify(
            _validated_canonical_payload(event),
            top_k=top_k,
        )
        for event in canonical_events
    ]
    return {"count": len(results), "results": results}


TOOLS = {
    "standardize_event": standardize_event,
    "standardize_batch": standardize_batch,
    "get_mapping_confidence": get_mapping_confidence,
    "detect_event": detect_event,
    "detect_batch": detect_batch,
    "classify_event": classify_event,
    "classify_batch": classify_batch,
}
register_tools(mcp_app, TOOLS)


def _prepare_stdio_runtime() -> None:
    """Importa XGBoost antes de arrancar el bucle stdio de FastMCP.

    En Windows, importar XGBoost por primera vez mientras FastMCP atiende una
    peticion puede bloquear la respuesta del transporte. Los artefactos siguen
    cargandose de forma perezosa: una estandarizacion no paga el coste de abrir
    modelos que no utiliza, y cada modelo se conserva durante toda la sesion.
    """
    import xgboost  # noqa: F401


if __name__ == "__main__":
    if mcp_app is None:
        raise SystemExit("SDK MCP no disponible: instala el proyecto con 'pip install .'.")
    _prepare_stdio_runtime()
    mcp_app.run()
