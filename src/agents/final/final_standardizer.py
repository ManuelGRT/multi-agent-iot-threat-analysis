# src/agents/final/final_standardizer.py
"""Agente final de estandarizacion: tool MCP ``standardize_event`` + sanitizacion.

Orden de resolucion de la tool (regla del handoff): cache Mistral primero,
LLM solo si se permite explicitamente, adapter determinista como fallback.
Este agente ademas:

- acepta entradas ya estandarizadas (``raw_input.canonical_event``) y las
  revalida/sanitiza sin repetir la estandarizacion (passthrough),
- elimina target leakage del evento canonico con ``predictive_sanitization``
  (label_raw, attack_family... nunca llegan como features a los modelos),
- decide la ruta: mapping_confidence < umbral -> juez (revision humana).
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.agents.final.base import FinalAgent
from src.agents.predictive_sanitization import scrub_predictive_payload
from src.contracts.agents import IngestOutput
from src.contracts.canonical import CanonicalEvent
from src.orchestration.state import OrchestratorState

REVIEW_MAPPING_CONFIDENCE = 0.5


def sanitize_canonical_event(event_data: dict[str, Any]) -> dict[str, Any]:
    """Devuelve el evento canonico sin campos target (leakage) y validado."""
    scrubbed = scrub_predictive_payload(event_data)
    event = CanonicalEvent(**scrubbed)
    return event.model_dump(mode="json")


class FinalStandardizer(FinalAgent):
    name = "final_standardizer"

    def __init__(self, client=None, allow_llm: bool = False):
        super().__init__(client)
        self.allow_llm = allow_llm

    def run(self, state: OrchestratorState) -> dict[str, Any]:
        raw_input = state.get("raw_input") or {}
        entry = self.start_entry(
            tool="standardize_event",
            dataset=raw_input.get("dataset"),
            row_id=raw_input.get("row_id"),
        )

        prestandardized = raw_input.get("canonical_event")
        if prestandardized:
            entry.tool = None
            entry.detail["mode"] = "prestandardized_passthrough"
            result: dict[str, Any] = {
                "ok": True,
                "canonical_event": prestandardized,
                "from_cache": False,
                "model": "prestandardized_passthrough",
                "mapping_confidence": float(
                    prestandardized.get("mapping_confidence", 0.0) or 0.0
                ),
            }
        else:
            result = self.call_tool(
                "inference",
                "standardize_event",
                dataset=raw_input.get("dataset", "generic"),
                row=raw_input.get("row"),
                text=raw_input.get("text"),
                source_file=raw_input.get("source_file", "api"),
                row_id=raw_input.get("row_id", 0),
                cache_key=raw_input.get("cache_key"),
                allow_llm=self.allow_llm,
            )

        if not result.get("ok", False):
            error = str(result.get("error") or "standardize_event sin resultado")
            return self.record_error(state, entry, error, {"route": "judge"})

        try:
            canonical = sanitize_canonical_event(result["canonical_event"])
        except (ValidationError, TypeError, KeyError) as exc:
            return self.record_error(
                state, entry, f"evento canonico invalido: {exc}", {"route": "judge"}
            )

        mapping_confidence = float(result.get("mapping_confidence", 0.0) or 0.0)
        notes = ["sanitized:no_target_leakage"]
        if result.get("from_cache"):
            notes.append("source:mistral_cache")
        output = IngestOutput(
            event_id=canonical["event_id"],
            normalized=True,
            mapping_confidence=mapping_confidence,
            schema_version="1.0",
            modality=str(canonical.get("modality") or "network_flow"),
            notes=notes,
        )
        ingest_output = output.model_dump(mode="json")
        # Metadatos extra para StandardizationInfo del CaseResult.
        ingest_output["model"] = result.get("model")
        ingest_output["from_cache"] = bool(result.get("from_cache", False))
        ingest_output["selected_columns"] = list(result.get("selected_columns") or [])

        route = "judge" if mapping_confidence < REVIEW_MAPPING_CONFIDENCE else "detect"
        entry.finish(
            status="ok",
            confidence=mapping_confidence,
            summary=(
                f"modelo={result.get('model')} cache={bool(result.get('from_cache'))} "
                f"mapping_confidence={mapping_confidence:.2f} ruta={route}"
            ),
        )
        update: dict[str, Any] = {
            "event_id": canonical["event_id"],
            "canonical_event": canonical,
            "ingest_output": ingest_output,
            "route": route,
        }
        return self.trace_update(state, entry, update)
