"""Agente final de estandarizacion sobre la tool MCP ``standardize_event``.

Para toda entrada cruda, la tool exige Mistral en vivo y nunca usa cache ni
adaptadores. Si la llamada o la validacion no pueden completarse, el agente se
abstiene y deriva el registro al juez para revision humana.

La sanitizacion no pertenece a este agente: es una politica tecnica de la
frontera MCP, aplicada antes del prompt, a la salida canonica y antes de la
featurizacion predictiva.
"""
from __future__ import annotations

from typing import Any

from src.agents.final.base import FinalAgent
from src.contracts.agents import IngestOutput
from src.contracts.case import TraceEntry
from src.mcp.standardization_contract import validate_standardization_success
from src.orchestration.state import OrchestratorState

REVIEW_MAPPING_CONFIDENCE = 0.5


class FinalStandardizer(FinalAgent):
    name = "final_standardizer"

    @staticmethod
    def _unstandardized_event_id(raw_input: dict[str, Any]) -> str:
        explicit = raw_input.get("event_id")
        if explicit is not None and str(explicit).strip():
            return str(explicit)
        return "unstandardized::{dataset}::{source_file}::{row_id}".format(
            dataset=raw_input.get("dataset", "generic"),
            source_file=raw_input.get("source_file", "api"),
            row_id=raw_input.get("row_id", 0),
        )

    def _abstain(
        self,
        state: OrchestratorState,
        entry: TraceEntry,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        raw_input = state.get("raw_input") or {}
        event_id = self._unstandardized_event_id(raw_input)
        failure_code = str(result.get("failure_code") or "standardization_failed")
        failure_reason = str(
            result.get("failure_reason")
            or result.get("error")
            or "standardize_event no produjo un evento canonico"
        )
        output = IngestOutput(
            event_id=event_id,
            normalized=False,
            mapping_confidence=0.0,
            schema_version="1.0",
            modality="unknown",
            notes=[failure_code],
            abstain=True,
            requires_human_review=True,
            failure_code=failure_code,
            failure_reason=failure_reason,
        ).model_dump(mode="json")
        prestandardized = raw_input.get("canonical_event") is not None
        source = "prestandardized" if prestandardized else "llm"
        provider = result.get("provider")
        if prestandardized or provider != "mistral":
            provider = None
        output.update(
            {
                "model": result.get("model"),
                "provider": provider,
                "source": source,
                "from_cache": False,
                "selected_columns": [],
            }
        )
        entry.finish(
            status="abstain",
            confidence=0.0,
            summary=f"abstencion={failure_code}; ruta=judge",
            error=failure_reason,
        )
        return self.trace_update(
            state,
            entry,
            {
                "event_id": event_id,
                "ingest_output": output,
                "needs_human_review": True,
                "route": "judge",
            },
        )

    def run(self, state: OrchestratorState) -> dict[str, Any]:
        raw_input = state.get("raw_input") or {}
        entry = self.start_entry(
            tool="standardize_event",
            dataset=raw_input.get("dataset"),
            row_id=raw_input.get("row_id"),
        )

        result = self.call_tool(
            "inference",
            "standardize_event",
            dataset=raw_input.get("dataset", "generic"),
            row=raw_input.get("row"),
            text=raw_input.get("text"),
            canonical_event=raw_input.get("canonical_event"),
            source_file=raw_input.get("source_file", "api"),
            row_id=raw_input.get("row_id", 0),
        )

        if result.get("ok") is not True:
            return self._abstain(state, entry, result)

        try:
            validated = validate_standardization_success(raw_input, result)
            canonical = validated["canonical_event"]
            event_id = str(canonical["event_id"] or "").strip()
            if not event_id:
                raise ValueError("canonical_event.event_id esta vacio")
            mapping_confidence = validated["mapping_confidence"]
            output = IngestOutput(
                event_id=event_id,
                normalized=True,
                mapping_confidence=mapping_confidence,
                schema_version="1.0",
                modality=str(canonical.get("modality") or "network_flow"),
                notes=validated["notes"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            return self._abstain(
                state,
                entry,
                {
                    **result,
                    "failure_code": "invalid_standardization_response",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                },
            )
        ingest_output = output.model_dump(mode="json")
        source = validated["source"]
        ingest_output.update(
            {
                "model": validated["model"],
                "provider": validated["provider"],
                "source": source,
                "from_cache": False,
                "selected_columns": validated["selected_columns"],
            }
        )

        route = "judge" if mapping_confidence < REVIEW_MAPPING_CONFIDENCE else "detect"
        entry.finish(
            status="ok",
            confidence=mapping_confidence,
            summary=(
                f"fuente={ingest_output['source']} modelo={result.get('model')} "
                f"mapping_confidence={mapping_confidence:.2f} ruta={route}"
            ),
        )
        update: dict[str, Any] = {
            "event_id": event_id,
            "canonical_event": canonical,
            "ingest_output": ingest_output,
            "route": route,
        }
        if route == "judge":
            update["needs_human_review"] = True
        return self.trace_update(state, entry, update)
