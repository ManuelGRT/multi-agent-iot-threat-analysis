"""Agente final de estandarizacion sobre la tool MCP ``standardize_event``.

Para una entrada cruda, el agente consulta primero la cache expuesta por el
servidor MCP ``case_memory``. Si no existe un duplicado exacto, exige Mistral
en vivo mediante ``inference`` y devuelve el exito al servidor de memoria para
persistirlo. Nunca usa adaptadores. Si no hay hit y la llamada o la validacion
fallan, se abstiene y deriva el registro al juez para revision humana.

La sanitizacion no pertenece al sistema multiagente. Las entradas operativas
se consideran ya preparadas; el runtime valida contratos y rechaza resultados
invalidos, pero no elimina campos silenciosamente.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from src.agents.final.base import FinalAgent
from src.contracts.agents import IngestOutput
from src.contracts.case import TraceEntry
from src.mcp.standardization_contract import validate_standardization_success
from src.orchestration.state import OrchestratorState

REVIEW_MAPPING_CONFIDENCE = 0.5


@dataclass(slots=True)
class _CacheFlight:
    lock: threading.Lock = field(default_factory=threading.Lock)
    references: int = 0


_CACHE_FLIGHTS_GUARD = threading.Lock()
_CACHE_FLIGHTS: dict[tuple[str, str], _CacheFlight] = {}


@contextmanager
def _singleflight(content_hash: str, pipeline_hash: str):
    """Serializa un miss duplicado sin acceder directamente al almacenamiento."""

    key = (content_hash, pipeline_hash)
    with _CACHE_FLIGHTS_GUARD:
        flight = _CACHE_FLIGHTS.get(key)
        if flight is None:
            flight = _CacheFlight()
            _CACHE_FLIGHTS[key] = flight
        flight.references += 1
    acquired = False
    try:
        flight.lock.acquire()
        acquired = True
        yield
    finally:
        if acquired:
            flight.lock.release()
        with _CACHE_FLIGHTS_GUARD:
            flight.references -= 1
            if flight.references == 0 and _CACHE_FLIGHTS.get(key) is flight:
                _CACHE_FLIGHTS.pop(key, None)


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

    @staticmethod
    def _raw_arguments(raw_input: dict[str, Any]) -> dict[str, Any]:
        return {
            "dataset": raw_input.get("dataset", "generic"),
            "row": raw_input.get("row"),
            "text": raw_input.get("text"),
            "source_file": raw_input.get("source_file", "api"),
            "row_id": raw_input.get("row_id", 0),
            "split": raw_input.get("split", "stream"),
        }

    def _lookup_cache(
        self,
        raw_input: dict[str, Any],
        arguments: dict[str, Any],
        tool_sequence: list[str],
    ) -> dict[str, Any]:
        tool_sequence.append("case_memory.lookup_standardization_cache")
        lookup = self.call_tool(
            "case_memory",
            "lookup_standardization_cache",
            **arguments,
        )
        if lookup.get("ok") is not True or lookup.get("hit") is not True:
            return lookup
        try:
            validate_standardization_success(raw_input, lookup)
        except (KeyError, TypeError, ValueError):
            content_hash = lookup.get("cache_content_hash")
            pipeline_hash = lookup.get("cache_pipeline_hash")
            if isinstance(content_hash, str) and isinstance(pipeline_hash, str):
                tool_sequence.append("case_memory.evict_standardization_cache")
                self.call_tool(
                    "case_memory",
                    "evict_standardization_cache",
                    content_hash=content_hash,
                    pipeline_hash=pipeline_hash,
                )
            return {**lookup, "hit": False, "invalidated": True}
        return lookup

    def _live_and_store(
        self,
        raw_input: dict[str, Any],
        arguments: dict[str, Any],
        lookup: dict[str, Any],
        tool_sequence: list[str],
    ) -> dict[str, Any]:
        lookup_status = (
            "invalidated"
            if lookup.get("invalidated") is True
            else "miss" if lookup.get("ok") is True else "error"
        )
        tool_sequence.append("inference.standardize_event")
        live = self.call_tool("inference", "standardize_event", **arguments)
        if live.get("ok") is not True:
            return {
                **live,
                "_cache_trace": {
                    "lookup": lookup_status,
                    "store": "not_attempted",
                },
            }
        try:
            validated = validate_standardization_success(raw_input, live)
        except (KeyError, TypeError, ValueError):
            # El llamador genera la abstencion controlada con el mismo detalle.
            return {
                **live,
                "_cache_trace": {
                    "lookup": lookup_status,
                    "store": "not_attempted",
                },
            }

        tool_sequence.append("case_memory.store_standardization_cache")
        stored = self.call_tool(
            "case_memory",
            "store_standardization_cache",
            **arguments,
            canonical_event=validated["canonical_event"],
            selected_columns=validated["selected_columns"],
            model=validated["model"],
        )
        winner = stored.get("winner")
        if (
            stored.get("ok") is True
            and stored.get("already_present") is True
            and isinstance(winner, dict)
        ):
            candidate = {"ok": True, **winner}
            try:
                validate_standardization_success(raw_input, candidate)
            except (KeyError, TypeError, ValueError):
                pass
            else:
                candidate["_cache_trace"] = {
                    "lookup": lookup_status,
                    "store": "already_exists",
                }
                return candidate

        # La cache es fail-open: un fallo de escritura nunca invalida un exito
        # Mistral. Se conservan los hashes calculados por case-memory cuando
        # estan disponibles para que la traza identifique la operacion.
        enriched = dict(live)
        enriched["_cache_trace"] = {
            "lookup": lookup_status,
            "store": (
                str(
                    stored.get("status")
                    or (
                        "inserted"
                        if stored.get("stored") is True
                        else "not_stored"
                    )
                )
                if stored.get("ok") is True
                else "error"
            ),
        }
        for key in ("cache_content_hash", "cache_pipeline_hash"):
            value = stored.get(key) or lookup.get(key)
            if isinstance(value, str):
                enriched[key] = value
        return enriched

    def _standardize(
        self,
        raw_input: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        arguments = self._raw_arguments(raw_input)
        tool_sequence: list[str] = []
        if raw_input.get("canonical_event") is not None:
            tool_sequence.append("inference.standardize_event")
            result = self.call_tool(
                "inference",
                "standardize_event",
                **arguments,
                canonical_event=raw_input.get("canonical_event"),
            )
            return {
                **result,
                "_cache_trace": {"lookup": "skipped", "store": "skipped"},
            }, tool_sequence

        lookup = self._lookup_cache(raw_input, arguments, tool_sequence)
        if lookup.get("ok") is True and lookup.get("hit") is True:
            return {
                **lookup,
                "_cache_trace": {"lookup": "hit", "store": "not_required"},
            }, tool_sequence

        content_hash = lookup.get("cache_content_hash")
        pipeline_hash = lookup.get("cache_pipeline_hash")
        has_flight_key = (
            isinstance(content_hash, str)
            and len(content_hash) == 64
            and isinstance(pipeline_hash, str)
            and len(pipeline_hash) == 64
        )
        if not has_flight_key:
            return (
                self._live_and_store(
                    raw_input,
                    arguments,
                    lookup,
                    tool_sequence,
                ),
                tool_sequence,
            )

        with _singleflight(content_hash, pipeline_hash):
            # Otro hilo del proceso puede haber completado el mismo contenido
            # mientras este esperaba el bloqueo.
            second_lookup = self._lookup_cache(
                raw_input,
                arguments,
                tool_sequence,
            )
            if (
                second_lookup.get("ok") is True
                and second_lookup.get("hit") is True
            ):
                return {
                    **second_lookup,
                    "_cache_trace": {
                        "lookup": "hit",
                        "store": "not_required",
                    },
                }, tool_sequence
            return (
                self._live_and_store(
                    raw_input,
                    arguments,
                    second_lookup,
                    tool_sequence,
                ),
                tool_sequence,
            )

    def run(self, state: OrchestratorState) -> dict[str, Any]:
        raw_input = state.get("raw_input") or {}
        entry = self.start_entry(
            tool="standardize_event",
            dataset=raw_input.get("dataset"),
            row_id=raw_input.get("row_id"),
            cache_server="case_memory",
        )

        result, tool_sequence = self._standardize(raw_input)
        entry.detail["tool_sequence"] = tool_sequence
        entry.detail["cache"] = result.pop(
            "_cache_trace",
            {"lookup": "unknown", "store": "unknown"},
        )
        entry.tool = (
            "lookup_standardization_cache"
            if result.get("from_cache") is True
            else "standardize_event"
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
                "from_cache": validated["from_cache"],
                "selected_columns": validated["selected_columns"],
                "cache_content_hash": validated["cache_content_hash"],
                "cache_pipeline_hash": validated["cache_pipeline_hash"],
            }
        )

        route = "judge" if mapping_confidence < REVIEW_MAPPING_CONFIDENCE else "detect"
        entry.finish(
            status="ok",
            confidence=mapping_confidence,
            summary=(
                f"fuente={ingest_output['source']} cache={ingest_output['from_cache']} "
                f"modelo={result.get('model')} "
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
