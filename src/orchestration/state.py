# src/orchestration/state.py
from __future__ import annotations

from typing import Any, Literal, TypedDict

class OrchestratorState(TypedDict, total=False):
    thread_id: str
    case_id: str
    event_id: str
    raw_input: dict[str, Any]

    canonical_event: dict[str, Any]
    latent_record: dict[str, Any]

    ingest_output: dict[str, Any]
    detection_output: dict[str, Any]
    classification_output: dict[str, Any]
    explanation_output: dict[str, Any]
    judge_output: dict[str, Any]

    route: Literal["ingest", "detect", "classify", "explain", "judge", "end"]
    needs_human_review: bool
    human_decision: dict[str, Any]
    errors: list[str]

    # Trazabilidad de caso (Fase 1 plan de cierre): lista de TraceEntry
    # serializados como dict. Ver src/contracts/case.py.
    trace: list[dict[str, Any]]


def append_trace(state: OrchestratorState, entry: dict[str, Any]) -> OrchestratorState:
    """Devuelve una actualizacion de estado que agrega ``entry`` a la traza.

    Pensado para usarse dentro de nodos del grafo:
        update.update(append_trace(state, trace_entry.model_dump(mode="json")))
    No muta el estado original.
    """
    existing = list(state.get("trace") or [])
    existing.append(entry)
    return {"trace": existing}
