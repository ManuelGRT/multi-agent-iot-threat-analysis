# src/orchestration/mcp_graph.py
"""Orquestador final sobre la capa MCP (Fase 3 del plan de cierre).

Reutiliza la topologia del grafo original (standardize -> detect -> classify
-> explain -> judge -> end) pero con los agentes finales que encapsulan los
modelos .joblib preparados via tools MCP. NO modifica ``graph.py``: la suite
previa sigue dependiendo de aquel; este modulo expone ``build_final_graph``
y el helper ``run_case`` que devuelve un ``CaseResult`` completo con traza.

Rutas de abstencion (revision humana):
- fallo del LLM de estandarizacion -> judge
- mapping_confidence < 0.5 en la estandarizacion -> judge
- probabilidad de deteccion en zona gris (0.4-0.6) -> judge
- errores de tool en cualquier agente -> judge
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - solo sin langgraph instalado
    END = "__end__"
    START = "__start__"
    StateGraph = None

from src.agents.final import (
    FinalClassifier,
    FinalDetector,
    FinalJudge,
    FinalMitigator,
    FinalStandardizer,
)
from src.agents.final.llm_mitigator import LLMMitigationAgent
from src.contracts.case import CaseResult, TraceEntry, new_case_id
from src.mcp.client import MCPToolClient
from src.orchestration.state import OrchestratorState


@dataclass
class FinalAgentBundle:
    """Agentes finales que comparten un mismo cliente MCP."""

    standardizer: FinalStandardizer
    detector: FinalDetector
    classifier: FinalClassifier
    mitigator: FinalMitigator
    judge: FinalJudge
    client: MCPToolClient


def _env_flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def _optional_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def default_mitigator_llm() -> LLMMitigationAgent:
    """Construye el backend LLM del mitigador desde variables de entorno.

    Patron identico al resto de agentes LLM: MITIGATOR_LLM_MODEL /
    MITIGATOR_LLM_PROVIDER / MITIGATOR_LLM_TIMEOUT_SECONDS, con LLM_PROVIDER
    como provider global. El proveedor final por defecto es Mistral: si no
    esta disponible, ``FinalMitigator`` conserva automaticamente el catalogo.
    """
    provider = (
        os.getenv("MITIGATOR_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "mistral"
    ).strip().lower()
    # El modelo por defecto se resuelve segun el proveedor EFECTIVO del
    # mitigador (no segun LLM_PROVIDER global, que puede ser otro).
    model = os.getenv("MITIGATOR_LLM_MODEL")
    if not model:
        if provider == "mistral":
            model = (
                os.getenv("MISTRAL_AGENT_MODEL")
                or os.getenv("INGEST_LLM_MODEL")
                or "mistral-small-2603"
            )
        elif provider == "openrouter":
            model = os.getenv("OPENROUTER_AGENT_MODEL", "openai/gpt-oss-120b:free")
        elif provider == "transformers":
            model = os.getenv("TRANSFORMERS_AGENT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        else:
            model = os.getenv("OLLAMA_AGENT_MODEL") or "gemma3:12b"
    return LLMMitigationAgent(
        model=model,
        base_url=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        timeout_seconds=_optional_float_env("MITIGATOR_LLM_TIMEOUT_SECONDS"),
        provider=provider,
    )


def default_final_agents(
    client: MCPToolClient | None = None,
    gray_low: float = 0.4,
    gray_high: float = 0.6,
    use_llm_mitigator: bool | None = None,
    mitigator_llm: LLMMitigationAgent | None = None,
) -> FinalAgentBundle:
    client = client or MCPToolClient(mode="inprocess")
    llm = mitigator_llm
    if llm is None:
        enabled = (
            use_llm_mitigator
            if use_llm_mitigator is not None
            else _env_flag("LLM_MITIGATOR_ENABLED")
        )
        if enabled:
            llm = default_mitigator_llm()
    return FinalAgentBundle(
        standardizer=FinalStandardizer(client=client),
        detector=FinalDetector(client=client, gray_low=gray_low, gray_high=gray_high),
        classifier=FinalClassifier(client=client),
        mitigator=FinalMitigator(client=client, llm=llm),
        judge=FinalJudge(client=client),
        client=client,
    )


def final_router(state: OrchestratorState) -> str:
    return state.get("route", "standardize")


def _route_map() -> dict[str, Any]:
    return {
        "standardize": "standardize",
        "detect": "detect",
        "classify": "classify",
        "explain": "explain",
        "judge": "judge",
        "end": END,
    }


class FinalLocalOrchestrator:
    """Fallback sin langgraph: mismo contrato ``invoke`` que el grafo compilado."""

    def __init__(self, agents: FinalAgentBundle | None = None, max_steps: int = 12):
        self.agents = agents or default_final_agents()
        self.max_steps = max_steps

    def invoke(
        self, state: OrchestratorState, config: dict[str, Any] | None = None
    ) -> OrchestratorState:
        del config
        current: OrchestratorState = dict(state)
        current.setdefault("route", "standardize")
        for _ in range(self.max_steps):
            route = final_router(current)
            if route == "end":
                return current
            if route == "standardize":
                update = self.agents.standardizer.run(current)
            elif route == "detect":
                update = self.agents.detector.run(current)
            elif route == "classify":
                update = self.agents.classifier.run(current)
            elif route == "explain":
                update = self.agents.mitigator.run(current)
            elif route == "judge":
                update = self.agents.judge.run(current)
            else:
                raise ValueError(f"Ruta desconocida: {route}")
            current.update(update)
        raise RuntimeError("El orquestador final supero max_steps sin terminar")


def build_final_graph(
    checkpointer: Any = None,
    agents: FinalAgentBundle | None = None,
    client: MCPToolClient | None = None,
):
    """Compila el grafo final (langgraph si esta disponible, local si no)."""
    agents = agents or default_final_agents(client=client)
    if StateGraph is None:
        return FinalLocalOrchestrator(agents)

    graph = StateGraph(OrchestratorState)
    graph.add_node("standardize", lambda state: agents.standardizer.run(state))
    graph.add_node("detect", lambda state: agents.detector.run(state))
    graph.add_node("classify", lambda state: agents.classifier.run(state))
    graph.add_node("explain", lambda state: agents.mitigator.run(state))
    graph.add_node("judge", lambda state: agents.judge.run(state))

    graph.add_conditional_edges(START, final_router, _route_map())
    for node in ["standardize", "detect", "classify", "explain", "judge"]:
        graph.add_conditional_edges(node, final_router, _route_map())
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Ejecucion de casos completos
# ---------------------------------------------------------------------------

class CasePersistenceError(RuntimeError):
    """Una operacion MCP impidio persistir el caso de forma completa."""


def _require_persistence_ok(operation: str, response: Any) -> dict[str, Any]:
    """Valida la respuesta MCP y detiene el flujo ante cualquier fallo."""
    if not isinstance(response, dict):
        raise CasePersistenceError(
            f"Persistencia {operation} devolvio una respuesta invalida: {type(response).__name__}"
        )
    if response.get("ok") is not True:
        detail = response.get("error") or "respuesta MCP sin ok=true"
        raise CasePersistenceError(f"Persistencia {operation} fallo: {detail}")
    return response


def _persist_case(
    memory_client: MCPToolClient,
    result: CaseResult,
    raw_input: dict[str, Any],
) -> None:
    """Persiste secuencialmente y nunca continua tras una operacion fallida.

    En particular, si ``create_case`` detecta una colision se aborta antes de
    anexar trazas o actualizar el registro que ya existia.
    """
    try:
        response = memory_client.call(
            "case_memory",
            "create_case",
            case_id=result.case_id,
            payload={"raw_input": raw_input},
        )
        _require_persistence_ok("create_case", response)

        for entry in result.trace:
            response = memory_client.call(
                "case_memory",
                "append_trace",
                case_id=result.case_id,
                entry=entry.model_dump(mode="json"),
            )
            _require_persistence_ok("append_trace", response)

        response = memory_client.call(
            "case_memory",
            "update_case",
            case_id=result.case_id,
            payload=result.model_dump(mode="json"),
            status=result.status,
        )
        _require_persistence_ok("update_case", response)
    except CasePersistenceError:
        raise
    except Exception as exc:
        raise CasePersistenceError(
            f"Persistencia MCP fallo con {type(exc).__name__}: {exc}"
        ) from exc


def run_case(
    raw_input: dict[str, Any],
    case_id: str | None = None,
    agents: FinalAgentBundle | None = None,
    client: MCPToolClient | None = None,
    use_llm_mitigator: bool | None = None,
    persist: bool = False,
    graph: Any = None,
    max_steps: int = 12,
) -> CaseResult:
    """Ejecuta un caso completo por el grafo final y devuelve el CaseResult.

    Con ``persist=True`` registra el caso en el servidor MCP de memoria de
    casos (create_case + append_trace por entrada + update_case final). Si una
    operacion falla lanza ``CasePersistenceError`` y no continua escribiendo.
    """
    agents = agents or default_final_agents(
        client=client, use_llm_mitigator=use_llm_mitigator
    )
    graph = graph or build_final_graph(agents=agents)
    case_id = case_id or new_case_id()

    opening = TraceEntry(
        agent="orchestrator",
        tool=None,
        status="started",
        detail={"dataset": raw_input.get("dataset")},
    )
    opening.finish(status="ok", summary=f"caso {case_id} creado")
    state: OrchestratorState = {
        "case_id": case_id,
        "raw_input": raw_input,
        "route": "standardize",
        "trace": [opening.model_dump(mode="json")],
    }

    try:
        final_state = graph.invoke(state, config={"recursion_limit": max_steps})
    except Exception as exc:  # el caso siempre devuelve un resultado auditable
        failed = dict(state)
        errors = list(failed.get("errors") or [])
        errors.append(f"orchestrator: {type(exc).__name__}: {exc}")
        failed["errors"] = errors
        final_state = failed

    result = CaseResult.from_orchestrator_state(final_state, case_id=case_id)

    if persist:
        _persist_case(agents.client, result, raw_input)
    return result
