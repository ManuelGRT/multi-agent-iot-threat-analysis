# src/agents/final/base.py
"""Base comun de los agentes finales: cliente MCP + trazabilidad."""
from __future__ import annotations

from typing import Any

from src.contracts.case import TraceEntry
from src.mcp.client import MCPToolClient
from src.orchestration.state import OrchestratorState, append_trace


class FinalAgent:
    """Agente final minimo: nombre, cliente MCP y helpers de traza.

    Cada subclase implementa ``run(state) -> update`` devolviendo una
    actualizacion de ``OrchestratorState`` que SIEMPRE incluye la traza
    extendida con su ``TraceEntry`` (inicio y fin en la misma entrada).
    """

    name = "final_agent"

    def __init__(self, client: MCPToolClient | None = None):
        self.client = client or MCPToolClient(mode="inprocess")

    # ------------------------------------------------------------------
    def start_entry(self, tool: str | None = None, **detail: Any) -> TraceEntry:
        return TraceEntry(agent=self.name, tool=tool, status="started", detail=detail)

    def call_tool(self, server: str, tool: str, **arguments: Any) -> dict[str, Any]:
        """Convierte fallos de transporte/protocolo en el contrato ``ok=false``.

        Asi cada agente puede cerrar su propia traza y el juez sigue viendo el
        estado parcial, en lugar de perderlo por una excepcion global del grafo.
        """
        try:
            result = self.client.call(server, tool, **arguments)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(result, dict):
            return {
                "ok": False,
                "error": f"Respuesta MCP invalida: {type(result).__name__}",
            }
        return result

    def trace_update(
        self,
        state: OrchestratorState,
        entry: TraceEntry,
        update: dict[str, Any],
    ) -> dict[str, Any]:
        """Cierra la actualizacion del nodo agregando la entrada de traza."""
        update.update(append_trace(state, entry.model_dump(mode="json")))
        return update

    def record_error(
        self,
        state: OrchestratorState,
        entry: TraceEntry,
        error: str,
        update: dict[str, Any],
    ) -> dict[str, Any]:
        """Registra un fallo de tool: traza con error + lista global de errores."""
        entry.finish(status="error", error=error, summary=f"{self.name} fallo")
        errors = list(state.get("errors") or [])
        errors.append(f"{self.name}: {error}")
        update["errors"] = errors
        return self.trace_update(state, entry, update)
