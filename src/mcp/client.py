# src/mcp/client.py
"""Cliente MCP para los agentes finales.

Dos modos con el mismo contrato de llamada:

- ``inprocess`` (por defecto): invoca directamente las tools registradas en
  cada modulo servidor. Es el modo usado por la API y los tests.
- ``stdio``: arranca cada servidor como proceso MCP real
  (``python -m src.mcp.<server>``) y llama a las tools via el protocolo MCP
  con el SDK oficial. Es el modo que demuestra la capa MCP autentica.

Uso:
    client = MCPToolClient()                       # in-process
    client = MCPToolClient(mode="stdio")           # MCP real
    result = client.call("inference", "detect_event", canonical_event={...})
"""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
from datetime import timedelta
from typing import Any

SERVER_MODULES: dict[str, str] = {
    "inference": "src.mcp.inference_server",
    "case_memory": "src.mcp.case_memory_server",
    "threat_intel": "src.mcp.threat_intel_server",
}

DEFAULT_STDIO_TIMEOUT_SECONDS = 120.0
STDIO_TIMEOUT_ENV = "MCP_STDIO_TIMEOUT_SECONDS"


def _resolve_stdio_timeout_seconds(timeout_seconds: float | None = None) -> float:
    """Resuelve un timeout stdio positivo y finito.

    El constructor tiene precedencia sobre ``MCP_STDIO_TIMEOUT_SECONDS``. Un
    valor invalido falla al configurar el cliente, nunca durante una llamada.
    """
    configured: Any = timeout_seconds
    if configured is None:
        configured = os.getenv(STDIO_TIMEOUT_ENV)
    if configured in (None, ""):
        configured = DEFAULT_STDIO_TIMEOUT_SECONDS
    try:
        value = float(configured)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Timeout MCP stdio invalido: {configured!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Timeout MCP stdio invalido: {configured!r}")
    return value


def _stdio_text_content(result: Any) -> str:
    texts = [
        str(getattr(item, "text", ""))
        for item in (getattr(result, "content", None) or [])
        if getattr(item, "type", None) == "text"
    ]
    return "\n".join(texts).strip()


def _decode_stdio_result(result: Any) -> dict[str, Any]:
    """Decodifica una respuesta MCP respetando el contrato JSON de las tools.

    La frontera es deliberadamente *fail-closed*: ``isError`` siempre produce
    ``ok=false`` y una respuesta correcta debe ser un objeto JSON con ``ok``
    booleano. Asi un error textual del protocolo no puede convertirse en una
    deteccion benigna por valores por defecto aguas abajo.
    """
    raw_text = _stdio_text_content(result)
    parsed: Any = None
    parse_error: Exception | None = None
    if raw_text:
        try:
            parsed = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError) as exc:
            parse_error = exc

    if bool(getattr(result, "isError", False)):
        if isinstance(parsed, dict):
            payload = dict(parsed)
            payload["ok"] = False
            payload.setdefault(
                "error", raw_text or "La tool MCP devolvio isError=true"
            )
            return payload
        detail = raw_text or str(getattr(result, "content", ""))
        return {
            "ok": False,
            "error": f"La tool MCP devolvio un error: {detail or 'sin detalle'}",
        }

    if not raw_text:
        return {
            "ok": False,
            "error": "Respuesta MCP sin contenido de texto JSON",
        }
    if parse_error is not None:
        return {
            "ok": False,
            "error": f"Respuesta MCP no JSON: {type(parse_error).__name__}",
            "text": raw_text,
        }
    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "error": "Respuesta MCP JSON no es un objeto",
            "content": parsed,
        }
    if not isinstance(parsed.get("ok"), bool):
        return {
            "ok": False,
            "error": "Respuesta MCP sin campo booleano 'ok'",
            "content": parsed,
        }
    return parsed


class MCPToolClient:
    def __init__(
        self,
        mode: str = "inprocess",
        python_executable: str | None = None,
        timeout_seconds: float | None = None,
    ):
        if mode not in {"inprocess", "stdio"}:
            raise ValueError(f"Modo no soportado: {mode}")
        self.mode = mode
        self.python_executable = python_executable or sys.executable
        # El entorno stdio solo se consulta en modo stdio para no hacer que un
        # valor ajeno rompa clientes puramente in-process.
        self.timeout_seconds = (
            _resolve_stdio_timeout_seconds(timeout_seconds)
            if mode == "stdio" or timeout_seconds is not None
            else DEFAULT_STDIO_TIMEOUT_SECONDS
        )

    # ------------------------------------------------------------------
    def call(self, server: str, tool: str, **arguments: Any) -> dict[str, Any]:
        if server not in SERVER_MODULES:
            raise ValueError(f"Servidor desconocido: {server}. Opciones: {sorted(SERVER_MODULES)}")
        if self.mode == "inprocess":
            return self._call_inprocess(server, tool, arguments)
        return self._call_stdio(server, tool, arguments)

    def list_tools(self, server: str) -> list[str]:
        module = importlib.import_module(SERVER_MODULES[server])
        return sorted(module.TOOLS.keys())

    # ------------------------------------------------------------------
    def _call_inprocess(self, server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        module = importlib.import_module(SERVER_MODULES[server])
        tools = getattr(module, "TOOLS", {})
        if tool not in tools:
            raise ValueError(f"Tool desconocida en {server}: {tool}. Opciones: {sorted(tools)}")
        return tools[tool](**arguments)

    def _call_stdio(self, server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        return asyncio.run(self._call_stdio_async(server, tool, arguments))

    async def _call_stdio_async(self, server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.python_executable,
            args=["-m", SERVER_MODULES[server]],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=self.timeout_seconds),
            ) as session:
                await asyncio.wait_for(
                    session.initialize(), timeout=self.timeout_seconds
                )
                result = await asyncio.wait_for(
                    session.call_tool(tool, arguments=arguments),
                    timeout=self.timeout_seconds,
                )
                return _decode_stdio_result(result)


async def list_tools_stdio(
    server: str,
    python_executable: str | None = None,
    timeout_seconds: float | None = None,
) -> list[str]:
    """Lista tools de un servidor via protocolo MCP real (para smoke tests)."""
    import asyncio

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    resolved_timeout = _resolve_stdio_timeout_seconds(timeout_seconds)

    params = StdioServerParameters(
        command=python_executable or sys.executable,
        args=["-m", SERVER_MODULES[server]],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(
            read,
            write,
            read_timeout_seconds=timedelta(seconds=resolved_timeout),
        ) as session:
            await asyncio.wait_for(session.initialize(), timeout=resolved_timeout)
            response = await asyncio.wait_for(
                session.list_tools(), timeout=resolved_timeout
            )
            return sorted(tool.name for tool in response.tools)
