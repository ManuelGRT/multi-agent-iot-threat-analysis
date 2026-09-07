# src/mcp/client.py
"""Cliente MCP para los agentes finales.

Dos modos con el mismo contrato de llamada:

- ``stdio`` (por defecto): arranca cada servidor utilizado como proceso MCP
  real (``python -m src.mcp.<server>``) y conserva su sesion durante la vida
  del cliente. Es el modo del runtime productivo.
- ``inprocess``: invoca directamente las tools registradas en cada modulo
  servidor. Se reserva para la demostracion rapida y las pruebas que no
  pretenden ejercitar el transporte.

``MCP_CLIENT_MODE`` permite seleccionar el modo del runtime sin cambiar codigo.
Su valor predeterminado es ``stdio``.

Uso:
    with MCPToolClient() as client:                # stdio / MCP real
        result = client.call(
            "inference", "detect_event", canonical_event={...}
        )

    client = MCPToolClient(mode="inprocess")      # demo y tests rapidos
"""
from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

SERVER_MODULES: dict[str, str] = {
    "inference": "src.mcp.inference_server",
    "case_memory": "src.mcp.case_memory_server",
    "threat_intel": "src.mcp.threat_intel_server",
}

DEFAULT_STDIO_TIMEOUT_SECONDS = 120.0
STDIO_TIMEOUT_ENV = "MCP_STDIO_TIMEOUT_SECONDS"
DEFAULT_CLIENT_MODE = "stdio"
CLIENT_MODE_ENV = "MCP_CLIENT_MODE"


def resolve_mcp_client_mode(mode: str | None = None) -> str:
    """Resuelve el transporte explicito o el configurado para el runtime."""
    configured = mode if mode is not None else os.getenv(
        CLIENT_MODE_ENV, DEFAULT_CLIENT_MODE
    )
    resolved = str(configured).strip().lower()
    if resolved not in {"inprocess", "stdio"}:
        raise ValueError(
            f"Modo MCP no soportado: {configured!r}. "
            "Opciones: ['inprocess', 'stdio']"
        )
    return resolved


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


@dataclass(slots=True)
class _StdioRequest:
    """Operacion sincrona entregada al bucle propietario de una sesion MCP."""

    operation: str
    tool: str | None
    arguments: dict[str, Any]
    future: Future[Any]


_STOP_STDIO_WORKER = object()


class _PersistentStdioServer:
    """Mantiene un proceso y una sesion MCP para un unico servidor.

    El SDK MCP 1.x crea *cancel scopes* de AnyIO al entrar en
    ``stdio_client`` y ``ClientSession``. Esos contextos deben cerrarse desde
    la misma tarea que los abrio. Por ello cada servidor tiene un hilo con un
    unico ``asyncio.run``: la corrutina raiz abre los dos contextos, atiende
    secuencialmente una cola thread-safe y finalmente los cierra. Las llamadas
    publicas siguen siendo sincronas y pueden proceder de varios hilos.
    """

    def __init__(
        self,
        *,
        server: str,
        python_executable: str,
        timeout_seconds: float,
    ) -> None:
        self.server = server
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds
        self._state_lock = threading.Lock()
        self._accepting = True
        self._terminated = threading.Event()
        self._loop_ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._requests: asyncio.Queue[_StdioRequest | object] | None = None
        self._pending: set[Future[Any]] = set()
        self._fatal_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"mcp-stdio-{server}",
            daemon=True,
        )
        self._thread.start()
        # El event loop y su cola se crean antes de iniciar el transporte. La
        # espera solo sincroniza la entrega thread-safe de la primera llamada.
        self._loop_ready.wait()

    @property
    def reusable(self) -> bool:
        with self._state_lock:
            return self._accepting and not self._terminated.is_set()

    def submit(
        self,
        operation: str,
        *,
        tool: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        response: Future[Any] = Future()
        request = _StdioRequest(
            operation=operation,
            tool=tool,
            arguments=arguments or {},
            future=response,
        )
        with self._state_lock:
            if not self._accepting:
                detail = (
                    f": {self._fatal_error}"
                    if self._fatal_error is not None
                    else ""
                )
                raise RuntimeError(
                    f"La sesion MCP stdio de {self.server} esta cerrada{detail}"
                )
            if self._loop is None or self._requests is None:
                raise RuntimeError(
                    f"No se pudo iniciar el worker MCP stdio de {self.server}"
                )
            self._pending.add(response)
            try:
                # El envio se serializa con close usando el mismo bloqueo: el
                # centinela siempre queda despues de toda peticion aceptada.
                self._loop.call_soon_threadsafe(
                    self._requests.put_nowait, request
                )
            except RuntimeError:
                self._pending.discard(response)
                raise
        return response.result()

    def close(self, *, join_timeout: float | None = None) -> None:
        """Deja terminar las peticiones aceptadas y cierra proceso y sesion."""
        with self._state_lock:
            if self._accepting:
                self._accepting = False
                if self._loop is not None and self._requests is not None:
                    try:
                        self._loop.call_soon_threadsafe(
                            self._requests.put_nowait, _STOP_STDIO_WORKER
                        )
                    except RuntimeError:
                        pass

        if threading.current_thread() is self._thread:
            return
        self._thread.join(timeout=join_timeout)
        if self._thread.is_alive():
            raise TimeoutError(
                f"No se pudo cerrar el servidor MCP stdio {self.server} "
                "dentro del plazo"
            )

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            with self._state_lock:
                self._fatal_error = exc
                self._accepting = False
            self._fail_pending(exc)
        finally:
            self._loop_ready.set()
            self._terminated.set()

    async def _serve(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._loop = asyncio.get_running_loop()
        self._requests = asyncio.Queue()
        self._loop_ready.set()

        params = StdioServerParameters(
            command=self.python_executable,
            args=["-m", SERVER_MODULES[self.server]],
            # El proceso hijo recibe la configuracion capturada al arrancar el
            # servidor: Mistral, rutas SQLite, artefactos y demas parametros.
            env=dict(os.environ),
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
                while True:
                    item = await self._requests.get()
                    if item is _STOP_STDIO_WORKER:
                        return
                    request = item
                    try:
                        if request.operation == "call":
                            result = await asyncio.wait_for(
                                session.call_tool(
                                    str(request.tool),
                                    arguments=request.arguments,
                                ),
                                timeout=self.timeout_seconds,
                            )
                            value: Any = _decode_stdio_result(result)
                        elif request.operation == "list_tools":
                            response = await asyncio.wait_for(
                                session.list_tools(),
                                timeout=self.timeout_seconds,
                            )
                            value = sorted(tool.name for tool in response.tools)
                        else:  # pragma: no cover - invariantes internas
                            raise ValueError(
                                f"Operacion MCP stdio desconocida: "
                                f"{request.operation}"
                            )
                    except BaseException as exc:
                        # Se deja de aceptar trabajo antes de publicar el
                        # fallo. Esto evita que otra llamada concurrente se
                        # encole en un worker que ya va a cerrar su sesion.
                        with self._state_lock:
                            self._fatal_error = exc
                            self._accepting = False
                            self._pending.discard(request.future)
                        if not request.future.done():
                            request.future.set_exception(exc)
                        self._fail_pending(exc)
                        # Tras un error de transporte o timeout no se reutiliza
                        # una sesion cuyo estado ya no puede garantizarse.
                        raise
                    else:
                        with self._state_lock:
                            self._pending.discard(request.future)
                        if not request.future.done():
                            request.future.set_result(value)

    def _fail_pending(self, cause: BaseException) -> None:
        with self._state_lock:
            pending = list(self._pending)
            self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(
                    RuntimeError(
                        f"La sesion MCP stdio de {self.server} termino: {cause}"
                    )
                )


class MCPToolClient:
    def __init__(
        self,
        mode: str | None = None,
        python_executable: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.mode = resolve_mcp_client_mode(mode)
        self.python_executable = python_executable or sys.executable
        # El entorno stdio solo se consulta en modo stdio para no hacer que un
        # valor ajeno rompa clientes puramente in-process.
        self.timeout_seconds = (
            _resolve_stdio_timeout_seconds(timeout_seconds)
            if self.mode == "stdio" or timeout_seconds is not None
            else DEFAULT_STDIO_TIMEOUT_SECONDS
        )
        self._state_lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._stdio_workers: dict[str, _PersistentStdioServer] = {}
        self._retired_stdio_workers: list[_PersistentStdioServer] = []

    def __enter__(self) -> "MCPToolClient":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.close()
        except Exception as close_error:
            # Un fallo secundario de apagado no debe ocultar, por ejemplo, un
            # CasePersistenceError que ya estuviera propagandose.
            if exc is None:
                raise
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(f"Fallo adicional al cerrar MCP: {close_error}")
        return False

    @property
    def closed(self) -> bool:
        with self._state_lock:
            return self._closed

    def close(self) -> None:
        """Cierra todas las sesiones y subprocesos stdio creados por el cliente.

        Es idempotente. Las peticiones ya aceptadas terminan antes de enviar el
        cierre al servidor; las nuevas llamadas se rechazan desde este punto.
        """
        with self._close_lock:
            with self._state_lock:
                # ``closed`` rechaza trabajo nuevo desde el primer intento,
                # pero los workers no se olvidan hasta confirmar su cierre.
                # De este modo un segundo close puede reintentar un join que
                # hubiera agotado el plazo.
                self._closed = True
                workers = list(
                    dict.fromkeys(
                        [
                            *self._stdio_workers.values(),
                            *self._retired_stdio_workers,
                        ]
                    )
                )
            if not workers:
                return

            errors: list[BaseException] = []
            closed_workers: set[_PersistentStdioServer] = set()
            join_timeout = self.timeout_seconds + 5.0
            for worker in workers:
                try:
                    worker.close(join_timeout=join_timeout)
                except BaseException as exc:  # se cierra el resto tambien
                    errors.append(exc)
                else:
                    closed_workers.add(worker)

            with self._state_lock:
                self._stdio_workers = {
                    server: worker
                    for server, worker in self._stdio_workers.items()
                    if worker not in closed_workers
                }
                self._retired_stdio_workers = [
                    worker
                    for worker in self._retired_stdio_workers
                    if worker not in closed_workers
                ]

            if errors:
                raise RuntimeError(
                    "No se pudieron cerrar todas las sesiones MCP stdio: "
                    + "; ".join(str(error) for error in errors)
                ) from errors[0]

    # ------------------------------------------------------------------
    def call(self, server: str, tool: str, **arguments: Any) -> dict[str, Any]:
        self._ensure_open()
        if server not in SERVER_MODULES:
            raise ValueError(f"Servidor desconocido: {server}. Opciones: {sorted(SERVER_MODULES)}")
        if self.mode == "inprocess":
            return self._call_inprocess(server, tool, arguments)
        return self._call_stdio(server, tool, arguments)

    def list_tools(self, server: str) -> list[str]:
        self._ensure_open()
        if server not in SERVER_MODULES:
            raise ValueError(
                f"Servidor desconocido: {server}. Opciones: {sorted(SERVER_MODULES)}"
            )
        if self.mode == "stdio":
            worker = self._stdio_worker(server)
            return worker.submit("list_tools")
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
        worker = self._stdio_worker(server)
        return worker.submit("call", tool=tool, arguments=arguments)

    async def _call_stdio_async(self, server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Adaptador async sobre el mismo worker persistente y thread-safe."""
        return await asyncio.to_thread(self._call_stdio, server, tool, arguments)

    def _ensure_open(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("El cliente MCP esta cerrado")

    def _stdio_worker(self, server: str) -> _PersistentStdioServer:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("El cliente MCP esta cerrado")
            worker = self._stdio_workers.get(server)
            if worker is None or not worker.reusable:
                if worker is not None:
                    self._retired_stdio_workers.append(worker)
                worker = _PersistentStdioServer(
                    server=server,
                    python_executable=self.python_executable,
                    timeout_seconds=self.timeout_seconds,
                )
                self._stdio_workers[server] = worker
            return worker


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
        env=dict(os.environ),
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
