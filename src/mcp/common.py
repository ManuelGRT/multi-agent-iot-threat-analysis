# src/mcp/common.py
"""Utilidades compartidas por los servidores MCP: rutas, envoltura de tools."""
from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterator

MCP_LAYER_VERSION = "0.2.0"


def repo_root() -> Path:
    """Localiza la raiz del repo (donde vive pyproject.toml).

    Se puede forzar con la variable de entorno ``TFM_REPO_ROOT``.
    """
    env = os.getenv("TFM_REPO_ROOT")
    if env:
        return Path(env)
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def artifacts_dir() -> Path:
    return Path(os.getenv("TFM_ARTIFACTS_DIR") or repo_root() / "artifacts")


def package_data_dir() -> Path:
    """Recursos de solo lectura distribuidos dentro del paquete."""
    return Path(__file__).resolve().parent / "data"


def state_dir() -> Path:
    """Directorio escribible para el estado operativo del servicio."""
    return Path(os.getenv("TFM_STATE_DIR") or artifacts_dir())


def data_dir() -> Path:
    return Path(os.getenv("TFM_DATA_DIR") or repo_root() / "data")


def resolve_confined_path(
    value: str | Path,
    root: str | Path,
    *,
    allow_absolute: bool = True,
) -> Path:
    """Resuelve una ruta sin permitir que escape de ``root``.

    Las rutas relativas se interpretan respecto a la raiz, no respecto al
    directorio de trabajo del proceso. ``resolve`` tambien impide escapar a
    traves de enlaces simbolicos ya existentes.
    """
    raw = Path(value).expanduser()
    if ".." in raw.parts:
        raise ValueError("No se permiten componentes '..' en la ruta")
    if raw.drive and not raw.is_absolute():
        raise ValueError("No se permiten rutas relativas a una unidad")
    if (raw.is_absolute() or raw.anchor) and not allow_absolute:
        raise ValueError("No se permiten rutas absolutas")

    safe_root = Path(root).expanduser().resolve()
    candidate = raw if raw.is_absolute() else safe_root / raw
    resolved = candidate.resolve()
    try:
        resolved.relative_to(safe_root)
    except ValueError as exc:
        raise ValueError("La ruta solicitada esta fuera de la raiz permitida") from exc
    return resolved


# Artefactos canonicos del proyecto (handoff 2026-08-02). Se pueden
# sobreescribir con variables de entorno para tests o entornos alternativos.
DEFAULT_PATHS: dict[str, Callable[[], Path]] = {
    "standardization_cache_db": lambda: Path(
        os.getenv("TFM_STANDARDIZATION_CACHE_DB")
        or state_dir() / "cache" / "mistral_standardization_v2.sqlite3"
    ),
    "mistral_cache": lambda: Path(
        os.getenv("TFM_MISTRAL_CACHE")
        or artifacts_dir() / "cache" / "mistral_prebalanced_no_simulated_logs_20260704.jsonl"
    ),
    "standardized_dataset": lambda: Path(
        os.getenv("TFM_STANDARDIZED_DATASET")
        or artifacts_dir() / "datasets" / "mistral_prebalanced_no_simulated_logs_standardized_20260704_all.jsonl"
    ),
    # Modelos de solo lectura incluidos en el paquete instalable.
    "detection_model": lambda: Path(
        os.getenv("TFM_DETECTION_MODEL")
        or package_data_dir()
        / "models"
        / "xgboost_detection_balanced_by_origin_20260905.joblib"
    ),
    "attack_type_model": lambda: Path(
        os.getenv("TFM_ATTACK_TYPE_MODEL")
        or package_data_dir()
        / "models"
        / "xgboost_attack_subtype_multidataset16_balanced500_20260906.joblib"
    ),
    "baselines": lambda: Path(
        os.getenv("TFM_BASELINES")
        or artifacts_dir() / "baselines" / "jorge_and_current_baselines.json"
    ),
    "case_memory_db": lambda: Path(
        os.getenv("TFM_CASE_MEMORY_DB") or state_dir() / "case_memory.db"
    ),
}


def resolve_path(key: str) -> Path:
    return DEFAULT_PATHS[key]()


def tool_result(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Decorador: toda tool devuelve JSON serializable con metadatos de traza.

    Anade ``tool_name``, ``tool_version`` y ``latency_ms``; captura errores y
    los devuelve como ``{"ok": false, "error": ...}`` en lugar de lanzar, para
    que los agentes puedan registrar el fallo en la traza del caso.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            payload = fn(*args, **kwargs)
            if not isinstance(payload, dict):
                payload = {"value": payload}
            payload.setdefault("ok", True)
        except Exception as exc:  # noqa: BLE001 - la tool nunca revienta al agente
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        payload["tool_name"] = fn.__name__
        payload["tool_version"] = MCP_LAYER_VERSION
        payload["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return payload

    return wrapper


def iter_jsonl(path: Path, limit: int | None = None, offset: int = 0) -> Iterator[dict[str, Any]]:
    emitted = 0
    with path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            if index < offset:
                continue
            if limit is not None and emitted >= limit:
                return
            line = line.strip()
            if not line:
                continue
            emitted += 1
            yield json.loads(line)


def register_tools(mcp_app: Any, tools: dict[str, Callable[..., Any]]) -> None:
    """Registra el diccionario de tools de un servidor en su instancia FastMCP.

    Si el SDK MCP no esta disponible (``mcp_app is None``) no registra nada:
    el modo in-process sigue funcionando via el diccionario ``TOOLS``.
    """
    if mcp_app is None:
        return
    for tool_fn in tools.values():
        mcp_app.tool()(tool_fn)
