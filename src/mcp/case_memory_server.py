# src/mcp/case_memory_server.py
"""Servidor MCP de persistencia: casos, trazas y cache en SQLite."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # SDK MCP oficial; opcional para el modo in-process
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - sin SDK solo se pierde el modo stdio
    FastMCP = None

from src.contracts.attack_taxonomy import MULTIDATASET_ATTACK_CLASSES
from src.contracts.canonical import CanonicalEvent
from src.mcp.common import (
    configured_ingest_model,
    register_tools,
    resolve_path,
    state_dir,
    tool_result,
)
from src.mcp.standardization_cache import (
    CachedStandardization,
    StandardizationCache,
    bind_event_identity,
    canonicalize_row,
    compute_content_hash,
    compute_pipeline_hash,
)
from src.mcp.standardization_contract import (
    validate_standardization_success,
    validate_target_free_canonical,
)

mcp_app = FastMCP("mcp-case-memory") if FastMCP is not None else None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    attack_type TEXT,
    schema_profile TEXT,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    seq INTEGER NOT NULL,
    entry TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_case ON traces(case_id, seq);
"""

_SCHEMA_VERSION = 1
_CACHE_INSTANCES: dict[Path, StandardizationCache] = {}
_CACHE_INSTANCES_LOCK = threading.Lock()


def _normalise_legacy_payload(payload: object) -> tuple[object, str | None, bool]:
    """Migra nombres antiguos sin convertir una familia amplia en un tipo.

    ``attack_subtype`` ya representaba el tipo concreto que ahora se publica
    como ``attack_type``. En cambio, ``attack_family`` se elimina sin usar su
    valor para completar el tipo, porque esa conversion seria ambigua.
    """

    if not isinstance(payload, dict):
        return payload, None, False

    changed = False
    classification = payload.get("classification")
    if isinstance(classification, dict):
        subtype = classification.get("attack_subtype")
        attack_type = classification.get("attack_type")
        if (
            not isinstance(attack_type, str)
            or not attack_type.strip()
        ) and isinstance(subtype, str) and subtype.strip():
            classification["attack_type"] = subtype
            changed = True

        if classification.get("model_task") in {"attack_family", "attack_subtype"}:
            # Actualiza el nombre de la tarea, pero nunca copia el valor de la
            # familia. Un caso antiguo solo-familia queda sin ``attack_type`` y
            # el auditor lo rechaza de forma controlada si era malicioso.
            classification["model_task"] = "attack_type"
            changed = True

        for legacy_key in (
            "attack_family",
            "attack_subtype",
            "family_confidence",
            "family_scores",
        ):
            if legacy_key in classification:
                classification.pop(legacy_key)
                changed = True

    explanation = payload.get("explanation")
    if isinstance(explanation, dict):
        subtype = explanation.get("attack_subtype")
        attack_type = explanation.get("attack_type")
        if (
            not isinstance(attack_type, str)
            or not attack_type.strip()
        ) and isinstance(subtype, str) and subtype.strip():
            explanation["attack_type"] = subtype
            changed = True

        for legacy_key in ("attack_family", "attack_subtype"):
            if legacy_key in explanation:
                explanation.pop(legacy_key)
                changed = True

        if explanation.get("catalog_scope") == "family":
            # No se puede certificar como catalogo tipado una mitigacion que
            # fue consultada por familia. Se conserva el caso, pero sin afirmar
            # una procedencia incompatible con el contrato actual.
            explanation["catalog_scope"] = None
            changed = True

    indexed_attack_type = None
    if isinstance(classification, dict):
        candidate = classification.get("attack_type")
        if isinstance(candidate, str) and candidate in MULTIDATASET_ATTACK_CLASSES:
            indexed_attack_type = candidate
    return payload, indexed_attack_type, changed


def _migrate_schema(connection: sqlite3.Connection) -> None:
    """Aplica una sola migracion transaccional, repetible tras un fallo."""

    current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current_version >= _SCHEMA_VERSION:
        return

    connection.execute("BEGIN IMMEDIATE")
    try:
        # Otro proceso puede haber completado la migracion mientras este
        # esperaba el bloqueo de escritura.
        current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if current_version >= _SCHEMA_VERSION:
            connection.commit()
            return

        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(cases)")
        }
        if "attack_type" not in columns:
            connection.execute("ALTER TABLE cases ADD COLUMN attack_type TEXT")

        # SQLite no permite eliminar la columna sin reconstruir la tabla. La
        # columna fisica heredada puede permanecer, pero deja de indexarse y de
        # intervenir en consultas o payloads operativos.
        connection.execute("DROP INDEX IF EXISTS idx_cases_family")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cases_attack_type ON cases(attack_type)"
        )

        rows = connection.execute(
            "SELECT case_id, attack_type, payload FROM cases"
        ).fetchall()
        for case_id, stored_attack_type, raw_payload in rows:
            try:
                decoded = json.loads(raw_payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            payload, indexed_attack_type, changed = _normalise_legacy_payload(decoded)
            if changed or stored_attack_type != indexed_attack_type:
                connection.execute(
                    "UPDATE cases SET attack_type = ?, payload = ? WHERE case_id = ?",
                    (
                        indexed_attack_type,
                        json.dumps(payload, ensure_ascii=False, default=str),
                        case_id,
                    ),
                )

        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _connect() -> sqlite3.Connection:
    path = resolve_path("case_memory_db")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.executescript(_SCHEMA)
        _migrate_schema(connection)
        return connection
    except Exception:
        connection.close()
        raise


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    """Entrega una transaccion y cierra siempre el descriptor SQLite.

    El context manager nativo de ``sqlite3.Connection`` confirma o revierte
    la transaccion, pero no cierra la conexion. El cierre explicito evita que
    Windows mantenga bloqueado el fichero entre casos o durante su limpieza.
    """

    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _standardization_cache() -> StandardizationCache:
    """Devuelve la cache cuya propiedad operativa pertenece a este servidor."""

    path = resolve_path("standardization_cache_db").resolve(strict=False)
    case_path = resolve_path("case_memory_db").resolve(strict=False)
    if path == case_path:
        raise ValueError(
            "case_memory.db y la cache deben ser ficheros distintos "
            "dentro del mismo directorio de estado"
        )
    with _CACHE_INSTANCES_LOCK:
        cache = _CACHE_INSTANCES.get(path)
        if cache is None:
            _copy_legacy_cache_if_needed(path)
            cache = StandardizationCache(path)
            _CACHE_INSTANCES[path] = cache
        return cache


def _copy_legacy_cache_if_needed(destination: Path) -> None:
    """Copia de forma consistente la cache del antiguo subdirectorio.

    La fuente se conserva intacta como respaldo. La migracion solo se aplica a
    las rutas predeterminadas; un override explicito nunca se interpreta ni se
    mueve automaticamente.
    """

    if os.getenv("TFM_STANDARDIZATION_CACHE_DB") or os.getenv("TFM_CASE_MEMORY_DB"):
        return
    legacy = (
        state_dir()
        / "cache"
        / "mistral_standardization_v2.sqlite3"
    ).resolve(strict=False)
    if destination.exists() or not legacy.is_file() or legacy == destination:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.migrating"
    )
    source_connection: sqlite3.Connection | None = None
    target_connection: sqlite3.Connection | None = None
    try:
        source_connection = sqlite3.connect(legacy, timeout=30.0)
        target_connection = sqlite3.connect(temporary, timeout=30.0)
        source_connection.backup(target_connection)
        target_connection.close()
        target_connection = None
        source_connection.close()
        source_connection = None
        if not StandardizationCache(temporary).available:
            return
        try:
            # El enlace crea el destino solo si aun no existe, tanto en
            # Windows como en POSIX; nunca sobrescribe una cache que otro
            # proceso haya creado mientras se realizaba el backup.
            os.link(temporary, destination)
        except FileExistsError:
            pass
    except (OSError, sqlite3.Error):
        # La cache es fail-open y la copia original nunca se elimina.
        return
    finally:
        if target_connection is not None:
            target_connection.close()
        if source_connection is not None:
            source_connection.close()
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _cache_coordinates(
    *,
    row: dict[str, Any] | None,
    text: str | None,
    model: str,
) -> tuple[dict[str, Any] | None, str, str]:
    if row is not None and not isinstance(row, dict):
        raise TypeError("row debe ser un objeto JSON")
    if text is not None and not isinstance(text, str):
        raise TypeError("text debe ser una cadena")
    if (row is None) == (text is None):
        raise ValueError("se requiere exactamente una entrada row o text")
    prompt_row = canonicalize_row(row) if row is not None else None
    content_hash = compute_content_hash(
        row=prompt_row,
        text=text if prompt_row is None else None,
    )
    return prompt_row, content_hash, compute_pipeline_hash(model)


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


def _cached_standardization_response(
    cached: CachedStandardization,
    *,
    dataset: str,
    source_file: str | None,
    row_id: int | str | None,
    split: str,
    row: dict[str, Any] | None,
) -> dict[str, Any]:
    if not _selected_columns_are_compatible(cached.selected_columns, row):
        raise ValueError("columnas de cache incompatibles con la entrada")
    rebound = bind_event_identity(
        cached.event_core,
        dataset=dataset,
        source_file=source_file,
        row_id=row_id,
        split=split,
        parser_version=cached.parser_version,
    )
    unknown = sorted(set(rebound).difference(CanonicalEvent.model_fields))
    if unknown:
        raise ValueError(
            "evento cacheado fuera del contrato: " + ", ".join(unknown)
        )
    canonical = CanonicalEvent(**rebound).model_dump(mode="json")
    validate_target_free_canonical(canonical)
    return {
        "canonical_event": canonical,
        "from_cache": True,
        "source": "llm",
        "provider": "mistral",
        "model": cached.model,
        "mapping_confidence": float(
            canonical.get("mapping_confidence", 0.0) or 0.0
        ),
        "selected_columns": list(cached.selected_columns),
        "notes": ["parsed_by_llm", "source:mistral_cache"],
        "cache_content_hash": cached.content_hash,
        "cache_pipeline_hash": cached.pipeline_hash,
        "cache_created_at": cached.created_at,
    }


@tool_result
def lookup_standardization_cache(
    dataset: str,
    row: dict[str, Any] | None = None,
    text: str | None = None,
    source_file: str | None = "api",
    row_id: int | str | None = 0,
    split: str = "stream",
) -> dict[str, Any]:
    """Recupera un exito Mistral por contenido y reconstruye su identidad."""

    model = configured_ingest_model()
    prompt_row, content_hash, pipeline_hash = _cache_coordinates(
        row=row,
        text=text,
        model=model,
    )
    cache = _standardization_cache()
    base = {
        "hit": False,
        "cache_available": cache.available,
        "cache_content_hash": content_hash,
        "cache_pipeline_hash": pipeline_hash,
        "model": model,
    }
    cached = cache.get(content_hash, pipeline_hash)
    if cached is None:
        return base
    try:
        if cached.model != model:
            raise ValueError("modelo de cache incompatible")
        response = _cached_standardization_response(
            cached,
            dataset=dataset,
            source_file=source_file,
            row_id=row_id,
            split=split,
            row=prompt_row,
        )
    except (TypeError, ValueError):
        cache.delete(content_hash, pipeline_hash)
        return {**base, "invalidated": True}
    return {**response, "hit": True, "cache_available": cache.available}


@tool_result
def store_standardization_cache(
    dataset: str,
    canonical_event: dict[str, Any],
    selected_columns: list[str],
    model: str,
    row: dict[str, Any] | None = None,
    text: str | None = None,
    source_file: str | None = "api",
    row_id: int | str | None = 0,
    split: str = "stream",
) -> dict[str, Any]:
    """Persiste solo una estandarizacion Mistral valida (*first success wins*)."""

    configured_model = configured_ingest_model()
    if model != configured_model:
        raise ValueError("el modelo a almacenar no coincide con el configurado")
    prompt_row, content_hash, pipeline_hash = _cache_coordinates(
        row=row,
        text=text,
        model=model,
    )
    if not isinstance(selected_columns, list) or not all(
        isinstance(column, str) and column.strip()
        for column in selected_columns
    ):
        raise ValueError("selected_columns debe ser una lista de cadenas")
    if not _selected_columns_are_compatible(selected_columns, prompt_row):
        raise ValueError("selected_columns no coincide con la entrada")
    validated = validate_standardization_success(
        {
            "dataset": dataset,
            "row": prompt_row,
            "text": text,
            "source_file": source_file,
            "row_id": row_id,
            "split": split,
        },
        {
            "ok": True,
            "canonical_event": canonical_event,
            "from_cache": False,
            "source": "llm",
            "provider": "mistral",
            "model": model,
            "mapping_confidence": canonical_event.get("mapping_confidence")
            if isinstance(canonical_event, dict)
            else None,
            "selected_columns": selected_columns,
            "notes": ["parsed_by_llm", "source:mistral_live"],
            "cache_content_hash": content_hash,
            "cache_pipeline_hash": pipeline_hash,
        },
    )
    canonical = validated["canonical_event"]
    parser_version = str(
        (canonical.get("provenance") or {}).get("parser_version") or ""
    ).strip()
    if not parser_version:
        raise ValueError("canonical_event no declara parser_version")

    cache = _standardization_cache()
    inserted = cache.put(
        content_hash,
        pipeline_hash,
        canonical,
        selected_columns,
        model,
        parser_version,
    )
    winner = cache.get(content_hash, pipeline_hash)
    response: dict[str, Any] = {
        "stored": inserted,
        "already_present": not inserted and winner is not None,
        "status": (
            "inserted"
            if inserted
            else "already_exists" if winner is not None else "write_failed"
        ),
        "cache_available": cache.available,
        "cache_content_hash": content_hash,
        "cache_pipeline_hash": pipeline_hash,
    }
    if winner is not None:
        response["winner"] = _cached_standardization_response(
            winner,
            dataset=dataset,
            source_file=source_file,
            row_id=row_id,
            split=split,
            row=prompt_row,
        )
    return response


@tool_result
def evict_standardization_cache(
    content_hash: str,
    pipeline_hash: str,
) -> dict[str, Any]:
    """Elimina de forma selectiva una entrada de cache invalida."""

    for name, value in (
        ("content_hash", content_hash),
        ("pipeline_hash", pipeline_hash),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{name} debe ser un SHA-256 hexadecimal")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"{name} debe ser un SHA-256 hexadecimal") from exc
    cache = _standardization_cache()
    return {
        "deleted": cache.delete(content_hash, pipeline_hash),
        "cache_available": cache.available,
    }


@tool_result
def create_case(case_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Registra un caso nuevo. payload es el CaseResult (parcial) serializado."""
    with _connection() as connection:
        existing = connection.execute(
            "SELECT case_id FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        if existing:
            raise ValueError(f"El caso ya existe: {case_id}")
        connection.execute(
            "INSERT INTO cases (case_id, created_at, updated_at, status, payload) VALUES (?, ?, ?, 'open', ?)",
            (case_id, _now(), _now(), json.dumps(payload or {}, ensure_ascii=False)),
        )
    return {"case_id": case_id, "created": True}


@tool_result
def update_case(case_id: str, payload: dict[str, Any], status: str | None = None) -> dict[str, Any]:
    """Actualiza el payload (CaseResult) y metadatos de busqueda del caso."""
    classification = payload.get("classification") or {}
    canonical = payload.get("canonical_event") or {}
    candidate_attack_type = classification.get("attack_type")
    indexed_attack_type = (
        candidate_attack_type
        if candidate_attack_type in MULTIDATASET_ATTACK_CLASSES
        else None
    )
    with _connection() as connection:
        updated = connection.execute(
            "UPDATE cases SET payload = ?, updated_at = ?, status = COALESCE(?, status), "
            "attack_type = ?, schema_profile = ? WHERE case_id = ?",
            (
                json.dumps(payload, ensure_ascii=False, default=str),
                _now(),
                status or payload.get("status"),
                indexed_attack_type,
                canonical.get("schema_profile"),
                case_id,
            ),
        )
        if updated.rowcount == 0:
            raise ValueError(f"Caso no encontrado: {case_id}")
    return {"case_id": case_id, "updated": True}


@tool_result
def append_trace(case_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Anade una entrada de traza (TraceEntry serializado) al caso."""
    with _connection() as connection:
        existing = connection.execute(
            "SELECT 1 FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        if existing is None:
            raise ValueError(f"Caso no encontrado: {case_id}")
        row = connection.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM traces WHERE case_id = ?", (case_id,)
        ).fetchone()
        seq = int(row[0]) + 1
        connection.execute(
            "INSERT INTO traces (case_id, seq, entry) VALUES (?, ?, ?)",
            (case_id, seq, json.dumps(entry, ensure_ascii=False, default=str)),
        )
    return {"case_id": case_id, "seq": seq}


@tool_result
def get_case(case_id: str, include_trace: bool = True) -> dict[str, Any]:
    """Recupera un caso y (opcionalmente) su traza completa."""
    with _connection() as connection:
        row = connection.execute(
            "SELECT payload, status, created_at, updated_at FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Caso no encontrado: {case_id}")
        payload = json.loads(row[0])
        result: dict[str, Any] = {
            "case_id": case_id,
            "status": row[1],
            "created_at": row[2],
            "updated_at": row[3],
            "case": payload,
        }
        if include_trace:
            entries = connection.execute(
                "SELECT entry FROM traces WHERE case_id = ? ORDER BY seq", (case_id,)
            ).fetchall()
            result["trace"] = [json.loads(item[0]) for item in entries]
    return result


@tool_result
def list_cases(limit: int = 20, status: str | None = None) -> dict[str, Any]:
    """Lista casos recientes, opcionalmente filtrados por estado."""
    query = "SELECT case_id, status, attack_type, created_at FROM cases"
    params: tuple[Any, ...] = ()
    if status:
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY created_at DESC LIMIT ?"
    params += (limit,)
    with _connection() as connection:
        rows = connection.execute(query, params).fetchall()
    return {
        "count": len(rows),
        "cases": [
            {"case_id": r[0], "status": r[1], "attack_type": r[2], "created_at": r[3]}
            for r in rows
        ],
    }


@tool_result
def retrieve_similar_cases(
    attack_type: str | None = None,
    schema_profile: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Casos previos similares por tipo de ataque y/o perfil de esquema."""
    if attack_type is not None and attack_type not in MULTIDATASET_ATTACK_CLASSES:
        raise ValueError(f"Tipo de ataque no soportado: {attack_type!r}")
    clauses: list[str] = []
    params: list[Any] = []
    if attack_type:
        clauses.append("attack_type = ?")
        params.append(attack_type)
    if schema_profile:
        clauses.append("schema_profile = ?")
        params.append(schema_profile)
    query = "SELECT case_id, status, attack_type, schema_profile, created_at FROM cases"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with _connection() as connection:
        rows = connection.execute(query, tuple(params)).fetchall()
    return {
        "count": len(rows),
        "cases": [
            {
                "case_id": r[0],
                "status": r[1],
                "attack_type": r[2],
                "schema_profile": r[3],
                "created_at": r[4],
            }
            for r in rows
        ],
    }


TOOLS = {
    "lookup_standardization_cache": lookup_standardization_cache,
    "store_standardization_cache": store_standardization_cache,
    "evict_standardization_cache": evict_standardization_cache,
    "create_case": create_case,
    "update_case": update_case,
    "append_trace": append_trace,
    "get_case": get_case,
    "list_cases": list_cases,
    "retrieve_similar_cases": retrieve_similar_cases,
}
register_tools(mcp_app, TOOLS)


if __name__ == "__main__":
    if mcp_app is None:
        raise SystemExit("SDK MCP no disponible: instala el proyecto con 'pip install .'.")
    mcp_app.run()
