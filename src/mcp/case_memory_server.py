# src/mcp/case_memory_server.py
"""Servidor MCP de memoria de casos: persistencia y trazas en SQLite."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

try:  # SDK MCP oficial; opcional para el modo in-process
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - sin SDK solo se pierde el modo stdio
    FastMCP = None

from src.contracts.attack_taxonomy import MULTIDATASET_ATTACK_CLASSES
from src.mcp.common import register_tools, resolve_path, tool_result

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
