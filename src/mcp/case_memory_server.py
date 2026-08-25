# src/mcp/case_memory_server.py
"""Servidor MCP de memoria de casos: persistencia y trazas en SQLite."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

try:  # SDK MCP oficial; opcional para el modo in-process
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - sin SDK solo se pierde el modo stdio
    FastMCP = None

from src.mcp.common import register_tools, resolve_path, tool_result

mcp_app = FastMCP("mcp-case-memory") if FastMCP is not None else None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    attack_family TEXT,
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
CREATE INDEX IF NOT EXISTS idx_cases_family ON cases(attack_family);
"""


def _connect() -> sqlite3.Connection:
    path = resolve_path("case_memory_db")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_SCHEMA)
    return connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@tool_result
def create_case(case_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Registra un caso nuevo. payload es el CaseResult (parcial) serializado."""
    with _connect() as connection:
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
    with _connect() as connection:
        updated = connection.execute(
            "UPDATE cases SET payload = ?, updated_at = ?, status = COALESCE(?, status), "
            "attack_family = ?, schema_profile = ? WHERE case_id = ?",
            (
                json.dumps(payload, ensure_ascii=False, default=str),
                _now(),
                status or payload.get("status"),
                classification.get("attack_family"),
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
    with _connect() as connection:
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
    with _connect() as connection:
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
    query = "SELECT case_id, status, attack_family, created_at FROM cases"
    params: tuple[Any, ...] = ()
    if status:
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY created_at DESC LIMIT ?"
    params += (limit,)
    with _connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return {
        "count": len(rows),
        "cases": [
            {"case_id": r[0], "status": r[1], "attack_family": r[2], "created_at": r[3]}
            for r in rows
        ],
    }


@tool_result
def retrieve_similar_cases(
    attack_family: str | None = None,
    schema_profile: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Casos previos similares por familia y/o perfil de esquema."""
    clauses: list[str] = []
    params: list[Any] = []
    if attack_family:
        clauses.append("attack_family = ?")
        params.append(attack_family)
    if schema_profile:
        clauses.append("schema_profile = ?")
        params.append(schema_profile)
    query = "SELECT case_id, status, attack_family, schema_profile, created_at FROM cases"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with _connect() as connection:
        rows = connection.execute(query, tuple(params)).fetchall()
    return {
        "count": len(rows),
        "cases": [
            {
                "case_id": r[0],
                "status": r[1],
                "attack_family": r[2],
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
        raise SystemExit("SDK MCP no disponible: instala mcp[cli]>=1.2 (extra [mcp]).")
    mcp_app.run()
