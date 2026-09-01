"""Persistent, identity-free cache for successful LLM standardizations.

The cache deliberately separates two concerns:

* the reusable technical mapping produced by the standardizer; and
* the identity and provenance of the row currently being processed.

Callers therefore fingerprint only the already-clean ``row`` or ``text``
payload, pass the pipeline fingerprint explicitly, and bind fresh identity
metadata after a cache hit.  This module does not sanitize inputs and has no
dependency on the runtime standardization guard.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


CACHE_SCHEMA_VERSION = 2
_TABLE_NAME = "standardization_cache_v2"
_IDENTITY_FIELDS = frozenset({"event_id", "origin", "provenance"})
_VALID_SPLITS = frozenset({"train", "val", "test", "stream"})
_SCHEMA_COLUMNS = (
    ("content_hash", "TEXT", 1, 1),
    ("pipeline_hash", "TEXT", 1, 2),
    ("event_core_json", "TEXT", 1, 0),
    ("selected_columns_json", "TEXT", 1, 0),
    ("model", "TEXT", 1, 0),
    ("parser_version", "TEXT", 1, 0),
    ("created_at", "TEXT", 1, 0),
)


@dataclass(slots=True)
class CachedStandardization:
    """A reusable standardization and the metadata needed to validate it."""

    content_hash: str
    pipeline_hash: str
    event_core: dict[str, Any]
    selected_columns: list[str]
    model: str
    parser_version: str
    created_at: str


@dataclass(slots=True)
class _SingleFlightEntry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    references: int = 0


_SINGLE_FLIGHT_GUARD = threading.Lock()
_SINGLE_FLIGHT_LOCKS: dict[tuple[str, str, str], _SingleFlightEntry] = {}


def _canonical_json(value: Any) -> str:
    _validate_json_mapping_keys(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must be finite and JSON serializable") from exc


def canonicalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the row in the exact key order used by its content hash.

    No field or value is filtered.  The JSON round-trip only enforces the
    operational input contract and recursively sorts object keys, so two rows
    sharing a cache key are also presented identically to the LLM.
    """

    if not isinstance(row, Mapping):
        raise TypeError("row must be a mapping")
    normalized = json.loads(_canonical_json(dict(row)))
    if not isinstance(normalized, dict):  # defensive; the input is a mapping
        raise TypeError("row must serialize as a JSON object")
    return normalized


def compute_content_hash(
    *,
    row: Mapping[str, Any] | None = None,
    text: str | None = None,
) -> str:
    """Return a stable SHA-256 for exactly one already-clean payload.

    Identity such as dataset, source file and row id is intentionally absent
    from this API.  Dictionary key order does not affect a row fingerprint;
    values and list order remain untouched.  JSON's object and string
    encodings distinguish a tabular row from text without adding an envelope.

    The payload must be JSON serializable.  No filtering, coercion or
    sanitization is performed here.
    """

    if (row is None) == (text is None):
        raise ValueError("provide exactly one of row or text")
    if row is not None and not isinstance(row, Mapping):
        raise TypeError("row must be a mapping")
    if text is not None and not isinstance(text, str):
        raise TypeError("text must be a string")

    payload: Any = dict(row) if row is not None else text
    serialized = _canonical_json(payload)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_json_mapping_keys(
    value: Any,
    active_containers: set[int] | None = None,
) -> None:
    """Reject JSON object keys that would be silently coerced to strings."""

    if not isinstance(value, (dict, list, tuple)):
        return
    active = active_containers if active_containers is not None else set()
    marker = id(value)
    if marker in active:
        raise ValueError("payload contains a circular reference")
    active.add(marker)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("all JSON object keys must be strings")
                _validate_json_mapping_keys(item, active)
        else:
            for item in value:
                _validate_json_mapping_keys(item, active)
    finally:
        active.remove(marker)


def strip_event_identity(canonical_event: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy without row-specific identity or provenance."""

    if not isinstance(canonical_event, Mapping):
        raise TypeError("canonical_event must be a mapping")
    event_core = copy.deepcopy(dict(canonical_event))
    for field_name in _IDENTITY_FIELDS:
        event_core.pop(field_name, None)
    return event_core


def bind_event_identity(
    event_core: Mapping[str, Any],
    *,
    dataset: str,
    source_file: str | None,
    row_id: str | int | None,
    split: str,
    parser_version: str,
    ingestion_ts: datetime | None = None,
) -> dict[str, Any]:
    """Bind fresh, trusted identity to a cached event core.

    The input mapping is never mutated and any stale identity fields are
    discarded defensively.  Supplying ``ingestion_ts`` makes the pure
    transformation deterministic for tests; otherwise a new UTC timestamp is
    generated for every invocation.
    """

    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty string")
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(_VALID_SPLITS)}")
    if not isinstance(parser_version, str) or not parser_version.strip():
        raise ValueError("parser_version must be a non-empty string")
    if source_file is not None and not isinstance(source_file, str):
        raise TypeError("source_file must be a string or None")
    if row_id is not None and (
        isinstance(row_id, bool) or not isinstance(row_id, (str, int))
    ):
        raise TypeError("row_id must be a string, integer or None")

    timestamp = ingestion_ts or datetime.now(timezone.utc)
    if not isinstance(timestamp, datetime):
        raise TypeError("ingestion_ts must be a datetime or None")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("ingestion_ts must be timezone-aware")
    timestamp = timestamp.astimezone(timezone.utc)

    rebound = strip_event_identity(event_core)
    dataset_name = dataset.strip()
    source_ref = source_file if source_file is not None else "inline"
    rebound["event_id"] = f"llm::{dataset_name}::{source_ref}::{row_id}"

    origin: dict[str, Any] = {
        "source_name": dataset_name,
        "source_file": source_ref,
        "row_id": row_id,
    }
    schema_profile = rebound.get("schema_profile")
    if schema_profile is not None:
        origin["schema_profile"] = schema_profile
    rebound["origin"] = origin
    rebound["provenance"] = {
        "dataset": dataset_name,
        "source_file": source_file,
        "row_id": row_id,
        "split": split,
        "parser_version": parser_version.strip(),
        "ingestion_ts": timestamp.isoformat(),
    }
    return rebound


class StandardizationCache:
    """SQLite v2 cache keyed by technical content and pipeline hashes.

    The database path is always supplied by the caller.  Database failures are
    fail-open for the application: ``get`` behaves as a miss and ``put`` or
    ``delete`` return ``False``.  Existing entries are never overwritten, so
    the first successfully persisted result wins even across processes.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or pathlib.Path")
        if not isinstance(timeout_seconds, (int, float)) or isinstance(
            timeout_seconds, bool
        ):
            raise TypeError("timeout_seconds must be numeric")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self.path = Path(path)
        self.timeout_seconds = float(timeout_seconds)
        self._initialization_lock = threading.Lock()
        self._ready = False
        self._lock_path = str(self.path.resolve(strict=False))
        self._ready = self._initialize()

    @property
    def available(self) -> bool:
        """Whether the v2 schema was initialized successfully."""

        return self._ready

    def get(
        self,
        content_hash: str,
        pipeline_hash: str,
    ) -> CachedStandardization | None:
        """Return a cached result or ``None`` on miss or storage failure."""

        if not self._valid_key(content_hash, pipeline_hash):
            return None
        if not self._ensure_ready():
            return None

        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            row = connection.execute(
                f"""
                SELECT event_core_json, selected_columns_json, model,
                       parser_version, created_at
                FROM {_TABLE_NAME}
                WHERE content_hash = ? AND pipeline_hash = ?
                """,
                (content_hash, pipeline_hash),
            ).fetchone()
            if row is None:
                return None

            event_core = json.loads(row[0])
            selected_columns = json.loads(row[1])
            if not isinstance(event_core, dict):
                return None
            if not isinstance(selected_columns, list) or not all(
                isinstance(column, str) for column in selected_columns
            ):
                return None
            if not isinstance(row[2], str) or not row[2].strip():
                return None
            if not isinstance(row[3], str) or not row[3].strip():
                return None
            if not isinstance(row[4], str) or not row[4].strip():
                return None

            return CachedStandardization(
                content_hash=content_hash,
                pipeline_hash=pipeline_hash,
                event_core=strip_event_identity(event_core),
                selected_columns=list(selected_columns),
                model=row[2],
                parser_version=row[3],
                created_at=row[4],
            )
        except (json.JSONDecodeError, sqlite3.Error, OSError, TypeError, ValueError):
            return None
        finally:
            if connection is not None:
                connection.close()

    def put(
        self,
        content_hash: str,
        pipeline_hash: str,
        canonical_event: Mapping[str, Any],
        selected_columns: list[str] | tuple[str, ...],
        model: str,
        parser_version: str,
    ) -> bool:
        """Persist a successful result without replacing an existing one.

        Returns ``True`` only when this call inserted the winning row.  A
        duplicate key, invalid value or storage failure returns ``False``.
        """

        if not self._valid_key(content_hash, pipeline_hash):
            return False
        if not isinstance(canonical_event, Mapping):
            return False
        if not isinstance(selected_columns, (list, tuple)) or not all(
            isinstance(column, str) and column.strip()
            for column in selected_columns
        ):
            return False
        if not isinstance(model, str) or not model.strip():
            return False
        if not isinstance(parser_version, str) or not parser_version.strip():
            return False
        if not self._ensure_ready():
            return False

        try:
            event_core_json = json.dumps(
                strip_event_identity(canonical_event),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            selected_columns_json = json.dumps(
                list(selected_columns),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            return False

        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            cursor = connection.execute(
                f"""
                INSERT OR IGNORE INTO {_TABLE_NAME} (
                    content_hash,
                    pipeline_hash,
                    event_core_json,
                    selected_columns_json,
                    model,
                    parser_version,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content_hash,
                    pipeline_hash,
                    event_core_json,
                    selected_columns_json,
                    model.strip(),
                    parser_version.strip(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()
            return cursor.rowcount == 1
        except (sqlite3.Error, OSError):
            return False
        finally:
            if connection is not None:
                connection.close()

    def delete(self, content_hash: str, pipeline_hash: str) -> bool:
        """Delete one entry, returning ``False`` on miss or storage failure."""

        if not self._valid_key(content_hash, pipeline_hash):
            return False
        if not self._ensure_ready():
            return False

        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            cursor = connection.execute(
                f"""
                DELETE FROM {_TABLE_NAME}
                WHERE content_hash = ? AND pipeline_hash = ?
                """,
                (content_hash, pipeline_hash),
            )
            connection.commit()
            return cursor.rowcount == 1
        except (sqlite3.Error, OSError):
            return False
        finally:
            if connection is not None:
                connection.close()

    @contextmanager
    def singleflight(
        self,
        content_hash: str,
        pipeline_hash: str,
    ) -> Iterator[None]:
        """Serialize work for one cache key within the current process.

        Callers should check ``get`` again after entering this context.  The
        database primary key still provides first-success-wins semantics
        across processes.
        """

        if not self._valid_key(content_hash, pipeline_hash):
            raise ValueError("content_hash and pipeline_hash must be non-empty strings")

        registry_key = (self._lock_path, content_hash, pipeline_hash)
        with _SINGLE_FLIGHT_GUARD:
            entry = _SINGLE_FLIGHT_LOCKS.get(registry_key)
            if entry is None:
                entry = _SingleFlightEntry()
                _SINGLE_FLIGHT_LOCKS[registry_key] = entry
            entry.references += 1

        acquired = False
        try:
            entry.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            with _SINGLE_FLIGHT_GUARD:
                entry.references -= 1
                if (
                    entry.references == 0
                    and _SINGLE_FLIGHT_LOCKS.get(registry_key) is entry
                ):
                    _SINGLE_FLIGHT_LOCKS.pop(registry_key, None)

    @staticmethod
    def _valid_key(content_hash: object, pipeline_hash: object) -> bool:
        return (
            isinstance(content_hash, str)
            and bool(content_hash.strip())
            and isinstance(pipeline_hash, str)
            and bool(pipeline_hash.strip())
        )

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True
        with self._initialization_lock:
            if not self._ready:
                self._ready = self._initialize()
        return self._ready

    def _initialize(self) -> bool:
        connection: sqlite3.Connection | None = None
        try:
            # A connection-per-operation design requires a filesystem-backed
            # database; accepting :memory: would report a misleading success.
            if str(self.path) == ":memory:":
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._connect()
            version_row = connection.execute("PRAGMA user_version").fetchone()
            current_version = int(version_row[0]) if version_row else 0
            if current_version not in (0, CACHE_SCHEMA_VERSION):
                return False

            schema_objects = connection.execute(
                """
                SELECT name, type
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
            if current_version == 0:
                # Only an actually empty database is initialized.  An
                # unversioned legacy schema is neither adopted nor migrated.
                if schema_objects:
                    return False
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute(
                    f"""
                    CREATE TABLE {_TABLE_NAME} (
                        content_hash TEXT NOT NULL,
                        pipeline_hash TEXT NOT NULL,
                        event_core_json TEXT NOT NULL,
                        selected_columns_json TEXT NOT NULL,
                        model TEXT NOT NULL,
                        parser_version TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (content_hash, pipeline_hash)
                    )
                    """
                )
                if not self._schema_matches_v2(connection):
                    return False
                connection.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION}")
            else:
                if (_TABLE_NAME, "table") not in schema_objects:
                    return False
                if not self._schema_matches_v2(connection):
                    return False
                connection.execute("PRAGMA journal_mode = WAL")
            connection.commit()
            return True
        except (sqlite3.Error, OSError, TypeError, ValueError):
            return False
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _schema_matches_v2(connection: sqlite3.Connection) -> bool:
        columns = connection.execute(
            f"PRAGMA table_info({_TABLE_NAME})"
        ).fetchall()
        observed = tuple(
            (str(column[1]), str(column[2]).upper(), int(column[3]), int(column[5]))
            for column in columns
        )
        return observed == _SCHEMA_COLUMNS

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout_seconds,
        )
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CachedStandardization",
    "StandardizationCache",
    "bind_event_identity",
    "canonicalize_row",
    "compute_content_hash",
    "strip_event_identity",
]
