from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from src.mcp.standardization_cache import (
    CACHE_SCHEMA_VERSION,
    StandardizationCache,
    bind_event_identity,
    canonicalize_row,
    compute_content_hash,
    strip_event_identity,
)


def _canonical_event(event_id: str, semantic_text: str) -> dict:
    return {
        "event_id": event_id,
        "modality": "network_flow",
        "schema_profile": "network_flow",
        "semantic_text": semantic_text,
        "mapping_confidence": 0.91,
        "telemetry": {"duration": 2.5},
        "origin": {
            "source_name": "old-dataset",
            "source_file": "old.csv",
            "row_id": 7,
        },
        "provenance": {
            "dataset": "old-dataset",
            "source_file": "old.csv",
            "row_id": 7,
            "split": "train",
            "parser_version": "old-parser",
            "ingestion_ts": "2025-01-01T00:00:00+00:00",
        },
    }


def test_content_hash_uses_exact_clean_payload_with_stable_dict_order():
    first = {"proto": "tcp", "nested": {"z": 2, "a": 1}, "ports": [80, 443]}
    reordered = {"ports": [80, 443], "nested": {"a": 1, "z": 2}, "proto": "tcp"}

    assert compute_content_hash(row=first) == compute_content_hash(row=reordered)
    exact_json = b'{"nested":{"a":1,"z":2},"ports":[80,443],"proto":"tcp"}'
    assert compute_content_hash(row=first) == hashlib.sha256(exact_json).hexdigest()
    assert compute_content_hash(row={"label": "Mirai"}) != compute_content_hash(
        row={"label": "DDoS"}
    )
    assert compute_content_hash(row={"value": "same"}) != compute_content_hash(
        text='{"value":"same"}'
    )

    with pytest.raises(ValueError, match="exactly one"):
        compute_content_hash()
    with pytest.raises(ValueError, match="exactly one"):
        compute_content_hash(row={}, text="event")
    with pytest.raises(ValueError, match="finite"):
        compute_content_hash(row={"duration": float("nan")})
    with pytest.raises(ValueError, match="keys must be strings"):
        compute_content_hash(row={1: "coerced by json"})
    with pytest.raises(ValueError, match="keys must be strings"):
        compute_content_hash(row={"nested": {1: "also ambiguous"}})


def test_canonical_row_matches_the_recursive_order_used_by_the_hash():
    original = {
        "z": {"last": 2, "first": 1},
        "a": [{"right": 2, "left": 1}],
    }

    normalized = canonicalize_row(original)

    assert list(normalized) == ["a", "z"]
    assert list(normalized["z"]) == ["first", "last"]
    assert list(normalized["a"][0]) == ["left", "right"]
    assert original == {
        "z": {"last": 2, "first": 1},
        "a": [{"right": 2, "left": 1}],
    }


def test_strip_event_identity_is_deep_and_does_not_mutate_input():
    event = _canonical_event("old-id", "technical event")

    core = strip_event_identity(event)
    core["telemetry"]["duration"] = 99

    assert set(core).isdisjoint({"event_id", "origin", "provenance"})
    assert event["event_id"] == "old-id"
    assert event["telemetry"]["duration"] == 2.5


def test_cache_v2_persists_only_identity_free_core_and_metadata(tmp_path):
    path = tmp_path / "standardization.sqlite3"
    content_hash = compute_content_hash(row={"proto": "tcp", "duration": 2.5})
    pipeline_hash = "pipeline-sha-from-caller"
    event = _canonical_event("llm::old::old.csv::7", "tcp flow")
    cache = StandardizationCache(path)

    assert cache.available is True
    assert cache.put(
        content_hash,
        pipeline_hash,
        event,
        ["proto", "duration"],
        "mistral-small-2603",
        "llm-0.2.0",
    ) is True

    cached = cache.get(content_hash, pipeline_hash)
    assert cached is not None
    assert cached.event_core["semantic_text"] == "tcp flow"
    assert set(cached.event_core).isdisjoint({"event_id", "origin", "provenance"})
    assert cached.selected_columns == ["proto", "duration"]
    assert cached.model == "mistral-small-2603"
    assert cached.parser_version == "llm-0.2.0"

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == CACHE_SCHEMA_VERSION
        raw_core, raw_columns = connection.execute(
            "SELECT event_core_json, selected_columns_json "
            "FROM standardization_cache_v2"
        ).fetchone()
    assert set(json.loads(raw_core)).isdisjoint({"event_id", "origin", "provenance"})
    assert json.loads(raw_columns) == ["proto", "duration"]

    reopened = StandardizationCache(path).get(content_hash, pipeline_hash)
    assert reopened is not None
    assert reopened.event_core == cached.event_core

    cached.event_core["telemetry"]["duration"] = 500
    cached.selected_columns.append("mutated")
    unchanged = cache.get(content_hash, pipeline_hash)
    assert unchanged is not None
    assert unchanged.event_core["telemetry"]["duration"] == 2.5
    assert unchanged.selected_columns == ["proto", "duration"]


def test_first_success_wins_and_pipeline_hash_is_an_external_key(tmp_path):
    cache = StandardizationCache(tmp_path / "standardization.sqlite3")
    content_hash = compute_content_hash(text="same technical event")
    first = _canonical_event("first-id", "first result")
    later = _canonical_event("later-id", "later result")

    assert cache.put(
        content_hash,
        "pipeline-A",
        first,
        [],
        "mistral-a",
        "parser-a",
    ) is True
    assert cache.put(
        content_hash,
        "pipeline-A",
        later,
        ["other"],
        "mistral-b",
        "parser-b",
    ) is False

    winner = cache.get(content_hash, "pipeline-A")
    assert winner is not None
    assert winner.event_core["semantic_text"] == "first result"
    assert winner.model == "mistral-a"

    assert cache.get(content_hash, "pipeline-B") is None
    assert cache.put(
        content_hash,
        "pipeline-B",
        later,
        [],
        "mistral-b",
        "parser-b",
    ) is True
    assert cache.get(content_hash, "pipeline-B").event_core[
        "semantic_text"
    ] == "later result"


def test_bind_event_identity_replaces_all_row_identity_with_fresh_values():
    stale = _canonical_event("stale-id", "cached result")
    core = strip_event_identity(stale)
    ingestion = datetime(2026, 8, 31, 10, 30, tzinfo=timezone.utc)

    rebound = bind_event_identity(
        core,
        dataset="iot23",
        source_file="capture.csv",
        row_id=42,
        split="test",
        parser_version="llm-0.2.0",
        ingestion_ts=ingestion,
    )

    assert rebound["event_id"] == "llm::iot23::capture.csv::42"
    assert rebound["origin"] == {
        "source_name": "iot23",
        "source_file": "capture.csv",
        "row_id": 42,
        "schema_profile": "network_flow",
    }
    assert rebound["provenance"] == {
        "dataset": "iot23",
        "source_file": "capture.csv",
        "row_id": 42,
        "split": "test",
        "parser_version": "llm-0.2.0",
        "ingestion_ts": "2026-08-31T10:30:00+00:00",
    }
    assert set(core).isdisjoint({"event_id", "origin", "provenance"})

    before = datetime.now(timezone.utc)
    with_new_ingestion = bind_event_identity(
        core,
        dataset="iot23",
        source_file=None,
        row_id="stream-9",
        split="stream",
        parser_version="llm-0.2.0",
    )
    after = datetime.now(timezone.utc)
    generated = datetime.fromisoformat(
        with_new_ingestion["provenance"]["ingestion_ts"]
    )
    assert before <= generated <= after
    assert with_new_ingestion["event_id"] == "llm::iot23::inline::stream-9"


def test_get_put_and_delete_are_tolerant(tmp_path):
    corrupt_path = tmp_path / "corrupt.sqlite3"
    corrupt_path.write_bytes(b"this is not a sqlite database")
    cache = StandardizationCache(corrupt_path, timeout_seconds=0.1)
    event = _canonical_event("id", "event")

    assert cache.available is False
    assert cache.get("content", "pipeline") is None
    assert cache.put(
        "content", "pipeline", event, [], "mistral", "parser"
    ) is False
    assert cache.delete("content", "pipeline") is False

    valid = StandardizationCache(tmp_path / "valid.sqlite3")
    assert valid.get("", "pipeline") is None
    assert valid.put("content", "pipeline", {"bad": object()}, [], "m", "p") is False
    assert valid.delete("missing", "pipeline") is False

    memory_only = StandardizationCache(":memory:")
    assert memory_only.available is False
    assert memory_only.get("content", "pipeline") is None


def test_delete_removes_only_the_requested_pipeline_entry(tmp_path):
    cache = StandardizationCache(tmp_path / "standardization.sqlite3")
    content_hash = compute_content_hash(text="event")
    event = _canonical_event("id", "event")
    for pipeline_hash in ("pipeline-A", "pipeline-B"):
        assert cache.put(
            content_hash,
            pipeline_hash,
            event,
            [],
            "mistral",
            "parser",
        ) is True

    assert cache.delete(content_hash, "pipeline-A") is True
    assert cache.delete(content_hash, "pipeline-A") is False
    assert cache.get(content_hash, "pipeline-A") is None
    assert cache.get(content_hash, "pipeline-B") is not None


def test_singleflight_allows_only_one_computation_for_a_key(tmp_path):
    path = tmp_path / "standardization.sqlite3"
    cache = StandardizationCache(path)
    second_instance = StandardizationCache(path)
    content_hash = compute_content_hash(row={"proto": "tcp"})
    pipeline_hash = "pipeline"
    worker_count = 8
    barrier = threading.Barrier(worker_count)
    counter_lock = threading.Lock()
    computations = 0

    def worker(index: int) -> str:
        nonlocal computations
        worker_cache = cache if index % 2 == 0 else second_instance
        barrier.wait()
        with worker_cache.singleflight(content_hash, pipeline_hash):
            cached = worker_cache.get(content_hash, pipeline_hash)
            if cached is None:
                with counter_lock:
                    computations += 1
                time.sleep(0.02)
                assert worker_cache.put(
                    content_hash,
                    pipeline_hash,
                    _canonical_event("winner", "computed once"),
                    ["proto"],
                    "mistral",
                    "parser",
                ) is True
            cached = worker_cache.get(content_hash, pipeline_hash)
            assert cached is not None
            return cached.event_core["semantic_text"]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(worker, range(worker_count)))

    assert computations == 1
    assert results == ["computed once"] * worker_count

    with pytest.raises(RuntimeError, match="body failed"):
        with cache.singleflight("exception-content", pipeline_hash):
            raise RuntimeError("body failed")
    with second_instance.singleflight("exception-content", pipeline_hash):
        pass


@pytest.mark.parametrize("legacy_version", [0, 1])
def test_legacy_schema_is_not_migrated_implicitly(tmp_path, legacy_version):
    path = tmp_path / f"legacy-v{legacy_version}.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE legacy_cache (cache_key TEXT PRIMARY KEY)")
        connection.execute(f"PRAGMA user_version = {legacy_version}")

    cache = StandardizationCache(path)

    assert cache.available is False
    assert cache.get("content", "pipeline") is None
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == legacy_version
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "legacy_cache" in table_names
    assert "standardization_cache_v2" not in table_names


def test_declared_v2_database_with_wrong_schema_is_rejected(tmp_path):
    path = tmp_path / "malformed-v2.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE standardization_cache_v2 (content_hash TEXT PRIMARY KEY)"
        )
        connection.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION}")

    cache = StandardizationCache(path)

    assert cache.available is False
    assert cache.get("content", "pipeline") is None
    assert cache.put(
        "content",
        "pipeline",
        _canonical_event("id", "event"),
        [],
        "mistral",
        "parser",
    ) is False
