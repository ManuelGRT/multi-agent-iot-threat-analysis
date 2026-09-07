# tests/test_mcp_servers.py
from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, closing
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.contracts.attack_taxonomy import (
    MULTIDATASET_ATTACK_CLASSES,
    MULTIDATASET_TAXONOMY_VERSION,
)
from src.contracts.canonical import CanonicalEvent
from src.mcp.client import (
    CLIENT_MODE_ENV,
    DEFAULT_CLIENT_MODE,
    DEFAULT_STDIO_TIMEOUT_SECONDS,
    MCPToolClient,
    SERVER_MODULES,
    _decode_stdio_result,
    list_tools_stdio,
    resolve_mcp_client_mode,
)

client = MCPToolClient(mode="inprocess")

EXPECTED_TYPE_REFERENCES = {
    "Backdoor": ({"T1105", "T1204"}, {"CAPEC-523"}, {"M1049"}),
    "DDoS_HTTP": ({"T1498"}, {"CAPEC-125", "CAPEC-488"}, {"M1037"}),
    "DDoS_ICMP": ({"T1498"}, {"CAPEC-125", "CAPEC-487"}, {"M1037"}),
    "DDoS_TCP": ({"T1498"}, {"CAPEC-125", "CAPEC-482"}, {"M1037"}),
    "DDoS_UDP": ({"T1498"}, {"CAPEC-125", "CAPEC-486"}, {"M1037"}),
    "Fingerprinting": ({"T1595"}, {"CAPEC-224"}, {"M1042"}),
    "MITM": ({"T1557"}, {"CAPEC-94"}, {"M1041"}),
    "Password": (
        {"T1110"},
        {"CAPEC-112", "CAPEC-49"},
        {"M1032", "M1027"},
    ),
    "Port_Scanning": ({"T1046"}, {"CAPEC-300"}, {"M1042"}),
    "Ransomware": ({"T1105", "T1204"}, {"CAPEC-542"}, {"M1049"}),
    "SQL_injection": ({"T1190"}, {"CAPEC-66"}, {"M1051"}),
    "Uploading": ({"T1190"}, {"CAPEC-242"}, {"M1051"}),
    "Vulnerability_scanner": ({"T1595"}, {"CAPEC-310"}, {"M1042"}),
    "XSS": ({"T1190"}, {"CAPEC-63"}, {"M1051"}),
    "DoS": ({"T1499"}, {"CAPEC-125"}, {"M1037"}),
    "Command_and_Control": ({"T1071"}, {"CAPEC-542"}, {"M1031"}),
}

CANONICAL_EVENT = {
    "event_id": "evt-test-1",
    "modality": "network_flow",
    "src_ip": "192.168.100.5",
    "dst_ip": "192.168.100.3",
    "src_port": 45312,
    "dst_port": 80,
    "transport_proto": "tcp",
    "packet_count": 120,
    "byte_count": 4096,
    "duration_ms": 1500.0,
    "schema_profile": "network_flow",
    "semantic_text": "tcp flow with high packet rate",
    "provenance": {"dataset": "edge_iiotset", "row_id": 1},
    "mapping_confidence": 0.9,
}


# ---------------------------------------------------------------------------
# Estructura general
# ---------------------------------------------------------------------------

def test_all_servers_expose_tools():
    expected = {
        "inference": {"standardize_event", "detect_event", "classify_event"},
        "case_memory": {"create_case", "append_trace", "get_case", "retrieve_similar_cases"},
        "threat_intel": {
            "list_attack_types",
            "map_attack_type_to_attack",
            "map_attack_type_to_capec",
            "suggest_mitigations",
            "get_multidataset_attack_type_coverage",
        },
    }
    assert set(SERVER_MODULES) == set(expected)
    for server, tools in expected.items():
        available = set(client.list_tools(server))
        assert tools <= available, f"{server}: faltan tools {tools - available}"


def test_tool_results_carry_trace_metadata():
    result = client.call("threat_intel", "list_attack_types")
    assert result["ok"] is True
    assert result["tool_name"] == "list_attack_types"
    assert "tool_version" in result and "latency_ms" in result


def test_tool_errors_are_returned_not_raised(case_db):
    result = client.call("case_memory", "get_case", case_id="no_existe")
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# threat intel
# ---------------------------------------------------------------------------

def test_threat_intel_lists_exactly_the_16_operational_attack_types():
    result = client.call("threat_intel", "list_attack_types")

    assert result["ok"] is True
    assert tuple(result["attack_types"]) == MULTIDATASET_ATTACK_CLASSES
    assert result["taxonomy_version"] == MULTIDATASET_TAXONOMY_VERSION
    assert result["version"] == "3.0"
    assert "families" not in result
    assert "families_by_attack_type" not in result


@pytest.mark.parametrize("attack_type", MULTIDATASET_ATTACK_CLASSES)
def test_threat_intel_resolves_specific_non_crossed_catalog_for_all_16_types(
    attack_type,
):
    expected_attack, expected_capec, expected_mitigations = (
        EXPECTED_TYPE_REFERENCES[attack_type]
    )

    attack = client.call(
        "threat_intel",
        "map_attack_type_to_attack",
        attack_type=attack_type,
    )
    capec = client.call(
        "threat_intel",
        "map_attack_type_to_capec",
        attack_type=attack_type,
    )
    result = client.call(
        "threat_intel",
        "suggest_mitigations",
        attack_type=attack_type,
        schema_profile="network_flow",
    )

    for response in (attack, capec, result):
        assert response["ok"] is True
        assert response["attack_type"] == attack_type
        assert response["catalog_scope"] == "attack_type"
        assert "family" not in response

    assert {item["id"] for item in attack["attack_techniques"]} == expected_attack
    assert {
        item["id"] for item in attack["attack_mitigation_refs"]
    } == expected_mitigations
    assert {item["id"] for item in capec["capec_patterns"]} == expected_capec

    references = result["references"]
    assert {
        item["id"] for item in references["attack_techniques"]
    } == expected_attack
    assert {
        item["id"] for item in references["capec_patterns"]
    } == expected_capec
    assert {
        item["id"] for item in references["attack_mitigation_refs"]
    } == expected_mitigations
    assert result["taxonomy_version"] == MULTIDATASET_TAXONOMY_VERSION

    by_phase = result["mitigations_by_phase"]
    assert all(by_phase[phase] for phase in ("containment", "eradication", "prevention"))
    ordered = [
        *by_phase["containment"],
        *by_phase["eradication"],
        *by_phase["prevention"],
    ]
    assert result["mitigations_ordered"] == ordered
    assert len(ordered) >= 5
    assert len(ordered) == len(set(ordered))
    assert result["profile_actions"]


def test_threat_intel_reports_complete_multidataset_catalog_coverage():
    result = client.call(
        "threat_intel", "get_multidataset_attack_type_coverage"
    )

    assert result["ok"] is True
    assert result["taxonomy_version"] == MULTIDATASET_TAXONOMY_VERSION
    assert result["total"] == len(MULTIDATASET_ATTACK_CLASSES) == 16
    assert result["covered"] == 16
    assert {row["attack_type"] for row in result["mappings"]} == set(
        MULTIDATASET_ATTACK_CLASSES
    )
    for row in result["mappings"]:
        assert "family" not in row
        assert row["catalog_scope"] == "attack_type"
        assert row["mitigations"] >= 5
        assert row["attack_techniques"] > 0
        assert row["capec_patterns"] > 0
        assert row["attack_mitigation_refs"] > 0


@pytest.mark.parametrize(
    "alias",
    [
        "command-and-control",
        "command and control",
        "command_control",
        "C&C",
        "C2",
    ],
)
def test_command_and_control_aliases_are_rejected(alias):
    result = client.call(
        "threat_intel",
        "suggest_mitigations",
        attack_type=alias,
    )

    assert result["ok"] is False
    assert "fuera de la taxonomia operativa" in result["error"]


def test_command_and_control_exact_type_keeps_specific_scope():
    result = client.call(
        "threat_intel",
        "suggest_mitigations",
        attack_type="Command_and_Control",
    )

    assert result["ok"] is True
    assert result["attack_type"] == "Command_and_Control"
    assert result["catalog_scope"] == "attack_type"
    assert {item["id"] for item in result["references"]["attack_techniques"]} == {
        "T1071"
    }


def test_family_argument_is_rejected_instead_of_affecting_type_selection():
    result = client.call(
        "threat_intel",
        "suggest_mitigations",
        family="malware",
        attack_type="DDoS_TCP",
    )

    assert result["ok"] is False
    assert "unexpected keyword argument 'family'" in result["error"]


def test_unknown_explicit_attack_type_fails_without_fallback():
    result = client.call(
        "threat_intel",
        "suggest_mitigations",
        attack_type="DDoS_QUIC",
    )

    assert result["ok"] is False
    assert "fuera de la taxonomia operativa" in result["error"]


def test_threat_intel_mappings_ddos_tcp_by_exact_type():
    attack = client.call(
        "threat_intel", "map_attack_type_to_attack", attack_type="DDoS_TCP"
    )
    assert any(t["id"] == "T1498" for t in attack["attack_techniques"])
    capec = client.call(
        "threat_intel", "map_attack_type_to_capec", attack_type="DDoS_TCP"
    )
    assert any(p["id"] == "CAPEC-482" for p in capec["capec_patterns"])


def test_threat_intel_suggest_mitigations_with_profile():
    result = client.call(
        "threat_intel",
        "suggest_mitigations",
        attack_type="Password",
        schema_profile="network_flow",
    )
    assert result["attack_type"] == "Password"
    assert result["mitigations_ordered"], "debe haber mitigaciones"
    assert result["profile_actions"], "debe haber acciones por perfil"
    assert result["references"]["attack_techniques"], "debe haber referencias ATT&CK"


def test_threat_intel_missing_attack_type_is_rejected():
    result = client.call("threat_intel", "suggest_mitigations")
    assert result["ok"] is False
    assert "missing 1 required positional argument" in result["error"]


def test_threat_intel_normal_is_not_a_seventeenth_attack_type():
    result = client.call(
        "threat_intel", "suggest_mitigations", attack_type="Normal"
    )
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# case memory
# ---------------------------------------------------------------------------

@pytest.fixture()
def case_db(tmp_path, monkeypatch):
    monkeypatch.setenv("TFM_CASE_MEMORY_DB", str(tmp_path / "cases.db"))
    yield


def test_case_memory_lifecycle(case_db):
    created = client.call("case_memory", "create_case", case_id="case-test-1", payload={"a": 1})
    assert created["ok"] and created["created"]

    duplicated = client.call("case_memory", "create_case", case_id="case-test-1")
    assert duplicated["ok"] is False

    client.call("case_memory", "append_trace", case_id="case-test-1", entry={"agent": "detector", "status": "ok"})
    client.call("case_memory", "append_trace", case_id="case-test-1", entry={"agent": "judge", "status": "ok"})

    payload = {
        "status": "completed",
        "classification": {"attack_type": "DDoS_TCP"},
        "canonical_event": {"schema_profile": "network_flow"},
    }
    updated = client.call("case_memory", "update_case", case_id="case-test-1", payload=payload)
    assert updated["ok"]

    fetched = client.call("case_memory", "get_case", case_id="case-test-1")
    assert fetched["ok"]
    assert fetched["status"] == "completed"
    assert len(fetched["trace"]) == 2
    assert fetched["trace"][0]["agent"] == "detector"

    similar = client.call(
        "case_memory", "retrieve_similar_cases", attack_type="DDoS_TCP"
    )
    assert similar["count"] == 1


def test_case_memory_preserves_but_never_indexes_an_invalid_attack_type(case_db):
    assert client.call(
        "case_memory", "create_case", case_id="case-invalid-type", payload={}
    )["ok"]
    payload = {
        "status": "completed",
        "classification": {"attack_type": "invented_attack"},
    }

    updated = client.call(
        "case_memory",
        "update_case",
        case_id="case-invalid-type",
        payload=payload,
    )
    fetched = client.call(
        "case_memory", "get_case", case_id="case-invalid-type", include_trace=False
    )
    listed = client.call("case_memory", "list_cases")
    invalid_search = client.call(
        "case_memory",
        "retrieve_similar_cases",
        attack_type="invented_attack",
    )

    assert updated["ok"] is True
    assert fetched["case"]["classification"]["attack_type"] == "invented_attack"
    indexed = next(
        item for item in listed["cases"] if item["case_id"] == "case-invalid-type"
    )
    assert indexed["attack_type"] is None
    assert invalid_search["ok"] is False
    assert "Tipo de ataque no soportado" in invalid_search["error"]


def test_case_memory_enforces_foreign_keys_and_rejects_orphan_writes(case_db):
    from src.mcp import case_memory_server

    with case_memory_server._connection() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    orphan_trace = client.call(
        "case_memory",
        "append_trace",
        case_id="case-missing",
        entry={"agent": "detector", "status": "error"},
    )
    assert orphan_trace["ok"] is False
    assert "Caso no encontrado: case-missing" in orphan_trace["error"]

    orphan_update = client.call(
        "case_memory",
        "update_case",
        case_id="case-missing",
        payload={"status": "completed"},
    )
    assert orphan_update["ok"] is False
    assert "Caso no encontrado: case-missing" in orphan_update["error"]


def test_case_memory_migrates_legacy_index_to_exact_attack_type(
    tmp_path, monkeypatch
):
    database = tmp_path / "legacy-cases.db"
    payload = {
        "case_id": "case-legacy",
        "status": "completed",
        # ``attack_subtype`` era el nombre publico anterior del mismo tipo
        # exacto; no se usa la antigua familia para completar este valor.
        "classification": {
            "attack_family": "ddos",
            "attack_subtype": "DDoS_TCP",
            "confidence": 0.95,
            "family_confidence": 0.95,
            "decision_threshold": 0.65,
            "family_scores": {"ddos": 0.95},
            "model_task": "attack_subtype",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
            "top_scores": {
                "DDoS_TCP": 0.95,
                "DDoS_UDP": 0.03,
                "DDoS_HTTP": 0.02,
            },
        },
        "detection": {"is_malicious": True, "probability": 0.9},
        "explanation": {
            "attack_family": "ddos",
            "attack_subtype": "DDoS_TCP",
            "catalog_scope": "attack_type",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
            "catalog_version": "test-v1",
            "catalog_taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
            "catalog_compatible_taxonomy_versions": [
                MULTIDATASET_TAXONOMY_VERSION
            ],
        },
        "judge": {
            "action": "approve",
            "approved": True,
            "final_label": "DDoS_TCP",
            "final_confidence": 0.95,
        },
        "canonical_event": {"schema_profile": "network_flow"},
    }
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.executescript(
                """
                CREATE TABLE cases (
                    case_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    attack_family TEXT,
                    schema_profile TEXT,
                    payload TEXT NOT NULL
                );
                CREATE TABLE traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    seq INTEGER NOT NULL,
                    entry TEXT NOT NULL
                );
                CREATE INDEX idx_cases_family ON cases(attack_family);
                """
            )
            connection.execute(
                "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "case-legacy",
                    "2026-09-07T00:00:00+00:00",
                    "2026-09-07T00:00:00+00:00",
                    "completed",
                    "ddos",
                    "network_flow",
                    json.dumps(payload),
                ),
            )
    monkeypatch.setenv("TFM_CASE_MEMORY_DB", str(database))

    similar = client.call(
        "case_memory", "retrieve_similar_cases", attack_type="DDoS_TCP"
    )

    assert similar["ok"] is True
    assert similar["count"] == 1
    assert similar["cases"][0]["attack_type"] == "DDoS_TCP"

    fetched = client.call(
        "case_memory", "get_case", case_id="case-legacy", include_trace=False
    )
    assert fetched["ok"] is True
    migrated_payload = fetched["case"]
    assert migrated_payload["classification"]["attack_type"] == "DDoS_TCP"
    assert migrated_payload["classification"]["model_task"] == "attack_type"
    assert migrated_payload["explanation"]["attack_type"] == "DDoS_TCP"
    for legacy_key in (
        "attack_family",
        "attack_subtype",
        "family_confidence",
        "family_scores",
    ):
        assert legacy_key not in migrated_payload["classification"]
    assert "attack_family" not in migrated_payload["explanation"]
    assert "attack_subtype" not in migrated_payload["explanation"]

    # La ruta de lectura completa ya satisface el contrato publico nuevo.
    from src.contracts.case import CaseResult
    from src.agents.final.auditor import CaseAuditor

    migrated_case = CaseResult.model_validate(migrated_payload)
    assert migrated_case.classification.attack_type == "DDoS_TCP"
    assert migrated_case.explanation.attack_type == "DDoS_TCP"
    migration_report = CaseAuditor().audit(migrated_case)
    assert migration_report.case_id == "case-legacy"
    assert all(
        check.passed
        for check in migration_report.checks
        if check.check
        in {
            "consistencia_modelo_por_tipo",
            "consistencia_tipo_ataque_presente",
            "consistencia_tipo_ataque_en_taxonomia",
            "consistencia_top_scores_tipos",
            "consistencia_tipo_ataque_vs_top_scores",
        }
    )
    with closing(sqlite3.connect(database)) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(cases)")
        }
        indexes = {
            row[1] for row in connection.execute("PRAGMA index_list(cases)")
        }
    assert "attack_type" in columns
    assert "idx_cases_attack_type" in indexes
    assert "idx_cases_family" not in indexes
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1

    # Las tools deben haber cerrado sus conexiones: Windows permite borrar el
    # fichero inmediatamente despues de la consulta.
    database.unlink()


def test_case_memory_migration_does_not_promote_broad_family_to_attack_type(
    tmp_path, monkeypatch
):
    database = tmp_path / "legacy-family-only.db"
    payload = {
        "case_id": "case-family-only",
        "status": "completed",
        "detection": {"is_malicious": True, "probability": 0.9},
        "classification": {
            "attack_family": "ddos",
            "attack_subtype": None,
            "model_task": "attack_family",
        },
        "explanation": {
            "attack_family": "ddos",
            "catalog_scope": "family",
        },
    }
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.executescript(
                """
                CREATE TABLE cases (
                    case_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    attack_family TEXT,
                    attack_type TEXT,
                    schema_profile TEXT,
                    payload TEXT NOT NULL
                );
                CREATE TABLE traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    seq INTEGER NOT NULL,
                    entry TEXT NOT NULL
                );
                CREATE INDEX idx_cases_family ON cases(attack_family);
                """
            )
            connection.execute(
                "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "case-family-only",
                    "2026-09-07T00:00:00+00:00",
                    "2026-09-07T00:00:00+00:00",
                    "completed",
                    "ddos",
                    None,
                    "network_flow",
                    json.dumps(payload),
                ),
            )
    monkeypatch.setenv("TFM_CASE_MEMORY_DB", str(database))

    # Dos lecturas demuestran que la migracion es idempotente.
    first = client.call(
        "case_memory", "get_case", case_id="case-family-only", include_trace=False
    )
    second = client.call(
        "case_memory", "get_case", case_id="case-family-only", include_trace=False
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert first["case"] == second["case"]
    classification = first["case"]["classification"]
    assert classification.get("attack_type") is None
    assert "attack_family" not in classification
    assert "attack_subtype" not in classification
    assert classification["model_task"] == "attack_type"
    assert first["case"]["explanation"]["catalog_scope"] is None

    from src.agents.final.auditor import CaseAuditor
    from src.contracts.case import CaseResult

    report = CaseAuditor().audit(CaseResult.model_validate(first["case"]))
    assert report.verdict == "reject"
    assert any(
        check.check == "consistencia_malicioso_con_tipo_ataque"
        and not check.passed
        for check in report.checks
    )


def test_legacy_unknown_subtype_is_not_added_to_attack_type_index():
    from src.mcp.case_memory_server import _normalise_legacy_payload

    payload, indexed_attack_type, changed = _normalise_legacy_payload(
        {
            "classification": {
                "attack_subtype": "invented_attack",
                "model_task": "attack_subtype",
            }
        }
    )

    assert changed is True
    assert payload["classification"]["attack_type"] == "invented_attack"
    assert payload["classification"]["model_task"] == "attack_type"
    assert indexed_attack_type is None


class FakeLLMParser:
    """Doble hermetico del parser Mistral; nunca usa red ni adaptadores."""

    def __init__(
        self,
        event: dict | None = None,
        error: Exception | None = None,
        selected_columns: list[str] | None = None,
    ):
        self.event = dict(event or CANONICAL_EVENT)
        self.error = error
        self.agent = SimpleNamespace(model="mistral-small-2603")
        self.last_column_selection = {
            "selected_columns": list(selected_columns or ["proto"])
        }
        self.received: dict | None = None

    async def parse(self, raw_input: dict) -> CanonicalEvent:
        self.received = raw_input
        if self.error is not None:
            raise self.error
        return CanonicalEvent(**self.event)


def test_standardize_event_forwards_input_without_runtime_sanitization(monkeypatch):
    import src.mcp.inference_server as inference_server

    parser = FakeLLMParser(selected_columns=["proto", "label"])
    monkeypatch.setattr(inference_server, "_build_llm_parser", lambda: parser)

    result = inference_server.standardize_event(
        dataset="iot23",
        row={"proto": "tcp", "label": "Mirai"},
        source_file="test.log",
        row_id=999999,
        cache_mode="bypass",
    )

    assert result["ok"] is True
    assert parser.received is not None
    assert parser.received["row"] == {"proto": "tcp", "label": "Mirai"}
    assert parser.received["use_column_selection"] is True
    assert result["from_cache"] is False
    assert result["model"] == "mistral-small-2603"
    assert result["selected_columns"] == ["proto", "label"]
    assert result["notes"] == ["parsed_by_llm", "source:mistral_live"]


def test_standardize_event_llm_failure_is_controlled_abstention(monkeypatch):
    import src.mcp.inference_server as inference_server

    parser = FakeLLMParser(error=RuntimeError("Mistral no disponible"))
    monkeypatch.setattr(inference_server, "_build_llm_parser", lambda: parser)

    result = inference_server.standardize_event(
        dataset="iot23",
        row={"proto": "tcp"},
        source_file="test.log",
        row_id=7,
        cache_mode="bypass",
    )

    assert result["ok"] is False
    assert result["abstain"] is True
    assert result["requires_human_review"] is True
    assert result["failure_code"] == "llm_standardization_failed"
    assert "Mistral no disponible" in result["error"]
    assert "canonical_event" not in result


def test_standardize_event_does_not_filter_target_like_input(monkeypatch):
    import src.mcp.inference_server as inference_server

    parser = FakeLLMParser(selected_columns=["risk", "family_hint"])
    monkeypatch.setattr(inference_server, "_build_llm_parser", lambda: parser)

    result = inference_server.standardize_event(
        dataset="iot23",
        row={"risk": "Mirai", "family_hint": "DDoS"},
        cache_mode="bypass",
    )

    assert result["ok"] is True
    assert parser.received["row"] == {"risk": "Mirai", "family_hint": "DDoS"}


def test_standardize_event_rejects_dirty_llm_output_instead_of_cleaning(monkeypatch):
    import src.mcp.inference_server as inference_server

    leaky = {
        **CANONICAL_EVENT,
        "label_raw": "Mirai",
        "telemetry": {"attack_type": "malware", "temperature": 21.5},
    }
    monkeypatch.setattr(
        inference_server,
        "_build_llm_parser",
        lambda: FakeLLMParser(leaky),
    )

    result = inference_server.standardize_event(
        dataset="iot23",
        row={"proto": "tcp"},
        cache_mode="bypass",
    )

    assert result["ok"] is False
    assert result["abstain"] is True
    assert result["requires_human_review"] is True
    assert "campo predictivo no permitido" in result["error"]


@pytest.mark.parametrize(
    "dirty_fields,expected_error",
    [
        (
            {"semantic_text": "tcp flow label=Mirai"},
            "patron predictivo no permitido",
        ),
        (
            {"telemetry": {"risk": "Mirai", "temperature": 21.5}},
            "valor predictivo no permitido",
        ),
    ],
)
def test_standardize_event_rejects_dirty_llm_values_without_rewriting(
    monkeypatch,
    dirty_fields,
    expected_error,
):
    import src.mcp.inference_server as inference_server

    dirty_event = {**CANONICAL_EVENT, **dirty_fields}
    monkeypatch.setattr(
        inference_server,
        "_build_llm_parser",
        lambda: FakeLLMParser(dirty_event),
    )

    result = inference_server.standardize_event(
        dataset="iot23",
        row={"proto": "tcp"},
        cache_mode="bypass",
    )

    assert result["ok"] is False
    assert result["abstain"] is True
    assert expected_error in result["error"]
    assert "canonical_event" not in result


@pytest.mark.asyncio
async def test_standardize_event_can_run_inside_an_active_event_loop(monkeypatch):
    import src.mcp.inference_server as inference_server

    parser = FakeLLMParser()
    monkeypatch.setattr(inference_server, "_build_llm_parser", lambda: parser)

    result = inference_server.standardize_event(
        dataset="iot23",
        row={"proto": "tcp"},
        cache_mode="bypass",
    )

    assert result["ok"] is True
    assert result["source"] == "llm"


def test_standardize_event_reuses_mistral_cache_for_exact_duplicate_and_rebinds_identity(
    monkeypatch,
    tmp_path,
):
    import src.mcp.inference_server as inference_server

    monkeypatch.setenv(
        "TFM_STANDARDIZATION_CACHE_DB",
        str(tmp_path / "mistral_standardization.sqlite3"),
    )
    inference_server._CACHE_INSTANCES.clear()
    parser = FakeLLMParser(selected_columns=["proto"])
    builds = 0

    def build_parser():
        nonlocal builds
        builds += 1
        return parser

    monkeypatch.setattr(inference_server, "_build_llm_parser", build_parser)

    first = inference_server.standardize_event(
        dataset="iot23",
        row={"z_metric": 2, "proto": "tcp"},
        source_file="first.csv",
        row_id=1,
        split="train",
    )
    second = inference_server.standardize_event(
        dataset="edge_iiotset",
        row={"proto": "tcp", "z_metric": 2},
        source_file="duplicate.csv",
        row_id=99,
        split="test",
    )

    assert first["ok"] is True and first["from_cache"] is False
    assert second["ok"] is True and second["from_cache"] is True
    assert builds == 1
    assert list(parser.received["row"]) == ["proto", "z_metric"]
    assert first["canonical_event"]["event_id"] == "llm::iot23::first.csv::1"
    assert first["canonical_event"]["origin"]["row_id"] == 1
    assert first["canonical_event"]["provenance"]["dataset"] == "iot23"
    assert first["canonical_event"]["provenance"]["split"] == "train"
    assert second["canonical_event"]["event_id"] == (
        "llm::edge_iiotset::duplicate.csv::99"
    )
    assert second["canonical_event"]["origin"]["row_id"] == 99
    assert second["canonical_event"]["provenance"]["dataset"] == "edge_iiotset"
    assert second["canonical_event"]["provenance"]["split"] == "test"
    assert second["cache_content_hash"] == first["cache_content_hash"]
    assert second["notes"] == ["parsed_by_llm", "source:mistral_cache"]


def test_invalid_blank_semantic_output_is_not_cached_for_later_duplicates(
    monkeypatch,
    tmp_path,
):
    import src.mcp.inference_server as inference_server

    monkeypatch.setenv(
        "TFM_STANDARDIZATION_CACHE_DB",
        str(tmp_path / "mistral_standardization.sqlite3"),
    )
    inference_server._CACHE_INSTANCES.clear()
    invalid_event = {**CANONICAL_EVENT, "semantic_text": "   "}
    parsers = iter(
        [
            FakeLLMParser(invalid_event, selected_columns=["proto"]),
            FakeLLMParser(selected_columns=["proto"]),
        ]
    )
    builds = 0

    def build_parser():
        nonlocal builds
        builds += 1
        return next(parsers)

    monkeypatch.setattr(inference_server, "_build_llm_parser", build_parser)

    first = inference_server.standardize_event(
        dataset="first_dataset",
        row={"proto": "tcp"},
        source_file="first.csv",
        row_id=1,
    )
    second = inference_server.standardize_event(
        dataset="second_dataset",
        row={"proto": "tcp"},
        source_file="second.csv",
        row_id=2,
    )

    assert first["ok"] is False
    assert second["ok"] is True
    assert second["from_cache"] is False
    assert builds == 2
    assert second["canonical_event"]["event_id"] == (
        "llm::second_dataset::second.csv::2"
    )


def test_standardize_event_rejects_dirty_prestandardized_without_llm(monkeypatch):
    import src.mcp.inference_server as inference_server

    def unexpected_llm():
        raise AssertionError("un CanonicalEvent no debe volver a enviarse a Mistral")

    monkeypatch.setattr(inference_server, "_build_llm_parser", unexpected_llm)
    leaky = {
        **CANONICAL_EVENT,
        "label_raw": "DDoS_UDP",
        "telemetry": {"attack_type": "ddos", "temperature": 20.0},
    }

    result = inference_server.standardize_event(
        dataset="edge_iiotset",
        canonical_event=leaky,
    )

    assert result["ok"] is False
    assert result["abstain"] is True
    assert result["failure_code"] == "canonical_event_invalid"
    assert "canonical_event" not in result


def test_standardize_event_rejects_unknown_provenance_field_without_rewriting(
    monkeypatch,
):
    import src.mcp.inference_server as inference_server

    monkeypatch.setattr(
        inference_server,
        "_build_llm_parser",
        lambda: pytest.fail("un CanonicalEvent no debe volver a Mistral"),
    )
    provenance = dict(CANONICAL_EVENT["provenance"])
    provenance["label"] = "Mirai"

    result = inference_server.standardize_event(
        dataset="edge_iiotset",
        canonical_event={**CANONICAL_EVENT, "provenance": provenance},
    )

    assert result["ok"] is False
    assert result["failure_code"] == "canonical_event_invalid"
    assert "extra_forbidden" in result["error"]


def test_standardize_event_rejects_ambiguous_raw_and_canonical_input(monkeypatch):
    import src.mcp.inference_server as inference_server

    monkeypatch.setattr(
        inference_server,
        "_build_llm_parser",
        lambda: pytest.fail("una entrada ambigua no debe invocar Mistral"),
    )

    result = inference_server.standardize_event(
        dataset="iot23",
        row={"proto": "tcp"},
        canonical_event=dict(CANONICAL_EVENT),
    )

    assert result["ok"] is False
    assert result["abstain"] is True
    assert result["requires_human_review"] is True
    assert result["failure_code"] == "standardization_input_ambiguous"


def test_standardize_event_rejects_ambiguous_row_and_text(monkeypatch):
    import src.mcp.inference_server as inference_server

    monkeypatch.setattr(
        inference_server,
        "_build_llm_parser",
        lambda: pytest.fail("una entrada ambigua no debe invocar Mistral"),
    )

    result = inference_server.standardize_event(
        dataset="iot23",
        row={"proto": "tcp"},
        text="proto=tcp",
    )

    assert result["ok"] is False
    assert result["abstain"] is True
    assert result["failure_code"] == "standardization_input_ambiguous"


def test_standardize_event_rejects_unsupported_read_only_cache_mode(monkeypatch):
    import src.mcp.inference_server as inference_server

    monkeypatch.setattr(
        inference_server,
        "_build_llm_parser",
        lambda: pytest.fail("un modo invalido no debe invocar Mistral"),
    )

    result = inference_server.standardize_event(
        dataset="iot23",
        row={"proto": "tcp"},
        cache_mode="read_only",
    )

    assert result["ok"] is False
    assert result["failure_code"] == "standardization_input_invalid"


def test_predictive_tool_rejects_dirty_canonical_event_without_sanitizing():
    import src.mcp.inference_server as inference_server

    result = inference_server.detect_event(
        {**CANONICAL_EVENT, "label_raw": "Mirai"}
    )

    assert result["ok"] is False
    assert "campo predictivo no permitido" in result["error"]
    assert "canonical_event" not in result


def _models_loadable() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    from src.mcp.common import resolve_path

    return resolve_path("detection_model").exists() and resolve_path("attack_type_model").exists()


@pytest.mark.skipif(not _models_loadable(), reason="xgboost o modelos .joblib no disponibles")
def test_detect_and_classify_with_prepared_models():
    detection = client.call("inference", "detect_event", canonical_event=CANONICAL_EVENT)
    assert detection["ok"], detection.get("error")
    assert isinstance(detection["is_malicious"], bool)
    assert 0.0 <= detection["probability"] <= 1.0

    classification = client.call("inference", "classify_event", canonical_event=CANONICAL_EVENT)
    assert classification["ok"], classification.get("error")
    assert classification["model_task"] == "attack_type"
    assert classification["taxonomy_version"] == MULTIDATASET_TAXONOMY_VERSION
    assert classification["attack_type"] in MULTIDATASET_ATTACK_CLASSES
    assert not {
        "attack_family",
        "attack_subtype",
        "family_confidence",
        "family_scores",
    } & classification.keys()
    assert 0.0 <= classification["confidence"] <= 1.0
    assert classification["decision_threshold"] == pytest.approx(0.65)
    assert len(classification["top_scores"]) == 3


# ---------------------------------------------------------------------------
# protocolo MCP real (stdio)
# ---------------------------------------------------------------------------

def test_stdio_is_the_runtime_default_and_inprocess_requires_explicit_selection(
    monkeypatch,
):
    monkeypatch.delenv(CLIENT_MODE_ENV, raising=False)

    assert DEFAULT_CLIENT_MODE == "stdio"
    assert resolve_mcp_client_mode() == "stdio"
    assert MCPToolClient().mode == "stdio"

    monkeypatch.setenv(CLIENT_MODE_ENV, " InProcess ")
    assert resolve_mcp_client_mode() == "inprocess"
    assert MCPToolClient().mode == "inprocess"
    assert MCPToolClient(mode="stdio").mode == "stdio"

    monkeypatch.setenv(CLIENT_MODE_ENV, "socket-magico")
    with pytest.raises(ValueError, match="Modo MCP no soportado"):
        MCPToolClient()

def _stdio_result(text: str | None, *, is_error: bool = False):
    content = [] if text is None else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(isError=is_error, content=content)


def test_stdio_decoder_is_fail_closed_for_protocol_error_even_with_ok_payload():
    result = _decode_stdio_result(
        _stdio_result('{"ok": true, "value": 1}', is_error=True)
    )
    assert result["ok"] is False
    assert "error" in result


def test_stdio_decoder_is_fail_closed_for_non_json_or_non_object_content():
    non_json = _decode_stdio_result(_stdio_result("fallo textual del servidor"))
    assert non_json["ok"] is False
    assert "no JSON" in non_json["error"]

    non_object = _decode_stdio_result(_stdio_result('[{"ok": true}]'))
    assert non_object["ok"] is False
    assert "no es un objeto" in non_object["error"]


def test_stdio_decoder_requires_boolean_ok_and_accepts_valid_tool_payload():
    missing_ok = _decode_stdio_result(_stdio_result('{"value": 1}'))
    assert missing_ok["ok"] is False
    assert "campo booleano" in missing_ok["error"]

    valid = _decode_stdio_result(_stdio_result('{"ok": true, "value": 1}'))
    assert valid == {"ok": True, "value": 1}


def test_stdio_timeout_is_finite_configurable_and_validated(monkeypatch):
    monkeypatch.delenv("MCP_STDIO_TIMEOUT_SECONDS", raising=False)
    default_client = MCPToolClient(mode="stdio")
    assert default_client.timeout_seconds == DEFAULT_STDIO_TIMEOUT_SECONDS
    assert math.isfinite(default_client.timeout_seconds)

    monkeypatch.setenv("MCP_STDIO_TIMEOUT_SECONDS", "7.5")
    assert MCPToolClient(mode="stdio").timeout_seconds == 7.5
    assert MCPToolClient(mode="stdio", timeout_seconds=3).timeout_seconds == 3.0

    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Timeout MCP stdio invalido"):
            MCPToolClient(mode="stdio", timeout_seconds=invalid)


@pytest.mark.asyncio
async def test_stdio_timeout_is_applied_to_session_and_tool_call(monkeypatch):
    observed: dict[str, object] = {}
    monkeypatch.setenv("TFM_CASE_MEMORY_DB", "ruta-heredada.db")

    class FakeServerParameters:
        def __init__(self, **kwargs):
            observed["params"] = kwargs

    class FakeSession:
        def __init__(self, read, write, read_timeout_seconds=None):
            observed["read_timeout"] = read_timeout_seconds

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def initialize(self):
            return None

        async def call_tool(self, tool, arguments):
            await asyncio.sleep(10)

    @asynccontextmanager
    async def fake_stdio_client(params):
        yield object(), object()

    fake_mcp = ModuleType("mcp")
    fake_mcp.ClientSession = FakeSession
    fake_mcp.StdioServerParameters = FakeServerParameters
    fake_mcp_client = ModuleType("mcp.client")
    fake_mcp_stdio = ModuleType("mcp.client.stdio")
    fake_mcp_stdio.stdio_client = fake_stdio_client
    fake_mcp.client = fake_mcp_client
    fake_mcp_client.stdio = fake_mcp_stdio
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", fake_mcp_client)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", fake_mcp_stdio)

    stdio_client_instance = MCPToolClient(mode="stdio", timeout_seconds=0.01)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await stdio_client_instance._call_stdio_async(
                "threat_intel", "list_attack_types", {}
            )
    finally:
        stdio_client_instance.close()

    assert observed["read_timeout"] == timedelta(seconds=0.01)
    assert observed["params"]["env"]["TFM_CASE_MEMORY_DB"] == "ruta-heredada.db"


def test_stdio_client_reuses_one_session_and_serializes_threaded_calls(monkeypatch):
    observed = {
        "transport_enter": 0,
        "transport_exit": 0,
        "session_enter": 0,
        "session_exit": 0,
        "initialize": 0,
        "calls": 0,
        "lists": 0,
        "in_flight": 0,
        "max_in_flight": 0,
        "owner_tasks": [],
    }

    class FakeServerParameters:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSession:
        def __init__(self, read, write, read_timeout_seconds=None):
            self.read_timeout_seconds = read_timeout_seconds

        async def __aenter__(self):
            observed["session_enter"] += 1
            observed["owner_tasks"].append(asyncio.current_task())
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            observed["session_exit"] += 1
            observed["owner_tasks"].append(asyncio.current_task())
            return False

        async def initialize(self):
            observed["initialize"] += 1

        async def call_tool(self, tool, arguments):
            observed["calls"] += 1
            observed["in_flight"] += 1
            observed["max_in_flight"] = max(
                observed["max_in_flight"], observed["in_flight"]
            )
            await asyncio.sleep(0.005)
            observed["in_flight"] -= 1
            return _stdio_result(
                '{"ok": true, "value": ' + str(arguments["value"]) + "}"
            )

        async def list_tools(self):
            observed["lists"] += 1
            return SimpleNamespace(tools=[SimpleNamespace(name="echo")])

    @asynccontextmanager
    async def fake_stdio_client(params):
        observed["transport_enter"] += 1
        observed["owner_tasks"].append(asyncio.current_task())
        try:
            yield object(), object()
        finally:
            observed["transport_exit"] += 1
            observed["owner_tasks"].append(asyncio.current_task())

    fake_mcp = ModuleType("mcp")
    fake_mcp.ClientSession = FakeSession
    fake_mcp.StdioServerParameters = FakeServerParameters
    fake_mcp_client = ModuleType("mcp.client")
    fake_mcp_stdio = ModuleType("mcp.client.stdio")
    fake_mcp_stdio.stdio_client = fake_stdio_client
    fake_mcp.client = fake_mcp_client
    fake_mcp_client.stdio = fake_mcp_stdio
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", fake_mcp_client)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", fake_mcp_stdio)

    with MCPToolClient(mode="stdio", timeout_seconds=1) as stdio_client_instance:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    stdio_client_instance.call,
                    "threat_intel",
                    "echo",
                    value=value,
                )
                for value in range(8)
            ]
            assert sorted(future.result()["value"] for future in futures) == list(
                range(8)
            )

        assert stdio_client_instance.list_tools("threat_intel") == ["echo"]

        assert observed["transport_enter"] == 1
        assert observed["session_enter"] == 1
        assert observed["initialize"] == 1
        assert observed["calls"] == 8
        assert observed["lists"] == 1
        assert observed["max_in_flight"] == 1

    assert observed["session_exit"] == 1
    assert observed["transport_exit"] == 1
    assert len({id(task) for task in observed["owner_tasks"]}) == 1
    assert stdio_client_instance.closed is True
    with pytest.raises(RuntimeError, match="cliente MCP esta cerrado"):
        stdio_client_instance.call("threat_intel", "echo", value=9)


def test_stdio_client_close_is_idempotent_before_starting_any_server():
    stdio_client_instance = MCPToolClient(mode="stdio")

    stdio_client_instance.close()
    stdio_client_instance.close()

    assert stdio_client_instance.closed is True
    with pytest.raises(RuntimeError, match="cliente MCP esta cerrado"):
        stdio_client_instance.list_tools("threat_intel")


def test_stdio_client_retains_a_worker_when_close_must_be_retried():
    class FlakyWorker:
        close_calls = 0

        def close(self, *, join_timeout=None):
            del join_timeout
            self.close_calls += 1
            if self.close_calls == 1:
                raise TimeoutError("cierre simulado")

    stdio_client_instance = MCPToolClient(mode="stdio", timeout_seconds=1)
    worker = FlakyWorker()
    stdio_client_instance._stdio_workers["threat_intel"] = worker

    with pytest.raises(RuntimeError, match="cerrar todas las sesiones"):
        stdio_client_instance.close()
    assert stdio_client_instance.closed is True
    assert stdio_client_instance._stdio_workers["threat_intel"] is worker

    stdio_client_instance.close()
    assert worker.close_calls == 2
    assert stdio_client_instance._stdio_workers == {}

@pytest.mark.asyncio
async def test_stdio_threat_intel_lists_tools_over_real_mcp():
    pytest.importorskip("mcp.server.fastmcp", reason="SDK MCP con FastMCP no instalado")
    tools = await list_tools_stdio("threat_intel")
    assert "suggest_mitigations" in tools


def test_stdio_client_lists_tools_through_real_mcp():
    with MCPToolClient(mode="stdio") as stdio_client_instance:
        tools = stdio_client_instance.list_tools("threat_intel")
    assert "suggest_mitigations" in tools


def test_stdio_inference_call_uses_real_mcp():
    pytest.importorskip("mcp.server.fastmcp", reason="SDK MCP con FastMCP no instalado")
    with MCPToolClient(mode="stdio") as stdio_client_instance:
        result = stdio_client_instance.call(
            "inference",
            "get_mapping_confidence",
            canonical_event={"mapping_confidence": 0.87},
        )

    assert result["ok"] is True
    assert result["mapping_confidence"] == pytest.approx(0.87)


def test_stdio_case_memory_roundtrip_uses_real_mcp(tmp_path, monkeypatch):
    pytest.importorskip("mcp.server.fastmcp", reason="SDK MCP con FastMCP no instalado")
    database = tmp_path / "stdio-case-memory.db"
    monkeypatch.setenv("TFM_CASE_MEMORY_DB", str(database))
    with MCPToolClient(mode="stdio") as stdio_client_instance:
        created = stdio_client_instance.call(
            "case_memory",
            "create_case",
            case_id="case-stdio-roundtrip",
            payload={"marker": "mcp-real"},
        )
        stored = stdio_client_instance.call(
            "case_memory",
            "get_case",
            case_id="case-stdio-roundtrip",
            include_trace=False,
        )

    assert created["ok"] is True
    assert stored["ok"] is True
    assert stored["case"] == {"marker": "mcp-real"}


def test_stdio_call_returns_same_payload_as_inprocess():
    pytest.importorskip("mcp.server.fastmcp", reason="SDK MCP con FastMCP no instalado")
    with MCPToolClient(mode="stdio") as stdio_client_instance:
        remote = stdio_client_instance.call(
            "threat_intel",
            "map_attack_type_to_capec",
            attack_type="SQL_injection",
        )
    local = client.call(
        "threat_intel",
        "map_attack_type_to_capec",
        attack_type="SQL_injection",
    )
    assert remote["ok"] is True, remote.get("error")
    assert remote["capec_patterns"] == local["capec_patterns"]


def test_stdio_protocol_error_is_fail_closed():
    pytest.importorskip("mcp.server.fastmcp", reason="SDK MCP con FastMCP no instalado")
    with MCPToolClient(mode="stdio") as stdio_client_instance:
        result = stdio_client_instance.call(
            "threat_intel", "tool_que_no_existe"
        )
    assert result["ok"] is False
    assert "Unknown tool" in result["error"]
