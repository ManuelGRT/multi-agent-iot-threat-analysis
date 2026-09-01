# tests/test_mcp_servers.py
from __future__ import annotations

import asyncio
import math
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.contracts.canonical import CanonicalEvent
from src.mcp.client import (
    DEFAULT_STDIO_TIMEOUT_SECONDS,
    MCPToolClient,
    SERVER_MODULES,
    _decode_stdio_result,
    list_tools_stdio,
)

client = MCPToolClient(mode="inprocess")

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
        "inference": {"standardize_event", "detect_event", "classify_event", "get_family_scores"},
        "case_memory": {"create_case", "append_trace", "get_case", "retrieve_similar_cases"},
        "threat_intel": {"map_family_to_attack", "map_family_to_capec", "suggest_mitigations"},
    }
    assert set(SERVER_MODULES) == set(expected)
    for server, tools in expected.items():
        available = set(client.list_tools(server))
        assert tools <= available, f"{server}: faltan tools {tools - available}"


def test_tool_results_carry_trace_metadata():
    result = client.call("threat_intel", "list_families")
    assert result["ok"] is True
    assert result["tool_name"] == "list_families"
    assert "tool_version" in result and "latency_ms" in result


def test_tool_errors_are_returned_not_raised(case_db):
    result = client.call("case_memory", "get_case", case_id="no_existe")
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# threat intel
# ---------------------------------------------------------------------------

def test_threat_intel_covers_all_corpus_families():
    corpus_families = ["benign", "injection", "ddos", "malware", "bruteforce", "scanning", "mitm"]
    listed = client.call("threat_intel", "list_families")["families"]
    for family in corpus_families:
        assert family in listed


def test_threat_intel_mappings_ddos():
    attack = client.call("threat_intel", "map_family_to_attack", family="ddos")
    assert any(t["id"] == "T1498" for t in attack["attack_techniques"])
    capec = client.call("threat_intel", "map_family_to_capec", family="ddos")
    assert any(p["id"] == "CAPEC-125" for p in capec["capec_patterns"])


def test_threat_intel_suggest_mitigations_with_profile():
    result = client.call(
        "threat_intel", "suggest_mitigations", family="bruteforce", schema_profile="network_flow"
    )
    assert result["family"] == "bruteforce"
    assert result["mitigations_ordered"], "debe haber mitigaciones"
    assert result["profile_actions"], "debe haber acciones por perfil"
    assert result["references"]["attack_techniques"], "debe haber referencias ATT&CK"


def test_threat_intel_alias_and_unknown():
    assert client.call("threat_intel", "map_family_to_attack", family="Mirai")["family"] == "botnet"
    assert (
        client.call("threat_intel", "suggest_mitigations", family="algo_rarisimo")["family"]
        == "unknown_attack"
    )


def test_threat_intel_benign_has_no_offensive_mitigations():
    result = client.call("threat_intel", "suggest_mitigations", family="benign")
    assert result["mitigations_by_phase"]["containment"] == []
    assert result["mitigations_by_phase"]["eradication"] == []


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
        "classification": {"attack_family": "ddos"},
        "canonical_event": {"schema_profile": "network_flow"},
    }
    updated = client.call("case_memory", "update_case", case_id="case-test-1", payload=payload)
    assert updated["ok"]

    fetched = client.call("case_memory", "get_case", case_id="case-test-1")
    assert fetched["ok"]
    assert fetched["status"] == "completed"
    assert len(fetched["trace"]) == 2
    assert fetched["trace"][0]["agent"] == "detector"

    similar = client.call("case_memory", "retrieve_similar_cases", attack_family="ddos")
    assert similar["count"] == 1


def test_case_memory_enforces_foreign_keys_and_rejects_orphan_writes(case_db):
    from src.mcp import case_memory_server

    with case_memory_server._connect() as connection:
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

    return resolve_path("detection_model").exists() and resolve_path("family_model").exists()


@pytest.mark.skipif(not _models_loadable(), reason="xgboost o modelos .joblib no disponibles")
def test_detect_and_classify_with_prepared_models():
    detection = client.call("inference", "detect_event", canonical_event=CANONICAL_EVENT)
    assert detection["ok"], detection.get("error")
    assert isinstance(detection["is_malicious"], bool)
    assert 0.0 <= detection["probability"] <= 1.0

    classification = client.call("inference", "classify_event", canonical_event=CANONICAL_EVENT)
    assert classification["ok"], classification.get("error")
    assert classification["attack_family"]
    assert 0.0 <= classification["confidence"] <= 1.0
    assert len(classification["top_scores"]) >= 1


# ---------------------------------------------------------------------------
# paridad de features con el script de entrenamiento
# ---------------------------------------------------------------------------

def test_feature_parity_with_training_script():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "train_xgboost_detection_standardized_datasets.py"
    spec = importlib.util.spec_from_file_location("_train_det_script", script)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # dependencia de entrenamiento ausente
        pytest.skip(f"script de entrenamiento no importable aqui: {exc}")

    from src.mcp.features import event_features
    from src.eval.data_sanitization import sanitize_canonical_event

    parity_event = {
        **CANONICAL_EVENT,
        "behavior_tags": ["Mirai"],
        "uncertainty": ["DDoS"],
        "asset_context": "malware",
        "service_context": {
            "protocol_family": "icmp",
            "risk": "DDoS",
            "family_hint": "Mirai",
            "role": "botnet",
        },
        "telemetry": {"temperature": 21.5, "risk": "Mirai"},
        "host": {"cpu": 12.0, "family_hint": "DDoS"},
    }
    # La sanitizacion pertenece al preprocesamiento de entrenamiento/evaluacion,
    # no a ``event_features`` ni al runtime.
    prepared_event = sanitize_canonical_event(parity_event)
    ours = event_features(prepared_event)
    theirs = module.event_features(parity_event)
    assert ours == theirs


# ---------------------------------------------------------------------------
# protocolo MCP real (stdio)
# ---------------------------------------------------------------------------

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
    with pytest.raises(asyncio.TimeoutError):
        await stdio_client_instance._call_stdio_async(
            "threat_intel", "list_families", {}
        )

    assert observed["read_timeout"] == timedelta(seconds=0.01)

@pytest.mark.asyncio
async def test_stdio_threat_intel_lists_tools_over_real_mcp():
    pytest.importorskip("mcp.server.fastmcp", reason="SDK MCP con FastMCP no instalado")
    tools = await list_tools_stdio("threat_intel")
    assert "suggest_mitigations" in tools


def test_stdio_call_returns_same_payload_as_inprocess():
    pytest.importorskip("mcp.server.fastmcp", reason="SDK MCP con FastMCP no instalado")
    stdio_client_instance = MCPToolClient(mode="stdio")
    remote = stdio_client_instance.call(
        "threat_intel", "map_family_to_capec", family="injection"
    )
    local = client.call("threat_intel", "map_family_to_capec", family="injection")
    assert remote["ok"] is True, remote.get("error")
    assert remote["capec_patterns"] == local["capec_patterns"]


def test_stdio_protocol_error_is_fail_closed():
    pytest.importorskip("mcp.server.fastmcp", reason="SDK MCP con FastMCP no instalado")
    result = MCPToolClient(mode="stdio").call(
        "threat_intel", "tool_que_no_existe"
    )
    assert result["ok"] is False
    assert "Unknown tool" in result["error"]
