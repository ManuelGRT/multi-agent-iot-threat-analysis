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
        "datasets": {"list_datasets", "load_sample", "load_batch", "get_schema_profile"},
        "inference": {"standardize_event", "detect_event", "classify_event", "get_family_scores"},
        "case_memory": {"create_case", "append_trace", "get_case", "retrieve_similar_cases"},
        "threat_intel": {"map_family_to_attack", "map_family_to_capec", "suggest_mitigations"},
        "evaluation": {"score_run", "compare_with_baseline", "export_report"},
    }
    for server, tools in expected.items():
        available = set(client.list_tools(server))
        assert tools <= available, f"{server}: faltan tools {tools - available}"


def test_tool_results_carry_trace_metadata():
    result = client.call("threat_intel", "list_families")
    assert result["ok"] is True
    assert result["tool_name"] == "list_families"
    assert "tool_version" in result and "latency_ms" in result


def test_tool_errors_are_returned_not_raised():
    result = client.call("datasets", "load_sample", dataset="no_existe")
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


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def test_evaluation_score_run_multiclass():
    result = client.call(
        "evaluation",
        "score_run",
        y_true=["ddos", "ddos", "scanning", "benign"],
        y_pred=["ddos", "scanning", "scanning", "benign"],
        task="multiclass",
    )
    assert result["ok"]
    assert result["n_samples"] == 4
    assert 0.0 <= result["f1"] <= 1.0


def test_evaluation_compare_with_baseline():
    result = client.call("evaluation", "compare_with_baseline", f1=0.9363, task="multiclass")
    if not result["ok"]:
        pytest.skip("baselines no congelados en este entorno")
    assert result["beats_jorge"] is True
    assert result["meets_audit_threshold"] is True
    low = client.call("evaluation", "compare_with_baseline", f1=0.60, task="multiclass")
    assert low["beats_jorge"] is False
    assert low["meets_audit_threshold"] is False


def test_evaluation_export_report(tmp_path, monkeypatch):
    monkeypatch.setenv("TFM_ARTIFACTS_DIR", str(tmp_path))
    result = client.call(
        "evaluation",
        "export_report",
        title="Informe de prueba",
        sections=[{"heading": "Resumen", "body": {"f1": 0.93}}],
        filename="informe_test.md",
    )
    assert result["ok"]
    content = Path(result["path"]).read_text(encoding="utf-8")
    assert "# Informe de prueba" in content and "0.93" in content


@pytest.mark.parametrize("filename", ["../escape.md", "subdir/../../escape.md"])
def test_evaluation_export_report_rejects_traversal(tmp_path, monkeypatch, filename):
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("TFM_ARTIFACTS_DIR", str(artifacts))

    result = client.call(
        "evaluation",
        "export_report",
        title="No debe escribirse",
        sections=[],
        filename=filename,
    )

    assert result["ok"] is False
    assert not (tmp_path / "escape.md").exists()


def test_evaluation_export_report_rejects_absolute_path(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    outside = tmp_path / "absolute.md"
    monkeypatch.setenv("TFM_ARTIFACTS_DIR", str(artifacts))

    result = client.call(
        "evaluation",
        "export_report",
        title="No debe escribirse",
        sections=[],
        filename=str(outside),
    )

    assert result["ok"] is False
    assert not outside.exists()


def test_evaluation_export_report_allows_nested_relative_path(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("TFM_ARTIFACTS_DIR", str(artifacts))

    result = client.call(
        "evaluation",
        "export_report",
        title="Informe anidado",
        sections=[],
        filename="reports/informe.md",
    )

    assert result["ok"] is True
    assert Path(result["path"]) == (artifacts / "reports" / "informe.md").resolve()


def test_datasets_explicit_path_is_confined_to_data_root(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    inside = data_root / "iot23.csv"
    inside.write_text("proto,label\ntcp,Mirai\n", encoding="utf-8")
    outside = tmp_path / "outside.csv"
    outside.write_text("proto,label\nudp,DDoS\n", encoding="utf-8")
    monkeypatch.setenv("TFM_DATA_DIR", str(data_root))

    allowed = client.call(
        "datasets", "load_sample", dataset="iot23", n=1, path=str(inside)
    )
    absolute_escape = client.call(
        "datasets", "load_sample", dataset="iot23", n=1, path=str(outside)
    )
    traversal = client.call(
        "datasets", "load_sample", dataset="iot23", n=1, path="../outside.csv"
    )

    assert allowed["ok"] is True
    assert allowed["count"] == 1
    assert absolute_escape["ok"] is False
    assert traversal["ok"] is False


def test_datasets_default_source_supports_relative_configured_root(tmp_path, monkeypatch):
    data_root = tmp_path / "data" / "IOT23"
    data_root.mkdir(parents=True)
    (data_root / "sample.csv").write_text(
        "proto,label\ntcp,Mirai\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TFM_DATA_DIR", "data")

    result = client.call("datasets", "load_sample", dataset="iot23", n=1)

    assert result["ok"] is True
    assert result["count"] == 1


# ---------------------------------------------------------------------------
# datasets + inference (dependen de artefactos/modelos locales)
# ---------------------------------------------------------------------------

def _standardized_available() -> bool:
    from src.mcp.common import resolve_path

    return resolve_path("standardized_dataset").exists()


def test_datasets_list_always_answers():
    result = client.call("datasets", "list_datasets")
    assert result["ok"]
    names = {d["name"] for d in result["datasets"]}
    assert "standardized_corpus" in names


@pytest.mark.skipif(not _standardized_available(), reason="corpus estandarizado no disponible")
def test_datasets_load_standardized_sample():
    result = client.call("datasets", "load_sample", dataset="standardized_corpus", n=2)
    assert result["ok"] and result["count"] == 2
    row = result["rows"][0]
    assert "canonical_event" in row and "target" in row


class FakeLLMParser:
    """Doble hermetico del parser Mistral; nunca usa red ni adaptadores."""

    def __init__(self, event: dict | None = None, error: Exception | None = None):
        self.event = dict(event or CANONICAL_EVENT)
        self.error = error
        self.agent = SimpleNamespace(model="mistral-small-2603")
        self.last_column_selection = {
            "selected_columns": ["proto", "duration", "orig_pkts"]
        }
        self.received: dict | None = None

    async def parse(self, raw_input: dict) -> CanonicalEvent:
        self.received = raw_input
        if self.error is not None:
            raise self.error
        return CanonicalEvent(**self.event)


def test_standardize_event_always_uses_mistral_and_sanitizes(monkeypatch):
    import src.mcp.inference_server as inference_server

    leaky = {
        **CANONICAL_EVENT,
        "label_raw": "Mirai",
        "attack_family": "malware",
        "semantic_text": "tcp flow label=Mirai",
        "telemetry": {"attack_type": "malware", "temperature": 21.5},
    }
    parser = FakeLLMParser(leaky)
    monkeypatch.setattr(inference_server, "_build_llm_parser", lambda: parser)

    result = inference_server.standardize_event(
        dataset="iot23",
        row={"proto": "tcp", "label": "Mirai"},
        source_file="test.log",
        row_id=999999,
    )

    assert result["ok"] is True
    assert parser.received is not None
    assert "label" not in parser.received["row"]
    assert parser.received["use_column_selection"] is True
    assert result["from_cache"] is False
    assert result["model"] == "mistral-small-2603"
    assert result["selected_columns"] == ["proto", "duration", "orig_pkts"]
    assert result["canonical_event"]["label_raw"] is None
    assert result["canonical_event"]["attack_family"] is None
    assert "Mirai" not in result["canonical_event"]["semantic_text"]
    assert "attack_type" not in result["canonical_event"]["telemetry"]
    assert result["canonical_event"]["telemetry"]["temperature"] == 21.5


def test_standardize_event_llm_failure_is_controlled_abstention(monkeypatch):
    import src.mcp.inference_server as inference_server

    parser = FakeLLMParser(error=RuntimeError("Mistral no disponible"))
    monkeypatch.setattr(inference_server, "_build_llm_parser", lambda: parser)

    result = inference_server.standardize_event(
        dataset="iot23",
        row={"proto": "tcp"},
        source_file="test.log",
        row_id=7,
    )

    assert result["ok"] is False
    assert result["abstain"] is True
    assert result["requires_human_review"] is True
    assert result["failure_code"] == "llm_standardization_failed"
    assert "Mistral no disponible" in result["error"]
    assert "canonical_event" not in result


def test_standardize_event_abstains_if_guard_removes_all_input_fields(monkeypatch):
    import src.mcp.inference_server as inference_server

    monkeypatch.setattr(
        inference_server,
        "_build_llm_parser",
        lambda: pytest.fail("no debe llamar a Mistral sin evidencia tecnica"),
    )

    result = inference_server.standardize_event(
        dataset="iot23",
        row={"risk": "Mirai", "family_hint": "DDoS"},
    )

    assert result["ok"] is False
    assert result["abstain"] is True
    assert result["requires_human_review"] is True
    assert (
        result["failure_code"]
        == "standardization_input_without_technical_fields"
    )


@pytest.mark.asyncio
async def test_standardize_event_can_run_inside_an_active_event_loop(monkeypatch):
    import src.mcp.inference_server as inference_server

    parser = FakeLLMParser()
    monkeypatch.setattr(inference_server, "_build_llm_parser", lambda: parser)

    result = inference_server.standardize_event(
        dataset="iot23",
        row={"proto": "tcp"},
    )

    assert result["ok"] is True
    assert result["source"] == "llm"


def test_standardize_event_prestandardized_is_guarded_without_llm(monkeypatch):
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

    assert result["ok"] is True
    assert result["model"] == "prestandardized_passthrough"
    assert result["from_cache"] is False
    assert result["canonical_event"]["label_raw"] is None
    assert "attack_type" not in result["canonical_event"]["telemetry"]


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


@pytest.mark.skipif(not _models_loadable() or not _standardized_available(), reason="modelos o corpus no disponibles")
def test_detection_agrees_with_corpus_labels_on_sample():
    """Sanidad: sobre 20 filas del corpus, el detector acierta la mayoria."""
    rows = client.call("datasets", "load_sample", dataset="standardized_corpus", n=20)["rows"]
    hits = 0
    total = 0
    for row in rows:
        target = row.get("target") or {}
        if "is_attack" not in target:
            continue
        detection = client.call("inference", "detect_event", canonical_event=row["canonical_event"])
        if not detection["ok"]:
            continue
        total += 1
        if detection["is_malicious"] == bool(target["is_attack"]):
            hits += 1
    assert total > 0
    assert hits / total >= 0.7, f"acierto {hits}/{total} demasiado bajo"


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
    ours = event_features(parity_event)
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
