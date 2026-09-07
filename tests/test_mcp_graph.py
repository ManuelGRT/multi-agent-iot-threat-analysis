# tests/test_mcp_graph.py
"""Tests del grafo final MCP (Fase 3): topologia, rutas y CaseResult E2E."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.contracts.case import CaseResult
from src.mcp.client import MCPToolClient
from src.mcp.common import resolve_path
from src.orchestration.mcp_graph import (
    CasePersistenceError,
    FinalAgentBundle,
    build_final_graph,
    default_final_agents,
    run_case,
)
from tests.test_final_agents import (
    CANONICAL_EVENT,
    StubClient,
    classify_ok,
    detect_ok,
    standardize_ok,
)

EDGE_STANDARDIZED = (
    resolve_path("standardized_dataset").parent
    / "edgeiiot_mistral_standardized_tfm_less_both_partial_cache_recalc_20260621.jsonl"
)

FINAL_AGENT_NAMES = [
    "final_standardizer",
    "final_detector",
    "final_classifier",
    "final_mitigator",
    "final_judge",
]

STDIO_E2E_EVENT = {
    "event_id": "evt-stdio-e2e",
    "modality": "network_flow",
    "src_ip": "192.168.0.128",
    "dst_ip": "192.168.0.170",
    "src_port": 60210,
    "dst_port": 4321,
    "transport_proto": "TCP",
    "packet_count": 1,
    "byte_count": 208,
    "duration_ms": 1.0,
    "telemetry": {
        "tcp.flags": 24.0,
        "tcp.flags.ack": 1.0,
        "tcp.len": 208.0,
        "tcp.dstport": 4321.0,
        "tcp.srcport": 60210.0,
    },
    "schema_profile": "network_packet",
    "semantic_text": "TCP packet with payload",
    "provenance": {"dataset": "stdio_smoke", "split": "stream"},
    "mapping_confidence": 0.95,
}


@pytest.fixture(autouse=True)
def isolate_case_memory(tmp_path, monkeypatch):
    """Evita que los tests del endpoint obligatorio ensucien la memoria real."""
    monkeypatch.setenv("TFM_CASE_MEMORY_DB", str(tmp_path / "cases.db"))


def _models_loadable() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return resolve_path("detection_model").exists() and resolve_path("attack_type_model").exists()


def stub_bundle(overrides: dict) -> FinalAgentBundle:
    """Bundle de agentes finales sobre un StubClient (hermetico, sin modelos)."""
    from src.agents.final import (
        FinalClassifier,
        FinalDetector,
        FinalJudge,
        FinalMitigator,
        FinalStandardizer,
    )

    stub = StubClient(overrides)
    return FinalAgentBundle(
        standardizer=FinalStandardizer(client=stub),
        detector=FinalDetector(client=stub),
        classifier=FinalClassifier(client=stub),
        mitigator=FinalMitigator(client=stub),
        judge=FinalJudge(client=stub),
        client=stub,
    )


# ---------------------------------------------------------------------------
# rutas del grafo (hermetico con stub)
# ---------------------------------------------------------------------------

def test_malicious_case_traverses_all_final_agents():
    agents = stub_bundle(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.97),
            ("inference", "classify_event"): classify_ok("DDoS_TCP", 0.95),
        }
    )
    case = run_case({"dataset": "iot23", "row": {"proto": "tcp"}}, agents=agents)

    assert isinstance(case, CaseResult)
    assert case.status == "completed"
    agents_in_trace = [entry.agent for entry in case.trace]
    for name in FINAL_AGENT_NAMES:
        assert name in agents_in_trace, f"falta {name} en la traza"
    assert len(case.trace) >= 5
    assert case.judge.action == "approve"
    assert case.classification.attack_type == "DDoS_TCP"
    assert case.explanation.references, "el mitigador debe adjuntar referencias"


def test_transport_failure_preserves_partial_trace_and_reaches_judge():
    def timeout(**_arguments):
        raise TimeoutError("transporte MCP bloqueado")

    agents = stub_bundle(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): timeout,
        }
    )
    case = run_case({"dataset": "iot23", "row": {"proto": "tcp"}}, agents=agents)

    assert case.status == "needs_human_review"
    assert any("TimeoutError" in error for error in case.errors)
    trace_agents = [entry.agent for entry in case.trace]
    assert trace_agents == [
        "orchestrator",
        "final_standardizer",
        "final_detector",
        "final_judge",
    ]
    assert case.judge.action == "human_interrupt"


def test_benign_case_skips_classification_but_is_judged():
    agents = stub_bundle(
        {
            ("inference", "standardize_event"): standardize_ok(
                selected_columns=["temp"],
                dataset="ton_iot",
            ),
            ("inference", "detect_event"): detect_ok(0.03),
        }
    )
    case = run_case({"dataset": "ton_iot", "row": {"temp": 21}}, agents=agents)

    assert case.status == "completed"
    agents_in_trace = [entry.agent for entry in case.trace]
    assert "final_classifier" not in agents_in_trace
    assert "final_judge" in agents_in_trace
    assert case.detection.is_malicious is False
    assert case.classification.attack_type is None
    assert case.judge.final_label == "benign"


def test_gray_zone_case_abstains_to_human_review():
    agents = stub_bundle(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.5),
        }
    )
    case = run_case({"dataset": "iot23", "row": {"proto": "tcp"}}, agents=agents)

    assert case.status == "needs_human_review"
    assert case.detection.abstain is True
    assert case.detection.is_malicious is None
    assert case.judge.action == "human_interrupt"
    assert "detector_abstained" in case.judge.issues
    assert "malicious_without_classification" not in case.judge.issues
    assert "malicious_without_mitigation" not in case.judge.issues
    assert case.judge.final_label is None
    assert case.judge.final_confidence == 0.0


def test_low_mapping_confidence_goes_straight_to_judge():
    agents = stub_bundle(
        {
            ("inference", "standardize_event"): standardize_ok(
                mapping_confidence=0.2,
                dataset="generic",
            )
        }
    )
    case = run_case({"dataset": "generic", "text": "algo raro"}, agents=agents)

    assert case.status == "needs_human_review"
    agents_in_trace = [entry.agent for entry in case.trace]
    assert "final_detector" not in agents_in_trace
    assert "mapping_confidence_below_review_threshold" in case.judge.issues


def test_llm_standardization_failure_abstains_before_detector():
    agents = stub_bundle(
        {
            ("inference", "standardize_event"): {
                "ok": False,
                "abstain": True,
                "requires_human_review": True,
                "failure_code": "llm_standardization_failed",
                "error": "TimeoutError: Mistral no responde",
            }
        }
    )

    case = run_case(
        {"dataset": "iot23", "row": {"proto": "tcp"}, "row_id": 7},
        agents=agents,
    )

    assert case.status == "needs_human_review"
    assert case.standardization.abstain is True
    assert case.standardization.requires_human_review is True
    assert case.standardization.failure_code == "llm_standardization_failed"
    assert case.canonical_event == {}
    assert case.detection.is_malicious is None
    assert case.judge.action == "human_interrupt"
    assert "standardizer_abstained" in case.judge.issues
    assert [entry.agent for entry in case.trace] == [
        "orchestrator",
        "final_standardizer",
        "final_judge",
    ]
    assert case.trace[1].status == "abstain"


def test_tool_error_produces_auditable_case():
    agents = stub_bundle(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): {"ok": False, "error": "modelo ausente"},
        }
    )
    case = run_case({"dataset": "iot23", "row": {"proto": "tcp"}}, agents=agents)

    assert case.status == "needs_human_review"
    assert case.errors
    assert "pipeline_errors_present" in case.judge.issues


def test_case_id_and_trace_lifecycle():
    agents = stub_bundle(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.02),
        }
    )
    case = run_case({"dataset": "iot23"}, case_id="case-fijo12345678", agents=agents)

    assert case.case_id == "case-fijo12345678"
    assert case.trace[0].agent == "orchestrator"
    assert all(entry.finished_at is not None for entry in case.trace)
    assert case.finished_at is not None
    payload = json.loads(case.model_dump_json())
    assert payload["case_id"] == "case-fijo12345678"


def test_build_final_graph_invocable_directly():
    agents = stub_bundle(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.02),
        }
    )
    graph = build_final_graph(agents=agents)
    final_state = graph.invoke(
        {"raw_input": {"dataset": "iot23"}, "route": "standardize", "trace": []}
    )
    assert final_state["route"] == "end"
    assert final_state["judge_output"]["action"] == "approve"


def test_build_final_graph_requires_an_explicit_lifecycle_owner():
    with pytest.raises(ValueError, match="requiere agents o client"):
        build_final_graph()

    with pytest.raises(ValueError, match="tambien debe inyectarse"):
        run_case({"dataset": "iot23"}, graph=object())


def test_default_agents_use_stdio_and_demo_can_select_inprocess(monkeypatch):
    monkeypatch.delenv("MCP_CLIENT_MODE", raising=False)
    assert default_final_agents().client.mode == "stdio"

    monkeypatch.setenv("MCP_CLIENT_MODE", "inprocess")
    assert default_final_agents().client.mode == "inprocess"


# ---------------------------------------------------------------------------
# persistencia en memoria de casos
# ---------------------------------------------------------------------------

def test_run_case_persists_to_case_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("TFM_CASE_MEMORY_DB", str(tmp_path / "cases.db"))
    agents = stub_bundle(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.97),
            ("inference", "classify_event"): classify_ok("DDoS_TCP", 0.95),
        }
    )
    case = run_case({"dataset": "iot23", "row": {"proto": "tcp"}}, agents=agents, persist=True)

    memory = MCPToolClient(mode="inprocess")
    stored = memory.call("case_memory", "get_case", case_id=case.case_id)
    assert stored["ok"], stored.get("error")
    assert stored["status"] == "completed"
    assert stored["case"]["classification"]["attack_type"] == "DDoS_TCP"
    assert len(stored["trace"]) == len(case.trace)


def test_run_case_closes_its_own_client_after_final_persistence(monkeypatch):
    import src.orchestration.mcp_graph as mcp_graph

    operations: list[str] = []

    class LifecycleStub(StubClient):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()
            return False

        def call(self, server, tool, **arguments):
            operations.append(tool)
            return super().call(server, tool, **arguments)

        def close(self):
            operations.append("close")

    lifecycle = LifecycleStub(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.02),
            ("case_memory", "create_case"): {"ok": True},
            ("case_memory", "append_trace"): {"ok": True},
            ("case_memory", "update_case"): {"ok": True},
        }
    )
    monkeypatch.setattr(mcp_graph, "MCPToolClient", lambda: lifecycle)

    case = run_case(
        {"dataset": "iot23", "row": {"proto": "tcp"}}, persist=True
    )

    assert case.status == "completed"
    assert operations.count("close") == 1
    assert operations[-2:] == ["update_case", "close"]


def test_run_case_closes_its_own_client_when_persistence_fails(monkeypatch):
    import src.orchestration.mcp_graph as mcp_graph

    operations: list[str] = []

    class LifecycleStub(StubClient):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()
            return False

        def call(self, server, tool, **arguments):
            operations.append(tool)
            return super().call(server, tool, **arguments)

        def close(self):
            operations.append("close")

    lifecycle = LifecycleStub(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.02),
            ("case_memory", "create_case"): {"ok": True},
            ("case_memory", "append_trace"): {
                "ok": False,
                "error": "fallo simulado",
            },
        }
    )
    monkeypatch.setattr(mcp_graph, "MCPToolClient", lambda: lifecycle)

    with pytest.raises(CasePersistenceError, match="append_trace"):
        run_case(
            {"dataset": "iot23", "row": {"proto": "tcp"}}, persist=True
        )

    assert operations[-1] == "close"
    assert operations.count("close") == 1


def test_run_case_does_not_close_an_injected_client():
    class LifecycleStub(StubClient):
        close_calls = 0

        def close(self):
            self.close_calls += 1

    lifecycle = LifecycleStub(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.02),
        }
    )

    case = run_case(
        {"dataset": "iot23", "row": {"proto": "tcp"}}, client=lifecycle
    )

    assert case.status == "completed"
    assert lifecycle.close_calls == 0


def test_run_case_collision_does_not_overwrite_existing_case(tmp_path, monkeypatch):
    monkeypatch.setenv("TFM_CASE_MEMORY_DB", str(tmp_path / "cases.db"))
    memory = MCPToolClient(mode="inprocess")
    case_id = "case-colision123456"
    original = {"sentinel": "registro-original"}
    created = memory.call(
        "case_memory", "create_case", case_id=case_id, payload=original
    )
    assert created["ok"] is True

    agents = stub_bundle(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.02),
        }
    )
    with pytest.raises(CasePersistenceError, match="create_case"):
        run_case(
            {"dataset": "iot23", "row": {"proto": "tcp"}},
            case_id=case_id,
            agents=agents,
            persist=True,
        )

    stored = memory.call("case_memory", "get_case", case_id=case_id)
    assert stored["ok"] is True
    assert stored["case"] == original
    assert stored["status"] == "open"
    assert stored["trace"] == []


@pytest.mark.parametrize("failing_tool", ["append_trace", "update_case"])
def test_run_case_stops_on_persistence_error(failing_tool):
    overrides = {
        ("inference", "standardize_event"): standardize_ok(),
        ("inference", "detect_event"): detect_ok(0.02),
        ("case_memory", "create_case"): {"ok": True, "created": True},
        ("case_memory", "append_trace"): {"ok": True, "seq": 1},
        ("case_memory", "update_case"): {"ok": True, "updated": True},
    }
    overrides[("case_memory", failing_tool)] = {
        "ok": False,
        "error": f"fallo simulado en {failing_tool}",
    }
    agents = stub_bundle(overrides)

    with pytest.raises(CasePersistenceError, match=failing_tool):
        run_case(
            {"dataset": "iot23", "row": {"proto": "tcp"}},
            agents=agents,
            persist=True,
        )

    persistence_calls = [
        tool for server, tool in agents.client.calls if server == "case_memory"
    ]
    assert failing_tool in persistence_calls
    if failing_tool == "append_trace":
        assert "update_case" not in persistence_calls


# ---------------------------------------------------------------------------
# criterio de aceptacion F3: evento Edge-IIoTset E2E determinista
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _models_loadable() or not EDGE_STANDARDIZED.exists(),
    reason="modelos .joblib o dataset Edge estandarizado no disponibles",
)
def test_edge_iiotset_event_full_pipeline_with_prepared_models():
    """Un ataque Edge-IIoTset recorre standardize->detect->classify->explain->judge."""
    chosen = None
    inprocess_client = MCPToolClient(mode="inprocess")
    with EDGE_STANDARDIZED.open("r", encoding="utf-8") as fh:
        for _ in range(40):
            line = fh.readline()
            if not line:
                break
            row = json.loads(line)
            if not (row.get("target") or {}).get("is_attack"):
                continue
            event = row.get("canonical_event") or {}
            if float(event.get("mapping_confidence", 0.0) or 0.0) < 0.5:
                continue
            case = run_case(
                {"dataset": "edge_iiotset", "canonical_event": event},
                client=inprocess_client,
            )
            agents_in_trace = [entry.agent for entry in case.trace]
            if all(name in agents_in_trace for name in FINAL_AGENT_NAMES):
                chosen = (row, case)
                break
    assert chosen is not None, "ninguna fila Edge de ataque atraveso el grafo completo"

    row, case = chosen
    assert case.status in {"completed", "needs_human_review"}
    assert len(case.trace) >= 5
    assert case.detection.is_malicious is True
    assert case.detection.probability > 0.6
    assert case.classification.attack_type is not None
    assert case.classification.top_scores
    assert case.explanation.mitigations
    assert case.explanation.references
    assert case.explanation.source == "catalog"
    # precondicion de entrada: el evento canonico del caso no arrastra targets
    assert case.canonical_event.get("label_raw") is None
    assert case.canonical_event.get("attack_family") is None

    # determinismo: repetir el caso da el mismo resultado
    repeat = run_case(
        {"dataset": "edge_iiotset", "canonical_event": row["canonical_event"]},
        client=inprocess_client,
    )
    assert repeat.detection.probability == pytest.approx(case.detection.probability)
    assert repeat.classification.attack_type == case.classification.attack_type


@pytest.mark.skipif(not _models_loadable(), reason="modelos productivos no disponibles")
def test_real_stdio_client_runs_full_graph_and_persists(tmp_path, monkeypatch):
    """Aceptacion: el grafo completo atraviesa los tres servidores MCP reales."""
    monkeypatch.setenv("TFM_CASE_MEMORY_DB", str(tmp_path / "stdio-cases.db"))
    monkeypatch.setenv(
        "TFM_STANDARDIZATION_CACHE_DB", str(tmp_path / "stdio-cache.db")
    )
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with MCPToolClient(mode="stdio", timeout_seconds=30) as client:
        case = run_case(
            {"dataset": "stdio_smoke", "canonical_event": STDIO_E2E_EVENT},
            case_id="case-stdio-e2e",
            client=client,
            use_llm_mitigator=False,
            persist=True,
        )
        stored = client.call(
            "case_memory", "get_case", case_id=case.case_id, include_trace=True
        )
        assert set(client._stdio_workers) == {
            "inference",
            "threat_intel",
            "case_memory",
        }
        assert client._retired_stdio_workers == []

    assert case.status == "completed"
    assert [entry.agent for entry in case.trace] == [
        "orchestrator",
        "final_standardizer",
        "final_detector",
        "final_classifier",
        "final_mitigator",
        "final_judge",
    ]
    assert case.standardization.source == "prestandardized"
    assert case.detection.is_malicious is True
    assert case.detection.probability > 0.6
    assert case.classification.attack_type == "Backdoor"
    assert (
        case.classification.confidence
        >= case.classification.decision_threshold
    )
    assert case.explanation.source == "catalog"
    assert case.explanation.references
    assert case.judge.approved is True
    assert stored["ok"] is True
    assert stored["case"]["case_id"] == case.case_id
    assert stored["status"] == case.status
    assert len(stored["trace"]) == len(case.trace)
    assert client.closed is True


# ---------------------------------------------------------------------------
# endpoint POST /cases/analyze
# ---------------------------------------------------------------------------

def test_cases_analyze_endpoint_requires_input():
    client = TestClient(app)
    response = client.post("/cases/analyze", json={"dataset": "generic"})
    assert response.status_code == 400


def test_cases_analyze_endpoint_returns_and_persists_case_result(monkeypatch):
    from src.api import routers

    def execute_without_external_llm(raw_input, **kwargs):
        assert raw_input["row"] == {"proto": "tcp"}
        assert kwargs == {"use_llm_mitigator": True, "persist": True}
        case = run_case(
            {"dataset": raw_input["dataset"], "canonical_event": dict(CANONICAL_EVENT)},
            client=MCPToolClient(mode="inprocess"),
            use_llm_mitigator=False,
            persist=True,
        )
        return case.model_dump(mode="json")

    monkeypatch.setattr(routers, "_run_final_case", execute_without_external_llm)
    client = TestClient(app)
    response = client.post(
        "/cases/analyze",
        json={
            "dataset": "iot23",
            "row": {"proto": "tcp"},
            "row_id": 42,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["case_id"].startswith("case-")
    assert payload["status"] in {"completed", "needs_human_review", "error"}
    assert payload["trace"], "el caso debe llevar traza"
    assert payload["trace"][0]["agent"] == "orchestrator"
    assert any(entry["agent"] == "final_judge" for entry in payload["trace"])

    stored = MCPToolClient(mode="inprocess").call(
        "case_memory", "get_case", case_id=payload["case_id"]
    )
    assert stored["ok"] is True
    assert stored["status"] == payload["status"]
    assert stored["case"] == payload
    assert len(stored["trace"]) == len(payload["trace"])


def test_cases_analyze_endpoint_rejects_canonical_event():
    client = TestClient(app)
    response = client.post(
        "/cases/analyze",
        json={"dataset": "edge_iiotset", "canonical_event": dict(CANONICAL_EVENT)},
    )
    assert response.status_code == 422


def test_cases_analyze_rejects_raw_and_canonical_input_combination():
    response = TestClient(app).post(
        "/cases/analyze",
        json={
            "dataset": "iot23",
            "row": {"proto": "tcp"},
            "canonical_event": dict(CANONICAL_EVENT),
        },
    )

    assert response.status_code == 422


def test_cases_analyze_raw_llm_failure_returns_reviewable_case(monkeypatch):
    import src.mcp.inference_server as inference_server

    # Esta prueba sustituye una dependencia Python local; no pretende validar
    # el transporte stdio, que se cubre en test_mcp_servers.py.
    monkeypatch.setenv("MCP_CLIENT_MODE", "inprocess")

    class FailingParser:
        async def parse(self, _raw_input):
            raise TimeoutError("Mistral no responde")

    monkeypatch.setattr(inference_server, "_build_llm_parser", FailingParser)

    response = TestClient(app).post(
        "/cases/analyze",
        json={"dataset": "iot23", "row": {"proto": "tcp"}, "row_id": 7},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "needs_human_review"
    assert payload["standardization"]["abstain"] is True
    assert payload["standardization"]["failure_code"] == "llm_standardization_failed"
    assert payload["detection"]["is_malicious"] is None
    assert payload["judge"]["action"] == "human_interrupt"
    assert [entry["agent"] for entry in payload["trace"]] == [
        "orchestrator",
        "final_standardizer",
        "final_judge",
    ]


def test_cases_analyze_empty_llm_extraction_abstains_before_detector(monkeypatch):
    import src.mcp.inference_server as inference_server
    from src.agents.llm_ingest_parser import LLMIngestParser

    # El parser simulado debe vivir en el mismo proceso que el servidor.
    monkeypatch.setenv("MCP_CLIENT_MODE", "inprocess")

    class EmptyExtractionAgent:
        model = "mistral-small-2603"

        async def invoke_json(self, system_prompt, user_payload, json_schema):
            del user_payload, json_schema
            if "preseleccion" in system_prompt:
                return {
                    "selected_columns": ["proto"],
                    "modality_guess": "network_flow",
                    "schema_profile_guess": "network_flow",
                    "rationale": "protocolo de transporte",
                }
            return {}

    parser = LLMIngestParser(
        provider="mistral",
        require_llm_column_selection=True,
        strict_output_validation=True,
        column_selection_threshold=1,
    )
    parser.agent = EmptyExtractionAgent()
    monkeypatch.setattr(inference_server, "_build_llm_parser", lambda: parser)

    response = TestClient(app).post(
        "/cases/analyze",
        json={"dataset": "iot23", "row": {"proto": "tcp"}, "row_id": 8},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "needs_human_review"
    assert payload["standardization"]["abstain"] is True
    assert payload["standardization"]["failure_code"] == "llm_standardization_failed"
    assert "faltan campos requeridos" in payload["standardization"]["failure_reason"]
    assert payload["detection"]["is_malicious"] is None
    assert payload["judge"]["action"] == "human_interrupt"
    assert [entry["agent"] for entry in payload["trace"]] == [
        "orchestrator",
        "final_standardizer",
        "final_judge",
    ]
