"""Un unico test E2E sobre las cuatro fuentes y las modalidades TON-IoT."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agents.final import CaseAuditor
from src.agents.llm_ingest_parser import LLM_PARSER_VERSION
from src.contracts.case import CaseResult
from src.contracts.leakage import is_predictive_target_field
from src.mcp.client import MCPToolClient
from src.mcp.common import configured_ingest_model
from src.mcp.standardization_cache import bind_event_identity
from src.mcp.standardization_contract import validate_target_free_canonical
from src.orchestration.mcp_graph import run_case


FIXTURES_PATH = Path(__file__).parent / "fixtures" / "e2e_sources.json"
EXPECTED_SOURCES = {
    "edge_iiotset",
    "iot23",
    "bot_iot",
    "ton_iot_network",
    "ton_iot_linux",
    "ton_iot_windows",
    "ton_iot_telemetry",
}


def _target_fields(value: Any, path: str = "") -> list[str]:
    """Localiza nombres de campos objetivo sin alterar la entrada."""

    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if is_predictive_target_field(key):
                found.append(child)
            found.extend(_target_fields(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_target_fields(item, f"{path}[{index}]"))
    return found


def test_end_to_end(tmp_path, monkeypatch):
    """Ejecuta por stdio un registro test de cada fuente/modalidad y lo audita."""

    monkeypatch.setenv("TFM_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("MCP_CLIENT_MODE", "stdio")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    assert {fixture["name"] for fixture in fixtures} == EXPECTED_SOURCES
    assert len(fixtures) == len(EXPECTED_SOURCES)

    cases: dict[str, CaseResult] = {}
    reports = {}
    with MCPToolClient(mode="stdio", timeout_seconds=60) as client:
        for fixture in fixtures:
            expected = fixture["expected"]
            raw_input = {
                "dataset": fixture["dataset"],
                "row": fixture["row"],
                "source_file": fixture["source_file"],
                "row_id": fixture["row_id"],
                "split": fixture["split"],
            }
            assert fixture["split"] == "test"
            assert _target_fields(raw_input["row"]) == []
            assert set(fixture["reference"]).isdisjoint(raw_input)

            canonical = bind_event_identity(
                fixture["canonical_event"],
                dataset=fixture["dataset"],
                source_file=fixture["source_file"],
                row_id=fixture["row_id"],
                split=fixture["split"],
                parser_version=LLM_PARSER_VERSION,
            )
            validate_target_free_canonical(canonical)
            cached = client.call(
                "case_memory",
                "store_standardization_cache",
                dataset=fixture["dataset"],
                row=fixture["row"],
                canonical_event=canonical,
                selected_columns=fixture["selected_columns"],
                model=configured_ingest_model(),
                source_file=fixture["source_file"],
                row_id=fixture["row_id"],
                split=fixture["split"],
            )
            assert cached["ok"] is True and cached["stored"] is True

            case = run_case(
                raw_input,
                case_id=f"case-e2e-{fixture['name']}",
                client=client,
                use_llm_mitigator=True,
                persist=True,
            )
            stored = client.call(
                "case_memory",
                "get_case",
                case_id=case.case_id,
                include_trace=True,
            )
            report = CaseAuditor().audit(CaseResult.model_validate(stored["case"]))
            cases[fixture["name"]] = case
            reports[fixture["name"]] = report

            assert case.status == expected["status"]
            assert [entry.agent for entry in case.trace] == expected["trace_agents"]
            assert case.standardization.source == "llm"
            assert case.standardization.from_cache is True
            assert case.standardization.model == fixture["standardization_model"]
            assert case.standardization.schema_profile == expected["schema_profile"]
            assert case.detection.model_name == "xgboost_detection_final"
            assert case.detection.is_malicious is expected["is_malicious"]
            assert case.detection.abstain is expected["detection_abstain"]
            assert case.judge.action == expected["judge_action"]
            if expected["attack_type"] is None:
                assert case.classification.attack_type is None
                assert case.classification.model_name is None
                assert case.explanation.attack_type is None
                assert case.explanation.mitigation_items == []
            else:
                assert case.classification.model_name == (
                    "xgboost_classification_final"
                )
                assert case.classification.attack_type == expected["attack_type"]
                assert case.explanation.attack_type == expected["attack_type"]
                assert case.explanation.source == "catalog"
                assert case.explanation.references
            assert stored["ok"] is True
            assert stored["status"] == case.status
            assert stored["case"] == case.model_dump(mode="json")
            assert len(stored["trace"]) == len(case.trace)
            assert report.verdict == expected["audit_verdict"], report.issues
            assert report.passed is True

        assert set(client._stdio_workers) == {
            "inference",
            "threat_intel",
            "case_memory",
        }
        assert client._retired_stdio_workers == []

    assert set(cases) == EXPECTED_SOURCES
    assert set(reports) == EXPECTED_SOURCES
    assert Path(tmp_path, "case_memory.db").is_file()
    assert Path(tmp_path, "mistral_standardization_v2.sqlite3").is_file()
    assert client.closed is True
