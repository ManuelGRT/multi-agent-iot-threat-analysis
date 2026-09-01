"""Contrato HTTP del analisis final y su auditoria posterior."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.mcp.client import MCPToolClient


def test_health_endpoint():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/events/analyze",
            {
                "dataset": "iot23",
                "row": {"proto": "tcp"},
                "use_llm": False,
            },
        ),
        (
            "/datasets/adapt",
            {
                "dataset": "iot23",
                "row": {"proto": "tcp"},
                "use_llm": False,
            },
        ),
        (
            "/datasets/analyze-file",
            {
                "path": "iot23.csv",
                "dataset": "iot23",
                "use_llm": False,
            },
        ),
    ],
)
def test_legacy_analysis_endpoints_are_closed_without_adapter_bypass(path, payload):
    response = TestClient(app).post(path, json=payload)

    assert response.status_code == 410
    assert "/cases/analyze" in response.json()["detail"]


def test_supported_datasets_is_metadata_only():
    response = TestClient(app).get("/datasets/supported")

    assert response.status_code == 200
    assert "iot23" in response.json()["datasets"]
    assert response.json()["scope"] == "legacy_adapter_metadata_only"
    assert response.json()["analysis_endpoint"] == "/cases/analyze"


def test_final_case_rejects_unknown_top_level_field_instead_of_dropping_it():
    response = TestClient(app).post(
        "/cases/analyze",
        json={
            "dataset": "iot23",
            "row": {"proto": "tcp"},
            "label": "Mirai",
        },
    )

    assert response.status_code == 422


def test_final_case_forwards_row_unchanged_without_runtime_sanitization(monkeypatch):
    from src.api import routers

    received = {}

    def fake_run(raw_input, **_kwargs):
        received.update(raw_input)
        return {"ok": True}

    monkeypatch.setattr(routers, "_run_final_case", fake_run)
    row = {"proto": "tcp", "label": "Mirai"}
    response = TestClient(app).post(
        "/cases/analyze",
        json={"dataset": "iot23", "row": row},
    )

    assert response.status_code == 200
    assert received["row"] == row


def test_final_case_always_enables_llm_mitigation_and_persistence(monkeypatch):
    from src.api import routers

    received = {}

    def fake_run(raw_input, **kwargs):
        received["raw_input"] = raw_input
        received.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(routers, "_run_final_case", fake_run)
    response = TestClient(app).post(
        "/cases/analyze",
        json={"dataset": "iot23", "row": {"proto": "tcp"}},
    )

    assert response.status_code == 200
    assert received["use_llm_mitigator"] is True
    assert received["persist"] is True


@pytest.mark.parametrize("field", ["use_llm_mitigator", "persist"])
def test_final_case_rejects_attempts_to_disable_operational_policies(field):
    response = TestClient(app).post(
        "/cases/analyze",
        json={
            "dataset": "iot23",
            "row": {"proto": "tcp"},
            field: False,
        },
    )

    assert response.status_code == 422


def test_audit_endpoint_checks_a_persisted_case_without_mutating_it(
    tmp_path,
    monkeypatch,
):
    from tests.test_auditor import clean_attack_case

    monkeypatch.setenv("TFM_CASE_MEMORY_DB", str(tmp_path / "cases.db"))
    case = clean_attack_case()
    payload = case.model_dump(mode="json")
    memory = MCPToolClient(mode="inprocess")
    assert memory.call(
        "case_memory", "create_case", case_id=case.case_id, payload={}
    )["ok"]
    for entry in case.trace:
        assert memory.call(
            "case_memory",
            "append_trace",
            case_id=case.case_id,
            entry=entry.model_dump(mode="json"),
        )["ok"]
    assert memory.call(
        "case_memory",
        "update_case",
        case_id=case.case_id,
        payload=payload,
        status=case.status,
    )["ok"]

    before = memory.call(
        "case_memory", "get_case", case_id=case.case_id, include_trace=True
    )
    response = TestClient(app).get(f"/cases/{case.case_id}/audit")
    after = memory.call(
        "case_memory", "get_case", case_id=case.case_id, include_trace=True
    )

    assert response.status_code == 200
    report = response.json()
    assert report["case_id"] == case.case_id
    assert report["verdict"] == "approve"
    assert report["passed"] is True
    assert report["issues"] == []
    assert report["checks"]
    assert before["case"] == after["case"]
    assert before["trace"] == after["trace"]
    assert all(entry["agent"] != "final_auditor" for entry in after["trace"])


def test_audit_endpoint_returns_404_for_unknown_case(tmp_path, monkeypatch):
    monkeypatch.setenv("TFM_CASE_MEMORY_DB", str(tmp_path / "cases.db"))

    response = TestClient(app).get("/cases/case-does-not-exist/audit")

    assert response.status_code == 404
    assert response.json()["detail"] == "Caso no encontrado: case-does-not-exist"


def test_audit_endpoint_rejects_invalid_persisted_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("TFM_CASE_MEMORY_DB", str(tmp_path / "cases.db"))
    memory = MCPToolClient(mode="inprocess")
    assert memory.call(
        "case_memory",
        "create_case",
        case_id="case-invalid",
        payload={"case_id": "case-invalid", "status": "estado-imposible"},
    )["ok"]

    response = TestClient(app).get("/cases/case-invalid/audit")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "El caso persistido no cumple el contrato CaseResult"
    )
