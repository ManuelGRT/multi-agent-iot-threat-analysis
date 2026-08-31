"""Contrato HTTP: solo /cases/analyze expone el flujo final de analisis."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


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
