"""Regresiones del runner de evaluación en vivo del mitigador."""
from __future__ import annotations

from types import SimpleNamespace

from scripts.evaluate_live_mitigator_multidataset16 import (
    DEFAULT_MCP_CLIENT_MODE,
    DEFAULT_MODEL,
    RecordingThreatIntelClient,
    case_metrics,
)
from src.agents.final.llm_mitigator import MITIGATION_SCHEMA


def test_live_mitigator_runner_keeps_the_production_mistral_model_as_default():
    assert DEFAULT_MODEL == "mistral-small-2603"


def test_live_mitigator_runner_uses_production_mcp_transport_by_default():
    assert DEFAULT_MCP_CLIENT_MODE == "stdio"


def test_live_runner_records_raw_context_from_threat_intel_tool():
    raw = {
        "risk_summary": "riesgo contextualizado",
        "mitigations": [{"base_id": 1, "text": "accion contextualizada"}],
        "confidence": 0.9,
        "requires_human_review": False,
    }

    class StubClient:
        def call(self, server, tool, **arguments):
            if tool == "suggest_mitigations":
                return {
                    "ok": True,
                    "attack_type": "DDoS_TCP",
                    "mitigations_by_phase": {
                        "containment": ["base 1"],
                        "eradication": ["base 2"],
                        "prevention": [],
                    },
                }
            return {
                "ok": True,
                "provider": "mistral",
                "model_name": "llm_mitigator::mistral::test-model",
                "base_count": 2,
                "latency_ms": 12.0,
                "contextualization": raw,
            }

    recorder = RecordingThreatIntelClient(StubClient(), MITIGATION_SCHEMA)
    recorder.call(
        "threat_intel",
        "suggest_mitigations",
        attack_type="DDoS_TCP",
    )
    response = recorder.call(
        "threat_intel",
        "contextualize_mitigations",
        attack_type="DDoS_TCP",
        canonical_event={"event_id": "evt-test"},
        detection={"is_malicious": True, "probability": 0.99},
        classification={"attack_type": "DDoS_TCP", "confidence": 0.95},
    )

    assert response["contextualization"] == raw
    assert len(recorder.records) == 1
    assert recorder.records[0]["status"] == "ok"
    assert recorder.records[0]["schema_valid"] is True
    assert recorder.records[0]["base_count"] == 2
    assert recorder.records[0]["payload"] == raw


def test_live_runner_preserves_invalid_raw_context_returned_by_threat_intel():
    invalid_raw = {
        "risk_summary": "riesgo contextualizado",
        "mitigations": [{"base_id": 1, "text": "accion contextualizada"}],
        "confidence": 0.9,
        "requires_human_review": "false",
    }

    class StubClient:
        def call(self, server, tool, **arguments):
            del server, arguments
            if tool == "suggest_mitigations":
                return {
                    "ok": True,
                    "attack_type": "DDoS_TCP",
                    "mitigations_by_phase": {
                        "containment": ["base 1"],
                        "eradication": [],
                        "prevention": [],
                    },
                }
            return {
                "ok": False,
                "error": "TypeError: contrato invalido",
                "provider": "mistral",
                "model_name": "llm_mitigator::mistral::test-model",
                "base_count": 1,
                "contextualization_schema_valid": False,
                "contextualization": invalid_raw,
            }

    recorder = RecordingThreatIntelClient(StubClient(), MITIGATION_SCHEMA)
    recorder.call(
        "threat_intel", "suggest_mitigations", attack_type="DDoS_TCP"
    )
    response = recorder.call(
        "threat_intel",
        "contextualize_mitigations",
        attack_type="DDoS_TCP",
        canonical_event={"event_id": "evt-invalid"},
        detection={"is_malicious": True, "probability": 0.99},
        classification={"attack_type": "DDoS_TCP", "confidence": 0.95},
    )

    assert response["ok"] is False
    assert recorder.records[0]["status"] == "error"
    assert recorder.records[0]["schema_valid"] is False
    assert recorder.records[0]["payload"] == invalid_raw
    assert "requires_human_review" in recorder.records[0]["schema_error"]


def test_case_metrics_does_not_coerce_invalid_raw_mitigation_types():
    explanation = SimpleNamespace(
        mitigation_items=[],
        references=[],
        evidence=[],
        model_name=None,
        first_five_catalog_anchored=False,
        llm_context_summary=None,
        review_reasons=[],
    )
    case = SimpleNamespace(
        case_id="case-invalid-raw",
        explanation=explanation,
        trace=[],
        standardization=SimpleNamespace(source="llm"),
        detection=SimpleNamespace(probability=0.99),
        classification=SimpleNamespace(
            attack_type="DDoS_TCP", confidence=0.95
        ),
        judge=SimpleNamespace(
            action="approve", requires_human_review=False, issues=[]
        ),
        status="completed",
    )
    audit = SimpleNamespace(verdict="approve", passed=True, hard_failures=[])
    raw_record = {
        "schema_valid": False,
        "status": "error",
        "payload": {
            "mitigations": 7,
            "requires_human_review": "false",
        },
    }

    result = case_metrics(
        {
            "manifest_id": "edge::1",
            "attack_type": "DDoS_TCP",
        },
        case,
        audit,
        raw_record,
    )

    assert result["raw_mitigation_count"] == 0
    assert result["raw_requires_human_review"] is False
