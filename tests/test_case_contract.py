# tests/test_case_contract.py
from __future__ import annotations

import json

from src.contracts.case import (
    CaseResult,
    ClassificationInfo,
    DetectionInfo,
    ExplanationInfo,
    JudgeInfo,
    StandardizationInfo,
    ThreatReference,
    TraceEntry,
    new_case_id,
)
from src.orchestration.state import append_trace


def test_new_case_id_format_and_uniqueness():
    first = new_case_id()
    second = new_case_id()
    assert first.startswith("case-")
    assert len(first) == len("case-") + 12
    assert first != second


def test_trace_entry_lifecycle():
    entry = TraceEntry(agent="detector", tool="detect_event", status="started")
    assert entry.finished_at is None
    assert entry.latency_ms is None

    entry.finish(status="ok", confidence=0.91, summary="malicious=True")
    assert entry.status == "ok"
    assert entry.finished_at is not None
    assert entry.latency_ms is not None and entry.latency_ms >= 0.0
    assert entry.confidence == 0.91


def test_case_result_full_serialization_roundtrip():
    case = CaseResult(
        raw_input={"dataset": "edge_iiotset", "row_id": 7},
        canonical_event={"event_id": "evt-1", "modality": "network_flow"},
        standardization=StandardizationInfo(
            model="mistral-small-latest", mapping_confidence=0.92, from_cache=True
        ),
        detection=DetectionInfo(is_malicious=True, probability=0.97, model_name="xgb_std_edge"),
        classification=ClassificationInfo(attack_family="ddos", confidence=0.88),
        explanation=ExplanationInfo(
            summary="Trafico DDoS",
            mitigations=["rate_limit_traffic"],
            references=[ThreatReference(attack_id="T1498", name="Network DoS")],
            confidence=0.85,
            source="hybrid",
        ),
        judge=JudgeInfo(action="approve", approved=True, final_label="ddos", final_confidence=0.88),
    )
    case.start_step("standardizer", tool="standardize_event").finish("ok", confidence=0.92)
    case.start_step("detector", tool="detect_event").finish("ok", confidence=0.97)
    case.close()

    assert case.status == "completed"
    assert case.finished_at is not None
    assert len(case.trace) == 2

    payload = json.loads(case.model_dump_json())
    assert payload["case_id"].startswith("case-")
    assert payload["explanation"]["references"][0]["attack_id"] == "T1498"

    restored = CaseResult(**payload)
    assert restored.case_id == case.case_id
    assert restored.trace[0].agent == "standardizer"
    assert restored.detection.probability == 0.97


def test_case_result_needs_human_review_when_judge_interrupts():
    case = CaseResult(judge=JudgeInfo(action="human_interrupt", approved=False, requires_human_review=True))
    case.close()
    assert case.status == "needs_human_review"


def test_case_result_error_status():
    case = CaseResult(errors=["standardizer_failed"])
    case.close()
    assert case.status == "error"


def test_from_orchestrator_state_bridges_existing_contract():
    state = {
        "case_id": "case-abc123def456",
        "event_id": "evt-9",
        "raw_input": {"dataset": "iot23"},
        "canonical_event": {
            "event_id": "evt-9",
            "modality": "network_flow",
            "schema_profile": "network_flow",
            "mapping_confidence": 0.9,
            "origin": {"parser": "adapter_iot23"},
        },
        "ingest_output": {"mapping_confidence": 0.9, "modality": "network_flow"},
        "detection_output": {
            "event_id": "evt-9",
            "is_malicious": True,
            "probability": 0.95,
            "evidence": ["dst_port=23"],
            "model_name": "rule_based",
            "next_route": "classify",
            "abstain": False,
        },
        "classification_output": {
            "event_id": "evt-9",
            "attack_family": "botnet",
            "confidence": 0.8,
            "next_route": "explain",
        },
        "explanation_output": {
            "risk_summary": "Posible botnet",
            "mitigations": ["isolate_device"],
            "confidence": 0.8,
        },
        "judge_output": {
            "action": "approve",
            "approved": True,
            "final_label": "botnet",
            "final_confidence": 0.8,
            "issues": [],
        },
        "trace": [
            {"agent": "ingest", "tool": "adapter_iot23", "status": "ok"},
            {"agent": "detector", "status": "ok", "confidence": 0.95},
        ],
    }

    case = CaseResult.from_orchestrator_state(state)

    assert case.case_id == "case-abc123def456"
    assert case.status == "completed"
    assert case.standardization.mapping_confidence == 0.9
    assert case.standardization.schema_profile == "network_flow"
    assert case.detection.is_malicious is True
    assert case.classification.attack_family == "botnet"
    assert case.explanation.mitigations == ["isolate_device"]
    assert case.judge.final_label == "botnet"
    assert len(case.trace) == 2
    assert case.trace[1].confidence == 0.95


def test_from_orchestrator_state_benign_case_without_classification():
    state = {
        "raw_input": {"dataset": "ton_iot"},
        "canonical_event": {"event_id": "evt-benign", "modality": "telemetry"},
        "ingest_output": {"mapping_confidence": 0.85, "modality": "telemetry"},
        "detection_output": {
            "is_malicious": False,
            "probability": 0.03,
            "next_route": "end",
        },
    }
    case = CaseResult.from_orchestrator_state(state)
    assert case.detection.is_malicious is False
    assert case.classification.attack_family is None
    assert case.status == "completed"
    assert case.case_id.startswith("case-")


def test_append_trace_state_helper_does_not_mutate():
    state = {"trace": [{"agent": "a", "status": "ok"}]}
    update = append_trace(state, {"agent": "b", "status": "ok"})
    assert len(update["trace"]) == 2
    assert len(state["trace"]) == 1
    empty_update = append_trace({}, {"agent": "first", "status": "started"})
    assert len(empty_update["trace"]) == 1
