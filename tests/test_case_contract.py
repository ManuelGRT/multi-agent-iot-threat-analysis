# tests/test_case_contract.py
from __future__ import annotations

import json

from src.contracts.attack_taxonomy import (
    JORGE_TAXONOMY_VERSION,
    MULTIDATASET_TAXONOMY_VERSION,
)
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
from src.mcp.threat_catalog import THREAT_INTEL_CATALOG_VERSION


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
        classification=ClassificationInfo(
            attack_type="DDoS_TCP",
            confidence=0.88,
            model_task="attack_type",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
        ),
        explanation=ExplanationInfo(
            summary="Trafico DDoS TCP",
            mitigations=["rate_limit_traffic"],
            references=[ThreatReference(attack_id="T1498", name="Network DoS")],
            confidence=0.85,
            attack_type="DDoS_TCP",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            catalog_scope="attack_type",
            catalog_version=THREAT_INTEL_CATALOG_VERSION,
            catalog_taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            catalog_compatible_taxonomy_versions=[
                MULTIDATASET_TAXONOMY_VERSION,
                JORGE_TAXONOMY_VERSION,
            ],
            reference_quality={"capec": "exact"},
            source="hybrid",
        ),
        judge=JudgeInfo(
            action="approve",
            approved=True,
            final_label="DDoS_TCP",
            final_confidence=0.88,
        ),
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
    assert payload["classification"]["attack_type"] == "DDoS_TCP"
    assert payload["explanation"]["attack_type"] == "DDoS_TCP"
    assert "attack_family" not in payload["classification"]
    assert payload["explanation"]["catalog_scope"] == "attack_type"
    assert (
        payload["explanation"]["catalog_version"]
        == THREAT_INTEL_CATALOG_VERSION
    )
    assert (
        payload["explanation"]["catalog_taxonomy_version"]
        == MULTIDATASET_TAXONOMY_VERSION
    )
    assert set(payload["explanation"]["catalog_compatible_taxonomy_versions"]) == {
        JORGE_TAXONOMY_VERSION,
        MULTIDATASET_TAXONOMY_VERSION,
    }

    restored = CaseResult(**payload)
    assert restored.case_id == case.case_id
    assert restored.trace[0].agent == "standardizer"
    assert restored.detection.probability == 0.97
    assert restored.explanation.attack_type == "DDoS_TCP"
    assert restored.explanation.taxonomy_version == MULTIDATASET_TAXONOMY_VERSION
    assert restored.explanation.catalog_scope == "attack_type"
    assert restored.explanation.catalog_version == THREAT_INTEL_CATALOG_VERSION
    assert set(restored.explanation.catalog_compatible_taxonomy_versions) == {
        JORGE_TAXONOMY_VERSION,
        MULTIDATASET_TAXONOMY_VERSION,
    }
    assert restored.explanation.reference_quality == {"capec": "exact"}


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
            "origin": {"parser": "mistral"},
        },
        "ingest_output": {"mapping_confidence": 0.9, "modality": "network_flow"},
        "detection_output": {
            "event_id": "evt-9",
            "is_malicious": True,
            "probability": 0.95,
            "evidence": ["dst_port=23"],
            "model_name": "xgboost_detection_balanced_by_origin_20260905",
            "next_route": "classify",
            "abstain": False,
        },
        "classification_output": {
            "event_id": "evt-9",
            "attack_type": "Command_and_Control",
            "confidence": 0.8,
            "model_task": "attack_type",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
            "next_route": "explain",
        },
        "explanation_output": {
            "risk_summary": "Posible canal de mando y control",
            "mitigations": ["isolate_device"],
            "confidence": 0.8,
            "attack_type": "Command_and_Control",
        },
        "judge_output": {
            "action": "approve",
            "approved": True,
            "final_label": "Command_and_Control",
            "final_confidence": 0.8,
            "issues": [],
        },
        "trace": [
            {
                "agent": "final_standardizer",
                "tool": "standardize_event",
                "status": "ok",
            },
            {
                "agent": "final_detector",
                "tool": "detect_event",
                "status": "ok",
                "confidence": 0.95,
            },
        ],
    }

    case = CaseResult.from_orchestrator_state(state)

    assert case.case_id == "case-abc123def456"
    assert case.status == "completed"
    assert case.standardization.mapping_confidence == 0.9
    assert case.standardization.schema_profile == "network_flow"
    assert case.detection.is_malicious is True
    assert case.classification.attack_type == "Command_and_Control"
    assert case.explanation.mitigations == ["isolate_device"]
    assert case.judge.final_label == "Command_and_Control"
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
    assert case.classification.attack_type is None
    assert case.status == "completed"
    assert case.case_id.startswith("case-")


def test_from_orchestrator_state_preserves_classifier_operational_contract():
    state = {
        "classification_output": {
            "attack_type": "DDoS_TCP",
            "confidence": 0.88,
            "decision_threshold": 0.81,
            "model_name": "xgboost_attack_subtype_multidataset16_balanced500_20260906",
            "model_task": "attack_type",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
            "top_scores": {
                "DDoS_TCP": 0.88,
                "DDoS_UDP": 0.06,
                "XSS": 0.03,
            },
        }
    }
    classification = CaseResult.from_orchestrator_state(state).classification
    assert classification.attack_type == "DDoS_TCP"
    assert classification.model_task == "attack_type"
    assert classification.taxonomy_version == MULTIDATASET_TAXONOMY_VERSION
    assert classification.decision_threshold == 0.81
    assert classification.model_name.endswith("_20260906")
    assert len(classification.top_scores) == 3
    assert not hasattr(classification, "attack_family")


def test_from_orchestrator_state_preserves_typed_mitigation_contract():
    state = {
        "classification_output": {
            "attack_type": "Command_and_Control",
            "confidence": 0.93,
            "model_task": "attack_type",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        },
        "explanation_output": {
            "risk_summary": "Canal de mando y control observado",
            "mitigations": ["aislar el dispositivo"],
            "confidence": 0.93,
            "attack_type": "Command_and_Control",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
            "catalog_scope": "attack_type",
            "catalog_version": THREAT_INTEL_CATALOG_VERSION,
            "catalog_taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
            "catalog_compatible_taxonomy_versions": [
                MULTIDATASET_TAXONOMY_VERSION,
                JORGE_TAXONOMY_VERSION,
            ],
            "reference_quality": {"capec": "generic_malware"},
            "source": "catalog",
        },
        "judge_output": {
            "action": "approve",
            "approved": True,
            "final_label": "Command_and_Control",
            "final_confidence": 0.93,
        },
    }

    case = CaseResult.from_orchestrator_state(state)

    assert case.classification.attack_type == "Command_and_Control"
    assert case.explanation.attack_type == "Command_and_Control"
    assert case.explanation.taxonomy_version == MULTIDATASET_TAXONOMY_VERSION
    assert case.explanation.catalog_scope == "attack_type"
    assert case.explanation.catalog_version == THREAT_INTEL_CATALOG_VERSION
    assert (
        case.explanation.catalog_taxonomy_version
        == MULTIDATASET_TAXONOMY_VERSION
    )
    assert set(case.explanation.catalog_compatible_taxonomy_versions) == {
        JORGE_TAXONOMY_VERSION,
        MULTIDATASET_TAXONOMY_VERSION,
    }
    assert (
        case.classification.taxonomy_version
        in case.explanation.catalog_compatible_taxonomy_versions
    )
    assert case.explanation.reference_quality == {"capec": "generic_malware"}

    restored = CaseResult.model_validate_json(case.model_dump_json())
    assert restored.explanation == case.explanation


def test_append_trace_state_helper_does_not_mutate():
    state = {"trace": [{"agent": "a", "status": "ok"}]}
    update = append_trace(state, {"agent": "b", "status": "ok"})
    assert len(update["trace"]) == 2
    assert len(state["trace"]) == 1
    empty_update = append_trace({}, {"agent": "first", "status": "started"})
    assert len(empty_update["trace"]) == 1
