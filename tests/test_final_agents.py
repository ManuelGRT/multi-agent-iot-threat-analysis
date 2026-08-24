# tests/test_final_agents.py
"""Tests de los agentes finales (Fase 3): unitarios con stub + integracion."""
from __future__ import annotations

import copy

import pytest

from src.agents.final import (
    FinalClassifier,
    FinalDetector,
    FinalJudge,
    FinalMitigator,
    FinalStandardizer,
)
from src.agents.final.final_standardizer import sanitize_canonical_event
from src.mcp.client import MCPToolClient


CANONICAL_EVENT = {
    "event_id": "evt-final-1",
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

LEAKY_EVENT = {
    **CANONICAL_EVENT,
    "label_raw": "DDoS_UDP",
    "attack_family": "ddos",
    "attack_subtype": "udp_flood",
    "semantic_text": "tcp flow label=DDoS_UDP with high packet rate",
    "telemetry": {"attack_type": "ddos", "fridge_temperature": 5.2},
}


class StubClient:
    """Cliente MCP de pruebas: respuestas fijas por (server, tool).

    Las tools no sobreescritas se delegan al cliente in-process real (util
    para threat_intel, que es local y determinista).
    """

    def __init__(self, overrides: dict | None = None):
        self.overrides = overrides or {}
        self.real = MCPToolClient(mode="inprocess")
        self.calls: list[tuple[str, str]] = []

    def call(self, server: str, tool: str, **arguments):
        self.calls.append((server, tool))
        key = (server, tool)
        if key in self.overrides:
            value = self.overrides[key]
            return value(**arguments) if callable(value) else copy.deepcopy(value)
        return self.real.call(server, tool, **arguments)


def standardize_ok(event=None, mapping_confidence=0.9, from_cache=False):
    return {
        "ok": True,
        "canonical_event": event or dict(CANONICAL_EVENT),
        "from_cache": from_cache,
        "cache_key": "stub::stub.csv::0",
        "model": "mistral-small-latest" if from_cache else "deterministic_adapter",
        "mapping_confidence": mapping_confidence,
    }


def detect_ok(probability: float):
    return {
        "ok": True,
        "is_malicious": probability >= 0.5,
        "probability": probability,
        "model_name": "xgboost_detection_standardized_with_edge_20260705",
    }


def classify_ok(family="ddos", confidence=0.95):
    return {
        "ok": True,
        "attack_family": family,
        "confidence": confidence,
        "top_scores": {family: confidence, "scanning": 0.02, "injection": 0.01},
        "model_name": "xgboost_attack_family_balanced_group_20260705",
    }


# ---------------------------------------------------------------------------
# sanitizacion (anti target-leakage)
# ---------------------------------------------------------------------------

def test_sanitize_removes_target_fields_and_scrubs_text():
    clean = sanitize_canonical_event(LEAKY_EVENT)
    assert clean["label_raw"] is None
    assert clean["attack_family"] is None
    assert clean["attack_subtype"] is None
    assert "DDoS_UDP" not in clean["semantic_text"]
    # los campos no-target sobreviven
    assert clean["event_id"] == "evt-final-1"
    assert clean["packet_count"] == 120
    assert clean["telemetry"].get("fridge_temperature") == 5.2
    assert "attack_type" not in clean["telemetry"]


# ---------------------------------------------------------------------------
# standardizer
# ---------------------------------------------------------------------------

def test_standardizer_sanitizes_and_routes_detect():
    stub = StubClient({("inference", "standardize_event"): standardize_ok(dict(LEAKY_EVENT))})
    agent = FinalStandardizer(client=stub)
    update = agent.run({"raw_input": {"dataset": "iot23", "row": {"proto": "tcp"}}})

    assert update["route"] == "detect"
    assert update["canonical_event"]["label_raw"] is None
    assert update["canonical_event"]["attack_family"] is None
    assert update["ingest_output"]["mapping_confidence"] == 0.9
    assert update["ingest_output"]["model"] == "deterministic_adapter"
    assert "sanitized:no_target_leakage" in update["ingest_output"]["notes"]
    assert update["trace"][-1]["agent"] == "final_standardizer"
    assert update["trace"][-1]["status"] == "ok"
    assert update["trace"][-1]["finished_at"] is not None


def test_standardizer_low_mapping_confidence_routes_judge():
    stub = StubClient(
        {("inference", "standardize_event"): standardize_ok(mapping_confidence=0.3)}
    )
    update = FinalStandardizer(client=stub).run({"raw_input": {"dataset": "generic"}})
    assert update["route"] == "judge"


def test_standardizer_prestandardized_passthrough():
    stub = StubClient()
    update = FinalStandardizer(client=stub).run(
        {"raw_input": {"dataset": "edge_iiotset", "canonical_event": dict(LEAKY_EVENT)}}
    )
    # no llama a la tool de estandarizacion: passthrough con sanitizacion
    assert ("inference", "standardize_event") not in stub.calls
    assert update["route"] == "detect"
    assert update["canonical_event"]["label_raw"] is None
    assert update["ingest_output"]["model"] == "prestandardized_passthrough"


def test_standardizer_tool_error_routes_judge_with_error():
    stub = StubClient(
        {("inference", "standardize_event"): {"ok": False, "error": "ValueError: boom"}}
    )
    state = {"raw_input": {"dataset": "no_existe"}}
    update = FinalStandardizer(client=stub).run(state)
    assert update["route"] == "judge"
    assert any("final_standardizer" in error for error in update["errors"])
    assert update["trace"][-1]["status"] == "error"


def test_standardizer_from_cache_flag_propagates():
    stub = StubClient({("inference", "standardize_event"): standardize_ok(from_cache=True)})
    update = FinalStandardizer(client=stub).run({"raw_input": {"dataset": "ton_iot"}})
    assert update["ingest_output"]["from_cache"] is True
    assert "source:mistral_cache" in update["ingest_output"]["notes"]


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------

def base_state_with_event():
    return {"canonical_event": dict(CANONICAL_EVENT), "trace": []}


def test_detector_malicious_routes_classify():
    stub = StubClient({("inference", "detect_event"): detect_ok(0.97)})
    update = FinalDetector(client=stub).run(base_state_with_event())
    output = update["detection_output"]
    assert output["is_malicious"] is True
    assert output["abstain"] is False
    assert update["route"] == "classify"
    assert update["trace"][-1]["confidence"] == pytest.approx(0.97)


def test_detector_benign_routes_judge_for_audit():
    stub = StubClient({("inference", "detect_event"): detect_ok(0.03)})
    update = FinalDetector(client=stub).run(base_state_with_event())
    assert update["detection_output"]["is_malicious"] is False
    assert update["route"] == "judge"
    assert update["trace"][-1]["confidence"] == pytest.approx(0.97)


@pytest.mark.parametrize("probability", [0.4, 0.5, 0.6])
def test_detector_gray_zone_abstains(probability):
    stub = StubClient({("inference", "detect_event"): detect_ok(probability)})
    update = FinalDetector(client=stub).run(base_state_with_event())
    output = update["detection_output"]
    assert output["abstain"] is True
    assert output["next_route"] == "judge"
    assert update["route"] == "judge"
    assert update["trace"][-1]["status"] == "abstain"


def test_detector_gray_zone_boundaries_exclusive():
    for probability, expected_route in [(0.39, "judge"), (0.61, "classify")]:
        stub = StubClient({("inference", "detect_event"): detect_ok(probability)})
        update = FinalDetector(client=stub).run(base_state_with_event())
        assert update["detection_output"]["abstain"] is False
        assert update["route"] == expected_route


def test_detector_tool_error_abstains_to_judge():
    stub = StubClient(
        {("inference", "detect_event"): {"ok": False, "error": "FileNotFoundError: modelo"}}
    )
    update = FinalDetector(client=stub).run(base_state_with_event())
    assert update["route"] == "judge"
    assert update["detection_output"]["abstain"] is True
    assert update["errors"]


def test_detector_invalid_gray_zone_rejected():
    with pytest.raises(ValueError):
        FinalDetector(gray_low=0.7, gray_high=0.3)


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------

def test_classifier_maps_family_and_top_scores():
    stub = StubClient({("inference", "classify_event"): classify_ok("ddos", 0.95)})
    update = FinalClassifier(client=stub).run(base_state_with_event())
    output = update["classification_output"]
    assert output["attack_family"] == "ddos"
    assert output["confidence"] == pytest.approx(0.95)
    assert output["top_scores"]["ddos"] == pytest.approx(0.95)
    assert output["next_route"] == "explain"
    assert update["route"] == "explain"


def test_classifier_tool_error_routes_judge():
    stub = StubClient({("inference", "classify_event"): {"ok": False, "error": "boom"}})
    update = FinalClassifier(client=stub).run(base_state_with_event())
    assert update["route"] == "judge"
    assert update["classification_output"]["attack_family"] == "unknown_attack"
    assert update["errors"]


# ---------------------------------------------------------------------------
# mitigator (base determinista sobre el catalogo threat intel real)
# ---------------------------------------------------------------------------

def test_mitigator_catalog_ddos_references_and_mitigations():
    state = {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": True, "probability": 0.97},
        "classification_output": {"attack_family": "ddos", "confidence": 0.95},
        "trace": [],
    }
    update = FinalMitigator(client=MCPToolClient()).run(state)
    output = update["explanation_output"]
    assert output["mitigations"], "debe haber mitigaciones del catalogo"
    reference_ids = {
        ref.get("attack_id") or ref.get("capec_id") for ref in output["references"]
    }
    assert "T1498" in reference_ids
    assert any(str(rid).startswith("CAPEC-") for rid in reference_ids)
    assert all(ref["source"] == "catalog" for ref in output["references"])
    assert output["source"] == "catalog"
    assert output["requires_human_review"] is False
    assert update["route"] == "judge"


def test_mitigator_benign_minimal_monitoring():
    state = {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": False, "probability": 0.02},
        "trace": [],
    }
    update = FinalMitigator(client=MCPToolClient()).run(state)
    output = update["explanation_output"]
    assert "benigno" in output["risk_summary"]
    assert output["requires_human_review"] is False
    # benigno: solo monitorizacion minima, sin contencion agresiva
    assert not any("aislar" in m or "bloquear" in m for m in output["mitigations"])


def test_mitigator_unknown_family_requires_review():
    state = {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": True, "probability": 0.8},
        "classification_output": {"attack_family": "familia_inexistente", "confidence": 0.4},
        "trace": [],
    }
    update = FinalMitigator(client=MCPToolClient()).run(state)
    assert update["explanation_output"]["requires_human_review"] is True
    assert update["needs_human_review"] is True


def test_mitigator_tool_error_falls_back_to_review():
    stub = StubClient({("threat_intel", "suggest_mitigations"): {"ok": False, "error": "x"}})
    state = {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": True, "probability": 0.9},
        "classification_output": {"attack_family": "ddos", "confidence": 0.9},
        "trace": [],
    }
    update = FinalMitigator(client=stub).run(state)
    assert update["explanation_output"]["requires_human_review"] is True
    assert update["errors"]


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------

def clean_state():
    return {
        "canonical_event": dict(CANONICAL_EVENT),
        "ingest_output": {"mapping_confidence": 0.9},
        "detection_output": {"is_malicious": True, "probability": 0.97, "abstain": False},
        "classification_output": {"attack_family": "ddos", "confidence": 0.95},
        "explanation_output": {"requires_human_review": False},
        "trace": [],
    }


def test_judge_approves_clean_case():
    update = FinalJudge().run(clean_state())
    output = update["judge_output"]
    assert output["action"] == "approve"
    assert output["approved"] is True
    assert output["final_label"] == "ddos"
    assert update["needs_human_review"] is False
    assert update["route"] == "end"


def test_judge_flags_detector_abstention():
    state = clean_state()
    state["detection_output"]["abstain"] = True
    update = FinalJudge().run(state)
    assert update["judge_output"]["action"] == "human_interrupt"
    assert "detector_abstained" in update["judge_output"]["issues"]
    assert update["needs_human_review"] is True


def test_judge_flags_low_mapping_confidence():
    state = clean_state()
    state["ingest_output"]["mapping_confidence"] = 0.2
    update = FinalJudge().run(state)
    assert "mapping_confidence_below_review_threshold" in update["judge_output"]["issues"]


def test_judge_flags_low_classification_confidence():
    state = clean_state()
    state["classification_output"]["confidence"] = 0.4
    update = FinalJudge().run(state)
    assert (
        "classification_confidence_below_review_threshold"
        in update["judge_output"]["issues"]
    )


def test_judge_flags_pipeline_errors():
    state = clean_state()
    state["errors"] = ["final_detector: boom"]
    update = FinalJudge().run(state)
    assert "pipeline_errors_present" in update["judge_output"]["issues"]


def test_judge_rejects_inconsistent_detection_semantics():
    state = clean_state()
    state["detection_output"] = {
        "is_malicious": False,
        "probability": 0.99,
        "abstain": False,
    }
    state.pop("classification_output")
    state.pop("explanation_output")

    update = FinalJudge().run(state)
    assert "detection_label_probability_mismatch" in update["judge_output"]["issues"]
    assert update["judge_output"]["approved"] is False


def test_judge_rejects_malicious_case_classified_as_benign():
    state = clean_state()
    state["classification_output"]["attack_family"] = "benign"

    update = FinalJudge().run(state)
    assert "malicious_classified_as_benign" in update["judge_output"]["issues"]


def test_judge_benign_case_label_and_confidence():
    state = clean_state()
    state["detection_output"] = {"is_malicious": False, "probability": 0.05, "abstain": False}
    state.pop("classification_output")
    state.pop("explanation_output")
    update = FinalJudge().run(state)
    output = update["judge_output"]
    assert output["action"] == "approve"
    assert output["final_label"] == "benign"
    assert output["final_confidence"] == pytest.approx(0.95)
