# tests/test_final_agents.py
"""Tests de los agentes finales (Fase 3): unitarios con stub + integracion."""
from __future__ import annotations

import copy

import pytest

from src.contracts.attack_taxonomy import (
    JORGE_TAXONOMY_VERSION,
    MULTIDATASET_TAXONOMY_VERSION,
)
from src.agents.final import (
    FinalClassifier,
    FinalDetector,
    FinalJudge,
    FinalMitigator,
    FinalStandardizer,
)
from src.mcp.client import MCPToolClient
from src.eval.data_sanitization import sanitize_canonical_event, sanitize_llm_input
from src.mcp.standardization_cache import bind_event_identity


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
        self.call_arguments: list[tuple[str, str, dict]] = []

    def call(self, server: str, tool: str, **arguments):
        self.calls.append((server, tool))
        self.call_arguments.append((server, tool, copy.deepcopy(arguments)))
        key = (server, tool)
        if key in self.overrides:
            value = self.overrides[key]
            return value(**arguments) if callable(value) else copy.deepcopy(value)
        return self.real.call(server, tool, **arguments)


def standardize_ok(
    event=None,
    mapping_confidence=0.9,
    *,
    model="mistral-small-2603",
    selected_columns=None,
    source="llm",
    provider="mistral",
    from_cache=False,
    dataset="iot23",
    source_file="api",
    row_id=0,
    split="stream",
    bind_identity=True,
):
    canonical = dict(event or CANONICAL_EVENT)
    canonical["mapping_confidence"] = mapping_confidence
    if source == "llm" and not from_cache and bind_identity:
        parser_version = str(
            (canonical.get("provenance") or {}).get("parser_version")
            or "llm-0.2.0"
        )
        canonical = bind_event_identity(
            canonical,
            dataset=dataset,
            source_file=source_file,
            row_id=row_id,
            split=split,
            parser_version=parser_version,
        )
    notes = (
        ["source:prestandardized"]
        if source == "prestandardized"
        else (
            ["parsed_by_llm", "source:mistral_cache"]
            if from_cache
            else ["parsed_by_llm", "source:mistral_live"]
        )
    )
    return {
        "ok": True,
        "canonical_event": canonical,
        "from_cache": from_cache,
        "source": source,
        "provider": provider,
        "model": model,
        "mapping_confidence": mapping_confidence,
        "selected_columns": list(
            ["proto"]
            if selected_columns is None
            else selected_columns
        ),
        "notes": notes,
        "cache_content_hash": "a" * 64 if from_cache else None,
        "cache_pipeline_hash": "b" * 64 if from_cache else None,
    }


def detect_ok(probability: float):
    return {
        "ok": True,
        "is_malicious": probability >= 0.5,
        "probability": probability,
        "model_name": "xgboost_detection_standardized_with_edge_20260705",
    }


def classify_ok(attack_type="DDoS_TCP", confidence=0.95):
    return {
        "ok": True,
        "attack_type": attack_type,
        "confidence": confidence,
        "decision_threshold": 0.65,
        "top_scores": {
            attack_type: confidence,
            "DDoS_UDP": 0.02,
            "XSS": 0.01,
        },
        "model_task": "attack_type",
        "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        "model_name": "xgboost_attack_subtype_multidataset16_balanced500_20260906",
    }


# ---------------------------------------------------------------------------
# preparacion offline para evaluacion (anti target-leakage)
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


def test_security_boundary_preserves_behavioral_indicators_without_mutation():
    original = {
        **LEAKY_EVENT,
        "attack_indicators": ["syn_flood_pattern", "label=Mirai"],
    }

    clean = sanitize_canonical_event(original)

    assert clean["attack_indicators"] == ["syn_flood_pattern", ""]
    assert original["label_raw"] == "DDoS_UDP"
    assert original["attack_indicators"][1] == "label=Mirai"


def test_security_boundary_scrubs_text_before_llm_prompt():
    clean = sanitize_llm_input(
        {
            "dataset": "generic",
            "text": "src_ip=10.0.0.1 label=Mirai proto=tcp",
        }
    )

    assert "label=Mirai" not in clean["text"]
    assert "src_ip=10.0.0.1" in clean["text"]


def test_security_boundary_scrubs_json_text_and_preserves_next_assignment():
    json_input = sanitize_llm_input(
        {"text": '{"label":"Mirai","proto":"tcp"}'}
    )
    flat_input = sanitize_llm_input(
        {"text": "src_ip=10.0.0.1 label=Mirai proto=tcp"}
    )
    list_input = sanitize_llm_input(
        {"text": 'label: ["Mirai","DDoS"], proto=tcp'}
    )
    embedded_json = sanitize_llm_input(
        {"text": 'payload={"label":"Mirai","proto":"tcp"}'}
    )
    query_string = sanitize_llm_input(
        {"text": "url=/login?label=Mirai&proto=tcp"}
    )

    assert json_input["text"] == '{"proto":"tcp"}'
    assert "label=Mirai" not in flat_input["text"]
    assert "src_ip=10.0.0.1" in flat_input["text"]
    assert "proto=tcp" in flat_input["text"]
    assert list_input["text"] == "proto=tcp"
    assert embedded_json["text"] == 'payload={"proto":"tcp"}'
    assert query_string["text"] == "url=/login?proto=tcp"


def test_security_boundary_keeps_allowlisted_protocol_fields():
    clean_input = sanitize_llm_input(
        {
            "row": {
                "protocol_family": "icmp",
                "dns.qry.type": "AAAA",
                "label": "Mirai",
            }
        }
    )
    clean_json_input = sanitize_llm_input(
        {"text": '{"protocol_family":"icmp","label":"Mirai"}'}
    )
    clean_event = sanitize_canonical_event(
        {
            **CANONICAL_EVENT,
            "service_context": {
                "protocol_family": "icmp",
                "dns": {"qry": {"type": "AAAA"}},
                "attack_family": "ddos",
            },
        }
    )

    assert clean_input["row"] == {
        "protocol_family": "icmp",
        "dns.qry.type": "AAAA",
    }
    assert clean_json_input["text"] == '{"protocol_family":"icmp"}'
    assert clean_event["service_context"]["protocol_family"] == "icmp"
    assert clean_event["service_context"]["dns"]["qry"]["type"] == "AAAA"
    assert "attack_family" not in clean_event["service_context"]


def test_security_boundary_removes_family_values_even_under_neutral_or_allowed_keys():
    clean_input = sanitize_llm_input(
        {
            "row": {
                "risk": "Mirai",
                "family_hint": "DDoS",
                "protocol_family": "Mirai",
                "metric_category": "DDoS",
                "proto": "tcp",
            }
        }
    )
    clean_event = sanitize_canonical_event(
        {
            **CANONICAL_EVENT,
            "telemetry": {"risk": "Mirai", "temperature": 21.5},
            "host": {"family_hint": "DDoS", "cpu": 12.0},
        }
    )

    assert clean_input["row"] == {"proto": "tcp"}
    assert clean_event["telemetry"] == {"temperature": 21.5}
    assert clean_event["host"] == {"cpu": 12.0}


# ---------------------------------------------------------------------------
# standardizer
# ---------------------------------------------------------------------------

def test_standardizer_uses_llm_mcp_result_and_routes_detect():
    sanitized = sanitize_canonical_event(LEAKY_EVENT)
    stub = StubClient({("inference", "standardize_event"): standardize_ok(sanitized)})
    agent = FinalStandardizer(client=stub)
    update = agent.run({"raw_input": {"dataset": "iot23", "row": {"proto": "tcp"}}})

    assert update["route"] == "detect"
    assert update["canonical_event"]["label_raw"] is None
    assert update["canonical_event"]["attack_family"] is None
    assert update["ingest_output"]["mapping_confidence"] == 0.9
    assert update["ingest_output"]["model"] == "mistral-small-2603"
    assert update["ingest_output"]["from_cache"] is False
    assert update["ingest_output"]["selected_columns"] == ["proto"]
    assert update["ingest_output"]["notes"] == [
        "parsed_by_llm",
        "source:mistral_live",
    ]
    _, _, arguments = stub.call_arguments[-1]
    assert "allow_llm" not in arguments
    assert "cache_key" not in arguments
    assert update["trace"][-1]["agent"] == "final_standardizer"
    assert update["trace"][-1]["status"] == "ok"
    assert update["trace"][-1]["finished_at"] is not None


def test_standardizer_low_mapping_confidence_routes_judge():
    stub = StubClient(
        {
            ("inference", "standardize_event"): standardize_ok(
                mapping_confidence=0.3,
                dataset="generic",
            )
        }
    )
    update = FinalStandardizer(client=stub).run({"raw_input": {"dataset": "generic"}})
    assert update["route"] == "judge"


def test_standardizer_prestandardized_passthrough():
    clean = sanitize_canonical_event(LEAKY_EVENT)
    stub = StubClient(
        {
            ("inference", "standardize_event"): standardize_ok(
                clean,
                model="prestandardized_passthrough",
                selected_columns=[],
                source="prestandardized",
                provider=None,
            )
        }
    )
    update = FinalStandardizer(client=stub).run(
        {"raw_input": {"dataset": "edge_iiotset", "canonical_event": dict(clean)}}
    )
    # El agente no limpia el evento: el limite MCP debe validarlo o rechazarlo.
    assert ("inference", "standardize_event") in stub.calls
    _, _, arguments = stub.call_arguments[-1]
    assert arguments["canonical_event"] == clean
    assert update["route"] == "detect"
    assert update["canonical_event"]["label_raw"] is None
    assert update["ingest_output"]["model"] == "prestandardized_passthrough"


def test_standardizer_accepts_valid_mistral_cache_hit():
    from src.mcp.standardization_cache import compute_content_hash

    cached_event = {
        **CANONICAL_EVENT,
        "event_id": "llm::iot23::duplicate.csv::17",
        "origin": {
            "source_name": "iot23",
            "source_file": "duplicate.csv",
            "row_id": 17,
            "schema_profile": "network_flow",
        },
        "provenance": {
            "dataset": "iot23",
            "source_file": "duplicate.csv",
            "row_id": 17,
            "split": "test",
            "parser_version": "llm-0.2.0",
        },
    }
    result = standardize_ok(cached_event, from_cache=True)
    result["cache_content_hash"] = compute_content_hash(row={"proto": "tcp"})
    stub = StubClient({("inference", "standardize_event"): result})

    update = FinalStandardizer(client=stub).run(
        {
            "raw_input": {
                "dataset": "iot23",
                "row": {"proto": "tcp"},
                "source_file": "duplicate.csv",
                "row_id": 17,
                "split": "test",
            }
        }
    )

    assert update["route"] == "detect"
    assert update["ingest_output"]["from_cache"] is True
    assert update["ingest_output"]["cache_content_hash"] == result["cache_content_hash"]
    assert update["ingest_output"]["cache_pipeline_hash"] == "b" * 64
    _, _, arguments = stub.call_arguments[-1]
    assert arguments["split"] == "test"


def test_standardizer_llm_failure_abstains_to_judge():
    stub = StubClient(
        {
            ("inference", "standardize_event"): {
                "ok": False,
                "abstain": True,
                "requires_human_review": True,
                "failure_code": "llm_standardization_failed",
                "error": "RuntimeError: Mistral no disponible",
            }
        }
    )
    state = {"raw_input": {"dataset": "no_existe"}}
    update = FinalStandardizer(client=stub).run(state)
    assert update["route"] == "judge"
    assert update["needs_human_review"] is True
    assert update["ingest_output"]["normalized"] is False
    assert update["ingest_output"]["abstain"] is True
    assert update["ingest_output"]["requires_human_review"] is True
    assert (
        update["ingest_output"]["failure_code"]
        == "llm_standardization_failed"
    )
    assert update["trace"][-1]["status"] == "abstain"


def test_standardizer_rejects_live_identity_from_a_different_record():
    result = standardize_ok(bind_identity=False)
    stub = StubClient({("inference", "standardize_event"): result})

    update = FinalStandardizer(client=stub).run(
        {
            "raw_input": {
                "dataset": "iot23",
                "row": {"proto": "tcp"},
                "source_file": "new.csv",
                "row_id": 77,
            }
        }
    )

    assert update["route"] == "judge"
    assert update["ingest_output"]["failure_code"] == (
        "invalid_standardization_response"
    )
    assert "event_id" in update["ingest_output"]["failure_reason"]


def test_standardizer_malformed_success_abstains_instead_of_crashing():
    stub = StubClient(
        {
            ("inference", "standardize_event"): {
                "ok": True,
                "canonical_event": {"modality": "network_flow"},
                "mapping_confidence": 1.5,
            }
        }
    )

    update = FinalStandardizer(client=stub).run(
        {"raw_input": {"dataset": "iot23", "row": {"proto": "tcp"}}}
    )

    assert update["route"] == "judge"
    assert update["needs_human_review"] is True
    assert update["ingest_output"]["failure_code"] == "invalid_standardization_response"
    assert update["trace"][-1]["status"] == "abstain"


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        {"source": "adapter", "provider": None},
        {"source": "prestandardized", "provider": None},
        {"from_cache": True},
        {"abstain": True},
        {"provider": "ollama"},
        {"selected_columns": 1},
        {"selected_columns": ["does_not_exist"]},
        {"selected_columns": ["attack_family"]},
        {"selected_columns": ["proto", "proto"]},
        {"abstain": 0},
        {"requires_human_review": True},
        {"failure_code": "llm_standardization_failed"},
        {"failure_reason": "Mistral unavailable"},
        {"error": "Mistral unavailable"},
        {"llm_error": "Mistral unavailable"},
        {"fallback": True},
        {"used_fallback": True},
        {"cache_key": "legacy-row"},
        {
            "notes": [
                "parsed_by_llm",
                "fallback_adapter",
            ]
        },
        {"notes": ["parsed_by_llm"]},
        {"mapping_confidence": "0.9"},
    ],
)
def test_standardizer_rejects_non_mistral_success_for_raw_input(invalid_metadata):
    result = standardize_ok()
    result.update(invalid_metadata)
    stub = StubClient({("inference", "standardize_event"): result})

    update = FinalStandardizer(client=stub).run(
        {"raw_input": {"dataset": "iot23", "row": {"proto": "tcp"}}}
    )

    assert update["route"] == "judge"
    assert update["needs_human_review"] is True
    assert update["ingest_output"]["normalized"] is False
    assert update["ingest_output"]["abstain"] is True
    assert update["ingest_output"]["failure_code"] == "invalid_standardization_response"
    assert "canonical_event" not in update
    assert update["trace"][-1]["status"] == "abstain"


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        {"source": "llm", "provider": "mistral"},
        {"source": "prestandardized", "provider": "mistral"},
        {"source": "prestandardized", "provider": None, "from_cache": True},
    ],
)
def test_standardizer_rejects_invalid_prestandardized_success(invalid_metadata):
    result = standardize_ok(
        source="prestandardized",
        provider=None,
        selected_columns=[],
    )
    result.update(invalid_metadata)
    stub = StubClient({("inference", "standardize_event"): result})

    update = FinalStandardizer(client=stub).run(
        {
            "raw_input": {
                "dataset": "edge_iiotset",
                "canonical_event": dict(CANONICAL_EVENT),
            }
        }
    )

    assert update["route"] == "judge"
    assert update["needs_human_review"] is True
    assert update["ingest_output"]["abstain"] is True
    assert update["ingest_output"]["source"] == "prestandardized"
    assert update["ingest_output"]["failure_code"] == "invalid_standardization_response"
    assert "canonical_event" not in update


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
    assert "sin_veredicto" in update["trace"][-1]["summary"]
    assert "malicioso=" not in update["trace"][-1]["summary"]


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

def test_classifier_requires_exact_top_three():
    with pytest.raises(ValueError, match="top-3"):
        FinalClassifier(top_k=2)

def test_classifier_persists_attack_type_and_top_scores():
    stub = StubClient(
        {("inference", "classify_event"): classify_ok("DDoS_TCP", 0.95)}
    )
    update = FinalClassifier(client=stub).run(base_state_with_event())
    output = update["classification_output"]
    assert output["attack_type"] == "DDoS_TCP"
    assert output["confidence"] == pytest.approx(0.95)
    assert output["top_scores"]["DDoS_TCP"] == pytest.approx(0.95)
    assert len(output["top_scores"]) == 3
    assert output["model_task"] == "attack_type"
    assert not {
        "attack_family",
        "attack_subtype",
        "family_confidence",
        "family_scores",
    } & output.keys()
    assert output["next_route"] == "explain"
    assert update["route"] == "explain"


def test_classifier_accepts_multidataset_attack_type_contract():
    response = {
        "ok": True,
        "attack_type": "Command_and_Control",
        "confidence": 0.90,
        "decision_threshold": 0.70,
        "top_scores": {
            "Command_and_Control": 0.90,
            "DoS": 0.06,
            "DDoS_TCP": 0.04,
        },
        "model_task": "attack_type",
        "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        "model_name": "multidataset16",
    }
    update = FinalClassifier(
        client=StubClient({("inference", "classify_event"): response})
    ).run(base_state_with_event())

    assert update["route"] == "explain"
    assert update["classification_output"]["attack_type"] == (
        "Command_and_Control"
    )
    assert "score_type=attack_type" in update["classification_output"]["reason"]


def test_classifier_rejects_old_family_task():
    response = {
        "ok": True,
        "attack_type": "DDoS_TCP",
        "confidence": 0.99,
        "top_scores": {"DDoS_TCP": 0.99, "DDoS_UDP": 0.01, "XSS": 0.0},
        "model_task": "attack_family",
        "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
    }
    update = FinalClassifier(
        client=StubClient({("inference", "classify_event"): response})
    ).run(base_state_with_event())
    assert update["route"] == "judge"
    assert update["classification_output"]["attack_type"] is None
    assert any("tarea_clasificador_no_soportada" in error for error in update["errors"])


def test_classifier_rejects_non_deployed_taxonomy():
    response = {
        "ok": True,
        "attack_type": "DDoS_TCP",
        "confidence": 0.99,
        "top_scores": {"DDoS_TCP": 0.99, "DDoS_UDP": 0.01, "XSS": 0.0},
        "model_task": "attack_type",
        "taxonomy_version": JORGE_TAXONOMY_VERSION,
    }
    update = FinalClassifier(
        client=StubClient({("inference", "classify_event"): response})
    ).run(base_state_with_event())
    assert update["route"] == "judge"
    assert any(
        "taxonomia_clasificador_no_soportada" in error
        for error in update["errors"]
    )


def test_classifier_rejects_incomplete_attack_type_scores():
    response = {
        "ok": True,
        "attack_type": "DDoS_TCP",
        "confidence": 0.91,
        "top_scores": {"DDoS_TCP": 0.91},
        "model_task": "attack_type",
        "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
    }
    update = FinalClassifier(
        client=StubClient({("inference", "classify_event"): response})
    ).run(base_state_with_event())
    assert update["route"] == "judge"
    assert any(
        "puntuaciones_clasificador_tipo_incoherentes" in error
        for error in update["errors"]
    )


def test_classifier_tool_error_routes_judge():
    stub = StubClient({("inference", "classify_event"): {"ok": False, "error": "boom"}})
    update = FinalClassifier(client=stub).run(base_state_with_event())
    assert update["route"] == "judge"
    assert update["classification_output"]["attack_type"] is None
    assert update["errors"]


# ---------------------------------------------------------------------------
# mitigator (base determinista sobre el catalogo threat intel real)
# ---------------------------------------------------------------------------

def test_mitigator_catalog_ddos_references_and_mitigations():
    state = {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": True, "probability": 0.97},
        "classification_output": {
            "attack_type": "DDoS_TCP",
            "confidence": 0.95,
            "model_task": "attack_type",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        },
        "trace": [],
    }
    update = FinalMitigator(client=MCPToolClient(mode="inprocess")).run(state)
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


@pytest.mark.parametrize(
    ("attack_type", "expected_reference"),
    [
        ("DDoS_TCP", "CAPEC-482"),
        ("DoS", "T1499"),
        ("Command_and_Control", "T1071"),
    ],
)
def test_mitigator_uses_typed_catalog_contract(attack_type, expected_reference):
    state = {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": True, "probability": 0.97},
        "classification_output": {
            "attack_type": attack_type,
            "confidence": 0.95,
            "model_task": "attack_type",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        },
        "trace": [],
    }

    update = FinalMitigator(client=MCPToolClient(mode="inprocess")).run(state)
    output = update["explanation_output"]
    reference_ids = {
        reference.get("attack_id") or reference.get("capec_id")
        for reference in output["references"]
    }

    assert output["attack_type"] == attack_type
    assert "attack_family" not in output
    assert "attack_subtype" not in output
    assert output["taxonomy_version"] == MULTIDATASET_TAXONOMY_VERSION
    assert output["catalog_scope"] == "attack_type"
    assert output["catalog_version"] == "3.0"
    assert output["catalog_taxonomy_version"] == MULTIDATASET_TAXONOMY_VERSION
    assert set(output["catalog_compatible_taxonomy_versions"]) == {
        JORGE_TAXONOMY_VERSION,
        MULTIDATASET_TAXONOMY_VERSION,
    }
    assert (
        MULTIDATASET_TAXONOMY_VERSION
        in output["catalog_compatible_taxonomy_versions"]
    )
    assert expected_reference in reference_ids
    assert attack_type in output["risk_summary"]
    assert output["requires_human_review"] is False
    assert output["review_reasons"] == []


def test_mitigator_requires_attack_type_without_family_fallback():
    state = {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": True, "probability": 0.97},
        "classification_output": {
            "attack_family": "malware",
            "confidence": 0.95,
            "model_task": "attack_type",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        },
        "trace": [],
    }

    update = FinalMitigator(client=MCPToolClient(mode="inprocess")).run(state)
    output = update["explanation_output"]

    assert output["attack_type"] is None
    assert output["catalog_scope"] is None
    assert output["requires_human_review"] is True
    assert "classification_attack_type_missing" in output["review_reasons"]
    assert update["needs_human_review"] is True


def test_mitigator_benign_minimal_monitoring():
    state = {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": False, "probability": 0.02},
        "trace": [],
    }
    update = FinalMitigator(client=MCPToolClient(mode="inprocess")).run(state)
    output = update["explanation_output"]
    assert "benigno" in output["risk_summary"]
    assert output["requires_human_review"] is False
    # benigno: solo monitorizacion minima, sin contencion agresiva
    assert not any("aislar" in m or "bloquear" in m for m in output["mitigations"])


def test_mitigator_unknown_attack_type_requires_review():
    state = {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": True, "probability": 0.8},
        "classification_output": {
            "attack_type": "tipo_inexistente",
            "confidence": 0.4,
            "model_task": "attack_type",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        },
        "trace": [],
    }
    update = FinalMitigator(client=MCPToolClient(mode="inprocess")).run(state)
    assert update["explanation_output"]["requires_human_review"] is True
    assert update["needs_human_review"] is True


def test_mitigator_tool_error_falls_back_to_review():
    stub = StubClient({("threat_intel", "suggest_mitigations"): {"ok": False, "error": "x"}})
    state = {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": True, "probability": 0.9},
        "classification_output": {
            "attack_type": "DDoS_TCP",
            "confidence": 0.9,
            "model_task": "attack_type",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        },
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
        "classification_output": {
            "attack_type": "DDoS_TCP",
            "confidence": 0.95,
            "decision_threshold": 0.65,
            "model_task": "attack_type",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
            "top_scores": {
                "DDoS_TCP": 0.95,
                "DDoS_UDP": 0.03,
                "XSS": 0.02,
            },
        },
        "explanation_output": {
            "attack_type": "DDoS_TCP",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
            "catalog_scope": "attack_type",
            "catalog_version": "2.0",
            "catalog_taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
            "catalog_compatible_taxonomy_versions": [
                MULTIDATASET_TAXONOMY_VERSION,
            ],
            "requires_human_review": False,
        },
        "trace": [],
    }


def configure_typed_mitigation(
    state,
    *,
    attack_type="DDoS_TCP",
    taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
):
    alternatives = [
        label
        for label in ("DDoS_UDP", "XSS", "DoS", "Command_and_Control")
        if label != attack_type
    ][:2]
    state["classification_output"].update(
        {
            "attack_type": attack_type,
            "model_task": "attack_type",
            "taxonomy_version": taxonomy_version,
            "top_scores": {
                attack_type: state["classification_output"]["confidence"],
                alternatives[0]: 0.03,
                alternatives[1]: 0.02,
            },
        }
    )
    state["explanation_output"].update(
        {
            "attack_type": attack_type,
            "taxonomy_version": taxonomy_version,
            "catalog_scope": "attack_type",
            "catalog_version": "2.0",
            "catalog_taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
            "catalog_compatible_taxonomy_versions": [
                MULTIDATASET_TAXONOMY_VERSION,
                JORGE_TAXONOMY_VERSION,
            ],
        }
    )
    return state


def set_classifier_confidence(state, confidence):
    state["classification_output"]["confidence"] = confidence
    attack_type = state["classification_output"]["attack_type"]
    state["classification_output"]["top_scores"][attack_type] = confidence
    return state


def test_judge_approves_clean_case():
    update = FinalJudge().run(clean_state())
    output = update["judge_output"]
    assert output["action"] == "approve"
    assert output["approved"] is True
    assert output["final_label"] == "DDoS_TCP"
    assert update["needs_human_review"] is False
    assert update["route"] == "end"


def test_judge_accepts_coherent_attack_type():
    state = configure_typed_mitigation(clean_state())
    update = FinalJudge().run(state)
    assert update["judge_output"]["action"] == "approve"
    assert update["judge_output"]["final_label"] == "DDoS_TCP"
    assert update["judge_output"]["final_confidence"] == pytest.approx(0.95)


def test_judge_accepts_multidataset_dos_attack_type_contract():
    state = configure_typed_mitigation(
        clean_state(),
        attack_type="DoS",
        taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
    )
    update = FinalJudge().run(state)
    assert update["judge_output"]["action"] == "approve"
    assert update["judge_output"]["final_label"] == "DoS"


def test_judge_uses_persisted_attack_type_threshold_and_contract():
    state = configure_typed_mitigation(clean_state())
    state["classification_output"]["decision_threshold"] = 0.81
    set_classifier_confidence(state, 0.80)
    update = FinalJudge().run(state)
    assert "classification_confidence_below_review_threshold" in update[
        "judge_output"
    ]["issues"]
    assert update["judge_output"]["final_label"] == "DDoS_TCP"


@pytest.mark.parametrize(
    ("confidence", "expected_action"),
    [(0.6499, "human_interrupt"), (0.65, "approve")],
)
def test_judge_applies_operational_classifier_threshold_boundary(
    confidence, expected_action
):
    state = configure_typed_mitigation(clean_state())
    state["classification_output"]["decision_threshold"] = 0.65
    set_classifier_confidence(state, confidence)

    update = FinalJudge().run(state)

    assert update["judge_output"]["action"] == expected_action
    assert (
        "classification_confidence_below_review_threshold"
        in update["judge_output"]["issues"]
    ) is (confidence < 0.65)


def test_judge_rejects_incomplete_attack_type_model_contract():
    state = clean_state()
    state["classification_output"].update(
        {
            "attack_type": None,
            "model_task": "attack_type",
            "taxonomy_version": None,
            "decision_threshold": 0.81,
        }
    )
    update = FinalJudge().run(state)
    assert "classification_attack_type_missing" in update["judge_output"]["issues"]
    assert "classification_taxonomy_version_invalid" in update["judge_output"]["issues"]


def test_judge_flags_attack_type_outside_taxonomy():
    state = configure_typed_mitigation(
        clean_state(),
        attack_type="Novel_Attack",
    )
    update = FinalJudge().run(state)
    assert "classification_attack_type_outside_taxonomy" in update[
        "judge_output"
    ]["issues"]


@pytest.mark.parametrize(
    ("field", "value", "expected_issue"),
    [
        ("attack_type", "DDoS_UDP", "mitigation_attack_type_mismatch"),
        (
            "taxonomy_version",
            JORGE_TAXONOMY_VERSION,
            "mitigation_taxonomy_version_mismatch",
        ),
        ("catalog_scope", "family", "mitigation_catalog_scope_invalid"),
        ("catalog_version", None, "mitigation_catalog_version_missing"),
        (
            "catalog_compatible_taxonomy_versions",
            [JORGE_TAXONOMY_VERSION],
            "mitigation_catalog_taxonomy_version_mismatch",
        ),
    ],
)
def test_judge_flags_typed_mitigation_contract_mismatch(
    field, value, expected_issue
):
    state = configure_typed_mitigation(clean_state())
    state["explanation_output"][field] = value

    update = FinalJudge().run(state)

    assert update["judge_output"]["action"] == "human_interrupt"
    assert expected_issue in update["judge_output"]["issues"]
    assert update["needs_human_review"] is True


def test_judge_rejects_legacy_model_task():
    state = clean_state()
    state["classification_output"]["model_task"] = "attack_subtype"
    update = FinalJudge().run(state)
    assert "classification_model_task_invalid" in update["judge_output"]["issues"]
    assert update["judge_output"]["final_label"] == "DDoS_TCP"


def test_judge_flags_detector_abstention():
    state = clean_state()
    state["detection_output"]["abstain"] = True
    state.pop("classification_output")
    state.pop("explanation_output")
    update = FinalJudge().run(state)
    assert update["judge_output"]["action"] == "human_interrupt"
    assert "detector_abstained" in update["judge_output"]["issues"]
    assert "malicious_without_classification" not in update["judge_output"]["issues"]
    assert "malicious_without_mitigation" not in update["judge_output"]["issues"]
    assert update["judge_output"]["final_label"] is None
    assert update["judge_output"]["final_confidence"] == 0.0
    assert update["needs_human_review"] is True


def test_judge_flags_standardizer_abstention_without_detection():
    state = {
        "event_id": "unstandardized-iot23-7",
        "ingest_output": {
            "normalized": False,
            "mapping_confidence": 0.0,
            "abstain": True,
            "requires_human_review": True,
            "failure_code": "llm_standardization_failed",
        },
        "needs_human_review": True,
        "trace": [],
    }
    update = FinalJudge().run(state)
    assert update["judge_output"]["action"] == "human_interrupt"
    assert "standardizer_abstained" in update["judge_output"]["issues"]
    assert update["needs_human_review"] is True


def test_judge_flags_low_mapping_confidence():
    state = clean_state()
    state["ingest_output"]["mapping_confidence"] = 0.2
    update = FinalJudge().run(state)
    assert "mapping_confidence_below_review_threshold" in update["judge_output"]["issues"]


def test_judge_flags_low_classification_confidence():
    state = clean_state()
    set_classifier_confidence(state, 0.4)
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


def test_judge_rejects_attack_type_outside_taxonomy():
    state = clean_state()
    state["classification_output"]["attack_type"] = "benign"

    update = FinalJudge().run(state)
    assert "classification_attack_type_outside_taxonomy" in update[
        "judge_output"
    ]["issues"]


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


def test_judge_requires_review_when_detector_has_no_verdict_without_abstaining():
    state = clean_state()
    state["detection_output"] = {
        "is_malicious": None,
        "probability": 0.0,
        "abstain": False,
    }
    state.pop("classification_output")
    state.pop("explanation_output")

    update = FinalJudge().run(state)
    output = update["judge_output"]

    assert output["action"] == "human_interrupt"
    assert output["approved"] is False
    assert output["final_label"] is None
    assert output["final_confidence"] == 0.0
    assert "detection_verdict_missing" in output["issues"]


def test_judge_never_uses_attack_type_when_detector_says_benign():
    state = clean_state()
    state["detection_output"] = {
        "is_malicious": False,
        "probability": 0.1,
        "abstain": False,
    }

    update = FinalJudge().run(state)
    output = update["judge_output"]

    assert output["action"] == "human_interrupt"
    assert output["approved"] is False
    assert output["final_label"] == "benign"
    assert output["final_confidence"] == pytest.approx(0.9)
    assert "benign_with_classification" in output["issues"]


def test_judge_requires_the_deployed_multidataset_taxonomy():
    state = configure_typed_mitigation(
        clean_state(), taxonomy_version=JORGE_TAXONOMY_VERSION
    )

    update = FinalJudge().run(state)

    assert update["judge_output"]["action"] == "human_interrupt"
    assert "classification_taxonomy_version_invalid" in update["judge_output"][
        "issues"
    ]


@pytest.mark.parametrize(
    "top_scores",
    [
        {},
        {"DDoS_TCP": 0.95, "XSS": 0.05},
        {"DDoS_TCP": 0.95, "XSS": 0.03, "invented_attack": 0.02},
        {"DDoS_TCP": 0.95, "XSS": 0.03, "DoS": 1.2},
        {"DDoS_TCP": 0.80, "XSS": 0.10, "DoS": 0.10},
        {"DDoS_TCP": 0.40, "XSS": 0.50, "DoS": 0.10},
    ],
)
def test_judge_rejects_invalid_attack_type_top_three(top_scores):
    state = clean_state()
    state["classification_output"]["top_scores"] = top_scores

    update = FinalJudge().run(state)

    assert update["judge_output"]["action"] == "human_interrupt"
    assert "classification_top_scores_invalid" in update["judge_output"]["issues"]
