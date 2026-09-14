"""Una prueba integral por cada agente del sistema."""
from __future__ import annotations

import copy

import pytest

import scripts.run_system_audit as audit_script
from src.agents.final import (
    FinalClassifier,
    FinalDetector,
    FinalJudge,
    FinalMitigator,
    FinalStandardizer,
)
from src.agents.final.llm_mitigator import LLMMitigationAgent
from src.contracts.attack_taxonomy import MULTIDATASET_ATTACK_CLASSES
from src.contracts.case import CaseResult
from src.mcp.client import MCPToolClient
from tests.helpers import (
    CANONICAL_EVENT,
    StubClient,
    classification_success,
    complete_malicious_state,
    detection_success,
    malicious_state,
    standardization_success,
)


def test_standardizer_agent(tmp_path, monkeypatch):
    """Mistral en vivo, reutilizacion por cache y abstencion por fallo."""

    from src.mcp import case_memory_server

    monkeypatch.setenv("TFM_STATE_DIR", str(tmp_path))
    case_memory_server._CACHE_INSTANCES.clear()

    class CacheIntegrationClient:
        def __init__(self):
            self.real = MCPToolClient(mode="inprocess")
            self.inference_calls = 0

        def call(self, server, tool, **arguments):
            if (server, tool) == ("inference", "standardize_event"):
                self.inference_calls += 1
                return standardization_success(
                    dataset=arguments["dataset"],
                    source_file=arguments["source_file"],
                    row_id=arguments["row_id"],
                    split=arguments["split"],
                    selected_columns=["proto"],
                )
            return self.real.call(server, tool, **arguments)

    client = CacheIntegrationClient()
    first = FinalStandardizer(client=client).run(
        {
            "raw_input": {
                "dataset": "iot23",
                "row": {"proto": "tcp"},
                "source_file": "first.csv",
                "row_id": 1,
                "split": "train",
            },
            "trace": [],
        }
    )
    duplicate = FinalStandardizer(client=client).run(
        {
            "raw_input": {
                "dataset": "edge_iiotset",
                "row": {"proto": "tcp"},
                "source_file": "duplicate.csv",
                "row_id": 99,
                "split": "test",
            },
            "trace": [],
        }
    )

    assert first["route"] == duplicate["route"] == "detect"
    assert first["ingest_output"]["from_cache"] is False
    assert duplicate["ingest_output"]["from_cache"] is True
    assert duplicate["canonical_event"]["event_id"] == (
        "llm::edge_iiotset::duplicate.csv::99"
    )
    assert client.inference_calls == 1

    failed = FinalStandardizer(
        client=StubClient(
            {
                ("inference", "standardize_event"): {
                    "ok": False,
                    "failure_code": "llm_standardization_failed",
                    "error": "Mistral no disponible",
                }
            }
        )
    ).run(
        {
            "raw_input": {"dataset": "iot23", "row": {"proto": "tcp"}},
            "trace": [],
        }
    )
    assert failed["route"] == "judge"
    assert failed["ingest_output"]["abstain"] is True
    assert failed["needs_human_review"] is True
    assert failed["trace"][-1]["status"] == "abstain"
    judged_failure = FinalJudge().run(failed)
    assert judged_failure["judge_output"]["action"] == "human_interrupt"
    assert "standardizer_abstained" in judged_failure["judge_output"]["issues"]

    raw_input = {
        "dataset": "iot23",
        "row": {"proto": "tcp"},
        "source_file": "leak.csv",
        "row_id": 7,
        "split": "test",
    }
    leaky = standardization_success(
        dataset="iot23",
        source_file="leak.csv",
        row_id=7,
        split="test",
    )
    leaky["canonical_event"]["label_raw"] = "DDoS_TCP"
    rejected_leakage = FinalStandardizer(
        client=StubClient({("inference", "standardize_event"): leaky})
    ).run({"raw_input": raw_input, "trace": []})
    assert rejected_leakage["route"] == "judge"
    assert rejected_leakage["ingest_output"]["failure_code"] == (
        "invalid_standardization_response"
    )


def test_detector_agent():
    """Rutas maliciosa, benigna, zona gris y error controlado."""

    assert audit_script._target_feature_names(
        ["label_raw=DDoS_TCP", "attack_type.present", "transport_proto=tcp"]
    ) == ["label_raw=DDoS_TCP", "attack_type.present"]

    expected = {
        0.90: (True, False, "classify", "ok"),
        0.10: (False, False, "judge", "ok"),
        0.50: (True, True, "judge", "abstain"),
    }
    for probability, contract in expected.items():
        update = FinalDetector(
            client=StubClient(
                {("inference", "detect_event"): detection_success(probability)}
            )
        ).run({"canonical_event": copy.deepcopy(CANONICAL_EVENT), "trace": []})
        output = update["detection_output"]
        assert (
            output["is_malicious"],
            output["abstain"],
            update["route"],
            update["trace"][-1]["status"],
        ) == contract
        assert output["model_name"] == "xgboost_detection_final"

    failed_state = {
        "canonical_event": copy.deepcopy(CANONICAL_EVENT),
        "ingest_output": {"mapping_confidence": 0.95},
        "trace": [],
    }
    failed = FinalDetector(
        client=StubClient(
            {("inference", "detect_event"): {"ok": False, "error": "modelo ausente"}}
        )
    ).run(failed_state)
    assert failed["route"] == "judge"
    assert failed["detection_output"]["abstain"] is True
    assert failed["trace"][-1]["status"] == "error"
    assert failed["errors"]
    judged_failure = FinalJudge().run({**failed_state, **failed})
    assert judged_failure["judge_output"]["action"] == "human_interrupt"
    assert "pipeline_errors_present" in judged_failure["judge_output"]["issues"]

    audit = audit_script.audit_detector()
    assert audit["semaforo"] == audit_script.GREEN
    assert audit["artifact"] == "xgboost_detection_final.joblib"
    assert audit["validation_rows"] == 34_635
    assert audit["corpus_rows"] == 23_604
    assert audit["splits"] == {"train": 16_496, "validation": 3_536, "test": 3_572}
    assert audit["feature_count"] == 5_264
    assert audit["target_feature_count"] == 0


def test_classifier_agent():
    """Contrato de 16 tipos, top-3 y rechazo de una tarea antigua."""

    update = FinalClassifier(
        client=StubClient(
            {("inference", "classify_event"): classification_success()}
        )
    ).run({"canonical_event": copy.deepcopy(CANONICAL_EVENT), "trace": []})
    output = update["classification_output"]
    assert update["route"] == "explain"
    assert output["attack_type"] == "DDoS_TCP"
    assert output["confidence"] == pytest.approx(0.95)
    assert output["decision_threshold"] == pytest.approx(0.65)
    assert len(output["top_scores"]) == 3
    assert output["model_name"] == "xgboost_classification_final"
    assert not {"attack_family", "family_scores"} & output.keys()

    obsolete = classification_success()
    obsolete["model_task"] = "attack_family"
    rejected = FinalClassifier(
        client=StubClient({("inference", "classify_event"): obsolete})
    ).run({"canonical_event": copy.deepcopy(CANONICAL_EVENT), "trace": []})
    assert rejected["route"] == "judge"
    assert rejected["classification_output"]["attack_type"] is None
    assert any("tarea_clasificador_no_soportada" in error for error in rejected["errors"])

    audit = audit_script.audit_classifier()
    assert audit["semaforo"] == audit_script.GREEN
    assert audit["artifact"] == "xgboost_classification_final.joblib"
    assert audit["validation_rows"] == 34_635
    assert audit["corpus_rows"] == 8_000
    assert audit["per_class"] == 500
    assert audit["splits"] == {"train": 5_600, "val": 1_200, "test": 1_200}
    assert audit["feature_count"] == 2_742
    assert audit["target_feature_count"] == 0


def test_mitigator_agent(monkeypatch):
    """Catalogo 16/16, anclaje de Mistral y fallback sin perder la base."""

    from src.mcp import threat_intel_server

    client = MCPToolClient(mode="inprocess")
    for attack_type in MULTIDATASET_ATTACK_CLASSES:
        update = FinalMitigator(client=client).run(malicious_state(attack_type))
        output = update["explanation_output"]
        assert output["attack_type"] == attack_type
        assert output["catalog_scope"] == "attack_type"
        assert output["source"] == "catalog"
        assert len(output["mitigation_items"]) >= 5
        assert output["references"]
        assert output["requires_human_review"] is False
        assert all(item["source"] == "catalog" for item in output["mitigation_items"])

    class Backend:
        def __init__(self, payload=None, error=None):
            self.payload = payload
            self.error = error
            self.calls = 0

        async def invoke_json(self, system_prompt, user_payload, json_schema):
            del system_prompt, user_payload, json_schema
            self.calls += 1
            if self.error:
                raise self.error
            return copy.deepcopy(self.payload) if self.payload is not None else {
                "risk_summary": "DDoS TCP contextualizado para el flujo observado.",
                "mitigations": [
                    {"base_id": None, "text": "recomendacion adicional"},
                    {"base_id": 5, "text": "contexto valido 5"},
                    {"base_id": 3, "text": "contexto valido 3"},
                    {"base_id": 1, "text": "contexto valido 1"},
                    {"base_id": 4, "text": "contexto valido 4"},
                    {"base_id": 2, "text": "contexto valido 2"},
                ],
                "confidence": 0.90,
                "requires_human_review": True,
            }

    backend = Backend()
    contextualizer = LLMMitigationAgent(
        model="mistral-small-2603",
        provider="mistral",
    )
    contextualizer.agent = backend
    monkeypatch.setattr(
        threat_intel_server,
        "_build_mitigation_llm",
        lambda: contextualizer,
    )
    contextualized = FinalMitigator(
        client=client,
        contextualize_with_llm=True,
    ).run(malicious_state())
    output = contextualized["explanation_output"]
    assert backend.calls == 1
    assert output["source"] == "hybrid"
    assert output["model_name"] == "llm_mitigator::mistral::mistral-small-2603"
    assert output["first_five_catalog_anchored"] is True
    assert output["has_llm_suggested"] is True
    assert output["requires_human_review"] is False
    assert output["review_reasons"] == []
    suggestions = [
        item
        for item in output["mitigation_items"]
        if item["source"] == "llm_suggested"
    ]
    assert suggestions == [
        {
            "base_id": None,
            "text": "recomendacion adicional",
            "phase": None,
            "source": "llm_suggested",
            "base": None,
            "context": None,
            "context_trusted": False,
        }
    ]
    assert "recomendacion adicional" in output["mitigations"]

    state_after_mitigator = {**malicious_state(), **contextualized}
    judged = FinalJudge().run(state_after_mitigator)
    assert judged["judge_output"]["action"] == "approve"
    case = CaseResult.from_orchestrator_state(
        {**state_after_mitigator, **judged},
        case_id="case-llm-suggested-roundtrip",
    )
    assert case.status == "completed"
    assert case.explanation.has_llm_suggested is True
    assert any(
        item.source == "llm_suggested"
        and item.text == "recomendacion adicional"
        for item in case.explanation.mitigation_items
    )

    mapping_review_state = {**state_after_mitigator}
    mapping_review_state["ingest_output"] = {"mapping_confidence": 0.49}
    reviewed_for_mapping = FinalJudge().run(mapping_review_state)
    assert reviewed_for_mapping["judge_output"]["action"] == "human_interrupt"
    assert "mapping_confidence_below_review_threshold" in reviewed_for_mapping[
        "judge_output"
    ]["issues"]
    assert "mitigator_requested_human_review" not in reviewed_for_mapping[
        "judge_output"
    ]["issues"]

    incomplete_payload = {
        "risk_summary": "C2 contextualizado para el evento.",
        "mitigations": [
            {"base_id": index, "text": f"contexto valido {index}"}
            for index in range(2, 7)
        ],
        "confidence": 0.90,
        "requires_human_review": True,
    }
    incomplete_backend = Backend(payload=incomplete_payload)
    incomplete_contextualizer = LLMMitigationAgent(
        model="mistral-small-2603",
        provider="mistral",
    )
    incomplete_contextualizer.agent = incomplete_backend
    monkeypatch.setattr(
        threat_intel_server,
        "_build_mitigation_llm",
        lambda: incomplete_contextualizer,
    )
    incomplete = FinalMitigator(
        client=client,
        contextualize_with_llm=True,
    ).run(malicious_state("Command_and_Control"))
    incomplete_output = incomplete["explanation_output"]
    assert sum(
        item["source"] == "llm"
        for item in incomplete_output["mitigation_items"]
    ) == 5
    assert incomplete_output["first_five_catalog_anchored"] is False
    assert incomplete_output["requires_human_review"] is True
    assert incomplete_output["review_reasons"] == ["llm_requested_human_review"]
    assert incomplete["needs_human_review"] is True

    unavailable = LLMMitigationAgent(
        model="mistral-small-2603",
        provider="mistral",
    )
    unavailable.agent = Backend(error=TimeoutError("Mistral no disponible"))
    monkeypatch.setattr(
        threat_intel_server,
        "_build_mitigation_llm",
        lambda: unavailable,
    )
    fallback = FinalMitigator(
        client=client,
        contextualize_with_llm=True,
    ).run(malicious_state())
    assert fallback["explanation_output"]["source"] == "catalog"
    assert fallback["needs_human_review"] is False
    assert fallback["trace"][-1]["status"] == "fallback"

    audit = audit_script.audit_catalog()
    assert audit["semaforo"] == audit_script.GREEN
    assert audit["artifact"] == "threat_intel_catalog.json"
    assert audit["version"] == "4.0"
    assert audit["covered_attack_types"] == 16
    assert set(audit["mitigations_per_attack_type"]) == set(
        MULTIDATASET_ATTACK_CLASSES
    )
    assert min(audit["mitigations_per_attack_type"].values()) >= 5


def test_judge_agent():
    """Aprobacion, frontera 0.65, benigno y abstencion operacional."""

    approved = FinalJudge().run(complete_malicious_state())
    assert approved["judge_output"]["action"] == "approve"
    assert approved["judge_output"]["final_label"] == "DDoS_TCP"
    assert approved["needs_human_review"] is False

    low_confidence = complete_malicious_state(confidence=0.6499)
    low_confidence["classification_output"]["top_scores"]["DDoS_TCP"] = 0.6499
    reviewed = FinalJudge().run(low_confidence)
    assert reviewed["judge_output"]["action"] == "human_interrupt"
    assert "classification_confidence_below_review_threshold" in reviewed[
        "judge_output"
    ]["issues"]

    benign = FinalJudge().run(
        {
            "canonical_event": copy.deepcopy(CANONICAL_EVENT),
            "ingest_output": {"mapping_confidence": 0.95},
            "detection_output": {
                "is_malicious": False,
                "probability": 0.03,
                "abstain": False,
            },
            "trace": [],
        }
    )
    assert benign["judge_output"]["action"] == "approve"
    assert benign["judge_output"]["final_label"] == "benign"

    abstention = FinalJudge().run(
        {
            "canonical_event": copy.deepcopy(CANONICAL_EVENT),
            "ingest_output": {"mapping_confidence": 0.95},
            "detection_output": {
                "is_malicious": True,
                "probability": 0.50,
                "abstain": True,
            },
            "trace": [],
        }
    )
    assert abstention["judge_output"]["action"] == "human_interrupt"
    assert abstention["judge_output"]["final_label"] is None
    assert "detector_abstained" in abstention["judge_output"]["issues"]

    pipeline_error = complete_malicious_state()
    pipeline_error["errors"] = ["final_detector: fallo controlado"]
    reviewed_error = FinalJudge().run(pipeline_error)
    assert reviewed_error["judge_output"]["action"] == "human_interrupt"
    assert "pipeline_errors_present" in reviewed_error["judge_output"]["issues"]


def test_auditor_agent():
    """Control limpio y las 16 mutaciones del contrato del auditor."""

    result = audit_script.audit_auditor()
    assert result["semaforo"] == audit_script.GREEN
    assert result["mutations"] == result["detected"] == 16
    assert result["false_rejects"] == 0
    assert all(row["detected"] for row in result["results"])
    leakage = next(
        row for row in result["results"] if row["id"] == "canonical_target_leakage"
    )
    assert leakage["expected_check"] == "leakage_evento_canonico"
    assert leakage["verdict"] == "reject"
