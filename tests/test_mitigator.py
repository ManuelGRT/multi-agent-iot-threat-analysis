# tests/test_mitigator.py
"""Tests del agente mitigador: catalogo y LLM anclado.

Criterios de aceptacion del plan:
(a) modo determinista devuelve mitigaciones y referencias correctas por tipo;
(b) modo LLM (mockeado) nunca produce referencias fuera del catalogo sin
    marcarlas ``llm_suggested``;
(c) benign -> mitigacion no aplicable, sin consultar el catalogo.

Ademas, comprueba que la estructura y la version del catalogo operativo sean
validas antes de exponerlo al mitigador.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.agents.final import FinalJudge, FinalMitigator
from src.agents.final.llm_mitigator import (
    LLMMitigationAgent,
    anchor_llm_payload,
    build_base_items,
    catalog_reference_ids,
    event_context,
)
from src.contracts.attack_taxonomy import (
    MULTIDATASET_ATTACK_CLASSES,
    MULTIDATASET_TAXONOMY_VERSION,
)
from src.contracts.case import CaseResult
from src.mcp.client import MCPToolClient
from src.mcp.threat_catalog import (
    THREAT_INTEL_CATALOG_VERSION,
    catalog,
    validate_catalog,
)
from tests.test_final_agents import CANONICAL_EVENT, StubClient

client = MCPToolClient(mode="inprocess")

CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "mcp" / "data" / "threat_intel_catalog.json"
)
CATALOG = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

EXPECTED_TYPE_REFERENCE_IDS = {
    "Backdoor": {"T1105", "T1204", "CAPEC-523", "M1049"},
    "DDoS_HTTP": {"T1498", "CAPEC-125", "CAPEC-488", "M1037"},
    "DDoS_ICMP": {"T1498", "CAPEC-125", "CAPEC-487", "M1037"},
    "DDoS_TCP": {"T1498", "CAPEC-125", "CAPEC-482", "M1037"},
    "DDoS_UDP": {"T1498", "CAPEC-125", "CAPEC-486", "M1037"},
    "Fingerprinting": {"T1595", "CAPEC-224", "M1042"},
    "MITM": {"T1557", "CAPEC-94", "M1041"},
    "Password": {"T1110", "CAPEC-112", "CAPEC-49", "M1032", "M1027"},
    "Port_Scanning": {"T1046", "CAPEC-300", "M1042"},
    "Ransomware": {"T1105", "T1204", "CAPEC-542", "M1049"},
    "SQL_injection": {"T1190", "CAPEC-66", "M1051"},
    "Uploading": {"T1190", "CAPEC-242", "M1051"},
    "Vulnerability_scanner": {"T1595", "CAPEC-310", "M1042"},
    "XSS": {"T1190", "CAPEC-63", "M1051"},
    "DoS": {"T1499", "CAPEC-125", "M1037"},
    "Command_and_Control": {"T1071", "CAPEC-542", "M1031"},
}


def malicious_state(
    attack_type: str = "DDoS_TCP", confidence: float = 0.95
) -> dict:
    alternatives = [
        label for label in MULTIDATASET_ATTACK_CLASSES if label != attack_type
    ][:2]
    remaining = max(0.0, 1.0 - confidence)
    top_scores = {
        attack_type: confidence,
        alternatives[0]: remaining * 0.6,
        alternatives[1]: remaining * 0.4,
    }
    return {
        "canonical_event": dict(CANONICAL_EVENT),
        "ingest_output": {"mapping_confidence": 0.95},
        "detection_output": {"is_malicious": True, "probability": 0.97},
        "classification_output": {
            "attack_type": attack_type,
            "confidence": confidence,
            "decision_threshold": 0.65,
            "model_task": "attack_type",
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
            "top_scores": top_scores,
        },
        "trace": [],
    }


def malicious_attack_type_state(
    attack_type: str, confidence: float = 0.95
) -> dict:
    """Alias descriptivo conservado para las pruebas específicas por tipo."""

    return malicious_state(attack_type, confidence)


def flattened_reference_ids(output: dict) -> set[str]:
    return {
        str(reference.get("attack_id") or reference.get("capec_id"))
        for reference in output["references"]
        if reference.get("attack_id") or reference.get("capec_id")
    }


class StubMitigationBackend:
    """Backend LLM de pruebas: registra llamadas y devuelve payload o error."""

    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls: list[dict] = []

    async def invoke_json(self, system_prompt, user_payload, json_schema):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_payload": user_payload,
                "json_schema": json_schema,
            }
        )
        if self.error:
            raise self.error
        return self.payload


def llm_with_stub(payload: dict | None = None, error: Exception | None = None) -> LLMMitigationAgent:
    agent = LLMMitigationAgent(model="test-model", provider="ollama")
    agent.agent = StubMitigationBackend(payload=payload, error=error)
    return agent


# ---------------------------------------------------------------------------
# (a) modo determinista: mitigaciones y referencias correctas por tipo
# ---------------------------------------------------------------------------

def test_catalog_v3_has_no_broad_family_fallback_data():
    assert CATALOG["version"] == "3.0"
    assert "families" not in CATALOG
    assert set(CATALOG["attack_types"]) == set(MULTIDATASET_ATTACK_CLASSES)
    assert all("family" not in entry for entry in CATALOG["attack_types"].values())


@pytest.mark.parametrize("attack_type", MULTIDATASET_ATTACK_CLASSES)
def test_catalog_mode_is_specific_and_complete_for_all_16_attack_types(attack_type):
    stub = StubClient()
    state = malicious_attack_type_state(attack_type)
    update = FinalMitigator(client=stub).run(state)
    output = update["explanation_output"]
    entry = CATALOG["attack_types"][attack_type]

    expected_mitigations = [
        *entry["mitigations"]["containment"],
        *entry["mitigations"]["eradication"],
        *entry["mitigations"]["prevention"],
    ]
    assert len(expected_mitigations) >= 5
    assert output["mitigations"][: len(expected_mitigations)] == expected_mitigations
    assert flattened_reference_ids(output) == EXPECTED_TYPE_REFERENCE_IDS[attack_type]
    assert all(item["source"] == "catalog" for item in output["mitigation_items"])
    assert all(ref["source"] == "catalog" for ref in output["references"])

    assert output["source"] == "catalog"
    assert output["attack_type"] == attack_type
    assert "attack_family" not in output
    assert "attack_subtype" not in output
    assert output["catalog_scope"] == "attack_type"
    assert output["catalog_version"] == CATALOG["version"]
    assert output["catalog_taxonomy_version"] == MULTIDATASET_TAXONOMY_VERSION
    assert output["requires_human_review"] is False
    assert output["review_reasons"] == []
    assert attack_type in output["risk_summary"]

    threat_call = next(
        arguments
        for server, tool, arguments in stub.call_arguments
        if (server, tool) == ("threat_intel", "suggest_mitigations")
    )
    assert threat_call["attack_type"] == attack_type
    assert "family" not in threat_call


def test_catalog_mode_missing_attack_type_requires_review_without_tool_call():
    stub = StubClient()
    state = malicious_state()
    state["classification_output"]["attack_type"] = None

    update = FinalMitigator(client=stub).run(state)
    output = update["explanation_output"]

    assert output["attack_type"] is None
    assert output["requires_human_review"] is True
    assert output["review_reasons"] == ["classification_attack_type_missing"]
    assert update["needs_human_review"] is True
    assert not any(
        server == "threat_intel" and tool == "suggest_mitigations"
        for server, tool, _ in stub.call_arguments
    )


def test_explicit_unknown_attack_type_never_falls_back_to_broad_family():
    state = malicious_attack_type_state("DDoS_TCP")
    state["classification_output"]["attack_type"] = "DDoS_QUIC"

    update = FinalMitigator(client=client).run(state)
    output = update["explanation_output"]

    assert output["attack_type"] == "DDoS_QUIC"
    assert output["catalog_scope"] is None
    assert output["requires_human_review"] is True
    assert output["review_reasons"] == ["catalog_attack_type_unavailable"]
    assert output["mitigations"] == ["escalate_to_analyst"]


@pytest.mark.parametrize("invalid_type", ["ddos", "DDoS TCP", "C2", "unknown"])
def test_catalog_rejects_family_names_and_aliases(invalid_type):
    result = client.call(
        "threat_intel", "suggest_mitigations", attack_type=invalid_type
    )
    assert result["ok"] is False
    assert "fuera de la taxonomia operativa" in result["error"]


# ---------------------------------------------------------------------------
# (c) benign -> monitorizacion minima
# ---------------------------------------------------------------------------

def test_benign_gets_minimal_monitoring_only():
    state = {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": False, "probability": 0.02},
        "trace": [],
    }
    update = FinalMitigator(client=client).run(state)
    output = update["explanation_output"]
    assert "benigno" in output["risk_summary"]
    assert output["mitigations"] == []
    assert output["references"] == []
    assert output["attack_type"] is None
    assert output["source"] == "rule_based"
    assert output["requires_human_review"] is False
    assert not any(entry.get("tool") == "suggest_mitigations" for entry in update["trace"])


def test_benign_never_calls_llm_even_if_configured():
    llm = llm_with_stub(payload={"risk_summary": "x", "mitigations": [], "confidence": 0.9, "requires_human_review": False})
    state = {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": False, "probability": 0.02},
        "trace": [],
    }
    update = FinalMitigator(client=client, llm=llm).run(state)
    assert llm.agent.calls == []
    assert update["explanation_output"]["source"] == "rule_based"


# ---------------------------------------------------------------------------
# (b) modo LLM anclado: nada fuera del catalogo sin marca llm_suggested
# ---------------------------------------------------------------------------

def hybrid_payload_ok() -> dict:
    return {
        "risk_summary": "Flujo tcp 45312->80 con tasa alta de paquetes: DDoS dirigido al servicio web.",
        "mitigations": [
            {"base_id": 1, "text": "aplicar rate limiting al origen 192.168.100.5 sobre el puerto 80"},
            {"base_id": 2, "text": "activar scrubbing DDoS upstream para el segmento del servidor web"},
        ],
        "confidence": 0.9,
        "requires_human_review": False,
    }


def five_catalog_anchored_payload(
    *, attack_type: str = "DDoS", requires_human_review: bool = False
) -> dict:
    return {
        "risk_summary": (
            f"{attack_type} contextualizado con cinco medidas del catalogo."
        ),
        "mitigations": [
            {"base_id": index, "text": f"contextualizacion catalogada {index}"}
            for index in range(1, 6)
        ],
        "confidence": 0.9,
        "requires_human_review": requires_human_review,
    }


def test_llm_contextualization_produces_hybrid_anchored_output():
    llm = llm_with_stub(payload=hybrid_payload_ok())
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("DDoS_TCP"))
    output = update["explanation_output"]

    assert output["source"] == "hybrid"
    assert output["model_name"] == "llm_mitigator::ollama::test-model"
    assert "45312->80" in output["llm_context_summary"]
    assert output["llm_context_trusted"] is False

    items = output["mitigation_items"]
    contextualized = [item for item in items if item["source"] == "llm"]
    assert len(contextualized) == 2
    assert all(item["text"] == item["base"] for item in contextualized)
    assert all(item["context"] and item["context_trusted"] is False for item in contextualized)

    # las mitigaciones del catalogo no cubiertas por el LLM entran literales
    typed_entry = CATALOG["attack_types"]["DDoS_TCP"]["mitigations"]
    for text in [*typed_entry["eradication"], *typed_entry["prevention"]]:
        assert any(item["text"] == text and item["source"] == "catalog" for item in items)

    # referencias: solo catalogo, todas marcadas catalog
    assert all(ref["source"] == "catalog" for ref in output["references"])


@pytest.mark.parametrize("attack_type", MULTIDATASET_ATTACK_CLASSES)
def test_all_16_attack_types_are_contextualized_and_approved_when_anchored(
    attack_type,
):
    stub = StubClient()
    llm = llm_with_stub(
        payload=five_catalog_anchored_payload(
            attack_type=attack_type,
            # La politica puede ignorar esta peticion generica solo cuando las
            # cinco primeras recomendaciones estan realmente ancladas.
            requires_human_review=True,
        )
    )
    state = malicious_attack_type_state(attack_type)
    update = FinalMitigator(client=stub, llm=llm).run(state)
    output = update["explanation_output"]

    assert output["source"] == "hybrid"
    assert output["attack_type"] == attack_type
    assert "attack_family" not in output
    assert "attack_subtype" not in output
    assert output["catalog_scope"] == "attack_type"
    assert output["first_five_catalog_anchored"] is True
    assert output.get("has_llm_suggested", False) is False
    assert output["requires_human_review"] is False
    assert output["review_reasons"] == []
    assert [
        item["source"] for item in output["mitigation_items"][:5]
    ] == ["llm"] * 5
    assert flattened_reference_ids(output) == EXPECTED_TYPE_REFERENCE_IDS[attack_type]
    assert all(reference["source"] == "catalog" for reference in output["references"])

    sent = llm.agent.calls[0]["user_payload"]
    assert sent["classification"]["attack_type"] == attack_type
    assert "attack_family" not in sent["classification"]
    assert "attack_subtype" not in sent["classification"]
    assert (
        sent["classification"]["taxonomy_version"]
        == MULTIDATASET_TAXONOMY_VERSION
    )
    assert sent["catalog"]["attack_type"] == attack_type
    assert sent["catalog"]["catalog_scope"] == "attack_type"
    assert sent["catalog"]["taxonomy_version"] == MULTIDATASET_TAXONOMY_VERSION

    judged = FinalJudge(client=stub).run({**state, **update})
    assert judged["judge_output"]["action"] == "approve"
    assert judged["judge_output"]["final_label"] == attack_type
    assert "mitigator_requested_human_review" not in judged["judge_output"]["issues"]
    assert judged["needs_human_review"] is False


def test_first_five_catalog_anchored_allow_judge_approval_with_later_suggestion():
    payload = five_catalog_anchored_payload(requires_human_review=True)
    payload["mitigations"].append(
        {"base_id": None, "text": "sexta recomendacion adicional sin respaldo"}
    )
    update = FinalMitigator(
        client=client, llm=llm_with_stub(payload=payload)
    ).run(malicious_state("DDoS_TCP"))
    output = update["explanation_output"]

    assert output["first_five_catalog_anchored"] is True
    assert output["has_llm_suggested"] is True
    assert output["requires_human_review"] is False
    assert output["review_reasons"] == []
    assert any(
        item["text"] == "sexta recomendacion adicional sin respaldo"
        and item["source"] == "llm_suggested"
        for item in output["mitigation_items"]
    )

    judge_state = malicious_state("DDoS_TCP")
    judge_state["ingest_output"] = {"mapping_confidence": 0.95}
    judge_state.update(update)
    judged = FinalJudge(client=client).run(judge_state)
    assert judged["judge_output"]["action"] == "approve"
    assert "mitigator_requested_human_review" not in judged["judge_output"]["issues"]
    assert judged["needs_human_review"] is False

    case = CaseResult.from_orchestrator_state({**judge_state, **judged})
    assert case.status == "completed"
    assert case.explanation.first_five_catalog_anchored is True
    assert case.explanation.has_llm_suggested is True
    assert case.explanation.review_reasons == []


def test_unanchored_fifth_recommendation_still_requires_human_review():
    payload = five_catalog_anchored_payload()
    payload["mitigations"][4] = {
        "base_id": None,
        "text": "quinta recomendacion sin respaldo",
    }
    update = FinalMitigator(
        client=client, llm=llm_with_stub(payload=payload)
    ).run(malicious_state("DDoS_TCP"))
    output = update["explanation_output"]

    assert output["first_five_catalog_anchored"] is False
    assert output["requires_human_review"] is True
    assert "unanchored_mitigation_in_review_window" in output["review_reasons"]


def test_fewer_than_five_do_not_override_explicit_llm_review_request():
    payload = five_catalog_anchored_payload(requires_human_review=True)
    payload["mitigations"] = payload["mitigations"][:4]
    update = FinalMitigator(
        client=client, llm=llm_with_stub(payload=payload)
    ).run(malicious_state("DDoS_TCP"))
    output = update["explanation_output"]

    assert output["first_five_catalog_anchored"] is False
    assert output["requires_human_review"] is True
    assert "llm_requested_human_review" in output["review_reasons"]


def test_first_five_anchored_do_not_hide_untrusted_additional_reference():
    payload = five_catalog_anchored_payload(requires_human_review=True)
    payload["additional_references"] = [
        {"attack_id": "T9999", "name": "Referencia no catalogada"}
    ]
    update = FinalMitigator(
        client=client, llm=llm_with_stub(payload=payload)
    ).run(malicious_state("DDoS_TCP"))
    output = update["explanation_output"]

    assert output["first_five_catalog_anchored"] is True
    assert output["requires_human_review"] is True
    assert "llm_requested_human_review" not in output["review_reasons"]
    assert "llm_reference_not_in_catalog" in output["review_reasons"]


def test_llm_cannot_smuggle_references_outside_catalog():
    payload = hybrid_payload_ok()
    payload["additional_references"] = [
        {"attack_id": "T9999", "name": "Tecnica Inventada", "url": "https://example.com/fake"},
        {"capec_id": "CAPEC-99999", "name": "Patron Inventado"},
        {"attack_id": "T1498", "name": "Network Denial of Service (duplicada)"},
    ]
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("DDoS_TCP"))
    references = update["explanation_output"]["references"]

    by_id = {(ref.get("attack_id") or ref.get("capec_id")): ref for ref in references}
    # inventadas: presentes SOLO como llm_suggested
    assert by_id["T9999"]["source"] == "llm_suggested"
    assert by_id["CAPEC-99999"]["source"] == "llm_suggested"
    # la duplicada del catalogo NO se degrada: se conserva la del catalogo
    assert by_id["T1498"]["source"] == "catalog"
    assert sum(1 for ref in references if (ref.get("attack_id") == "T1498")) == 1
    # y el resto del catalogo sigue intacto
    catalog_ids = catalog_reference_ids(
        client.call(
            "threat_intel", "suggest_mitigations", attack_type="DDoS_TCP"
        )["references"]
    )
    present = {ref.get("attack_id") or ref.get("capec_id") for ref in references}
    assert catalog_ids <= present


def test_llm_free_mitigations_are_marked_llm_suggested():
    payload = hybrid_payload_ok()
    payload["mitigations"].append({"base_id": None, "text": "instalar honeypot en la DMZ"})
    payload["mitigations"].append({"text": "regla extra sin base_id"})
    payload["mitigations"].append({"base_id": 999, "text": "base_id inexistente"})
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("DDoS_TCP"))
    items = update["explanation_output"]["mitigation_items"]

    suggested = {item["text"] for item in items if item["source"] == "llm_suggested"}
    assert "instalar honeypot en la DMZ" in suggested
    assert "regla extra sin base_id" in suggested
    assert "base_id inexistente" in suggested
    assert update["explanation_output"]["has_llm_suggested"] is True
    assert update["explanation_output"]["requires_human_review"] is True


def test_llm_failure_falls_back_to_catalog_completely():
    llm = llm_with_stub(error=TimeoutError("llm caido"))
    with_llm = FinalMitigator(client=client, llm=llm).run(malicious_state("DDoS_TCP"))
    without_llm = FinalMitigator(client=client).run(malicious_state("DDoS_TCP"))

    assert with_llm["explanation_output"]["source"] == "catalog"
    assert (
        with_llm["explanation_output"]["mitigations"]
        == without_llm["explanation_output"]["mitigations"]
    )
    assert with_llm["explanation_output"]["references"] == without_llm["explanation_output"]["references"]
    # la traza registra el fallback sin romper el caso
    fallback_entries = [
        entry for entry in with_llm["trace"] if entry.get("status") == "fallback"
    ]
    assert len(fallback_entries) == 1
    assert "TimeoutError" in fallback_entries[0]["error"]
    assert with_llm["needs_human_review"] is False


def test_llm_payload_has_no_target_leakage_nor_provenance():
    llm = llm_with_stub(payload=hybrid_payload_ok())
    state = malicious_state("DDoS_TCP")
    # inyectamos deliberadamente campos prohibidos: el allowlist debe filtrarlos
    state["canonical_event"]["origin"] = {"source_name": "EDGE_IIOTSET"}
    state["canonical_event"]["provenance"] = {"dataset": "EDGE_IIOTSET", "row_id": 4}
    state["canonical_event"]["label_raw"] = "DDoS_UDP"
    state["canonical_event"]["attack_family"] = "ddos"
    state["canonical_event"]["attack_subtype"] = "udp_flood"
    FinalMitigator(client=client, llm=llm).run(state)

    sent = llm.agent.calls[0]["user_payload"]
    assert "origin" not in sent["event"]
    assert "provenance" not in sent["event"]
    assert "label_raw" not in sent["event"]
    assert "attack_family" not in sent["event"]
    assert "attack_subtype" not in sent["event"]
    # el catalogo numerado si viaja al LLM (es su ancla)
    assert sent["catalog"]["base_mitigations"]
    assert all("id" in item for item in sent["catalog"]["base_mitigations"])


def test_llm_trace_records_both_steps():
    llm = llm_with_stub(payload=hybrid_payload_ok())
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("DDoS_TCP"))
    tools = [entry.get("tool") for entry in update["trace"]]
    assert "suggest_mitigations" in tools
    assert "llm_contextualize" in tools


# ---------------------------------------------------------------------------
# (b) endurecido: contrabando por texto libre y tipos crudos (review F4)
# ---------------------------------------------------------------------------

def test_llm_risk_summary_with_invented_ids_is_discarded():
    payload = hybrid_payload_ok()
    payload["risk_summary"] = "Aplicar la tecnica T1337 y el patron CAPEC-777 cuanto antes."
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("DDoS_TCP"))
    output = update["explanation_output"]

    assert "T1337" not in output["risk_summary"]
    assert "CAPEC-777" not in output["risk_summary"]
    assert output.get("llm_context_summary") is None
    assert output["has_llm_suggested"] is True
    assert output["requires_human_review"] is True
    assert any("risk_summary_llm_descartado" in note for note in output["evidence"])


def test_llm_item_text_with_invented_id_is_downgraded():
    payload = hybrid_payload_ok()
    payload["mitigations"] = [
        {"base_id": 1, "text": "bloquear conforme a la tecnica T4242 el origen"},
    ]
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("DDoS_TCP"))
    items = update["explanation_output"]["mitigation_items"]

    downgraded = next(item for item in items if "T4242" in item["text"])
    assert downgraded["source"] == "llm_suggested"
    # la mitigacion base del catalogo reentra literal como catalog
    ddos_containment = CATALOG["attack_types"]["DDoS_TCP"]["mitigations"][
        "containment"
    ][0]
    assert any(
        item["text"] == ddos_containment and item["source"] == "catalog" for item in items
    )
    assert any("mitigacion_llm_degradada" in n for n in update["explanation_output"]["evidence"])


def test_llm_item_citing_catalog_ids_is_not_downgraded():
    payload = hybrid_payload_ok()
    payload["mitigations"] = [
        {"base_id": 4, "text": "M1037 Filter Network Traffic aplicado al segmento del sensor"},
    ]
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("DDoS_TCP"))
    items = update["explanation_output"]["mitigation_items"]
    contextualized = next(item for item in items if "segmento del sensor" in item["context"])
    # M1037 esta en el catalogo de ddos: la contextualizacion es legitima
    assert contextualized["source"] == "llm"


def test_reference_from_sibling_attack_type_is_not_treated_as_catalog_anchored():
    payload = five_catalog_anchored_payload(attack_type="DDoS_HTTP")
    payload["mitigations"][0]["text"] = (
        "aplicar CAPEC-486, que corresponde a UDP, a este supuesto HTTP"
    )
    update = FinalMitigator(
        client=client,
        llm=llm_with_stub(payload=payload),
    ).run(malicious_attack_type_state("DDoS_HTTP"))
    output = update["explanation_output"]

    crossed = next(
        item
        for item in output["mitigation_items"]
        if "CAPEC-486" in item["text"]
    )
    assert crossed["source"] == "llm_suggested"
    assert "CAPEC-486" not in flattened_reference_ids(output)
    assert output["first_five_catalog_anchored"] is False
    assert output["requires_human_review"] is True
    assert "unanchored_mitigation_in_review_window" in output["review_reasons"]


@pytest.mark.parametrize("foreign_reference", ["capec-486", "t9999", "m9999"])
def test_lowercase_reference_cannot_bypass_catalog_anchor(foreign_reference):
    payload = five_catalog_anchored_payload(attack_type="DDoS_HTTP")
    payload["mitigations"][0]["text"] = (
        f"aplicar la referencia ajena {foreign_reference} al evento HTTP"
    )
    output = FinalMitigator(
        client=client,
        llm=llm_with_stub(payload=payload),
    ).run(malicious_attack_type_state("DDoS_HTTP"))["explanation_output"]

    crossed = next(
        item
        for item in output["mitigation_items"]
        if foreign_reference in item["text"]
    )
    assert crossed["source"] == "llm_suggested"
    assert output["first_five_catalog_anchored"] is False
    assert output["requires_human_review"] is True
    assert "unanchored_mitigation_in_review_window" in output["review_reasons"]


def test_llm_raw_typed_references_do_not_crash_case_result():
    payload = hybrid_payload_ok()
    payload["additional_references"] = [
        {"attack_id": 12345, "name": ["lista", "rara"], "url": 99},
        {"capec_id": {"objeto": "raro"}},
        "no soy un dict",
    ]
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("DDoS_TCP"))

    state = malicious_state("DDoS_TCP")
    state["explanation_output"] = update["explanation_output"]
    state["judge_output"] = {"action": "approve", "approved": True}
    case = CaseResult.from_orchestrator_state(state)  # no debe lanzar
    coerced = next(ref for ref in case.explanation.references if ref.attack_id == "12345")
    assert coerced.source == "llm_suggested"
    assert coerced.name is None  # la lista cruda no se cuela como nombre


def test_llm_nan_confidence_falls_back_without_breaking_case():
    payload = hybrid_payload_ok()
    payload["confidence"] = float("nan")
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("DDoS_TCP"))
    output = update["explanation_output"]
    assert output["source"] == "hybrid"
    assert 0.0 <= output["confidence"] <= 1.0


def test_llm_bool_base_id_is_not_anchored():
    payload = hybrid_payload_ok()
    payload["mitigations"] = [{"base_id": True, "text": "colada booleana"}]
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("DDoS_TCP"))
    item = next(
        i for i in update["explanation_output"]["mitigation_items"]
        if i["text"] == "colada booleana"
    )
    assert item["source"] == "llm_suggested"


def test_llm_cannot_replace_catalog_action_with_contradictory_text():
    payload = hybrid_payload_ok()
    payload["mitigations"] = [
        {"base_id": 1, "text": "desactivar el firewall y permitir todo el trafico"},
    ]
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("DDoS_TCP"))
    item = next(
        candidate
        for candidate in update["explanation_output"]["mitigation_items"]
        if candidate["source"] == "llm"
    )

    assert item["text"] == item["base"]
    assert "desactivar" not in item["text"]
    assert "desactivar" in item["context"]
    assert item["context_trusted"] is False
    assert not any(
        "desactivar" in action
        for action in update["explanation_output"]["mitigations"]
    )


def test_llm_case_variant_reference_not_duplicated():
    payload = hybrid_payload_ok()
    payload["additional_references"] = [{"attack_id": "t1498", "name": "minusculas"}]
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("DDoS_TCP"))
    references = update["explanation_output"]["references"]
    t1498 = [r for r in references if (r.get("attack_id") or "").upper() == "T1498"]
    assert len(t1498) == 1
    assert t1498[0]["source"] == "catalog"


def test_llm_is_not_called_for_an_invalid_attack_type():
    payload = hybrid_payload_ok()
    payload["requires_human_review"] = False
    llm = llm_with_stub(payload=payload)
    state = malicious_state("DDoS_QUIC")
    update = FinalMitigator(client=client, llm=llm).run(state)
    assert update["explanation_output"]["requires_human_review"] is True
    assert update["explanation_output"]["review_reasons"] == [
        "catalog_attack_type_unavailable"
    ]
    assert update["needs_human_review"] is True
    assert llm.agent.calls == []


def test_catalog_error_fallback_items_consistent_with_flat_list():
    from tests.test_final_agents import StubClient

    stub = StubClient({("threat_intel", "suggest_mitigations"): {"ok": False, "error": "x"}})
    update = FinalMitigator(client=stub).run(malicious_state("DDoS_TCP"))
    output = update["explanation_output"]
    assert output["mitigations"] == [item["text"] for item in output["mitigation_items"]]
    assert output["mitigation_items"][0]["source"] == "fallback"


def test_llm_failure_case_level_stays_completed():
    from src.orchestration.mcp_graph import MultiAgentComponents, run_case
    from src.agents.final import (
        FinalClassifier,
        FinalDetector,
        FinalJudge,
        FinalStandardizer,
    )
    from tests.test_final_agents import StubClient, classify_ok, detect_ok, standardize_ok

    stub = StubClient(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.97),
            ("inference", "classify_event"): classify_ok("DDoS_TCP", 0.95),
        }
    )
    llm = llm_with_stub(error=ConnectionError("api caida"))
    agents = MultiAgentComponents(
        standardizer=FinalStandardizer(client=stub),
        detector=FinalDetector(client=stub),
        classifier=FinalClassifier(client=stub),
        mitigator=FinalMitigator(client=stub, llm=llm),
        judge=FinalJudge(client=stub),
        client=stub,
    )
    case = run_case({"dataset": "iot23", "row": {"proto": "tcp"}}, agents=agents)
    assert case.status == "completed"  # el fallback no rompe ni marca el caso
    assert case.errors == []
    assert case.explanation.source == "catalog"
    assert case.explanation.model_name is None
    fallback_steps = [e for e in case.trace if e.status == "fallback"]
    assert len(fallback_steps) == 1


def test_llm_suggested_marks_survive_case_result_boundary():
    from src.orchestration.mcp_graph import MultiAgentComponents, run_case
    from src.agents.final import (
        FinalClassifier,
        FinalDetector,
        FinalJudge,
        FinalStandardizer,
    )
    from tests.test_final_agents import StubClient, classify_ok, detect_ok, standardize_ok

    payload = hybrid_payload_ok()
    payload["additional_references"] = [{"attack_id": "T9999", "name": "Inventada"}]
    payload["mitigations"].append({"base_id": None, "text": "honeypot sin respaldo"})
    stub = StubClient(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.97),
            ("inference", "classify_event"): classify_ok("DDoS_TCP", 0.95),
        }
    )
    agents = MultiAgentComponents(
        standardizer=FinalStandardizer(client=stub),
        detector=FinalDetector(client=stub),
        classifier=FinalClassifier(client=stub),
        mitigator=FinalMitigator(client=stub, llm=llm_with_stub(payload=payload)),
        judge=FinalJudge(client=stub),
        client=stub,
    )
    case = run_case({"dataset": "iot23", "row": {"proto": "tcp"}}, agents=agents)
    payload_json = case.model_dump(mode="json")  # frontera API

    refs = payload_json["explanation"]["references"]
    assert any(r["attack_id"] == "T9999" and r["source"] == "llm_suggested" for r in refs)
    items = payload_json["explanation"]["mitigation_items"]
    assert any(
        i["text"] == "honeypot sin respaldo" and i["source"] == "llm_suggested"
        for i in items
    )


# ---------------------------------------------------------------------------
# anchor_llm_payload: robustez ante payloads deformes
# ---------------------------------------------------------------------------

def anchor_fixture():
    result = client.call(
        "threat_intel", "suggest_mitigations", attack_type="DDoS_TCP"
    )
    base_items = build_base_items(result)
    from src.agents.final.final_mitigator import catalog_references

    refs = catalog_references(result["references"])
    known = catalog_reference_ids(result["references"])
    return base_items, refs, known


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"mitigations": "no soy una lista", "confidence": "alta"},
        {"mitigations": [{"base_id": 1}], "additional_references": "tampoco"},
        {"mitigations": [{"base_id": 1, "text": "  "}, "item raro", None]},
    ],
)
def test_anchor_survives_malformed_payloads(payload):
    base_items, refs, known = anchor_fixture()
    anchored = anchor_llm_payload(
        payload,
        base_items=base_items,
        catalog_references=refs,
        known_reference_ids=known,
        fallback_summary="resumen determinista",
        fallback_confidence=0.8,
    )
    # todas las mitigaciones del catalogo presentes pase lo que pase
    assert len(anchored["mitigation_items"]) >= len(base_items)
    base_texts = {item["text"] for item in base_items}
    anchored_texts = {
        item["base"] or item["text"] for item in anchored["mitigation_items"]
    }
    assert base_texts <= anchored_texts
    assert anchored["risk_summary"] == "resumen determinista" or anchored["risk_summary"]
    assert 0.0 <= anchored["confidence"] <= 1.0
    assert anchored["source"] == "hybrid"


def test_anchor_duplicate_base_id_marked_suggested():
    base_items, refs, known = anchor_fixture()
    payload = {
        "risk_summary": "r",
        "mitigations": [
            {"base_id": 1, "text": "primera contextualizacion"},
            {"base_id": 1, "text": "segunda con el mismo base_id"},
        ],
        "confidence": 0.7,
        "requires_human_review": False,
    }
    anchored = anchor_llm_payload(
        payload, base_items=base_items, catalog_references=refs,
        known_reference_ids=known, fallback_summary="s", fallback_confidence=0.5,
    )
    sources = [
        item["source"]
        for item in anchored["mitigation_items"]
        if (item.get("context") or item["text"]).startswith(("primera", "segunda"))
    ]
    assert sources == ["llm", "llm_suggested"]


# ---------------------------------------------------------------------------
# contrato estructural del catalogo operativo
# ---------------------------------------------------------------------------

def test_catalog_v3_passes_the_reusable_structural_validator():
    validate_catalog(copy.deepcopy(CATALOG))
    assert CATALOG["version"] == THREAT_INTEL_CATALOG_VERSION
    assert "jorge_tfm_capec_table" not in CATALOG


def test_catalog_validator_rejects_an_obsolete_catalog_version():
    invalid = copy.deepcopy(CATALOG)
    invalid["version"] = "2.0"

    with pytest.raises(ValueError, match="Version del catalogo incompatible"):
        validate_catalog(invalid)


def test_catalog_validator_rejects_historical_or_unknown_top_level_blocks():
    invalid = copy.deepcopy(CATALOG)
    invalid["jorge_tfm_capec_table"] = {"mappings": []}

    with pytest.raises(ValueError, match="Campos de primer nivel"):
        validate_catalog(invalid)


def test_catalog_cache_cannot_be_mutated_by_a_consumer():
    first = catalog()
    first["attack_types"]["DDoS_TCP"]["mitigations"]["containment"][0] = (
        "accion alterada"
    )

    second = catalog()

    assert (
        second["attack_types"]["DDoS_TCP"]["mitigations"]["containment"][0]
        != "accion alterada"
    )


def test_catalog_validator_checks_urls_without_a_domain_allowlist():
    group = "attack_techniques"
    reference_id = next(iter(CATALOG["reference_catalog"][group]))
    legitimate = copy.deepcopy(CATALOG)
    legitimate["reference_catalog"][group][reference_id]["url"] = (
        "https://example.org/authoritative-reference"
    )
    validate_catalog(legitimate)

    invalid = copy.deepcopy(CATALOG)
    invalid["reference_catalog"][group][reference_id]["url"] = (
        "javascript:alert(1)"
    )
    with pytest.raises(ValueError, match="Referencia invalida"):
        validate_catalog(invalid)


def test_edge_attack_types_require_the_exact_contractual_spelling():
    exact = client.call(
        "threat_intel", "suggest_mitigations", attack_type="DDoS_UDP"
    )
    alias = client.call(
        "threat_intel", "suggest_mitigations", attack_type="ddos-udp"
    )

    assert exact["ok"] is True
    assert exact["attack_type"] == "DDoS_UDP"
    assert alias["ok"] is False


# ---------------------------------------------------------------------------
# integracion con el grafo final
# ---------------------------------------------------------------------------

def test_run_case_with_llm_mitigator_produces_hybrid_case():
    from src.orchestration.mcp_graph import MultiAgentComponents, run_case
    from src.agents.final import (
        FinalClassifier,
        FinalDetector,
        FinalJudge,
        FinalStandardizer,
    )
    from tests.test_final_agents import classify_ok, detect_ok, standardize_ok

    stub = StubClient(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.97),
            ("inference", "classify_event"): classify_ok("DDoS_TCP", 0.95),
        }
    )
    llm = llm_with_stub(payload=hybrid_payload_ok())
    agents = MultiAgentComponents(
        standardizer=FinalStandardizer(client=stub),
        detector=FinalDetector(client=stub),
        classifier=FinalClassifier(client=stub),
        mitigator=FinalMitigator(client=stub, llm=llm),
        judge=FinalJudge(client=stub),
        client=stub,
    )
    case = run_case({"dataset": "iot23", "row": {"proto": "tcp"}}, agents=agents)

    assert isinstance(case, CaseResult)
    assert case.status == "completed"
    assert case.explanation.source == "hybrid"
    assert case.explanation.model_name == "llm_mitigator::ollama::test-model"
    assert case.explanation.mitigation_items
    assert any(item.source == "llm" for item in case.explanation.mitigation_items)
    assert all(ref.source == "catalog" for ref in case.explanation.references)
    tools_in_trace = [entry.tool for entry in case.trace]
    assert "llm_contextualize" in tools_in_trace


def test_default_final_agents_env_flag_enables_llm(monkeypatch):
    from src.orchestration.mcp_graph import default_final_agents

    for name in (
        "LLM_MITIGATOR_ENABLED",
        "MITIGATOR_LLM_MODEL",
        "MISTRAL_AGENT_MODEL",
        "INGEST_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    assert default_final_agents().mitigator.llm is None

    monkeypatch.setenv("LLM_MITIGATOR_ENABLED", "true")
    bundle = default_final_agents()
    assert bundle.mitigator.llm is not None
    assert bundle.mitigator.llm.model_name == "llm_mitigator::mistral::mistral-small-2603"

    # el parametro explicito manda sobre el entorno
    assert default_final_agents(use_llm_mitigator=False).mitigator.llm is None


def test_explicit_llm_mitigation_defaults_to_mistral(monkeypatch):
    from src.orchestration.mcp_graph import default_final_agents

    for name in (
        "MITIGATOR_LLM_PROVIDER",
        "LLM_PROVIDER",
        "MITIGATOR_LLM_MODEL",
        "MISTRAL_AGENT_MODEL",
        "INGEST_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    bundle = default_final_agents(use_llm_mitigator=True)

    assert bundle.mitigator.llm is not None
    assert (
        bundle.mitigator.llm.model_name
        == "llm_mitigator::mistral::mistral-small-2603"
    )


def test_online_mitigator_provider_cannot_override_mistral(monkeypatch):
    from src.orchestration.mcp_graph import default_final_agents

    monkeypatch.setenv("MITIGATOR_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    bundle = default_final_agents(use_llm_mitigator=True)

    assert bundle.mitigator.llm is not None
    assert bundle.mitigator.llm.model_name.startswith("llm_mitigator::mistral::")


def test_event_context_drops_empty_and_forbidden_fields():
    event = dict(CANONICAL_EVENT)
    event["origin"] = {"source_name": "X"}
    event["provenance"] = {"dataset": "X"}
    event["telemetry"] = {}
    context = event_context(event)
    assert "origin" not in context
    assert "provenance" not in context
    assert "telemetry" not in context  # vacio: no aporta
    assert context["event_id"] == event["event_id"]
