# tests/test_mitigator.py
"""Tests del agente de mitigacion Fase 4: catalogo + LLM anclado.

Criterios de aceptacion del plan:
(a) modo determinista devuelve mitigaciones y referencias correctas por familia;
(b) modo LLM (mockeado) nunca produce referencias fuera del catalogo sin
    marcarlas ``llm_suggested``;
(c) benign -> mitigaciones de monitorizacion minima.

Ademas: cobertura del mapeo CAPEC del TFM de Jorge (su unica documentacion de
mitigacion) — este catalogo debe ser superset de su Tabla 3.4.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.final import FinalMitigator
from src.agents.final.llm_mitigator import (
    LLMMitigationAgent,
    anchor_llm_payload,
    build_base_items,
    catalog_reference_ids,
    event_context,
)
from src.contracts.case import CaseResult
from src.mcp.client import MCPToolClient
from tests.test_final_agents import CANONICAL_EVENT, StubClient

client = MCPToolClient(mode="inprocess")

CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "mcp" / "data" / "threat_intel_catalog.json"
)
CATALOG = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

ATTACK_FAMILIES = [
    "ddos",
    "scanning",
    "botnet",
    "bruteforce",
    "injection",
    "malware",
    "exfiltration",
    "mitm",
]


def malicious_state(family: str = "ddos", confidence: float = 0.95) -> dict:
    return {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": True, "probability": 0.97},
        "classification_output": {"attack_family": family, "confidence": confidence},
        "trace": [],
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
# (a) modo determinista: mitigaciones y referencias correctas por familia
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ATTACK_FAMILIES)
def test_catalog_mode_returns_expected_mitigations_and_references(family):
    update = FinalMitigator(client=client).run(malicious_state(family))
    output = update["explanation_output"]
    entry = CATALOG["families"][family]

    expected_mitigations = [
        *entry["mitigations"].get("containment", []),
        *entry["mitigations"].get("eradication", []),
        *entry["mitigations"].get("prevention", []),
    ]
    for text in expected_mitigations:
        assert text in output["mitigations"], f"{family}: falta '{text}'"

    reference_ids = {
        ref.get("attack_id") or ref.get("capec_id") for ref in output["references"]
    }
    for technique in entry["attack_techniques"]:
        assert technique["id"] in reference_ids
    for pattern in entry["capec_patterns"]:
        assert pattern["id"] in reference_ids
    assert all(ref["source"] == "catalog" for ref in output["references"])
    assert all(item["source"] == "catalog" for item in output["mitigation_items"])
    assert output["source"] == "catalog"


def test_catalog_mode_unknown_family_requires_review():
    update = FinalMitigator(client=client).run(malicious_state("familia_marciana"))
    assert update["explanation_output"]["requires_human_review"] is True
    assert update["needs_human_review"] is True


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
    benign_expected = CATALOG["families"]["benign"]["mitigations"]["prevention"]
    for text in benign_expected:
        assert text in output["mitigations"]
    assert "benigno" in output["risk_summary"]
    assert output["requires_human_review"] is False
    # sin acciones de contencion/erradicacion agresivas
    phases = {item["phase"] for item in output["mitigation_items"]}
    assert "containment" not in phases
    assert "eradication" not in phases


def test_benign_never_calls_llm_even_if_configured():
    llm = llm_with_stub(payload={"risk_summary": "x", "mitigations": [], "confidence": 0.9, "requires_human_review": False})
    state = {
        "canonical_event": dict(CANONICAL_EVENT),
        "detection_output": {"is_malicious": False, "probability": 0.02},
        "trace": [],
    }
    update = FinalMitigator(client=client, llm=llm).run(state)
    assert llm.agent.calls == []
    assert update["explanation_output"]["source"] == "catalog"


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


def test_llm_contextualization_produces_hybrid_anchored_output():
    llm = llm_with_stub(payload=hybrid_payload_ok())
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("ddos"))
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
    ddos = CATALOG["families"]["ddos"]["mitigations"]
    for text in [*ddos["eradication"], *ddos["prevention"]]:
        assert any(item["text"] == text and item["source"] == "catalog" for item in items)

    # referencias: solo catalogo, todas marcadas catalog
    assert all(ref["source"] == "catalog" for ref in output["references"])


def test_llm_cannot_smuggle_references_outside_catalog():
    payload = hybrid_payload_ok()
    payload["additional_references"] = [
        {"attack_id": "T9999", "name": "Tecnica Inventada", "url": "https://example.com/fake"},
        {"capec_id": "CAPEC-99999", "name": "Patron Inventado"},
        {"attack_id": "T1498", "name": "Network Denial of Service (duplicada)"},
    ]
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("ddos"))
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
        client.call("threat_intel", "suggest_mitigations", family="ddos")["references"]
    )
    present = {ref.get("attack_id") or ref.get("capec_id") for ref in references}
    assert catalog_ids <= present


def test_llm_free_mitigations_are_marked_llm_suggested():
    payload = hybrid_payload_ok()
    payload["mitigations"].append({"base_id": None, "text": "instalar honeypot en la DMZ"})
    payload["mitigations"].append({"text": "regla extra sin base_id"})
    payload["mitigations"].append({"base_id": 999, "text": "base_id inexistente"})
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("ddos"))
    items = update["explanation_output"]["mitigation_items"]

    suggested = {item["text"] for item in items if item["source"] == "llm_suggested"}
    assert "instalar honeypot en la DMZ" in suggested
    assert "regla extra sin base_id" in suggested
    assert "base_id inexistente" in suggested
    assert update["explanation_output"]["has_llm_suggested"] is True
    assert update["explanation_output"]["requires_human_review"] is True


def test_llm_failure_falls_back_to_catalog_completely():
    llm = llm_with_stub(error=TimeoutError("llm caido"))
    with_llm = FinalMitigator(client=client, llm=llm).run(malicious_state("ddos"))
    without_llm = FinalMitigator(client=client).run(malicious_state("ddos"))

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
    state = malicious_state("ddos")
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
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("ddos"))
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
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("ddos"))
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
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("ddos"))
    items = update["explanation_output"]["mitigation_items"]

    downgraded = next(item for item in items if "T4242" in item["text"])
    assert downgraded["source"] == "llm_suggested"
    # la mitigacion base del catalogo reentra literal como catalog
    ddos_containment = CATALOG["families"]["ddos"]["mitigations"]["containment"][0]
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
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("ddos"))
    items = update["explanation_output"]["mitigation_items"]
    contextualized = next(item for item in items if "segmento del sensor" in item["context"])
    # M1037 esta en el catalogo de ddos: la contextualizacion es legitima
    assert contextualized["source"] == "llm"


def test_llm_raw_typed_references_do_not_crash_case_result():
    payload = hybrid_payload_ok()
    payload["additional_references"] = [
        {"attack_id": 12345, "name": ["lista", "rara"], "url": 99},
        {"capec_id": {"objeto": "raro"}},
        "no soy un dict",
    ]
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("ddos"))

    state = malicious_state("ddos")
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
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("ddos"))
    output = update["explanation_output"]
    assert output["source"] == "hybrid"
    assert 0.0 <= output["confidence"] <= 1.0


def test_llm_bool_base_id_is_not_anchored():
    payload = hybrid_payload_ok()
    payload["mitigations"] = [{"base_id": True, "text": "colada booleana"}]
    llm = llm_with_stub(payload=payload)
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("ddos"))
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
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("ddos"))
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
    update = FinalMitigator(client=client, llm=llm).run(malicious_state("ddos"))
    references = update["explanation_output"]["references"]
    t1498 = [r for r in references if (r.get("attack_id") or "").upper() == "T1498"]
    assert len(t1498) == 1
    assert t1498[0]["source"] == "catalog"


def test_llm_cannot_clear_unknown_attack_review_flag():
    payload = hybrid_payload_ok()
    payload["requires_human_review"] = False
    llm = llm_with_stub(payload=payload)
    state = malicious_state("familia_marciana")
    update = FinalMitigator(client=client, llm=llm).run(state)
    assert update["explanation_output"]["requires_human_review"] is True
    assert update["needs_human_review"] is True


def test_catalog_error_fallback_items_consistent_with_flat_list():
    from tests.test_final_agents import StubClient

    stub = StubClient({("threat_intel", "suggest_mitigations"): {"ok": False, "error": "x"}})
    update = FinalMitigator(client=stub).run(malicious_state("ddos"))
    output = update["explanation_output"]
    assert output["mitigations"] == [item["text"] for item in output["mitigation_items"]]
    assert output["mitigation_items"][0]["source"] == "fallback"


def test_llm_failure_case_level_stays_completed():
    from src.orchestration.mcp_graph import FinalAgentBundle, run_case
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
            ("inference", "classify_event"): classify_ok("ddos", 0.95),
        }
    )
    llm = llm_with_stub(error=ConnectionError("api caida"))
    agents = FinalAgentBundle(
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
    from src.orchestration.mcp_graph import FinalAgentBundle, run_case
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
            ("inference", "classify_event"): classify_ok("ddos", 0.95),
        }
    )
    agents = FinalAgentBundle(
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
    result = client.call("threat_intel", "suggest_mitigations", family="ddos")
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
# cobertura del TFM de Jorge (su documentacion de mitigacion: MITRE CAPEC)
# ---------------------------------------------------------------------------

def test_catalog_covers_all_jorge_capec_mappings():
    """El catalogo debe ser superset de la Tabla 3.4 del TFM de Jorge."""
    coverage = client.call("threat_intel", "get_jorge_capec_coverage")
    assert coverage["ok"]
    assert coverage["total"] == 14
    assert coverage["covered"] == 14, [
        row for row in coverage["mappings"] if not row["covered"]
    ]
    # y ademas anadimos lo que Jorge no tenia: tecnicas ATT&CK y mitigaciones M-*
    for row in coverage["mappings"]:
        assert row["extra_attack_techniques"], row
        assert row["extra_attack_mitigations"], row


def test_edge_attack_type_aliases_resolve_to_catalog_families():
    for attack_type, family in [
        ("DDoS_UDP", "ddos"),
        ("DDoS_ICMP", "ddos"),
        ("SQL_injection", "injection"),
        ("XSS", "injection"),
        ("Uploading", "injection"),
        ("Password", "bruteforce"),
        ("Vulnerability_scanner", "scanning"),
        ("Fingerprinting", "scanning"),
        ("Backdoor", "malware"),
        ("Ransomware", "malware"),
        ("MITM", "mitm"),
    ]:
        resolved = client.call("threat_intel", "suggest_mitigations", family=attack_type)
        assert resolved["family"] == family, f"{attack_type} -> {resolved['family']}"


# ---------------------------------------------------------------------------
# integracion con el grafo final
# ---------------------------------------------------------------------------

def test_run_case_with_llm_mitigator_produces_hybrid_case():
    from src.orchestration.mcp_graph import FinalAgentBundle, run_case
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
            ("inference", "classify_event"): classify_ok("ddos", 0.95),
        }
    )
    llm = llm_with_stub(payload=hybrid_payload_ok())
    agents = FinalAgentBundle(
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

    monkeypatch.delenv("LLM_MITIGATOR_ENABLED", raising=False)
    assert default_final_agents().mitigator.llm is None

    monkeypatch.setenv("LLM_MITIGATOR_ENABLED", "true")
    monkeypatch.setenv("MITIGATOR_LLM_PROVIDER", "mistral")
    bundle = default_final_agents()
    assert bundle.mitigator.llm is not None
    assert bundle.mitigator.llm.model_name == "llm_mitigator::mistral::mistral-small-latest"

    # el parametro explicito manda sobre el entorno
    assert default_final_agents(use_llm_mitigator=False).mitigator.llm is None


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
