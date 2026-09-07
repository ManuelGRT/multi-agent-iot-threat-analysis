# tests/test_auditor.py
"""Tests del agente de auditoria (Fase 5a) con casos sinteticos.

Incluye el caso trampa con target leakage que el auditor DEBE detectar
(criterio de aceptacion del plan).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.final.auditor import (
    CaseAuditor,
    canonical_leakage_issues,
    feature_leakage_issues,
)
from src.contracts.attack_taxonomy import (
    JORGE_TAXONOMY_VERSION,
    MULTIDATASET_TAXONOMY_VERSION,
)
from src.contracts.leakage import contains_predictive_target_text
from src.contracts.case import (
    CaseResult,
    ClassificationInfo,
    DetectionInfo,
    ExplanationInfo,
    JudgeInfo,
    StandardizationInfo,
    ThreatReference,
    TraceEntry,
)
from tests.test_final_agents import CANONICAL_EVENT

auditor = CaseAuditor()

T0 = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)


def make_trace(agents: list[str], start: datetime = T0) -> list[TraceEntry]:
    """Traza sintetica cerrada y ordenada (1s por agente)."""
    entries = []
    for index, agent in enumerate(agents):
        started = start + timedelta(seconds=index)
        entries.append(
            TraceEntry(
                agent=agent,
                status="ok",
                started_at=started,
                finished_at=started + timedelta(milliseconds=500),
            )
        )
    return entries


FULL_PATH = [
    "orchestrator",
    "final_standardizer",
    "final_detector",
    "final_classifier",
    "final_mitigator",
    "final_judge",
]
BENIGN_PATH = ["orchestrator", "final_standardizer", "final_detector", "final_judge"]


def clean_attack_case(**overrides) -> CaseResult:
    defaults = dict(
        canonical_event=dict(CANONICAL_EVENT),
        standardization=StandardizationInfo(mapping_confidence=0.9, schema_profile="network_flow"),
        detection=DetectionInfo(is_malicious=True, probability=0.97, model_name="xgb"),
        classification=ClassificationInfo(
            attack_type="DDoS_TCP",
            confidence=0.95,
            decision_threshold=0.65,
            model_task="attack_type",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            top_scores={"DDoS_TCP": 0.95, "DDoS_UDP": 0.03, "XSS": 0.02},
        ),
        explanation=ExplanationInfo(
            summary="DDoS TCP",
            mitigations=["rate_limit"],
            references=[ThreatReference(attack_id="T1498", source="catalog")],
            confidence=0.95,
            attack_type="DDoS_TCP",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            catalog_scope="attack_type",
            catalog_version="2.0",
            catalog_taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            catalog_compatible_taxonomy_versions=[
                MULTIDATASET_TAXONOMY_VERSION,
            ],
            source="catalog",
        ),
        judge=JudgeInfo(
            action="approve",
            approved=True,
            final_label="DDoS_TCP",
            final_confidence=0.95,
        ),
        trace=make_trace(FULL_PATH),
    )
    defaults.update(overrides)
    case = CaseResult(**defaults)
    return case.close()


def typed_explanation(
    *,
    attack_type: str = "DDoS_TCP",
    taxonomy_version: str = MULTIDATASET_TAXONOMY_VERSION,
) -> ExplanationInfo:
    return ExplanationInfo(
        summary=attack_type,
        mitigations=["rate_limit"],
        references=[ThreatReference(attack_id="T1498", source="catalog")],
        confidence=0.95,
        attack_type=attack_type,
        taxonomy_version=taxonomy_version,
        catalog_scope="attack_type",
        catalog_version="2.0",
        catalog_taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
        catalog_compatible_taxonomy_versions=[
            MULTIDATASET_TAXONOMY_VERSION,
            JORGE_TAXONOMY_VERSION,
        ],
        source="catalog",
    )


def clean_benign_case(**overrides) -> CaseResult:
    defaults = dict(
        canonical_event=dict(CANONICAL_EVENT),
        standardization=StandardizationInfo(mapping_confidence=0.9),
        detection=DetectionInfo(is_malicious=False, probability=0.03),
        judge=JudgeInfo(action="approve", approved=True, final_label="benign", final_confidence=0.97),
        trace=make_trace(BENIGN_PATH),
    )
    defaults.update(overrides)
    return CaseResult(**defaults).close()


# ---------------------------------------------------------------------------
# casos limpios
# ---------------------------------------------------------------------------

def test_clean_attack_case_is_approved():
    report = auditor.audit(clean_attack_case())
    assert report.verdict == "approve", report.issues
    assert report.passed
    assert report.issues == []
    catalog_check = next(
        check
        for check in report.checks
        if check.check == "consistencia_catalogo_sin_llm_suggested"
    )
    assert catalog_check.detail is None


def test_auditor_validates_attack_type_and_top_score_coherence():
    coherent = clean_attack_case(
        classification=ClassificationInfo(
            attack_type="DDoS_TCP",
            confidence=0.91,
            model_task="attack_type",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            top_scores={"DDoS_TCP": 0.91, "DDoS_UDP": 0.06, "XSS": 0.03},
        ),
        explanation=typed_explanation(),
        judge=JudgeInfo(
            action="approve",
            approved=True,
            final_label="DDoS_TCP",
            final_confidence=0.91,
        ),
    )
    assert auditor.audit(coherent).verdict == "approve"

    mismatch = clean_attack_case(
        classification=ClassificationInfo(
            attack_type="DDoS_TCP",
            confidence=0.91,
            model_task="attack_type",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            top_scores={"DDoS_UDP": 0.91, "DDoS_TCP": 0.06, "XSS": 0.03},
        ),
        explanation=typed_explanation(),
        judge=JudgeInfo(
            action="approve",
            approved=True,
            final_label="DDoS_TCP",
            final_confidence=0.91,
        ),
    )
    report = auditor.audit(mismatch)
    assert report.verdict == "reject"
    assert any(
        "consistencia_tipo_ataque_vs_top_scores" in issue for issue in report.issues
    )


def test_auditor_accepts_multidataset_attack_type_contract():
    case = clean_attack_case(
        classification=ClassificationInfo(
            attack_type="DoS",
            confidence=0.88,
            model_task="attack_type",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            top_scores={"DoS": 0.88, "DDoS_TCP": 0.07, "Command_and_Control": 0.05},
        ),
        explanation=typed_explanation(
            attack_type="DoS",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
        ),
        judge=JudgeInfo(
            action="approve",
            approved=True,
            final_label="DoS",
            final_confidence=0.88,
        ),
    )

    assert auditor.audit(case).verdict == "approve"


def test_auditor_validates_full_attack_type_contract_and_dynamic_threshold():
    case = clean_attack_case(
        classification=ClassificationInfo(
            attack_type="DDoS_TCP",
            confidence=0.80,
            decision_threshold=0.81,
            model_name="xgboost_attack_subtype_multidataset16_balanced500_20260906",
            model_task="attack_type",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            top_scores={"DDoS_TCP": 0.80, "DDoS_UDP": 0.12, "XSS": 0.08},
        ),
        explanation=typed_explanation(),
        judge=JudgeInfo(
            action="human_interrupt",
            approved=False,
            requires_human_review=True,
            final_label="DDoS_TCP",
            final_confidence=0.80,
        ),
    )
    report = auditor.audit(case)
    assert report.verdict == "review", report.issues
    threshold_check = next(
        check
        for check in report.checks
        if check.check == "umbral_confianza_clasificacion_derivada"
    )
    assert threshold_check.passed
    assert "threshold=0.81" in (threshold_check.detail or "")


def test_auditor_rejects_wrong_attack_type_taxonomy_and_invalid_top_score():
    case = clean_attack_case(
        classification=ClassificationInfo(
            attack_type="DDoS_TCP",
            confidence=0.91,
            model_task="attack_type",
            taxonomy_version="obsolete",
            top_scores={
                "DDoS_TCP": 0.91,
                "Novel_Attack": 0.06,
                "XSS": 0.03,
            },
        ),
        explanation=typed_explanation(taxonomy_version="obsolete"),
        judge=JudgeInfo(
            action="approve",
            approved=True,
            final_label="DDoS_TCP",
            final_confidence=0.91,
        ),
    )
    report = auditor.audit(case)
    assert report.verdict == "reject"
    assert any("consistencia_version_taxonomia" in issue for issue in report.issues)
    assert any("consistencia_top_scores_tipos" in issue for issue in report.issues)


@pytest.mark.parametrize(
    ("field", "value", "expected_check"),
    [
        ("attack_type", "DDoS_UDP", "consistencia_mitigacion_tipo_ataque"),
        (
            "taxonomy_version",
            JORGE_TAXONOMY_VERSION,
            "consistencia_mitigacion_taxonomia",
        ),
        ("catalog_scope", "family", "consistencia_mitigacion_catalogo_tipado"),
        ("catalog_version", None, "consistencia_mitigacion_catalogo_tipado"),
        (
            "catalog_compatible_taxonomy_versions",
            [JORGE_TAXONOMY_VERSION],
            "consistencia_mitigacion_catalogo_tipado",
        ),
    ],
)
def test_auditor_rejects_typed_mitigation_contract_mismatch(
    field, value, expected_check
):
    explanation = typed_explanation()
    setattr(explanation, field, value)
    case = clean_attack_case(
        classification=ClassificationInfo(
            attack_type="DDoS_TCP",
            confidence=0.91,
            model_task="attack_type",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            top_scores={"DDoS_TCP": 0.91, "DDoS_UDP": 0.06, "XSS": 0.03},
        ),
        explanation=explanation,
        judge=JudgeInfo(
            action="approve",
            approved=True,
            final_label="DDoS_TCP",
            final_confidence=0.91,
        ),
    )

    report = auditor.audit(case)

    assert report.verdict == "reject"
    failed = next(check for check in report.checks if check.check == expected_check)
    assert failed.passed is False


def test_clean_benign_case_is_approved():
    report = auditor.audit(clean_benign_case())
    assert report.verdict == "approve", report.issues


def test_properly_flagged_abstention_is_review_not_reject():
    case = clean_benign_case(
        detection=DetectionInfo(is_malicious=False, probability=0.5, abstain=True),
        judge=JudgeInfo(
            action="human_interrupt",
            approved=False,
            requires_human_review=True,
            issues=["detector_abstained"],
        ),
    )
    report = auditor.audit(case)
    assert case.status == "needs_human_review"
    assert report.verdict == "review"
    assert report.passed


def test_standardizer_abstention_without_detector_is_valid_review():
    trace = make_trace(["orchestrator", "final_standardizer", "final_judge"])
    trace[1].status = "abstain"
    case = CaseResult(
        raw_input={"dataset": "iot23", "row_id": 7},
        standardization=StandardizationInfo(
            model="mistral-small-2603",
            mapping_confidence=0.0,
            abstain=True,
            requires_human_review=True,
            failure_code="llm_standardization_failed",
            failure_reason="RuntimeError: Mistral no disponible",
        ),
        judge=JudgeInfo(
            action="human_interrupt",
            approved=False,
            requires_human_review=True,
            issues=["standardizer_abstained", "detection_missing"],
        ),
        trace=trace,
    ).close()

    report = auditor.audit(case)

    assert case.status == "needs_human_review"
    assert "final_detector" not in [entry.agent for entry in case.trace]
    assert report.verdict == "review", report.issues
    assert report.passed
    assert not any("traza_detector_presente" in issue for issue in report.issues)
    canonical_check = next(
        check
        for check in report.checks
        if check.check == "leakage_evento_canonico_presente"
    )
    assert canonical_check.passed
    assert canonical_check.detail is None


# ---------------------------------------------------------------------------
# consistencia
# ---------------------------------------------------------------------------

def test_benign_with_attack_type_is_rejected():
    case = clean_benign_case(
        classification=ClassificationInfo(
            attack_type="DDoS_TCP",
            confidence=0.9,
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
        ),
    )
    report = auditor.audit(case)
    assert report.verdict == "reject"
    assert any(
        "consistencia_benigno_sin_tipo_ataque" in issue for issue in report.issues
    )


def test_detection_label_must_match_probability():
    case = clean_benign_case(
        detection=DetectionInfo(is_malicious=False, probability=0.99),
    )
    report = auditor.audit(case)
    assert report.verdict == "reject"
    assert any("consistencia_etiqueta_vs_probabilidad" in issue for issue in report.issues)


def test_malicious_case_cannot_use_type_outside_taxonomy():
    case = clean_attack_case(
        classification=ClassificationInfo(
            attack_type="benign",
            confidence=0.95,
            model_task="attack_type",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            top_scores={"benign": 0.95, "DDoS_TCP": 0.03, "XSS": 0.02},
        ),
    )
    report = auditor.audit(case)
    assert report.verdict == "reject"
    assert any(
        "consistencia_tipo_ataque_en_taxonomia" in issue for issue in report.issues
    )


def test_unflagged_abstention_is_rejected():
    case = clean_attack_case(
        detection=DetectionInfo(is_malicious=True, probability=0.55, abstain=True),
    )
    report = auditor.audit(case)
    assert case.status == "completed"  # el juez aprobo indebidamente
    assert report.verdict == "reject"
    assert any("consistencia_abstencion_flaggeada" in issue for issue in report.issues)


def test_gray_zone_probability_without_abstain_is_rejected():
    case = clean_attack_case(
        detection=DetectionInfo(is_malicious=True, probability=0.55, abstain=False),
    )
    report = auditor.audit(case)
    assert report.verdict == "reject"
    assert any("umbral_zona_gris_abstiene" in issue for issue in report.issues)


def test_completed_malicious_without_attack_type_is_rejected():
    case = clean_attack_case(classification=ClassificationInfo())
    report = auditor.audit(case)
    assert report.verdict == "reject"
    assert any(
        "consistencia_malicioso_con_tipo_ataque" in issue for issue in report.issues
    )


def test_catalog_source_with_llm_suggested_reference_is_rejected():
    case = clean_attack_case(
        explanation=ExplanationInfo(
            summary="x",
            references=[ThreatReference(attack_id="T9999", source="llm_suggested")],
            source="catalog",
        )
    )
    report = auditor.audit(case)
    assert report.verdict == "reject"
    assert any("consistencia_catalogo_sin_llm_suggested" in issue for issue in report.issues)
    catalog_check = next(
        check
        for check in report.checks
        if check.check == "consistencia_catalogo_sin_llm_suggested"
    )
    assert catalog_check.detail == "referencias llm_suggested con source=catalog"


def test_completed_status_with_rejecting_judge_is_rejected():
    # close() marca 'completed' aunque el juez rechace: solo este check lo caza
    case = clean_attack_case(
        judge=JudgeInfo(
            action="reject", approved=False, final_label="DDoS_TCP"
        )
    )
    assert case.status == "completed"
    report = auditor.audit(case)
    assert report.verdict == "reject"
    assert any("consistencia_estado_vs_juez" in issue for issue in report.issues)


def test_error_status_case_is_rejected():
    case = CaseResult(
        canonical_event=dict(CANONICAL_EVENT),
        standardization=StandardizationInfo(mapping_confidence=0.9),
        errors=["final_detector: crash"],
        trace=make_trace(["orchestrator", "final_standardizer", "final_detector"]),
    ).close()
    assert case.status == "error"
    report = auditor.audit(case)
    assert report.verdict == "reject"
    assert any("consistencia_sin_estado_error" in issue for issue in report.issues)


def test_completed_case_with_unflagged_errors_is_rejected():
    # close() ya deriva esto a status=error; el escenario solo llega con
    # datos deserializados/mutados que declaran completed con errores
    case = clean_attack_case(errors=["final_detector: fallo silencioso"], status="completed")
    assert case.status == "completed"
    report = auditor.audit(case)
    assert report.verdict == "reject"
    assert any("consistencia_errores_derivados" in issue for issue in report.issues)


def test_completed_case_with_null_detection_verdict_is_rejected():
    case = clean_attack_case(
        detection=DetectionInfo(is_malicious=None, probability=0.55, abstain=False),
    )
    report = auditor.audit(case)
    assert report.verdict == "reject"
    assert any(
        "consistencia_veredicto_deteccion_presente" in issue for issue in report.issues
    )


def test_out_of_range_probability_is_rejected():
    # pydantic valida en construccion; el check del auditor es defensivo
    # frente a mutacion post-construccion o datos deserializados sin validar
    case = clean_attack_case()
    case.detection.probability = 1.5  # validate_assignment esta desactivado
    report = auditor.audit(case)
    assert report.verdict == "reject"
    assert any("consistencia_probabilidad_en_rango" in issue for issue in report.issues)


# ---------------------------------------------------------------------------
# trazabilidad
# ---------------------------------------------------------------------------


def test_empty_trace_is_rejected():
    report = auditor.audit(clean_attack_case(trace=[]))
    assert report.verdict == "reject"
    assert any(issue.startswith("traza_no_vacia") for issue in report.issues)


def test_classifier_without_detector_in_trace_is_rejected():
    path = ["orchestrator", "final_standardizer", "final_classifier",
            "final_mitigator", "final_judge"]
    report = auditor.audit(clean_attack_case(trace=make_trace(path)))
    assert report.verdict == "reject"
    assert any("traza_detector_antes_de_clasificar" in issue for issue in report.issues)


def test_error_trace_entry_in_completed_case_is_rejected():
    trace = make_trace(FULL_PATH)
    trace[2].status = "error"
    trace[2].error = "boom"
    report = auditor.audit(clean_attack_case(trace=trace))
    assert report.verdict == "reject"
    assert any("traza_errores_derivados" in issue for issue in report.issues)


def test_error_trace_entry_flagged_for_review_is_acceptable():
    trace = make_trace(BENIGN_PATH)
    trace[2].status = "error"
    case = clean_benign_case(
        trace=trace,
        detection=DetectionInfo(is_malicious=False, probability=0.0, abstain=True),
        judge=JudgeInfo(action="human_interrupt", approved=False, requires_human_review=True),
        errors=["final_detector: tool caida"],
    )
    report = auditor.audit(case)
    assert report.verdict == "review"
    assert report.passed

def test_unfinished_trace_entry_is_rejected():
    trace = make_trace(FULL_PATH)
    trace[2] = TraceEntry(agent="final_detector", status="started", started_at=T0 + timedelta(seconds=2))
    report = auditor.audit(clean_attack_case(trace=trace))
    assert report.verdict == "reject"
    assert any("traza_entradas_cerradas" in issue for issue in report.issues)


def test_out_of_order_trace_is_rejected():
    trace = make_trace(FULL_PATH)
    trace[3].started_at = T0 - timedelta(seconds=30)  # viaja al pasado
    report = auditor.audit(clean_attack_case(trace=trace))
    assert report.verdict == "reject"
    assert any("traza_orden_temporal" in issue for issue in report.issues)


def test_negative_duration_is_rejected():
    trace = make_trace(FULL_PATH)
    trace[1].finished_at = trace[1].started_at - timedelta(seconds=5)
    report = auditor.audit(clean_attack_case(trace=trace))
    assert report.verdict == "reject"
    assert any("traza_sin_duraciones_negativas" in issue for issue in report.issues)


def test_missing_judge_in_trace_is_rejected():
    trace = make_trace(FULL_PATH[:-1])  # sin final_judge
    report = auditor.audit(clean_attack_case(trace=trace))
    assert report.verdict == "reject"
    assert any("traza_juez_presente" in issue for issue in report.issues)


def test_wrong_agent_order_is_rejected():
    path = ["orchestrator", "final_standardizer", "final_classifier", "final_detector",
            "final_mitigator", "final_judge"]  # clasificador antes que detector
    report = auditor.audit(clean_attack_case(trace=make_trace(path)))
    assert report.verdict == "reject"
    assert any("traza_orden_de_agentes" in issue for issue in report.issues)


def test_classifier_missing_for_completed_attack_is_rejected():
    report = auditor.audit(clean_attack_case(trace=make_trace(BENIGN_PATH)))
    assert report.verdict == "reject"
    assert any("traza_clasificador_presente" in issue for issue in report.issues)


def test_completed_case_requires_standardizer_and_detector_trace():
    trace = make_trace(["orchestrator", "final_classifier", "final_mitigator", "final_judge"])
    report = auditor.audit(clean_attack_case(trace=trace))
    assert report.verdict == "reject"
    assert any("traza_estandarizador_presente" in issue for issue in report.issues)
    assert any("traza_detector_presente" in issue for issue in report.issues)


def test_completed_attack_requires_mitigator_trace():
    trace = make_trace(["orchestrator", "final_standardizer", "final_detector", "final_classifier", "final_judge"])
    report = auditor.audit(clean_attack_case(trace=trace))
    assert report.verdict == "reject"
    assert any("traza_mitigador_presente" in issue for issue in report.issues)


# ---------------------------------------------------------------------------
# TARGET LEAKAGE — incluido el caso trampa del plan
# ---------------------------------------------------------------------------

def leaky_event() -> dict:
    return {
        **CANONICAL_EVENT,
        "label_raw": "DDoS_UDP",
        "attack_family": "ddos",
        "telemetry": {"attack_type": "ddos", "fridge_temperature": 5.0},
        "semantic_text": "udp flow label=DDoS_UDP high rate",
    }


def test_leakage_trap_case_is_detected_and_rejected():
    """Caso trampa del plan: evento con leakage debe ser cazado por el auditor."""
    report = auditor.audit(clean_attack_case(canonical_event=leaky_event()))
    assert report.verdict == "reject"
    assert any("leakage_evento_canonico" in issue for issue in report.issues)


def test_canonical_leakage_issues_enumerates_problems():
    issues = canonical_leakage_issues(leaky_event())
    kinds = {issue.split(":")[0] for issue in issues}
    assert "campo_target_con_valor" in kinds
    assert "clave_target_anidada" in kinds
    assert "semantic_text_contiene_patron_target" in kinds
    assert len(issues) >= 4  # label_raw, attack_family, telemetry, semantic_text


def test_clean_event_has_no_leakage_issues():
    assert canonical_leakage_issues(dict(CANONICAL_EVENT)) == []


def test_feature_leakage_guard_on_clean_and_invalid_events():
    assert feature_leakage_issues(dict(CANONICAL_EVENT)) == []
    broken = feature_leakage_issues({"no": "es un evento"})
    assert broken and broken[0].startswith("evento_no_featurizable")


def test_top_level_target_key_beyond_named_fields_is_detected():
    """Cualquier clave target del nivel superior cuenta, no solo label_raw."""
    event = {**CANONICAL_EVENT, "label": "DDoS_UDP", "class": "attack"}
    issues = canonical_leakage_issues(event)
    flagged = {issue.split("=")[0] for issue in issues}
    assert "campo_target_con_valor:label" in flagged
    assert "campo_target_con_valor:class" in flagged


def test_common_target_aliases_are_detected_and_removed_during_offline_preparation():
    aliases = ("ground_truth", "target_value", "class_id", "outcome", "y")
    event = {**CANONICAL_EVENT, "telemetry": {key: 1 for key in aliases}}

    issues = canonical_leakage_issues(event)
    for key in aliases:
        assert any(f"telemetry.{key}" in issue for issue in issues)

    # La proyeccion runtime asume entrada limpia. Es el preprocesamiento de
    # evaluacion el que retira los aliases antes de generar features.
    from src.eval.data_sanitization import sanitize_canonical_event
    from src.mcp.features import event_features

    features = event_features(sanitize_canonical_event(event))
    assert "group_mean.other" not in features


def test_attack_indicators_are_allowlisted_behavioral_signal():
    event = {**CANONICAL_EVENT, "attack_indicators": ["syn_flood_pattern"]}
    assert canonical_leakage_issues(event) == []


def test_offline_preparation_keeps_label_like_context_out_of_features():
    from src.eval.data_sanitization import sanitize_canonical_event
    from src.mcp.features import event_features

    base_event = {
        **CANONICAL_EVENT,
        "telemetry": {"temperature": 21.5},
        "host": {"cpu": 12.0},
        "service_context": {
            "protocol_family": "icmp",
            "service": "telnet",
            "payload": "username=admin password=test <script>xss</script>",
        },
    }
    event = {
        **base_event,
        "telemetry": {**base_event["telemetry"], "risk": "Mirai"},
        "host": {**base_event["host"], "family_hint": "DDoS"},
        "behavior_tags": ["Mirai"],
        "uncertainty": ["DDoS"],
        "asset_context": "malware",
        "service_context": {
            **base_event["service_context"],
            "threat": "malware",
            "risk": "DDoS",
            "family_hint": "Mirai",
            "classification": "malicious",
            "role": "botnet",
        },
    }

    expected = event_features(sanitize_canonical_event(base_event))
    features = event_features(sanitize_canonical_event(event))
    serialized = " ".join(f"{key}={value}" for key, value in features.items()).casefold()

    assert features == expected
    assert "mirai" not in serialized
    assert "ddos" not in serialized
    assert "malware" not in serialized
    assert not any(
        key.startswith(("behavior.", "uncertainty.", "asset_context."))
        for key in features
    )
    assert "behavior_tag_count" not in features
    assert "uncertainty_count" not in features
    assert features["context.service.protocol_family.icmp"] == 1
    assert features["context.service.service.telnet"] == 1
    assert any("password" in key and "xss" in key for key in features)

    issues = canonical_leakage_issues(event)
    assert any("service_context.risk" in issue for issue in issues)


def test_completed_case_without_canonical_event_is_rejected():
    report = auditor.audit(clean_attack_case(canonical_event={}))
    assert report.verdict == "reject"
    assert any("leakage_evento_canonico_presente" in issue for issue in report.issues)
    canonical_check = next(
        check
        for check in report.checks
        if check.check == "leakage_evento_canonico_presente"
    )
    assert canonical_check.detail == "caso completed sin evento canonico"


def test_feature_leakage_check_fires_on_contaminated_features(monkeypatch):
    """Modo fallo del check de features: si event_features regresionara y
    dejara pasar claves target, el auditor DEBE emitir feature_target y reject."""
    import src.mcp.features as features_mod

    monkeypatch.setattr(
        features_mod,
        "event_features",
        lambda event: {"attack_family": "ddos", "packets": 1},
    )
    assert feature_leakage_issues(dict(CANONICAL_EVENT)) == ["feature_target:attack_family"]
    report = auditor.audit(clean_attack_case())
    assert report.verdict == "reject"
    assert any("leakage_features_modelos" in issue for issue in report.issues)


def test_contains_predictive_target_text():
    assert contains_predictive_target_text("src=1.2.3.4 label=Mirai proto=tcp")
    assert contains_predictive_target_text("attack_type: ddos")
    assert contains_predictive_target_text("ground_truth=1")
    assert contains_predictive_target_text("target-value: attack")
    assert not contains_predictive_target_text("tcp flow with high packet rate")
    assert not contains_predictive_target_text(None)


# ---------------------------------------------------------------------------
# umbrales
# ---------------------------------------------------------------------------

def test_low_mapping_confidence_unflagged_is_rejected():
    case = clean_attack_case(
        standardization=StandardizationInfo(mapping_confidence=0.2),
    )
    report = auditor.audit(case)
    assert report.verdict == "reject"
    assert any("umbral_mapping_confidence_derivado" in issue for issue in report.issues)


def test_low_mapping_confidence_flagged_is_review():
    case = clean_attack_case(
        standardization=StandardizationInfo(mapping_confidence=0.2),
        judge=JudgeInfo(
            action="human_interrupt",
            approved=False,
            requires_human_review=True,
            final_label="DDoS_TCP",
            final_confidence=0.95,
        ),
    )
    report = auditor.audit(case)
    assert report.verdict == "review"
    assert report.passed


def test_low_classification_confidence_unflagged_is_review_not_reject():
    case = clean_attack_case(
        classification=ClassificationInfo(
            attack_type="DDoS_TCP",
            confidence=0.4,
            decision_threshold=0.65,
            model_task="attack_type",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            top_scores={"DDoS_TCP": 0.4, "DDoS_UDP": 0.35, "XSS": 0.25},
        ),
        explanation=typed_explanation(),
        judge=JudgeInfo(
            action="approve",
            approved=True,
            final_label="DDoS_TCP",
            final_confidence=0.4,
        ),
    )
    report = auditor.audit(case)
    assert report.verdict == "review"  # fallo blando: deriva, no invalida
    assert report.passed


# ---------------------------------------------------------------------------
# batch + integracion con el grafo final
# ---------------------------------------------------------------------------

def test_audit_batch_summary_counts():
    cases = [
        clean_attack_case(),
        clean_benign_case(),
        clean_benign_case(
            classification=ClassificationInfo(
                attack_type="DDoS_TCP", confidence=0.9
            )
        ),
    ]
    summary = auditor.audit_batch(cases)
    assert summary["total"] == 3
    assert summary["by_verdict"]["approve"] == 2
    assert summary["by_verdict"]["reject"] == 1
    assert summary["all_valid"] is False
    assert len(summary["rejected"]) == 1


def test_real_final_graph_case_passes_audit():
    from src.orchestration.mcp_graph import run_case
    from tests.test_mcp_graph import stub_bundle
    from tests.test_final_agents import classify_ok, detect_ok, standardize_ok

    agents = stub_bundle(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.97),
            ("inference", "classify_event"): classify_ok("DDoS_TCP", 0.95),
        }
    )
    case = run_case({"dataset": "iot23", "row": {"proto": "tcp"}}, agents=agents)
    report = auditor.audit(case)
    assert report.verdict in {"approve", "review"}, report.issues
    assert report.passed


def test_real_abstention_case_passes_audit_as_review():
    from src.orchestration.mcp_graph import run_case
    from tests.test_mcp_graph import stub_bundle
    from tests.test_final_agents import detect_ok, standardize_ok

    agents = stub_bundle(
        {
            ("inference", "standardize_event"): standardize_ok(),
            ("inference", "detect_event"): detect_ok(0.5),
        }
    )
    case = run_case({"dataset": "iot23", "row": {"proto": "tcp"}}, agents=agents)
    report = auditor.audit(case)
    assert report.verdict == "review", report.issues
    assert report.passed
