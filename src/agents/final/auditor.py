# src/agents/final/auditor.py
"""Agente de auditoria E2E posterior al cierre de cada caso.

Valida un ``CaseResult`` terminado en cuatro dimensiones:

1. **Consistencia**: benigno sin tipo de ataque, abstenciones correctamente
   señaladas, malicioso completado con un tipo válido, estado coherente con el juez.
2. **Trazabilidad**: agentes esperados presentes, todas las entradas cerradas,
   orden temporal sin retrocesos.
3. **Target leakage**: el evento canonico del caso no arrastra campos de
   label/attack (ni en claves, ni en semantic_text, ni en las features que
   veran los modelos). Solo detecta y reporta; nunca modifica el caso.
4. **Umbrales**: mapping_confidence, zona gris de deteccion y confianza de
   clasificacion deben haberse traducido en revision humana cuando toca.

Veredicto: ``approve`` (limpio) / ``review`` (correctamente derivado a humano)
/ ``reject`` (fallo estructural: el caso no es fiable). La auditoria portable
de ``scripts/run_system_audit.py`` usa este mismo agente.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.contracts.attack_taxonomy import (
    MULTIDATASET_ATTACK_CLASSES,
    MULTIDATASET_TAXONOMY_VERSION,
)
from src.contracts.leakage import (
    contains_predictive_target_text,
    is_allowed_canonical_target_path,
    is_predictive_target_field,
    is_predictive_target_value,
)
from src.contracts.case import CaseResult
from src.mcp.threat_catalog import (
    THREAT_INTEL_CATALOG_VERSION,
    catalog_output_issues,
)

GRAY_ZONE_LOW = 0.4
GRAY_ZONE_HIGH = 0.6
REVIEW_MAPPING_CONFIDENCE = 0.5
REVIEW_CLASSIFICATION_CONFIDENCE = 0.65

# Orden canonico de los agentes finales en la traza (los presentes deben
# respetarlo; no todos son obligatorios en todas las rutas).
FINAL_AGENT_ORDER = [
    "orchestrator",
    "final_standardizer",
    "final_detector",
    "final_classifier",
    "final_mitigator",
    "final_judge",
]


class CheckResult(BaseModel):
    check: str
    passed: bool
    hard: bool = True  # un fallo "hard" invalida el caso (reject)
    detail: str | None = None


class CaseAuditReport(BaseModel):
    case_id: str
    verdict: Literal["approve", "review", "reject"]
    passed: bool
    issues: list[str] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def hard_failures(self) -> list[CheckResult]:
        return [check for check in self.checks if check.hard and not check.passed]


# ---------------------------------------------------------------------------
# Leakage sobre eventos canonicos (reutilizable por el script batch)
# ---------------------------------------------------------------------------

# Campos del CanonicalEvent cuyo nombre dispara el predicado target pero son
# senales de comportamiento legitimas (NUNCA llegan como features: features.py
# excluye attack_indicator.* explicitamente).
BEHAVIORAL_FIELD_ALLOWLIST = {"attack_indicators"}


def canonical_leakage_issues(canonical_event: dict[str, Any]) -> list[str]:
    """Problemas de target leakage en un evento canonico ya preparado.

    Un evento del flujo final NO debe llevar valores en NINGUNA clave target
    del nivel superior (label_raw, attack_family, label, class...), ni
    patrones ``label=X`` en semantic_text, ni claves target anidadas con
    valor en telemetry/host/contextos.
    """
    issues: list[str] = []
    for field, value in canonical_event.items():
        if field in BEHAVIORAL_FIELD_ALLOWLIST:
            continue
        if is_predictive_target_field(field) and value not in (None, "", [], {}):
            issues.append(f"campo_target_con_valor:{field}={value!r}")

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_path = f"{path}.{key}" if path else str(key)
                if (
                    is_predictive_target_field(key)
                    and not is_allowed_canonical_target_path(key_path)
                    and item not in (None, "", [], {})
                ):
                    issues.append(f"clave_target_anidada:{key_path}={item!r}")
                walk(item, key_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif is_predictive_target_value(value):
            issues.append(f"valor_target_anidado:{path}={value!r}")

    for container in ("telemetry", "host", "service_context", "host_context", "telemetry_context"):
        walk(canonical_event.get(container) or {}, container)

    if contains_predictive_target_text(str(canonical_event.get("semantic_text") or "")):
        issues.append("semantic_text_contiene_patron_target")
    if contains_predictive_target_text(str(canonical_event.get("anomaly_summary") or "")):
        issues.append("anomaly_summary_contiene_patron_target")
    return issues


def feature_leakage_issues(canonical_event: dict[str, Any]) -> list[str]:
    """Claves target que llegarian como features a los modelos (debe ser 0)."""
    from src.mcp.features import event_features

    try:
        features = event_features(canonical_event)
    except Exception as exc:  # evento invalido: se reporta como problema
        return [f"evento_no_featurizable:{type(exc).__name__}"]
    return [
        f"feature_target:{key}"
        for key in features
        if is_predictive_target_field(key)
    ]


# ---------------------------------------------------------------------------
# Auditor por caso
# ---------------------------------------------------------------------------

class CaseAuditor:
    name = "final_auditor"

    def __init__(
        self,
        gray_low: float = GRAY_ZONE_LOW,
        gray_high: float = GRAY_ZONE_HIGH,
        check_features: bool = True,
    ):
        self.gray_low = gray_low
        self.gray_high = gray_high
        self.check_features = check_features

    # ------------------------------------------------------------------
    def audit(self, case: CaseResult) -> CaseAuditReport:
        checks: list[CheckResult] = []
        issues: list[str] = []

        def add(check: str, passed: bool, hard: bool = True, detail: str | None = None) -> None:
            checks.append(CheckResult(check=check, passed=passed, hard=hard, detail=detail))
            if not passed:
                issues.append(f"{check}{f': {detail}' if detail else ''}")

        flagged_for_review = (
            case.status == "needs_human_review"
            or case.judge.requires_human_review
            or case.judge.action in {"human_interrupt", "abstain"}
        )

        # ---------------- 1. consistencia ----------------
        detection = case.detection
        classification = case.classification
        standardizer_abstained = case.standardization.abstain
        # un caso que termino en error nunca puede certificarse como fiable
        add(
            "consistencia_sin_estado_error",
            case.status != "error",
            detail=f"status={case.status} errors={case.errors[:3]}",
        )
        # errores de pipeline solo son admisibles si el caso quedo derivado
        add(
            "consistencia_errores_derivados",
            not case.errors or flagged_for_review or case.status == "error",
            detail=f"errors={case.errors[:3]} status={case.status}",
        )
        # un caso certificado como completed exige veredicto de deteccion
        if case.status == "completed":
            add(
                "consistencia_veredicto_deteccion_presente",
                detection.is_malicious is not None,
                detail=f"is_malicious={detection.is_malicious} probability={detection.probability}",
            )
        if detection.is_malicious is False:
            add(
                "consistencia_benigno_sin_tipo_ataque",
                classification.attack_type is None,
                detail=f"tipo={classification.attack_type}",
            )
        if detection.abstain:
            add(
                "consistencia_abstencion_flaggeada",
                flagged_for_review,
                detail=f"status={case.status} action={case.judge.action}",
            )
        if standardizer_abstained:
            add(
                "consistencia_abstencion_estandarizador_flaggeada",
                flagged_for_review
                and case.standardization.requires_human_review
                and case.judge.action == "human_interrupt",
                detail=(
                    f"status={case.status} action={case.judge.action} "
                    f"failure_code={case.standardization.failure_code}"
                ),
            )
        if case.status == "completed" and detection.is_malicious:
            add(
                "consistencia_malicioso_con_tipo_ataque",
                classification.attack_type is not None,
            )
        # defensivo: pydantic ya valida en construccion, pero el auditor
        # tambien recibe casos deserializados/mutados fuera de ese camino
        add(
            "consistencia_probabilidad_en_rango",
            0.0 <= detection.probability <= 1.0,
            detail=f"probability={detection.probability}",
        )
        if detection.is_malicious is not None and not detection.abstain:
            add(
                "consistencia_etiqueta_vs_probabilidad",
                detection.is_malicious == (detection.probability >= 0.5),
                detail=(
                    f"is_malicious={detection.is_malicious} "
                    f"probability={detection.probability}"
                ),
            )
        attack_classes: tuple[str, ...] = ()
        if detection.is_malicious:
            add(
                "consistencia_modelo_por_tipo",
                classification.model_task == "attack_type",
                detail=f"model_task={classification.model_task}",
            )
            taxonomy_supported = (
                classification.taxonomy_version
                == MULTIDATASET_TAXONOMY_VERSION
            )
            if taxonomy_supported:
                attack_classes = MULTIDATASET_ATTACK_CLASSES
            add(
                "consistencia_tipo_ataque_presente",
                classification.attack_type is not None,
                detail=f"tipo={classification.attack_type}",
            )
            add(
                "consistencia_version_taxonomia",
                taxonomy_supported,
                detail=(
                    f"recibida={classification.taxonomy_version} "
                    f"esperada={MULTIDATASET_TAXONOMY_VERSION} "
                    f"clases={len(MULTIDATASET_ATTACK_CLASSES)}"
                ),
            )
            add(
                "consistencia_umbral_clasificacion_operativo",
                abs(
                    float(classification.decision_threshold)
                    - REVIEW_CLASSIFICATION_CONFIDENCE
                )
                <= 1e-9,
                detail=(
                    f"recibido={classification.decision_threshold} "
                    f"esperado={REVIEW_CLASSIFICATION_CONFIDENCE}"
                ),
            )
            invalid_scores = {
                str(label): score
                for label, score in classification.top_scores.items()
                if label not in attack_classes
                or isinstance(score, bool)
                or not 0.0 <= float(score) <= 1.0
            }
            add(
                "consistencia_top_scores_tipos",
                len(classification.top_scores) == 3
                and classification.attack_type in classification.top_scores
                and not invalid_scores,
                detail=(
                    f"cantidad={len(classification.top_scores)} "
                    f"incluye_elegido={classification.attack_type in classification.top_scores} "
                    f"valores_invalidos={invalid_scores}"
                ),
            )
        if detection.is_malicious and classification.attack_type is not None:
            if classification.attack_type not in attack_classes:
                add(
                    "consistencia_tipo_ataque_en_taxonomia",
                    False,
                    detail=f"tipo={classification.attack_type}",
                )
            else:
                add(
                    "consistencia_tipo_ataque_en_taxonomia",
                    True,
                    detail=f"tipo={classification.attack_type}",
                )
                if classification.top_scores:
                    top_label, top_score = max(
                        classification.top_scores.items(), key=lambda item: item[1]
                    )
                else:
                    top_label, top_score = None, None
                add(
                    "consistencia_tipo_ataque_vs_top_scores",
                    top_label == classification.attack_type
                    and top_score is not None
                    and abs(float(top_score) - classification.confidence) <= 1e-6,
                    detail=(
                        f"tipo={classification.attack_type} "
                        f"top={top_label} score={top_score} "
                        f"confidence={classification.confidence}"
                    ),
                )
        if (
            detection.is_malicious
            and classification.model_task == "attack_type"
            and classification.attack_type is not None
        ):
            add(
                "consistencia_mitigacion_tipo_ataque",
                case.explanation.attack_type == classification.attack_type,
                detail=(
                    f"mitigador={case.explanation.attack_type} "
                    f"clasificador={classification.attack_type}"
                ),
            )
            add(
                "consistencia_mitigacion_taxonomia",
                case.explanation.taxonomy_version
                == classification.taxonomy_version,
                detail=(
                    f"mitigador={case.explanation.taxonomy_version} "
                    f"clasificador={classification.taxonomy_version}"
                ),
            )
            add(
                "consistencia_mitigacion_catalogo_tipado",
                case.explanation.catalog_scope == "attack_type"
                and case.explanation.catalog_version
                == THREAT_INTEL_CATALOG_VERSION
                and case.explanation.catalog_taxonomy_version
                == MULTIDATASET_TAXONOMY_VERSION
                and classification.taxonomy_version
                == MULTIDATASET_TAXONOMY_VERSION
                and classification.taxonomy_version
                in case.explanation.catalog_compatible_taxonomy_versions,
                detail=(
                    f"scope={case.explanation.catalog_scope} "
                    f"version={case.explanation.catalog_version} "
                    f"taxonomia_catalogo={case.explanation.catalog_taxonomy_version} "
                    "taxonomias_compatibles="
                    f"{case.explanation.catalog_compatible_taxonomy_versions} "
                    f"taxonomia_clasificador={classification.taxonomy_version}"
                ),
            )
            catalog_issues = catalog_output_issues(
                case.explanation,
                attack_type=classification.attack_type,
            )
            add(
                "consistencia_contenido_catalogo_tipado",
                not catalog_issues,
                detail=("; ".join(catalog_issues) if catalog_issues else None),
            )
            add(
                "consistencia_veredicto_juez_tipo_ataque",
                case.judge.final_label == classification.attack_type
                and abs(
                    float(case.judge.final_confidence)
                    - float(classification.confidence)
                )
                <= 1e-6,
                detail=(
                    f"juez={case.judge.final_label}/{case.judge.final_confidence} "
                    f"clasificador={classification.attack_type}/"
                    f"{classification.confidence}"
                ),
            )
        add(
            "consistencia_estado_vs_juez",
            case.status != "completed" or (case.judge.approved and not case.judge.requires_human_review),
            detail=f"status={case.status} approved={case.judge.approved}",
        )
        if case.explanation.source == "catalog":
            has_non_catalog_reference = any(
                ref.source != "catalog" for ref in case.explanation.references
            )
            add(
                "consistencia_catalogo_sin_llm_suggested",
                not has_non_catalog_reference,
                detail=(
                    "referencias llm_suggested con source=catalog"
                    if has_non_catalog_reference
                    else None
                ),
            )

        # ---------------- 2. trazabilidad ----------------
        trace = case.trace
        add("traza_no_vacia", len(trace) > 0)
        unfinished = [entry.agent for entry in trace if entry.finished_at is None]
        add("traza_entradas_cerradas", not unfinished, detail=f"abiertas={unfinished}")
        negative = [
            entry.agent
            for entry in trace
            if entry.finished_at is not None and entry.finished_at < entry.started_at
        ]
        add("traza_sin_duraciones_negativas", not negative, detail=f"{negative}")
        starts = [entry.started_at for entry in trace]
        add(
            "traza_orden_temporal",
            all(a <= b for a, b in zip(starts, starts[1:])),
        )

        agents_in_trace = [entry.agent for entry in trace]
        known_positions = [
            FINAL_AGENT_ORDER.index(agent)
            for agent in agents_in_trace
            if agent in FINAL_AGENT_ORDER
        ]
        add(
            "traza_orden_de_agentes",
            all(a <= b for a, b in zip(known_positions, known_positions[1:])),
            detail=f"agentes={agents_in_trace}",
        )
        if case.status != "error":
            add("traza_juez_presente", "final_judge" in agents_in_trace)
            add("traza_estandarizador_presente", "final_standardizer" in agents_in_trace)
            if not standardizer_abstained:
                add("traza_detector_presente", "final_detector" in agents_in_trace)
        if detection.is_malicious and not detection.abstain and case.status != "error":
            add("traza_clasificador_presente", "final_classifier" in agents_in_trace)
            add("traza_mitigador_presente", "final_mitigator" in agents_in_trace)
        # ruta coherente: si hay clasificacion hubo deteccion antes
        if "final_classifier" in agents_in_trace:
            add("traza_detector_antes_de_clasificar", "final_detector" in agents_in_trace)
        # entradas de traza en error solo son admisibles si el caso quedo
        # derivado a humano o termino en estado error
        error_entries = [entry.agent for entry in trace if entry.status == "error"]
        add(
            "traza_errores_derivados",
            not error_entries or flagged_for_review or case.status == "error",
            detail=f"agentes_con_error={error_entries}",
        )

        # ---------------- 3. target leakage ----------------
        if case.canonical_event:
            leakage = canonical_leakage_issues(case.canonical_event)
            add("leakage_evento_canonico", not leakage, detail="; ".join(leakage[:5]) or None)
            if self.check_features:
                feature_leaks = feature_leakage_issues(case.canonical_event)
                add(
                    "leakage_features_modelos",
                    not feature_leaks,
                    detail="; ".join(feature_leaks[:5]) or None,
                )
        else:
            # sin evento canonico no hay nada que auditar contra leakage:
            # solo es admisible si el caso NO quedo certificado como completed
            completed_without_canonical_event = case.status == "completed"
            add(
                "leakage_evento_canonico_presente",
                not completed_without_canonical_event,
                detail=(
                    "caso completed sin evento canonico"
                    if completed_without_canonical_event
                    else None
                ),
            )

        # ---------------- 4. umbrales ----------------
        mapping = case.standardization.mapping_confidence
        if mapping < REVIEW_MAPPING_CONFIDENCE:
            add(
                "umbral_mapping_confidence_derivado",
                flagged_for_review,
                detail=f"mapping_confidence={mapping}",
            )
        if (
            detection.is_malicious is not None
            and self.gray_low <= detection.probability <= self.gray_high
        ):
            add(
                "umbral_zona_gris_abstiene",
                detection.abstain,
                detail=f"probability={detection.probability}",
            )
        if (
            classification.attack_type is not None
            and classification.confidence < classification.decision_threshold
        ):
            add(
                "umbral_confianza_clasificacion_derivada",
                flagged_for_review,
                hard=False,
                detail=(
                    f"confidence={classification.confidence} "
                    f"threshold={classification.decision_threshold}"
                ),
            )

        hard_failed = any(check.hard and not check.passed for check in checks)
        soft_failed = any(not check.hard and not check.passed for check in checks)
        if hard_failed:
            verdict: Literal["approve", "review", "reject"] = "reject"
        elif flagged_for_review or soft_failed:
            verdict = "review"
        else:
            verdict = "approve"
        return CaseAuditReport(
            case_id=case.case_id,
            verdict=verdict,
            passed=not hard_failed,
            issues=issues,
            checks=checks,
        )

    # ------------------------------------------------------------------
    def audit_batch(self, cases: list[CaseResult]) -> dict[str, Any]:
        """Resume los veredictos de auditoria de varios casos cerrados."""
        reports = [self.audit(case) for case in cases]
        by_verdict = {"approve": 0, "review": 0, "reject": 0}
        for report in reports:
            by_verdict[report.verdict] += 1
        rejected = [
            {"case_id": report.case_id, "issues": report.issues}
            for report in reports
            if report.verdict == "reject"
        ]
        return {
            "total": len(reports),
            "by_verdict": by_verdict,
            "all_valid": by_verdict["reject"] == 0,
            "rejected": rejected,
            "reports": reports,
        }
