# src/agents/final/auditor.py
"""Agente de auditoria E2E — auditoria por caso (Fase 5a del plan de cierre).

Valida un ``CaseResult`` terminado en cuatro dimensiones:

1. **Consistencia**: benigno sin familia de ataque, abstenciones correctamente
   flaggeadas, malicioso completado con familia, estado coherente con el juez.
2. **Trazabilidad**: agentes esperados presentes, todas las entradas cerradas,
   orden temporal sin retrocesos.
3. **Target leakage**: el evento canonico del caso no arrastra campos de
   label/attack (ni en claves, ni en semantic_text, ni en las features que
   veran los modelos). Solo detecta y reporta; nunca modifica el caso.
4. **Umbrales**: mapping_confidence, zona gris de deteccion y confianza de
   clasificacion deben haberse traducido en revision humana cuando toca.

Veredicto: ``approve`` (limpio) / ``review`` (correctamente derivado a humano)
/ ``reject`` (fallo duro: el caso no es fiable). La auditoria batch de sistema
(Fase 5b) vive en ``scripts/run_system_audit.py`` y usa este mismo auditor.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.contracts.leakage import (
    contains_predictive_target_text,
    is_allowed_canonical_target_path,
    is_predictive_target_field,
    is_predictive_target_value,
)
from src.contracts.case import CaseResult

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
                "consistencia_benigno_sin_familia",
                classification.attack_family is None,
                detail=f"familia={classification.attack_family}",
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
                "consistencia_malicioso_con_familia",
                classification.attack_family is not None,
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
        if detection.is_malicious and classification.attack_family is not None:
            add(
                "consistencia_malicioso_no_benigno",
                classification.attack_family.strip().lower() not in {"benign", "normal"},
                detail=f"familia={classification.attack_family}",
            )
        add(
            "consistencia_estado_vs_juez",
            case.status != "completed" or (case.judge.approved and not case.judge.requires_human_review),
            detail=f"status={case.status} approved={case.judge.approved}",
        )
        if case.explanation.source == "catalog":
            add(
                "consistencia_catalogo_sin_llm_suggested",
                all(ref.source == "catalog" for ref in case.explanation.references),
                detail="referencias llm_suggested con source=catalog",
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
            add(
                "leakage_evento_canonico_presente",
                case.status != "completed",
                detail="caso completed sin evento canonico",
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
            classification.attack_family is not None
            and classification.confidence < REVIEW_CLASSIFICATION_CONFIDENCE
        ):
            add(
                "umbral_confianza_clasificacion_derivada",
                flagged_for_review,
                hard=False,
                detail=f"confidence={classification.confidence}",
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
        """Resumen agregado para la auditoria de sistema (Fase 5b)."""
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
