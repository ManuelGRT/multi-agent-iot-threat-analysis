# src/contracts/case.py
"""Contrato de caso con trazabilidad completa (Fase 1 del plan de cierre).

Un *caso* es la unidad auditable del sistema multiagente: agrupa la entrada
cruda, el evento canonico, las salidas de cada agente y una traza ordenada de
todo lo que ocurrio. Este contrato sigue la forma recomendada en el handoff
(handoff_llm_terminar_tfm_multiagente_mcp_20260802) y reutiliza los contratos
existentes de ``src/contracts/agents.py`` en lugar de crear otros paralelos.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

TraceStatus = Literal["started", "ok", "error", "fallback", "abstain", "skipped"]
JudgeAction = Literal["approve", "rework", "human_interrupt", "reject", "abstain"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_case_id() -> str:
    """Genera un identificador de caso unico con el prefijo del handoff."""
    return f"case-{uuid.uuid4().hex[:12]}"


class TraceEntry(BaseModel):
    """Registro atomico de la traza: una accion de un agente o tool."""

    agent: str
    tool: str | None = None
    status: TraceStatus = "ok"
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    summary: str | None = None
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def latency_ms(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds() * 1000.0

    def finish(
        self,
        status: TraceStatus = "ok",
        confidence: float | None = None,
        summary: str | None = None,
        error: str | None = None,
    ) -> "TraceEntry":
        self.finished_at = utcnow()
        self.status = status
        if confidence is not None:
            self.confidence = confidence
        if summary is not None:
            self.summary = summary
        if error is not None:
            self.error = error
        return self


class StandardizationInfo(BaseModel):
    model: str | None = None
    provider: str | None = None
    source: Literal["llm", "prestandardized"] | None = None
    mapping_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    selected_columns: list[str] = Field(default_factory=list)
    from_cache: bool = False
    cache_content_hash: str | None = None
    cache_pipeline_hash: str | None = None
    schema_profile: str | None = None
    modality: str | None = None
    abstain: bool = False
    requires_human_review: bool = False
    failure_code: str | None = None
    failure_reason: str | None = None


class DetectionInfo(BaseModel):
    is_malicious: bool | None = None
    probability: float = Field(default=0.0, ge=0.0, le=1.0)
    model_name: str | None = None
    abstain: bool = False
    next_route: str | None = None
    evidence: list[str] = Field(default_factory=list)


class ClassificationInfo(BaseModel):
    attack_type: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    decision_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    model_name: str | None = None
    model_task: Literal["attack_type"] = "attack_type"
    taxonomy_version: str | None = None
    top_scores: dict[str, float] = Field(default_factory=dict)
    reason: list[str] = Field(default_factory=list)


class ThreatReference(BaseModel):
    """Referencia auditable a conocimiento externo (MITRE ATT&CK / CAPEC)."""

    attack_id: str | None = None
    capec_id: str | None = None
    name: str | None = None
    url: str | None = None
    source: Literal["catalog", "llm_suggested"] = "catalog"


class MitigationItem(BaseModel):
    """Mitigacion individual con procedencia auditable (Fase 4).

    - ``catalog``: texto literal del catalogo threat intel.
    - ``llm``: la accion ``text``/``base`` permanece literal del catalogo;
      la redaccion del LLM vive solo en ``context`` y nunca es confiable.
    - ``llm_suggested``: aportacion del LLM SIN respaldo en el catalogo;
      nunca se presenta como conocimiento auditado.
    - ``fallback``: accion de emergencia del sistema (catalogo no disponible).
    """

    text: str
    phase: str | None = None
    source: Literal["catalog", "llm", "llm_suggested", "fallback"] = "catalog"
    base: str | None = None
    context: str | None = None
    context_trusted: bool = False


class ExplanationInfo(BaseModel):
    summary: str = ""
    evidence: list[str] = Field(default_factory=list)
    mitigations: list[str] = Field(default_factory=list)
    mitigation_items: list[MitigationItem] = Field(default_factory=list)
    references: list[ThreatReference] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    attack_type: str | None = None
    taxonomy_version: str | None = None
    catalog_scope: Literal["attack_type"] | None = None
    catalog_version: str | None = None
    catalog_taxonomy_version: str | None = None
    catalog_compatible_taxonomy_versions: list[str] = Field(default_factory=list)
    reference_quality: dict[str, str] = Field(default_factory=dict)
    source: Literal["catalog", "llm", "hybrid", "rule_based"] = "rule_based"
    model_name: str | None = None
    llm_context_summary: str | None = None
    llm_context_trusted: bool = False
    has_llm_suggested: bool = False
    first_five_catalog_anchored: bool = False
    review_reasons: list[str] = Field(default_factory=list)


class JudgeInfo(BaseModel):
    action: JudgeAction = "approve"
    approved: bool = True
    requires_human_review: bool = False
    final_label: str | None = None
    final_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    """Resultado final y auditable de un caso completo."""

    case_id: str = Field(default_factory=new_case_id)
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    status: Literal["open", "completed", "needs_human_review", "error"] = "open"

    raw_input: dict[str, Any] = Field(default_factory=dict)
    canonical_event: dict[str, Any] = Field(default_factory=dict)

    standardization: StandardizationInfo = Field(default_factory=StandardizationInfo)
    detection: DetectionInfo = Field(default_factory=DetectionInfo)
    classification: ClassificationInfo = Field(default_factory=ClassificationInfo)
    explanation: ExplanationInfo = Field(default_factory=ExplanationInfo)
    judge: JudgeInfo = Field(default_factory=JudgeInfo)

    trace: list[TraceEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def append_trace(self, entry: TraceEntry) -> TraceEntry:
        self.trace.append(entry)
        return entry

    def start_step(self, agent: str, tool: str | None = None, **detail: Any) -> TraceEntry:
        entry = TraceEntry(agent=agent, tool=tool, status="started", detail=detail)
        return self.append_trace(entry)

    def close(self) -> "CaseResult":
        self.finished_at = utcnow()
        if self.status == "open":
            if self.judge.action in {"human_interrupt", "abstain"} or self.judge.requires_human_review:
                self.status = "needs_human_review"
            elif self.errors:
                self.status = "error"
            else:
                self.status = "completed"
        return self

    # ------------------------------------------------------------------
    # Puente con el orquestador existente (OrchestratorState)
    # ------------------------------------------------------------------
    @classmethod
    def from_orchestrator_state(
        cls,
        state: dict[str, Any],
        case_id: str | None = None,
    ) -> "CaseResult":
        """Construye un CaseResult desde un OrchestratorState ya ejecutado.

        No exige que todas las salidas esten presentes: un caso benigno no
        tendra classification_output y un caso abstenido puede carecer de
        explanation_output.
        """
        ingest = state.get("ingest_output") or {}
        detection = state.get("detection_output") or {}
        classification = state.get("classification_output") or {}
        explanation = state.get("explanation_output") or {}
        judge = state.get("judge_output") or {}
        canonical = state.get("canonical_event") or {}

        raw_trace = state.get("trace") or []
        trace = [
            entry if isinstance(entry, TraceEntry) else TraceEntry(**entry)
            for entry in raw_trace
        ]

        result = cls(
            case_id=case_id or state.get("case_id") or new_case_id(),
            raw_input=state.get("raw_input") or {},
            canonical_event=canonical,
            standardization=StandardizationInfo(
                model=ingest.get("model")
                or (canonical.get("origin") or {}).get("parser")
                or (canonical.get("provenance") or {}).get("parser_version"),
                provider=ingest.get("provider"),
                source=ingest.get("source"),
                mapping_confidence=float(ingest.get("mapping_confidence", canonical.get("mapping_confidence", 0.0) or 0.0)),
                selected_columns=list(ingest.get("selected_columns") or []),
                from_cache=bool(ingest.get("from_cache", False)),
                cache_content_hash=ingest.get("cache_content_hash"),
                cache_pipeline_hash=ingest.get("cache_pipeline_hash"),
                schema_profile=canonical.get("schema_profile"),
                modality=ingest.get("modality") or canonical.get("modality"),
                abstain=bool(ingest.get("abstain", False)),
                requires_human_review=bool(
                    ingest.get("requires_human_review", False)
                ),
                failure_code=ingest.get("failure_code"),
                failure_reason=ingest.get("failure_reason"),
            ),
            detection=DetectionInfo(
                # La clase binaria de trabajo se conserva en el estado interno,
                # pero una abstencion no es un veredicto publico del caso.
                is_malicious=(
                    None
                    if detection.get("abstain")
                    else detection.get("is_malicious")
                ),
                probability=float(detection.get("probability", 0.0) or 0.0),
                model_name=detection.get("model_name"),
                abstain=bool(detection.get("abstain", False)),
                next_route=detection.get("next_route"),
                evidence=list(detection.get("evidence") or []),
            ),
            classification=ClassificationInfo(
                attack_type=classification.get("attack_type"),
                confidence=float(classification.get("confidence", 0.0) or 0.0),
                decision_threshold=float(
                    classification.get("decision_threshold", 0.65)
                ),
                model_name=classification.get("model_name"),
                model_task=classification.get("model_task") or "attack_type",
                taxonomy_version=classification.get("taxonomy_version"),
                top_scores={
                    str(name): float(score)
                    for name, score in (classification.get("top_scores") or {}).items()
                },
                reason=list(classification.get("reason") or []),
            ),
            explanation=ExplanationInfo(
                summary=explanation.get("risk_summary") or "",
                evidence=list(explanation.get("evidence") or []),
                mitigations=list(explanation.get("mitigations") or []),
                mitigation_items=[
                    item if isinstance(item, MitigationItem) else MitigationItem(**item)
                    for item in (explanation.get("mitigation_items") or [])
                ],
                references=[
                    ref if isinstance(ref, ThreatReference) else ThreatReference(**ref)
                    for ref in (explanation.get("references") or [])
                ],
                confidence=float(explanation.get("confidence", 0.0) or 0.0),
                attack_type=explanation.get("attack_type"),
                taxonomy_version=explanation.get("taxonomy_version"),
                catalog_scope=explanation.get("catalog_scope"),
                catalog_version=explanation.get("catalog_version"),
                catalog_taxonomy_version=explanation.get(
                    "catalog_taxonomy_version"
                ),
                catalog_compatible_taxonomy_versions=[
                    str(version)
                    for version in (
                        explanation.get("catalog_compatible_taxonomy_versions") or []
                    )
                ],
                reference_quality={
                    str(name): str(quality)
                    for name, quality in (
                        explanation.get("reference_quality") or {}
                    ).items()
                },
                model_name=explanation.get("model_name"),
                source=explanation.get("source") or "rule_based",
                llm_context_summary=explanation.get("llm_context_summary"),
                llm_context_trusted=bool(explanation.get("llm_context_trusted", False)),
                has_llm_suggested=bool(explanation.get("has_llm_suggested", False)),
                first_five_catalog_anchored=bool(
                    explanation.get("first_five_catalog_anchored", False)
                ),
                review_reasons=list(explanation.get("review_reasons") or []),
            ),
            judge=JudgeInfo(
                action=judge.get("action") or "approve",
                approved=bool(judge.get("approved", True)),
                requires_human_review=bool(
                    state.get("needs_human_review", False)
                    or judge.get("action") == "human_interrupt"
                ),
                final_label=judge.get("final_label"),
                final_confidence=float(judge.get("final_confidence", 0.0) or 0.0),
                issues=list(judge.get("issues") or []),
            ),
            trace=trace,
            errors=list(state.get("errors") or []),
        )
        return result.close()
