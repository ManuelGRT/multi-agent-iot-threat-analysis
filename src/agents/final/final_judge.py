# src/agents/final/final_judge.py
"""Juez final por caso: reglas de consistencia sobre el estado completo.

Extiende los umbrales del juez original (mapping_confidence, abstencion del
detector, confianza de clasificacion, revision pedida por el mitigador) con
chequeos propios del flujo final: errores de pipeline y deteccion ausente.
La auditoria E2E completa llega en la Fase 5 (auditor).
"""
from __future__ import annotations

from typing import Any

from src.agents.final.base import FinalAgent
from src.contracts.agents import JudgeOutput
from src.orchestration.state import OrchestratorState

REVIEW_MAPPING_CONFIDENCE = 0.5
REVIEW_CLASSIFICATION_CONFIDENCE = 0.65


class FinalJudge(FinalAgent):
    name = "final_judge"

    def run(self, state: OrchestratorState) -> dict[str, Any]:
        canonical = state.get("canonical_event") or {}
        event_id = str(canonical.get("event_id") or state.get("event_id") or "unknown")
        entry = self.start_entry(tool=None, event_id=event_id)

        ingest = state.get("ingest_output") or {}
        detection = state.get("detection_output") or {}
        classification = state.get("classification_output") or {}
        explanation = state.get("explanation_output") or {}
        errors = list(state.get("errors") or [])

        issues: list[str] = []
        standardizer_abstained = bool(ingest.get("abstain"))
        if errors:
            issues.append("pipeline_errors_present")
        if standardizer_abstained:
            issues.append("standardizer_abstained")
        else:
            if (
                float(ingest.get("mapping_confidence", 0.0) or 0.0)
                < REVIEW_MAPPING_CONFIDENCE
            ):
                issues.append("mapping_confidence_below_review_threshold")
            if not detection:
                issues.append("detection_missing")
            elif detection.get("is_malicious") is not None:
                probability = float(detection.get("probability", 0.0) or 0.0)
                if bool(detection.get("is_malicious")) != (probability >= 0.5):
                    issues.append("detection_label_probability_mismatch")
        if detection.get("abstain"):
            issues.append("detector_abstained")
        if (
            detection.get("is_malicious")
            and classification
            and float(classification.get("confidence", 1.0) or 0.0)
            < REVIEW_CLASSIFICATION_CONFIDENCE
        ):
            issues.append("classification_confidence_below_review_threshold")
        if detection.get("is_malicious") and not classification:
            issues.append("malicious_without_classification")
        if detection.get("is_malicious") and str(
            classification.get("attack_family") or ""
        ).strip().lower() in {"benign", "normal"}:
            issues.append("malicious_classified_as_benign")
        if detection.get("is_malicious") and not explanation:
            issues.append("malicious_without_mitigation")
        if explanation.get("requires_human_review"):
            issues.append("mitigator_requested_human_review")

        probability = float(detection.get("probability", 0.0) or 0.0)
        if classification.get("attack_family"):
            final_label = str(classification["attack_family"])
            final_confidence = float(classification.get("confidence", 0.0) or 0.0)
        elif detection:
            final_label = "malicious" if detection.get("is_malicious") else "benign"
            final_confidence = probability if detection.get("is_malicious") else 1.0 - probability
        else:
            final_label = None
            final_confidence = 0.0

        action = "human_interrupt" if issues else "approve"
        output = JudgeOutput(
            event_id=event_id,
            approved=not issues,
            final_label=final_label,
            final_confidence=min(max(final_confidence, 0.0), 1.0),
            issues=issues,
            action=action,
        )
        entry.finish(
            status="ok" if not issues else "abstain",
            confidence=output.final_confidence,
            summary=f"accion={action} etiqueta={final_label} issues={issues or 'ninguno'}",
        )
        update: dict[str, Any] = {
            "judge_output": output.model_dump(mode="json"),
            "needs_human_review": bool(issues) or bool(state.get("needs_human_review")),
            "route": "end",
        }
        return self.trace_update(state, entry, update)
