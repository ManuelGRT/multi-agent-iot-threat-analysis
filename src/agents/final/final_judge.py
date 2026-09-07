# src/agents/final/final_judge.py
"""Juez final por caso: reglas de consistencia sobre el estado completo.

Extiende los umbrales del juez original (mapping_confidence, abstencion del
detector, confianza de clasificacion, revision pedida por el mitigador) con
chequeos propios del flujo final: errores de pipeline y deteccion ausente.
La auditoria E2E completa llega en la Fase 5 (auditor).
"""
from __future__ import annotations

from typing import Any

from src.contracts.attack_taxonomy import (
    MULTIDATASET_TAXONOMY_VERSION,
    attack_classes_for_taxonomy,
    broad_family_for_attack_type,
)
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
        model_task = str(
            classification.get("model_task")
            or classification.get("score_type")
            or "attack_family"
        )
        subtype_classes: tuple[str, ...] = ()
        try:
            classification_threshold = float(
                classification.get(
                    "decision_threshold", REVIEW_CLASSIFICATION_CONFIDENCE
                )
            )
        except (TypeError, ValueError):
            classification_threshold = REVIEW_CLASSIFICATION_CONFIDENCE
            issues.append("classification_threshold_invalid")
        else:
            if not 0.0 <= classification_threshold <= 1.0:
                classification_threshold = REVIEW_CLASSIFICATION_CONFIDENCE
                issues.append("classification_threshold_invalid")
        if classification and model_task not in {"attack_family", "attack_subtype"}:
            issues.append("classification_model_task_invalid")
        if classification and model_task == "attack_subtype":
            subtype_value = classification.get("attack_subtype")
            if not subtype_value:
                issues.append("classification_subtype_missing")
            try:
                subtype_classes = attack_classes_for_taxonomy(
                    classification.get("taxonomy_version")
                )
            except ValueError:
                issues.append("classification_taxonomy_version_invalid")
            else:
                if subtype_value and str(subtype_value) not in subtype_classes:
                    issues.append("classification_subtype_outside_taxonomy")
        if (
            classification
            and model_task == "attack_family"
            and classification.get("attack_subtype") is not None
        ):
            issues.append("classification_subtype_unexpected_for_family_model")
        standardizer_abstained = bool(ingest.get("abstain"))
        detector_abstained = bool(detection.get("abstain"))
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
        if detector_abstained:
            issues.append("detector_abstained")
        if (
            not detector_abstained
            and detection.get("is_malicious")
            and classification
            and float(classification.get("confidence", 1.0) or 0.0)
            < classification_threshold
        ):
            issues.append("classification_confidence_below_review_threshold")
        if (
            not detector_abstained
            and detection.get("is_malicious")
            and not classification
        ):
            issues.append("malicious_without_classification")
        if (
            not detector_abstained
            and detection.get("is_malicious")
            and str(classification.get("attack_family") or "").strip().lower()
            in {"benign", "normal"}
        ):
            issues.append("malicious_classified_as_benign")
        subtype = classification.get("attack_subtype")
        if (
            not detector_abstained
            and detection.get("is_malicious")
            and subtype
            and str(subtype) in subtype_classes
        ):
            try:
                expected_family = broad_family_for_attack_type(str(subtype))
            except ValueError:
                issues.append("classification_subtype_outside_taxonomy")
            else:
                if classification.get("attack_family") != expected_family:
                    issues.append("classification_subtype_family_mismatch")
        if (
            not detector_abstained
            and detection.get("is_malicious")
            and not explanation
        ):
            issues.append("malicious_without_mitigation")
        if (
            not detector_abstained
            and detection.get("is_malicious")
            and model_task == "attack_subtype"
            and subtype
            and explanation
        ):
            if explanation.get("attack_subtype") != subtype:
                issues.append("mitigation_attack_subtype_mismatch")
            if explanation.get("attack_family") != classification.get("attack_family"):
                issues.append("mitigation_attack_family_mismatch")
            if explanation.get("taxonomy_version") != classification.get(
                "taxonomy_version"
            ):
                issues.append("mitigation_taxonomy_version_mismatch")
            if (
                explanation.get("catalog_taxonomy_version")
                != MULTIDATASET_TAXONOMY_VERSION
                or classification.get("taxonomy_version")
                not in (
                    explanation.get("catalog_compatible_taxonomy_versions") or []
                )
            ):
                issues.append("mitigation_catalog_taxonomy_version_mismatch")
            if explanation.get("catalog_scope") != "attack_type":
                issues.append("mitigation_catalog_scope_invalid")
            if not explanation.get("catalog_version"):
                issues.append("mitigation_catalog_version_missing")
        if explanation.get("requires_human_review"):
            issues.append("mitigator_requested_human_review")

        probability = float(detection.get("probability", 0.0) or 0.0)
        if standardizer_abstained or detector_abstained:
            final_label = None
            final_confidence = 0.0
        elif model_task == "attack_subtype" and classification.get("attack_subtype"):
            # La confianza del clasificador tipado corresponde al subtipo, no a
            # la probabilidad agregada de su familia amplia.
            final_label = str(classification["attack_subtype"])
            final_confidence = float(classification.get("confidence", 0.0) or 0.0)
        elif model_task == "attack_family" and classification.get("attack_family"):
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
