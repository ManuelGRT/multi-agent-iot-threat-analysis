# src/agents/final/final_judge.py
"""Juez operacional: aplica umbrales y reglas al estado completo del caso.

Comprueba la confianza de estandarizacion y clasificacion, la abstencion del
detector, las peticiones del mitigador, los errores del flujo y la presencia
de las salidas obligatorias. El auditor posterior es un agente independiente.
"""
from __future__ import annotations

import math
from typing import Any

from src.contracts.attack_taxonomy import (
    MULTIDATASET_ATTACK_CLASSES,
    MULTIDATASET_TAXONOMY_VERSION,
)
from src.agents.final.base import FinalAgent
from src.contracts.agents import JudgeOutput
from src.mcp.threat_catalog import (
    THREAT_INTEL_CATALOG_VERSION,
    catalog_output_issues,
)
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
            or "attack_type"
        )
        attack_classes = MULTIDATASET_ATTACK_CLASSES
        classification_confidence = 0.0
        if classification and "decision_threshold" not in classification:
            issues.append("classification_threshold_missing")
        raw_classification_threshold = classification.get(
            "decision_threshold", REVIEW_CLASSIFICATION_CONFIDENCE
        )
        try:
            received_classification_threshold = float(raw_classification_threshold)
        except (TypeError, ValueError):
            issues.append("classification_threshold_invalid")
        else:
            if (
                not math.isfinite(received_classification_threshold)
                or not 0.0 <= received_classification_threshold <= 1.0
            ):
                issues.append("classification_threshold_invalid")
            elif (
                abs(
                    received_classification_threshold
                    - REVIEW_CLASSIFICATION_CONFIDENCE
                )
                > 1e-9
            ):
                issues.append("classification_threshold_mismatch")
        # El umbral es parte del contrato desplegado, no un parametro que una
        # salida persistida pueda rebajar para evitar la revision humana.
        classification_threshold = REVIEW_CLASSIFICATION_CONFIDENCE
        if classification and model_task != "attack_type":
            issues.append("classification_model_task_invalid")
        if classification:
            attack_type = classification.get("attack_type")
            if not attack_type:
                issues.append("classification_attack_type_missing")
            if classification.get("taxonomy_version") != MULTIDATASET_TAXONOMY_VERSION:
                issues.append("classification_taxonomy_version_invalid")
            if attack_type and str(attack_type) not in attack_classes:
                issues.append("classification_attack_type_outside_taxonomy")

            try:
                classification_confidence = float(
                    classification.get("confidence", 0.0)
                )
            except (TypeError, ValueError):
                issues.append("classification_confidence_invalid")
            else:
                if not math.isfinite(classification_confidence) or not (
                    0.0 <= classification_confidence <= 1.0
                ):
                    issues.append("classification_confidence_invalid")
                    classification_confidence = 0.0

            raw_top_scores = classification.get("top_scores")
            top_scores: dict[str, float] = {}
            top_scores_valid = isinstance(raw_top_scores, dict)
            if top_scores_valid:
                for label, raw_score in raw_top_scores.items():
                    if (
                        not isinstance(label, str)
                        or label not in attack_classes
                        or isinstance(raw_score, bool)
                    ):
                        top_scores_valid = False
                        break
                    try:
                        score = float(raw_score)
                    except (TypeError, ValueError):
                        top_scores_valid = False
                        break
                    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                        top_scores_valid = False
                        break
                    top_scores[label] = score

            if top_scores_valid and len(top_scores) == 3:
                top_label, top_confidence = max(
                    top_scores.items(), key=lambda item: item[1]
                )
                top_scores_valid = (
                    top_label == attack_type
                    and math.isfinite(classification_confidence)
                    and abs(top_confidence - classification_confidence) <= 1e-6
                )
            else:
                top_scores_valid = False
            if not top_scores_valid:
                issues.append("classification_top_scores_invalid")
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
            elif detection.get("is_malicious") is None:
                issues.append("detection_verdict_missing")
            else:
                probability = float(detection.get("probability", 0.0) or 0.0)
                if bool(detection.get("is_malicious")) != (probability >= 0.5):
                    issues.append("detection_label_probability_mismatch")
        if detector_abstained:
            issues.append("detector_abstained")
        if (
            not detector_abstained
            and detection.get("is_malicious")
            and classification
            and classification_confidence < classification_threshold
        ):
            issues.append("classification_confidence_below_review_threshold")
        if (
            not detector_abstained
            and detection.get("is_malicious")
            and not classification
        ):
            issues.append("malicious_without_classification")
        attack_type = classification.get("attack_type")
        if (
            not detector_abstained
            and detection.get("is_malicious")
            and not explanation
        ):
            issues.append("malicious_without_mitigation")
        if detection.get("is_malicious") is False and classification:
            issues.append("benign_with_classification")
        if (
            not detector_abstained
            and detection.get("is_malicious")
            and model_task == "attack_type"
            and attack_type
            and explanation
        ):
            if explanation.get("attack_type") != attack_type:
                issues.append("mitigation_attack_type_mismatch")
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
            if (
                explanation.get("catalog_version")
                != THREAT_INTEL_CATALOG_VERSION
            ):
                issues.append("mitigation_catalog_version_mismatch")
            catalog_issues = catalog_output_issues(
                explanation,
                attack_type=str(attack_type),
            )
            if catalog_issues:
                issues.extend(
                    f"mitigation_catalog_content_{issue}"
                    for issue in catalog_issues
                )
        if explanation.get("requires_human_review"):
            issues.append("mitigator_requested_human_review")

        probability = float(detection.get("probability", 0.0) or 0.0)
        if standardizer_abstained or detector_abstained:
            final_label = None
            final_confidence = 0.0
        elif (
            detection.get("is_malicious") is True
            and classification.get("attack_type")
        ):
            # La etiqueta final es el tipo concreto predicho entre los 16
            # desplegados; el flujo no deriva una familia amplia paralela.
            final_label = str(classification["attack_type"])
            final_confidence = classification_confidence
        elif detection.get("is_malicious") is True:
            final_label = "malicious"
            final_confidence = probability
        elif detection.get("is_malicious") is False:
            final_label = "benign"
            final_confidence = 1.0 - probability
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
