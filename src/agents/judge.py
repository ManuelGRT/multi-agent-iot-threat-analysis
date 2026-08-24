# src/agents/judge.py
from __future__ import annotations

from typing import Any

from src.contracts.agents import JudgeOutput


class RuleBasedJudge:
    def judge(self, state: dict[str, Any]) -> JudgeOutput:
        event_id = state.get("event_id") or state.get("canonical_event", {}).get("event_id") or "unknown"
        issues: list[str] = []

        ingest = state.get("ingest_output") or {}
        detection = state.get("detection_output") or {}
        classification = state.get("classification_output") or {}
        explanation = state.get("explanation_output") or {}

        mapping_confidence = float(ingest.get("mapping_confidence", 0.0))
        if mapping_confidence < 0.5:
            issues.append("mapping_confidence_below_review_threshold")

        if detection.get("abstain"):
            issues.append("detector_abstained")

        if classification and classification.get("confidence", 1.0) < 0.65:
            issues.append("classification_confidence_below_review_threshold")

        if explanation.get("requires_human_review"):
            issues.append("explanation_requested_human_review")

        if issues:
            return JudgeOutput(
                event_id=event_id,
                approved=False,
                final_label=classification.get("attack_family"),
                final_confidence=float(classification.get("confidence") or detection.get("probability") or 0.0),
                issues=issues,
                action="human_interrupt",
            )

        final_label = classification.get("attack_family")
        if final_label is None and detection:
            final_label = "malicious" if detection.get("is_malicious") else "benign"

        return JudgeOutput(
            event_id=event_id,
            approved=True,
            final_label=final_label,
            final_confidence=float(classification.get("confidence") or detection.get("probability") or 1.0),
            issues=[],
            action="approve",
        )
