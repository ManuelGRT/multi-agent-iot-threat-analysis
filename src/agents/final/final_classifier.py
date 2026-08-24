# src/agents/final/final_classifier.py
"""Agente final de clasificacion: tool MCP ``classify_event``.

Usa el modelo XGBoost de familia balanceado por grupo y anota la familia,
la confianza y las top-k puntuaciones en el estado del caso.
"""
from __future__ import annotations

from typing import Any

from src.agents.final.base import FinalAgent
from src.contracts.agents import ClassificationOutput
from src.orchestration.state import OrchestratorState


class FinalClassifier(FinalAgent):
    name = "final_classifier"

    def __init__(self, client=None, top_k: int = 3):
        super().__init__(client)
        self.top_k = top_k

    def run(self, state: OrchestratorState) -> dict[str, Any]:
        canonical = state.get("canonical_event") or {}
        event_id = str(canonical.get("event_id") or state.get("event_id") or "unknown")
        entry = self.start_entry(tool="classify_event", event_id=event_id)

        result = self.call_tool(
            "inference", "classify_event", canonical_event=canonical, top_k=self.top_k
        )
        if not result.get("ok", False):
            error = str(result.get("error") or "classify_event sin resultado")
            fallback = ClassificationOutput(
                event_id=event_id,
                attack_family="unknown_attack",
                confidence=0.0,
                reason=[f"error:{error}"],
                next_route="judge",
            )
            return self.record_error(
                state,
                entry,
                error,
                {"classification_output": fallback.model_dump(mode="json"), "route": "judge"},
            )

        family = str(result.get("attack_family") or "unknown_attack")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        top_scores = {
            str(name): float(score)
            for name, score in (result.get("top_scores") or {}).items()
        }
        output = ClassificationOutput(
            event_id=event_id,
            attack_family=family,
            attack_subtype=None,
            confidence=confidence,
            reason=[
                f"model={result.get('model_name')}",
                f"top_scores={sorted(top_scores, key=top_scores.get, reverse=True)}",
            ],
            next_route="explain",
        )
        classification_output = output.model_dump(mode="json")
        # Metadatos extra para ClassificationInfo del CaseResult.
        classification_output["top_scores"] = top_scores
        classification_output["model_name"] = result.get("model_name")

        entry.finish(
            status="ok",
            confidence=confidence,
            summary=f"familia={family} confianza={confidence:.4f}",
        )
        update: dict[str, Any] = {
            "classification_output": classification_output,
            "route": "explain",
        }
        return self.trace_update(state, entry, update)
