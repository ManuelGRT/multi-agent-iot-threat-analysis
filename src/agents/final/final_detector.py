# src/agents/final/final_detector.py
"""Agente final de deteccion: XGBoost global balanceado por origen.

Politica de abstencion: probabilidad en zona gris [gray_low, gray_high]
(por defecto 0.4-0.6) -> abstain=True y ruta al juez (revision humana).
Benigno tambien pasa por el juez para que todo caso quede auditado.
"""
from __future__ import annotations

from typing import Any

from src.agents.final.base import FinalAgent
from src.contracts.agents import DetectionOutput
from src.orchestration.state import OrchestratorState

GRAY_ZONE_LOW = 0.4
GRAY_ZONE_HIGH = 0.6


class FinalDetector(FinalAgent):
    name = "final_detector"

    def __init__(self, client=None, gray_low: float = GRAY_ZONE_LOW, gray_high: float = GRAY_ZONE_HIGH):
        super().__init__(client)
        if not 0.0 <= gray_low <= gray_high <= 1.0:
            raise ValueError(f"Zona gris invalida: [{gray_low}, {gray_high}]")
        self.gray_low = gray_low
        self.gray_high = gray_high

    def run(self, state: OrchestratorState) -> dict[str, Any]:
        canonical = state.get("canonical_event") or {}
        event_id = str(canonical.get("event_id") or state.get("event_id") or "unknown")
        entry = self.start_entry(tool="detect_event", event_id=event_id)

        result = self.call_tool("inference", "detect_event", canonical_event=canonical)
        if not result.get("ok", False):
            error = str(result.get("error") or "detect_event sin resultado")
            fallback = DetectionOutput(
                event_id=event_id,
                is_malicious=False,
                probability=0.0,
                evidence=[f"error:{error}"],
                model_name="unavailable",
                next_route="judge",
                abstain=True,
            )
            return self.record_error(
                state,
                entry,
                error,
                {"detection_output": fallback.model_dump(mode="json"), "route": "judge"},
            )

        probability = float(result.get("probability", 0.0) or 0.0)
        is_malicious = bool(result.get("is_malicious", probability >= 0.5))
        abstain = self.gray_low <= probability <= self.gray_high
        if abstain:
            next_route = "judge"
        elif is_malicious:
            next_route = "classify"
        else:
            next_route = "judge"  # benigno: el juez cierra el caso auditado

        output = DetectionOutput(
            event_id=event_id,
            is_malicious=is_malicious,
            probability=probability,
            evidence=[
                f"probability={probability:.4f}",
                f"gray_zone=[{self.gray_low}, {self.gray_high}]",
            ],
            model_name=str(
                result.get("model_name")
                or "xgboost_detection_balanced_by_origin_20260905"
            ),
            next_route=next_route,
            abstain=abstain,
        )
        confidence = probability if is_malicious else 1.0 - probability
        summary = (
            f"sin_veredicto p={probability:.4f} "
            f"abstencion=True ruta={next_route}"
            if abstain
            else (
                f"malicioso={is_malicious} p={probability:.4f} "
                f"abstencion=False ruta={next_route}"
            )
        )
        entry.finish(
            status="abstain" if abstain else "ok",
            confidence=confidence,
            summary=summary,
        )
        update: dict[str, Any] = {
            "detection_output": output.model_dump(mode="json"),
            "route": next_route,
        }
        return self.trace_update(state, entry, update)
