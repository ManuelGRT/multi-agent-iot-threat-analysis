"""Agente final de clasificacion: tool MCP ``classify_event``.

El agente predice directamente uno de los 16 tipos de ataque desplegados y
persiste su confianza junto con el top-3 de tipos. No deriva ni agrega
familias amplias.
"""
from __future__ import annotations

from typing import Any

from src.agents.final.base import FinalAgent
from src.contracts.attack_taxonomy import (
    MULTIDATASET_TAXONOMY_VERSION,
    attack_classes_for_taxonomy,
)
from src.contracts.agents import ClassificationOutput
from src.orchestration.state import OrchestratorState


class FinalClassifier(FinalAgent):
    name = "final_classifier"

    def __init__(self, client=None, top_k: int = 3):
        super().__init__(client)
        if isinstance(top_k, bool) or top_k != 3:
            raise ValueError("El agente final requiere exactamente un top-3")
        self.top_k = top_k

    def run(self, state: OrchestratorState) -> dict[str, Any]:
        canonical = state.get("canonical_event") or {}
        event_id = str(canonical.get("event_id") or state.get("event_id") or "unknown")
        entry = self.start_entry(tool="classify_event", event_id=event_id)

        def fail(error: str) -> dict[str, Any]:
            fallback = ClassificationOutput(
                event_id=event_id,
                attack_type=None,
                confidence=0.0,
                reason=[f"error:{error}"],
                next_route="judge",
            )
            return self.record_error(
                state,
                entry,
                error,
                {
                    "classification_output": fallback.model_dump(mode="json"),
                    "route": "judge",
                },
            )

        result = self.call_tool(
            "inference", "classify_event", canonical_event=canonical, top_k=self.top_k
        )
        if not result.get("ok", False):
            return fail(str(result.get("error") or "classify_event sin resultado"))

        model_task = str(result.get("model_task") or "")
        if model_task != "attack_type":
            return fail(f"tarea_clasificador_no_soportada:{model_task}")

        taxonomy_version = result.get("taxonomy_version")
        if taxonomy_version != MULTIDATASET_TAXONOMY_VERSION:
            return fail(
                "taxonomia_clasificador_no_soportada:"
                f"{taxonomy_version!r}"
            )
        attack_classes = attack_classes_for_taxonomy(taxonomy_version)

        attack_type_raw = result.get("attack_type")
        attack_type = (
            str(attack_type_raw) if attack_type_raw not in (None, "") else None
        )
        if attack_type is None or attack_type not in attack_classes:
            return fail(
                "contrato_tipo_ataque_invalido: "
                f"tipo={attack_type!r} taxonomia={taxonomy_version!r}"
            )

        try:
            confidence = float(result.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            return fail(f"confianza_clasificador_invalida:{result.get('confidence')!r}")
        if not 0.0 <= confidence <= 1.0:
            return fail(f"confianza_clasificador_fuera_de_rango:{confidence!r}")

        try:
            decision_threshold = float(result.get("decision_threshold", 0.65))
        except (TypeError, ValueError):
            decision_threshold = -1.0
        if not 0.0 <= decision_threshold <= 1.0:
            return fail(
                f"umbral_clasificador_invalido:{result.get('decision_threshold')!r}"
            )

        raw_top_scores = result.get("top_scores") or {}
        try:
            top_scores = {
                str(name): float(score) for name, score in raw_top_scores.items()
            }
        except (AttributeError, TypeError, ValueError):
            return fail("puntuaciones_clasificador_tipo_incoherentes")

        top_label, top_confidence = (
            max(top_scores.items(), key=lambda item: item[1])
            if top_scores
            else (None, None)
        )
        invalid_contract = (
            len(top_scores) != 3
            or attack_type not in top_scores
            or any(label not in attack_classes for label in top_scores)
            or any(not 0.0 <= score <= 1.0 for score in top_scores.values())
            or top_label != attack_type
            or top_confidence is None
            or abs(top_confidence - confidence) > 1e-6
            or sum(top_scores.values()) > 1.0 + 1e-5
        )
        if invalid_contract:
            return fail("puntuaciones_clasificador_tipo_incoherentes")

        output = ClassificationOutput(
            event_id=event_id,
            attack_type=attack_type,
            confidence=confidence,
            decision_threshold=decision_threshold,
            model_name=result.get("model_name"),
            model_task="attack_type",
            taxonomy_version=str(taxonomy_version),
            top_scores=top_scores,
            reason=[
                f"model={result.get('model_name')}",
                "score_type=attack_type",
                f"top_scores={sorted(top_scores, key=top_scores.get, reverse=True)}",
            ],
            next_route="explain",
        )
        classification_output = output.model_dump(mode="json")

        entry.finish(
            status="ok",
            confidence=confidence,
            summary=f"tipo={attack_type} confianza={confidence:.4f}",
        )
        return self.trace_update(
            state,
            entry,
            {"classification_output": classification_output, "route": "explain"},
        )
