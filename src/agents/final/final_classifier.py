# src/agents/final/final_classifier.py
"""Agente final de clasificacion: tool MCP ``classify_event``.

Usa el modelo XGBoost de familia balanceado por grupo y anota la familia,
la confianza y las top-k puntuaciones en el estado del caso.
"""
from __future__ import annotations

from typing import Any

from src.agents.final.base import FinalAgent
from src.contracts.attack_taxonomy import (
    attack_classes_for_taxonomy,
    broad_family_for_attack_type,
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
        subtype_raw = result.get("attack_subtype")
        subtype = str(subtype_raw) if subtype_raw not in (None, "") else None
        model_task = str(
            result.get("model_task")
            or result.get("score_type")
            or "attack_family"
        )
        taxonomy_version = result.get("taxonomy_version")
        if model_task not in {"attack_family", "attack_subtype"}:
            error = f"tarea_clasificador_no_soportada:{model_task}"
            return self.record_error(
                state,
                entry,
                error,
                {
                    "classification_output": ClassificationOutput(
                        event_id=event_id,
                        attack_family="unknown_attack",
                        confidence=0.0,
                        reason=[f"error:{error}"],
                        next_route="judge",
                    ).model_dump(mode="json"),
                    "route": "judge",
                },
            )
        subtype_classes: tuple[str, ...] = ()
        if model_task == "attack_subtype":
            try:
                subtype_classes = attack_classes_for_taxonomy(taxonomy_version)
            except ValueError:
                subtype_classes = ()
            if subtype is None or subtype not in subtype_classes:
                error = (
                    "contrato_subtipo_invalido: "
                    f"subtipo={subtype!r} taxonomia={taxonomy_version!r}"
                )
                return self.record_error(
                    state,
                    entry,
                    error,
                    {
                        "classification_output": ClassificationOutput(
                            event_id=event_id,
                            attack_family="unknown_attack",
                            confidence=0.0,
                            reason=[f"error:{error}"],
                            next_route="judge",
                        ).model_dump(mode="json"),
                        "route": "judge",
                    },
                )
        if model_task == "attack_family" and subtype is not None:
            error = f"subtipo_inesperado_para_modelo_familia:{subtype}"
            return self.record_error(
                state,
                entry,
                error,
                {
                    "classification_output": ClassificationOutput(
                        event_id=event_id,
                        attack_family="unknown_attack",
                        confidence=0.0,
                        reason=[f"error:{error}"],
                        next_route="judge",
                    ).model_dump(mode="json"),
                    "route": "judge",
                },
            )
        if subtype is not None:
            try:
                expected_family = broad_family_for_attack_type(subtype)
            except ValueError as exc:
                return self.record_error(
                    state,
                    entry,
                    str(exc),
                    {
                        "classification_output": ClassificationOutput(
                            event_id=event_id,
                            attack_family="unknown_attack",
                            confidence=0.0,
                            reason=[f"error:{exc}"],
                            next_route="judge",
                        ).model_dump(mode="json"),
                        "route": "judge",
                    },
                )
            if family != expected_family:
                error = (
                    f"subtipo_familia_incoherentes:{subtype}->{expected_family}, "
                    f"recibido={family}"
                )
                return self.record_error(
                    state,
                    entry,
                    error,
                    {
                        "classification_output": ClassificationOutput(
                            event_id=event_id,
                            attack_family="unknown_attack",
                            confidence=0.0,
                            reason=[f"error:{error}"],
                            next_route="judge",
                        ).model_dump(mode="json"),
                        "route": "judge",
                    },
                )
        confidence = float(result.get("confidence", 0.0) or 0.0)
        try:
            decision_threshold = float(result.get("decision_threshold", 0.65))
        except (TypeError, ValueError):
            decision_threshold = -1.0
        if not 0.0 <= decision_threshold <= 1.0:
            error = f"umbral_clasificador_invalido:{result.get('decision_threshold')!r}"
            return self.record_error(
                state,
                entry,
                error,
                {
                    "classification_output": ClassificationOutput(
                        event_id=event_id,
                        attack_family="unknown_attack",
                        confidence=0.0,
                        reason=[f"error:{error}"],
                        next_route="judge",
                    ).model_dump(mode="json"),
                    "route": "judge",
                },
            )
        top_scores = {
            str(name): float(score)
            for name, score in (result.get("top_scores") or {}).items()
        }
        family_scores = {
            str(name): float(score)
            for name, score in (result.get("family_scores") or {}).items()
        }
        family_confidence_raw = result.get("family_confidence")
        family_confidence = (
            float(family_confidence_raw)
            if family_confidence_raw is not None
            else family_scores.get(family)
        )
        if model_task == "attack_subtype":
            top_label, top_confidence = (
                max(top_scores.items(), key=lambda item: item[1])
                if top_scores
                else (None, None)
            )
            family_score = family_scores.get(family)
            invalid_contract = (
                len(top_scores) != 3
                or subtype not in top_scores
                or any(label not in subtype_classes for label in top_scores)
                or any(not 0.0 <= score <= 1.0 for score in top_scores.values())
                or top_label != subtype
                or top_confidence is None
                or abs(top_confidence - confidence) > 1e-6
                or not family_scores
                or any(not 0.0 <= score <= 1.0 for score in family_scores.values())
                or abs(sum(family_scores.values()) - 1.0) > 1e-5
                or family_score is None
                or family_confidence is None
                or abs(family_score - family_confidence) > 1e-6
            )
            if invalid_contract:
                error = "puntuaciones_clasificador_subtipo_incoherentes"
                return self.record_error(
                    state,
                    entry,
                    error,
                    {
                        "classification_output": ClassificationOutput(
                            event_id=event_id,
                            attack_family="unknown_attack",
                            confidence=0.0,
                            reason=[f"error:{error}"],
                            next_route="judge",
                        ).model_dump(mode="json"),
                        "route": "judge",
                    },
                )
        output = ClassificationOutput(
            event_id=event_id,
            attack_family=family,
            attack_subtype=subtype,
            confidence=confidence,
            family_confidence=family_confidence,
            decision_threshold=decision_threshold,
            model_name=result.get("model_name"),
            model_task=model_task,
            taxonomy_version=(
                str(taxonomy_version) if taxonomy_version is not None else None
            ),
            top_scores=top_scores,
            family_scores=family_scores,
            reason=[
                f"model={result.get('model_name')}",
                f"score_type={model_task}",
                f"top_scores={sorted(top_scores, key=top_scores.get, reverse=True)}",
            ],
            next_route="explain",
        )
        classification_output = output.model_dump(mode="json")

        entry.finish(
            status="ok",
            confidence=confidence,
            summary=(
                f"tipo={subtype} familia={family} confianza={confidence:.4f}"
                if subtype is not None
                else f"familia={family} confianza={confidence:.4f}"
            ),
        )
        update: dict[str, Any] = {
            "classification_output": classification_output,
            "route": "explain",
        }
        return self.trace_update(state, entry, update)
