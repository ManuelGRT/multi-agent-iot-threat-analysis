# src/agents/final/final_mitigator.py
"""Agente mitigador: catalogo verificable y contextualizacion LLM anclada.

Flujo en dos pasos:
1. Determinista: ``threat_intel.suggest_mitigations(attack_type, ...)``
   devuelve la base auditable especifica para una de las 16 clases operativas
   (mitigaciones por fase + referencias MITRE ATT&CK/CAPEC). El tipo es
   obligatorio y nunca se sustituye por una familia amplia. Este paso siempre
   se ejecuta para los eventos maliciosos; en los benignos no hay mitigacion.
2. LLM (opcional, ``llm=LLMMitigationAgent``): contextualiza las mitigaciones
   del catalogo al evento concreto (puertos, protocolo, telemetria). Regla
   dura: no puede inventar tecnicas ni referencias fuera del catalogo; toda
   aportacion sin respaldo queda marcada ``llm_suggested``. Si el LLM falla,
   fallback total al modo catalogo (la demo nunca se rompe).

``source`` de la salida: ``catalog`` (solo paso 1) o ``hybrid`` (1+2).
"""
from __future__ import annotations

from typing import Any

from src.agents.final.base import FinalAgent
from src.agents.final.llm_mitigator import (
    LLMMitigationAgent,
    anchor_llm_payload,
    build_base_items,
    catalog_reference_ids,
)
from src.contracts.agents import ExplanationOutput
from src.orchestration.state import OrchestratorState


def catalog_references(references: dict[str, Any]) -> list[dict[str, Any]]:
    """Convierte las referencias del catalogo en ThreatReference serializados."""
    items: list[dict[str, Any]] = []
    for technique in references.get("attack_techniques") or []:
        items.append(
            {
                "attack_id": technique.get("id"),
                "capec_id": None,
                "name": technique.get("name"),
                "url": technique.get("url"),
                "source": "catalog",
            }
        )
    for pattern in references.get("capec_patterns") or []:
        items.append(
            {
                "attack_id": None,
                "capec_id": pattern.get("id"),
                "name": pattern.get("name"),
                "url": pattern.get("url"),
                "source": "catalog",
            }
        )
    for mitigation in references.get("attack_mitigation_refs") or []:
        items.append(
            {
                "attack_id": mitigation.get("id"),
                "capec_id": None,
                "name": mitigation.get("name"),
                "url": mitigation.get("url"),
                "source": "catalog",
            }
        )
    return items


class FinalMitigator(FinalAgent):
    name = "final_mitigator"

    def __init__(self, client=None, llm: LLMMitigationAgent | None = None):
        super().__init__(client)
        self.llm = llm

    def run(self, state: OrchestratorState) -> dict[str, Any]:
        canonical = state.get("canonical_event") or {}
        detection = state.get("detection_output") or {}
        classification = state.get("classification_output") or {}
        event_id = str(canonical.get("event_id") or state.get("event_id") or "unknown")

        raw_attack_type = classification.get("attack_type")
        attack_type = str(raw_attack_type) if raw_attack_type else None
        taxonomy_version = classification.get("taxonomy_version")
        schema_profile = canonical.get("schema_profile")

        # El detector gestiona la clase Normal. Esta rama defensiva conserva
        # el comportamiento benigno sin inventar un decimoséptimo tipo ni
        # consultar una entrada familiar del catalogo.
        if not bool(detection.get("is_malicious", False)):
            confidence = 1.0 - float(detection.get("probability", 0.0) or 0.0)
            confidence = min(max(confidence, 0.0), 1.0)
            entry = self.start_entry(tool=None, event_id=event_id)
            risk_summary = (
                f"Evento {event_id} evaluado como benigno "
                f"(p_maliciosa={float(detection.get('probability', 0.0) or 0.0):.4f}). "
                "No requiere mitigacion."
            )
            output = ExplanationOutput(
                event_id=event_id,
                risk_summary=risk_summary,
                mitigations=[],
                confidence=confidence,
                requires_human_review=False,
                next_route="judge",
            ).model_dump(mode="json")
            output.update(
                {
                    "references": [],
                    "mitigation_items": [],
                    "source": "rule_based",
                    "evidence": [],
                    "has_llm_suggested": False,
                    "first_five_catalog_anchored": False,
                    "review_reasons": [],
                }
            )
            entry.finish(
                status="ok",
                confidence=confidence,
                summary="evento benigno; mitigacion no aplicable",
            )
            return self.trace_update(
                state,
                entry,
                {
                    "explanation_output": output,
                    "needs_human_review": False,
                    "route": "judge",
                },
            )

        # ------------------------------------------------------------------
        # Paso 1 (siempre): base determinista del catalogo threat intel
        # ------------------------------------------------------------------
        entry = self.start_entry(
            tool="suggest_mitigations",
            event_id=event_id,
            attack_type=attack_type,
        )

        if attack_type is None:
            error = "El clasificador no proporciono un tipo de ataque"
            fallback = ExplanationOutput(
                event_id=event_id,
                risk_summary="No puede consultarse el catalogo sin un tipo de ataque.",
                mitigations=["escalate_to_analyst"],
                confidence=0.0,
                requires_human_review=True,
                next_route="judge",
            ).model_dump(mode="json")
            fallback.update(
                {
                    "references": [],
                    "mitigation_items": [
                        {
                            "text": "escalate_to_analyst",
                            "phase": None,
                            "source": "fallback",
                            "base": None,
                        }
                    ],
                    "source": "catalog",
                    "has_llm_suggested": False,
                    "first_five_catalog_anchored": False,
                    "review_reasons": ["classification_attack_type_missing"],
                    "attack_type": None,
                    "taxonomy_version": taxonomy_version,
                    "catalog_scope": None,
                    "catalog_version": None,
                    "catalog_taxonomy_version": None,
                    "catalog_compatible_taxonomy_versions": [],
                    "reference_quality": {},
                }
            )
            return self.record_error(
                state,
                entry,
                error,
                {
                    "explanation_output": fallback,
                    "needs_human_review": True,
                    "route": "judge",
                },
            )

        result = self.call_tool(
            "threat_intel",
            "suggest_mitigations",
            schema_profile=schema_profile,
            attack_type=attack_type,
        )
        if not result.get("ok", False):
            error = str(result.get("error") or "suggest_mitigations sin resultado")
            fallback = ExplanationOutput(
                event_id=event_id,
                risk_summary=(
                    "Sin catalogo threat intel disponible para el tipo "
                    f"{attack_type}."
                ),
                mitigations=["escalate_to_analyst"],
                confidence=0.0,
                requires_human_review=True,
                next_route="judge",
            )
            payload = fallback.model_dump(mode="json")
            payload["references"] = []
            payload["mitigation_items"] = [
                {
                    "text": "escalate_to_analyst",
                    "phase": None,
                    "source": "fallback",
                    "base": None,
                }
            ]
            payload["source"] = "catalog"
            payload["has_llm_suggested"] = False
            payload["first_five_catalog_anchored"] = False
            payload["review_reasons"] = ["catalog_attack_type_unavailable"]
            payload["attack_type"] = attack_type
            payload["taxonomy_version"] = taxonomy_version
            payload["catalog_scope"] = None
            payload["catalog_version"] = None
            payload["catalog_taxonomy_version"] = None
            payload["catalog_compatible_taxonomy_versions"] = []
            payload["reference_quality"] = {}
            return self.record_error(
                state,
                entry,
                error,
                {
                    "explanation_output": payload,
                    "needs_human_review": True,
                    "route": "judge",
                },
            )

        normalized_attack_type = (
            str(result["attack_type"])
            if result.get("attack_type") is not None
            else None
        )
        catalog_scope = str(result.get("catalog_scope") or "attack_type")
        catalog_version = (
            str(result["catalog_version"])
            if result.get("catalog_version") is not None
            else None
        )
        catalog_taxonomy_version = (
            str(result["taxonomy_version"])
            if result.get("taxonomy_version") is not None
            else None
        )
        catalog_compatible_taxonomy_versions = [
            str(version)
            for version in (result.get("compatible_taxonomy_versions") or [])
        ]
        base_items = build_base_items(result)
        references = catalog_references(result.get("references") or {})
        known_ids = catalog_reference_ids(result.get("references") or {})
        mitigation_items = [
            {"text": item["text"], "phase": item["phase"], "source": "catalog", "base": None}
            for item in base_items
        ]
        if classification.get("confidence") is not None:
            confidence = float(classification["confidence"])
        elif not detection.get("is_malicious"):
            confidence = 1.0 - float(detection.get("probability", 0.0) or 0.0)
        else:
            confidence = float(detection.get("probability", 0.0) or 0.0)
        confidence = min(max(confidence, 0.0), 1.0)
        review_reasons: list[str] = []
        if normalized_attack_type != attack_type:
            review_reasons.append("catalog_attack_type_mismatch")
        if catalog_scope != "attack_type":
            review_reasons.append("catalog_attack_type_scope_missing")
        if taxonomy_version not in catalog_compatible_taxonomy_versions:
            review_reasons.append("catalog_taxonomy_version_mismatch")
        requires_review = bool(review_reasons)

        attack_ids = [ref["attack_id"] for ref in references if ref.get("attack_id")]
        risk_summary = (
            f"Evento {event_id} clasificado como {normalized_attack_type} "
            f"(confianza {float(classification.get('confidence', 0.0) or 0.0):.4f}) "
            f"sobre perfil {schema_profile or 'unknown'}. "
            f"Referencias ATT&CK: {', '.join(attack_ids[:3]) or 'n/a'}."
        )

        catalog_payload = {
            "risk_summary": risk_summary,
            "mitigation_items": mitigation_items,
            "mitigations": [item["text"] for item in mitigation_items],
            "references": references,
            "confidence": confidence,
            "requires_human_review": requires_review,
            "has_llm_suggested": False,
            "first_five_catalog_anchored": False,
            "review_reasons": review_reasons,
            "source": "catalog",
            "attack_type": normalized_attack_type,
            "taxonomy_version": taxonomy_version,
            "catalog_scope": catalog_scope,
            "catalog_version": catalog_version,
            "catalog_taxonomy_version": catalog_taxonomy_version,
            "catalog_compatible_taxonomy_versions": (
                catalog_compatible_taxonomy_versions
            ),
            "reference_quality": dict(result.get("reference_quality") or {}),
        }
        entry.finish(
            status="ok",
            confidence=confidence,
            summary=(
                f"tipo={normalized_attack_type or 'n/a'} "
                f"mitigaciones={len(mitigation_items)} "
                f"referencias={len(references)}"
            ),
        )
        trace_entries = [entry]

        # ------------------------------------------------------------------
        # Paso 2 (opcional): contextualizacion LLM anclada al catalogo
        # ------------------------------------------------------------------
        final_payload = catalog_payload
        model_name: str | None = None
        if self.llm is not None:
            llm_entry = self.start_entry(
                tool="llm_contextualize",
                event_id=event_id,
                attack_type=normalized_attack_type,
            )
            try:
                raw = self.llm.contextualize(
                    canonical_event=canonical,
                    detection=detection,
                    classification=classification,
                    catalog_result=result,
                    base_items=base_items,
                )
                anchored = anchor_llm_payload(
                    raw,
                    base_items=base_items,
                    catalog_references=references,
                    known_reference_ids=known_ids,
                    fallback_summary=risk_summary,
                    fallback_confidence=confidence,
                )
                review_reasons = list(anchored.get("review_reasons") or [])
                for reason in catalog_payload["review_reasons"]:
                    if reason not in review_reasons:
                        review_reasons.append(reason)
                anchored["review_reasons"] = review_reasons
                anchored["requires_human_review"] = bool(review_reasons)
                for key in (
                    "attack_type",
                    "taxonomy_version",
                    "catalog_scope",
                    "catalog_version",
                    "catalog_taxonomy_version",
                    "catalog_compatible_taxonomy_versions",
                    "reference_quality",
                ):
                    anchored[key] = catalog_payload[key]
                final_payload = anchored
                model_name = self.llm.model_name
                suggested = sum(
                    1
                    for item in anchored["mitigation_items"]
                    if item["source"] == "llm_suggested"
                )
                llm_entry.finish(
                    status="ok",
                    confidence=anchored["confidence"],
                    summary=(
                        f"contextualizadas={sum(1 for i in anchored['mitigation_items'] if i['source'] == 'llm')} "
                        f"llm_suggested={suggested} "
                        f"primeras_5_ancladas={anchored['first_five_catalog_anchored']} "
                        f"modelo={model_name}"
                    ),
                )
            except Exception as exc:  # fallback total: la demo nunca se rompe
                llm_entry.finish(
                    status="fallback",
                    summary="LLM no disponible; se mantiene el modo catalogo",
                    error=f"{type(exc).__name__}: {exc}",
                )
            trace_entries.append(llm_entry)

        output = ExplanationOutput(
            event_id=event_id,
            risk_summary=final_payload["risk_summary"],
            mitigations=final_payload["mitigations"],
            confidence=final_payload["confidence"],
            requires_human_review=final_payload["requires_human_review"],
            next_route="judge",
            model_name=model_name,
        )
        explanation_output = output.model_dump(mode="json")
        # Metadatos extra para ExplanationInfo del CaseResult.
        explanation_output["references"] = final_payload["references"]
        explanation_output["mitigation_items"] = final_payload["mitigation_items"]
        explanation_output["source"] = final_payload["source"]
        explanation_output["evidence"] = list(final_payload.get("evidence") or [])
        if final_payload.get("llm_context_summary"):
            explanation_output["llm_context_summary"] = final_payload["llm_context_summary"]
            explanation_output["llm_context_trusted"] = False
        if final_payload.get("has_llm_suggested"):
            explanation_output["has_llm_suggested"] = True
        explanation_output["first_five_catalog_anchored"] = bool(
            final_payload.get("first_five_catalog_anchored", False)
        )
        explanation_output["review_reasons"] = list(
            final_payload.get("review_reasons") or []
        )
        explanation_output["attack_type"] = final_payload.get("attack_type")
        explanation_output["taxonomy_version"] = final_payload.get("taxonomy_version")
        explanation_output["catalog_scope"] = final_payload.get("catalog_scope")
        explanation_output["catalog_version"] = final_payload.get("catalog_version")
        explanation_output["catalog_taxonomy_version"] = final_payload.get(
            "catalog_taxonomy_version"
        )
        explanation_output["catalog_compatible_taxonomy_versions"] = list(
            final_payload.get("catalog_compatible_taxonomy_versions") or []
        )
        explanation_output["reference_quality"] = dict(
            final_payload.get("reference_quality") or {}
        )

        existing_trace = list(state.get("trace") or [])
        existing_trace.extend(e.model_dump(mode="json") for e in trace_entries)
        return {
            "explanation_output": explanation_output,
            "needs_human_review": final_payload["requires_human_review"],
            "route": "judge",
            "trace": existing_trace,
        }
