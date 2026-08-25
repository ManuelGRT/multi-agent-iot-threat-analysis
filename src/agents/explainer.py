# src/agents/explainer.py
from __future__ import annotations

import os
from typing import Any

from src.agents.base import OllamaChatAgent, OpenRouterChatAgent, TransformersChatAgent
from src.contracts.agents import ClassificationOutput, ExplanationOutput
from src.contracts.canonical import CanonicalEvent


EXPLAINER_SYSTEM = """
Eres el agente de mitigacion y explicacion.
Recibes un evento canonico ya estandarizado y una clasificacion de amenaza.
Tu trabajo es producir una explicacion breve del riesgo y una lista de mitigaciones accionables.
Usa el perfil de esquema, la familia de ataque, las senales de comportamiento y el contexto tecnico.
No uses nombres de datasets, ficheros u origenes como reglas de decision.
Si la clasificacion es benigna o no hay amenaza, devuelve mitigaciones vacias o de monitorizacion minima.
Si falta contexto importante o la confianza es baja, marca requires_human_review=true.
Responde exclusivamente con JSON valido.
"""


EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "event_id": {"type": "string"},
        "risk_summary": {"type": "string"},
        "mitigations": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "requires_human_review": {"type": "boolean"},
        "next_route": {"type": "string", "enum": ["judge", "end"]},
        "model_name": {"type": ["string", "null"]},
    },
    "required": [
        "event_id",
        "risk_summary",
        "mitigations",
        "confidence",
        "requires_human_review",
        "next_route",
    ],
}


MITIGATIONS_BY_FAMILY = {
    "ddos": ["rate_limit_traffic", "review_firewall_rules", "enable_upstream_ddos_protection"],
    "scanning": ["block_source_if_repeated", "harden_exposed_services", "increase_scan_monitoring"],
    "botnet": ["isolate_device", "rotate_credentials", "inspect_c2_indicators"],
    "bruteforce": ["enforce_lockout_policy", "rotate_weak_credentials", "enable_mfa_where_possible"],
    "injection": ["patch_service", "validate_inputs", "review_application_logs"],
    "malware": ["isolate_host", "run_forensics", "restore_from_clean_backup_if_needed"],
    "exfiltration": ["contain_endpoint", "review_egress_rules", "audit_data_access"],
    "mitm": ["inspect_arp_tables", "enforce_tls_certificate_validation", "segment_untrusted_networks"],
    "unknown_attack": ["escalate_to_analyst", "capture_additional_context", "increase_monitoring"],
}


MITIGATIONS_BY_SCHEMA_PROFILE = {
    "network_flow": ["capture_flow_window", "review_network_policy"],
    "network_packet": ["capture_packet_sample", "inspect_protocol_anomalies"],
    "host_metrics": ["collect_endpoint_artifacts", "check_process_integrity"],
    "iot_telemetry": ["validate_sensor_baseline", "check_device_state"],
    "alert_text": ["correlate_alert_context", "verify_rule_trigger"],
    "pcap_ref": ["preserve_pcap_evidence", "replay_sample_in_sandbox"],
    "unknown": ["capture_additional_context"],
}


class RuleBasedExplainer:
    def explain(self, event: CanonicalEvent, classification: ClassificationOutput) -> ExplanationOutput:
        family = classification.attack_family or "unknown_attack"
        schema_profile = event.schema_profile or "unknown"
        mitigations = [
            *MITIGATIONS_BY_FAMILY.get(family, MITIGATIONS_BY_FAMILY["unknown_attack"]),
            *MITIGATIONS_BY_SCHEMA_PROFILE.get(schema_profile, MITIGATIONS_BY_SCHEMA_PROFILE["unknown"]),
        ]
        mitigations = list(dict.fromkeys(mitigations))
        summary = (
            f"Evento {event.event_id} clasificado como {family}. "
            f"Confianza={classification.confidence:.2f}; schema_profile={schema_profile}."
        )
        return ExplanationOutput(
            event_id=event.event_id,
            risk_summary=summary,
            mitigations=mitigations,
            confidence=classification.confidence,
            requires_human_review=classification.confidence < 0.65,
            next_route="judge",
            model_name="rule_based_explainer_v0",
        )


class LLMExplanationAgent:
    def __init__(
        self,
        model: str = "gemma3:12b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float | None = None,
        provider: str | None = None,
        fallback_explainer: RuleBasedExplainer | None = None,
    ):
        provider_name = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
        self.model_name = f"llm_explainer::{provider_name}::{model}"
        if provider_name == "openrouter":
            self.agent = OpenRouterChatAgent(
                model=model,
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                timeout_seconds=timeout_seconds,
            )
        elif provider_name == "transformers":
            self.agent = TransformersChatAgent(
                model=model,
                adapter_path=os.getenv("EXPLAINER_LORA_ADAPTER") or os.getenv("MITIGATOR_LORA_ADAPTER") or os.getenv("TRANSFORMERS_ADAPTER_PATH"),
                timeout_seconds=timeout_seconds,
            )
        else:
            self.agent = OllamaChatAgent(model=model, base_url=base_url, timeout_seconds=timeout_seconds)
        self.fallback_explainer = fallback_explainer or RuleBasedExplainer()

    def explain(self, event: CanonicalEvent, classification: ClassificationOutput) -> ExplanationOutput:
        fallback = self.fallback_explainer.explain(event, classification)
        return fallback.model_copy(
            update={
                "model_name": f"{self.model_name}|sync_fallback:{fallback.model_name}",
                "risk_summary": f"LLM explanation requires async execution. {fallback.risk_summary}",
            }
        )

    async def explain_async(self, event: CanonicalEvent, classification: ClassificationOutput) -> ExplanationOutput:
        try:
            payload = await self.agent.invoke_json(
                system_prompt=EXPLAINER_SYSTEM,
                user_payload=explanation_input(event, classification),
                json_schema=EXPLANATION_SCHEMA,
            )
            return normalize_explanation_payload(payload, event, classification, self.model_name)
        except Exception as exc:
            fallback = self.fallback_explainer.explain(event, classification)
            return fallback.model_copy(
                update={
                    "model_name": f"{self.model_name}|fallback:{fallback.model_name}",
                    "risk_summary": f"llm_explainer_failed={type(exc).__name__}. {fallback.risk_summary}",
                }
            )


def explanation_input(event: CanonicalEvent, classification: ClassificationOutput) -> dict[str, Any]:
    return {
        "event": {
            "event_id": event.event_id,
            "modality": event.modality,
            "schema_profile": event.schema_profile,
            "src_ip": event.src_ip,
            "dst_ip": event.dst_ip,
            "src_port": event.src_port,
            "dst_port": event.dst_port,
            "transport_proto": event.transport_proto,
            "app_proto": event.app_proto,
            "packet_count": event.packet_count,
            "byte_count": event.byte_count,
            "duration_ms": event.duration_ms,
            "telemetry": event.telemetry,
            "host": event.host,
            "severity": event.severity,
            "behavior_tags": event.behavior_tags,
            "attack_indicators": event.attack_indicators,
            "asset_context": event.asset_context,
            "uncertainty": event.uncertainty,
            "anomaly_summary": event.anomaly_summary,
            "service_context": event.service_context,
            "host_context": event.host_context,
            "telemetry_context": event.telemetry_context,
            "semantic_text": event.semantic_text,
            "mapping_confidence": event.mapping_confidence,
            "missing_fields": event.missing_fields,
        },
        "classification": classification.model_dump(mode="json"),
        "mitigation_policy": {
            "prefer_actionable_controls": True,
            "avoid_dataset_specific_rules": True,
            "escalate_when_confidence_below": 0.65,
        },
    }


def normalize_explanation_payload(
    payload: dict[str, Any],
    event: CanonicalEvent,
    classification: ClassificationOutput,
    model_name: str,
) -> ExplanationOutput:
    mitigations = payload.get("mitigations")
    if not isinstance(mitigations, list):
        mitigations = []
    mitigations = [str(item).strip() for item in mitigations if str(item).strip()]

    confidence = _coerce_confidence(payload.get("confidence"), default=classification.confidence)
    requires_review = bool(payload.get("requires_human_review", confidence < 0.65))
    next_route = str(payload.get("next_route") or "judge").strip()
    if next_route not in {"judge", "end"}:
        next_route = "judge"
    if requires_review:
        next_route = "judge"

    summary = str(payload.get("risk_summary") or "").strip()
    if not summary:
        family = classification.attack_family or "benign"
        summary = f"Evento {event.event_id} clasificado como {family}; confianza={confidence:.2f}."

    reported_model = str(payload.get("model_name") or "").strip()
    if reported_model and reported_model != model_name:
        mitigations.append(f"llm_reported_model_name={reported_model}")

    return ExplanationOutput(
        event_id=str(payload.get("event_id") or event.event_id),
        risk_summary=summary,
        mitigations=list(dict.fromkeys(mitigations)),
        confidence=confidence,
        requires_human_review=requires_review,
        next_route=next_route,
        model_name=model_name,
    )


def _coerce_confidence(value: Any, default: float = 0.5) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))
