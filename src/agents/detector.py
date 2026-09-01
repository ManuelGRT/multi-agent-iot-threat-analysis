# src/agents/detector.py
from __future__ import annotations

import os
from typing import Any

from src.agents.base import (
    GoogleAIStudioChatAgent,
    GroqChatAgent,
    MistralChatAgent,
    OllamaChatAgent,
    OpenRouterChatAgent,
    TransformersChatAgent,
)
from src.eval.predictive_sanitization import scrub_predictive_payload
from src.contracts.agents import DetectionOutput
from src.contracts.canonical import CanonicalEvent


DETECTOR_SYSTEM = """
Eres el agente detector.
Tu trabajo es decidir si un evento es malicioso o benigno.
Debes responder exclusivamente con JSON valido.
Usa el esquema canonico, la vista semantica y las senales normalizadas.
No hagas clasificacion detallada de familia; eso lo hara otro agente.
Usa schema_profile para decidir que senales tecnicas son relevantes, sin aplicar reglas por nombre de origen.
Si schema_profile falta o es unknown, infierelo desde modalidad, nombres de campos disponibles y valores.
La procedencia no es una senal decisional: no uses nombres de datasets, ficheros u origenes para decidir.

Reglas de decision:
- Basa la decision en senales observables: trafico, puertos, protocolos, volumen, estados, metricas de host, telemetria, alertas operacionales y comportamiento normalizado.
- La ausencia de campos tecnicos debe bajar probability y puede requerir abstain.
- Si las senales disponibles son contradictorias o debiles, usa abstain=true y next_route='judge' en lugar de forzar una decision.

Si la evidencia es insuficiente, marca abstain=true y next_route='judge'.
Si el evento es claramente benigno, usa is_malicious=false y next_route='end'.
Si el evento es claramente malicioso, usa is_malicious=true y next_route='classify'.
"""


DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "event_id": {"type": "string"},
        "is_malicious": {"type": "boolean"},
        "probability": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "model_name": {"type": "string"},
        "next_route": {"type": "string", "enum": ["classify", "judge", "end"]},
        "abstain": {"type": "boolean"},
    },
    "required": ["event_id", "is_malicious", "probability", "evidence", "model_name", "next_route", "abstain"],
}


def detection_input(event: CanonicalEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "modality": event.modality,
        "schema_profile": event.schema_profile,
        "ts": event.ts.isoformat() if event.ts else None,
        "network": {
            "src_ip": event.src_ip,
            "dst_ip": event.dst_ip,
            "src_port": event.src_port,
            "dst_port": event.dst_port,
            "transport_proto": event.transport_proto,
            "app_proto": event.app_proto,
            "packet_count": event.packet_count,
            "byte_count": event.byte_count,
            "duration_ms": event.duration_ms,
        },
        "telemetry": scrub_predictive_payload(event.telemetry),
        "host": scrub_predictive_payload(event.host),
        "behavior": {
            "behavior_tags": scrub_predictive_payload(event.behavior_tags),
            "attack_indicators": scrub_predictive_payload(event.attack_indicators),
            "asset_context": event.asset_context,
            "uncertainty": scrub_predictive_payload(event.uncertainty),
            "anomaly_summary": scrub_predictive_payload(event.anomaly_summary),
            "service_context": scrub_predictive_payload(event.service_context),
            "host_context": scrub_predictive_payload(event.host_context),
            "telemetry_context": scrub_predictive_payload(event.telemetry_context),
        },
        "alert_context": {"severity": event.severity},
        "semantic_text": scrub_predictive_payload(event.semantic_text),
        "mapping_confidence": event.mapping_confidence,
        "missing_fields": scrub_predictive_payload(event.missing_fields),
    }


def normalize_detection_payload(payload: dict[str, Any], event: CanonicalEvent, model_name: str) -> DetectionOutput:
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    evidence = [str(item) for item in evidence if str(item).strip()]
    reported_model = str(payload.get("model_name") or "").strip()
    if reported_model and reported_model != model_name:
        evidence.append(f"llm_reported_model_name={reported_model}")

    probability = _coerce_probability(payload.get("probability"))
    abstain = bool(payload.get("abstain", False))
    is_malicious = bool(payload.get("is_malicious", False))

    next_route = payload.get("next_route")
    if abstain:
        next_route = "judge"
        if probability in {0.0, 1.0}:
            probability = 0.5
    elif is_malicious:
        next_route = "classify"
        probability = max(probability, 0.5)
    else:
        next_route = "end"
        probability = min(probability, 0.5)

    if next_route not in {"classify", "judge", "end"}:
        next_route = "judge" if abstain else "classify" if is_malicious else "end"

    return DetectionOutput(
        event_id=str(payload.get("event_id") or event.event_id),
        is_malicious=is_malicious,
        probability=probability,
        evidence=evidence or ["llm_detection_no_evidence"],
        model_name=model_name,
        next_route=next_route,
        abstain=abstain,
    )


def _coerce_probability(value: Any) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        probability = 0.5
    return max(0.0, min(1.0, probability))


class RuleBasedDetector:
    model_name = "rule_based_detector_v0"

    def detect(self, event: CanonicalEvent) -> DetectionOutput:
        evidence: list[str] = []

        if event.mapping_confidence < 0.5:
            evidence.append(f"low_mapping_confidence={event.mapping_confidence}")
            return DetectionOutput(
                event_id=event.event_id,
                is_malicious=False,
                probability=0.5,
                evidence=evidence,
                model_name=self.model_name,
                next_route="judge",
                abstain=True,
            )

        indicators = [str(item) for item in event.attack_indicators if str(item).strip()]
        if indicators:
            evidence.append(f"behavior_indicators={','.join(indicators[:5])}")
            return DetectionOutput(
                event_id=event.event_id,
                is_malicious=True,
                probability=0.7,
                evidence=evidence,
                model_name=self.model_name,
                next_route="classify",
                abstain=False,
            )

        if self._has_suspicious_network_shape(event):
            evidence.append("suspicious_network_shape")
            return DetectionOutput(
                event_id=event.event_id,
                is_malicious=True,
                probability=0.65,
                evidence=evidence,
                model_name=self.model_name,
                next_route="judge",
                abstain=False,
            )

        evidence.append("insufficient_evidence")
        return DetectionOutput(
            event_id=event.event_id,
            is_malicious=False,
            probability=0.5,
            evidence=evidence,
            model_name=self.model_name,
            next_route="judge",
            abstain=True,
        )

    @staticmethod
    def _has_suspicious_network_shape(event: CanonicalEvent) -> bool:
        high_volume = (event.packet_count or 0) > 10_000 or (event.byte_count or 0) > 10_000_000
        sensitive_port = event.dst_port in {22, 23, 2323, 3389, 445, 1433, 3306}
        return bool(high_volume or sensitive_port)


class LLMDetectionAgent:
    def __init__(
        self,
        model: str = "gemma3:12b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float | None = None,
        provider: str | None = None,
        fallback_detector: RuleBasedDetector | None = None,
    ):
        provider_name = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
        self.model_name = f"llm_detector::{provider_name}::{model}"
        if provider_name == "openrouter":
            self.agent = OpenRouterChatAgent(
                model=model,
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                timeout_seconds=timeout_seconds,
            )
        elif provider_name == "groq":
            self.agent = GroqChatAgent(
                model=model,
                base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                timeout_seconds=timeout_seconds,
            )
        elif provider_name == "mistral":
            self.agent = MistralChatAgent(
                model=model,
                base_url=os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
                timeout_seconds=timeout_seconds,
            )
        elif provider_name in {"google", "gemini", "google_ai_studio"}:
            provider_name = "google"
            self.model_name = f"llm_detector::{provider_name}::{model}"
            self.agent = GoogleAIStudioChatAgent(
                model=model,
                base_url=os.getenv("GOOGLE_AI_STUDIO_BASE_URL"),
                timeout_seconds=timeout_seconds,
            )
        elif provider_name == "transformers":
            self.agent = TransformersChatAgent(
                model=model,
                adapter_path=os.getenv("DETECTOR_LORA_ADAPTER") or os.getenv("TRANSFORMERS_ADAPTER_PATH"),
                timeout_seconds=timeout_seconds,
            )
        else:
            self.agent = OllamaChatAgent(model=model, base_url=base_url, timeout_seconds=timeout_seconds)
        self.fallback_detector = fallback_detector or RuleBasedDetector()

    def detect(self, event: CanonicalEvent) -> DetectionOutput:
        fallback = self.fallback_detector.detect(event)
        return fallback.model_copy(
            update={
                "model_name": f"{self.model_name}|sync_fallback:{fallback.model_name}",
                "evidence": ["llm_detector_requires_async", *fallback.evidence],
            }
        )

    async def detect_async(self, event: CanonicalEvent) -> DetectionOutput:
        try:
            payload = await self.agent.invoke_json(
                system_prompt=DETECTOR_SYSTEM,
                user_payload=detection_input(event),
                json_schema=DETECTION_SCHEMA,
            )
            return normalize_detection_payload(payload, event, self.model_name)
        except Exception as exc:
            fallback = self.fallback_detector.detect(event)
            return fallback.model_copy(
                update={
                    "model_name": f"{self.model_name}|fallback:{fallback.model_name}",
                    "evidence": [
                        f"llm_detector_failed={type(exc).__name__}",
                        *fallback.evidence,
                    ],
                }
            )
