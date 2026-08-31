# src/agents/classifier.py
from __future__ import annotations

import os
from typing import Any

from src.agents.base import OllamaChatAgent, OpenRouterChatAgent, TransformersChatAgent
from src.eval.predictive_sanitization import scrub_predictive_payload
from src.contracts.agents import ClassificationOutput, DetectionOutput
from src.contracts.canonical import CanonicalEvent
from src.contracts.taxonomy import ATTACK_FAMILY_KEYWORDS, infer_label_info


CLASSIFIER_SYSTEM = """
Eres el agente clasificador de amenazas.
Recibes un evento canonico y una decision previa del detector.
Tu trabajo es catalogar la amenaza en una familia y, si es posible, un subtipo.
No decidas si el evento es benigno o malicioso salvo para respetar la decision del detector.
Usa schema_profile para interpretar que campos son relevantes: comunicaciones, paquetes, host, telemetria fisica o alerta textual.
No apliques reglas por nombre de origen. Los nombres de datasets, ficheros u origenes no son senales de clasificacion.

Familias preferidas:
- mitm
- ddos
- scanning
- botnet
- bruteforce
- injection
- malware
- exfiltration
- unknown_attack

Si el detector dice que no es malicioso, devuelve attack_family=null, attack_subtype=null y next_route='end'.
Si no hay evidencia suficiente para catalogar una amenaza maliciosa, usa next_route='judge' y confidence menor que 0.5.
Si puedes catalogarla, usa next_route='explain'.
Responde exclusivamente con JSON valido.
"""


CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "event_id": {"type": "string"},
        "attack_family": {"type": ["string", "null"]},
        "attack_subtype": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
        "cross_dataset_neighbors": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "array", "items": {"type": "string"}},
        "next_route": {"type": "string", "enum": ["explain", "judge", "end"]},
    },
    "required": [
        "event_id",
        "attack_family",
        "attack_subtype",
        "confidence",
        "cross_dataset_neighbors",
        "reason",
        "next_route",
    ],
}


def classification_input(event: CanonicalEvent, detection: DetectionOutput) -> dict[str, Any]:
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
            "telemetry": scrub_predictive_payload(event.telemetry),
            "host": scrub_predictive_payload(event.host),
            "severity": event.severity,
            "behavior_tags": scrub_predictive_payload(event.behavior_tags),
            "attack_indicators": scrub_predictive_payload(event.attack_indicators),
            "asset_context": event.asset_context,
            "uncertainty": scrub_predictive_payload(event.uncertainty),
            "anomaly_summary": scrub_predictive_payload(event.anomaly_summary),
            "service_context": scrub_predictive_payload(event.service_context),
            "host_context": scrub_predictive_payload(event.host_context),
            "telemetry_context": scrub_predictive_payload(event.telemetry_context),
            "semantic_text": scrub_predictive_payload(event.semantic_text),
            "mapping_confidence": event.mapping_confidence,
            "missing_fields": scrub_predictive_payload(event.missing_fields),
        },
        "detection": detection.model_dump(mode="json"),
        "taxonomy": {
            "families": sorted(ATTACK_FAMILY_KEYWORDS),
            "keywords": ATTACK_FAMILY_KEYWORDS,
        },
    }


def normalize_classification_payload(
    payload: dict[str, Any],
    event: CanonicalEvent,
    detection: DetectionOutput,
    model_name: str,
) -> ClassificationOutput:
    if not detection.is_malicious:
        return ClassificationOutput(
            event_id=event.event_id,
            attack_family=None,
            attack_subtype=None,
            confidence=max(0.0, min(1.0, 1.0 - detection.probability)),
            reason=["detector_marked_event_benign", f"llm_classifier_model={model_name}"],
            next_route="end",
        )

    family = _normalize_family(payload.get("attack_family"))
    subtype = _clean_optional_text(payload.get("attack_subtype"))
    reason = payload.get("reason")
    if not isinstance(reason, list):
        reason = []
    reason = [str(item) for item in reason if str(item).strip()]

    confidence = _coerce_confidence(payload.get("confidence"))
    next_route = str(payload.get("next_route") or "").strip()
    if family is None:
        next_route = "judge"
        confidence = min(confidence, 0.45)
    elif next_route not in {"explain", "judge", "end"}:
        next_route = "explain" if confidence >= 0.5 else "judge"

    neighbors = payload.get("cross_dataset_neighbors")
    if not isinstance(neighbors, list):
        neighbors = []

    reason.append(f"llm_classifier_model={model_name}")
    return ClassificationOutput(
        event_id=str(payload.get("event_id") or event.event_id),
        attack_family=family,
        attack_subtype=subtype,
        confidence=confidence,
        cross_dataset_neighbors=[str(item) for item in neighbors],
        reason=reason or ["llm_classification_no_reason"],
        next_route=next_route,
    )


def _normalize_family(value: Any) -> str | None:
    text = _clean_optional_text(value)
    if text is None:
        return None
    normalized = text.lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "dos": "ddos",
        "recon": "scanning",
        "reconnaissance": "scanning",
        "credential_access": "bruteforce",
        "brute_force": "bruteforce",
        "data_theft": "exfiltration",
        "theft": "exfiltration",
        "c2": "botnet",
        "c&c": "botnet",
        "backdoor": "malware",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in ATTACK_FAMILY_KEYWORDS or normalized == "unknown_attack":
        return normalized
    inferred = infer_label_info(normalized).family
    return inferred or normalized


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "n/a", "-"}:
        return None
    return text


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.5
    return max(0.0, min(1.0, confidence))


class RuleBasedClassifier:
    def classify(self, event: CanonicalEvent, detection: DetectionOutput) -> ClassificationOutput:
        if not detection.is_malicious:
            return ClassificationOutput(
                event_id=event.event_id,
                attack_family=None,
                attack_subtype=None,
                confidence=1.0 - detection.probability,
                reason=["event_detected_as_benign"],
                next_route="end",
            )

        family = self._family_from_behavior(event)
        if family is None:
            return ClassificationOutput(
                event_id=event.event_id,
                attack_family=None,
                attack_subtype=None,
                confidence=0.45,
                reason=["insufficient_behavioral_evidence"],
                next_route="judge",
            )

        confidence = 0.7 if family != "unknown_attack" else 0.5
        return ClassificationOutput(
            event_id=event.event_id,
            attack_family=family,
            attack_subtype=family,
            confidence=confidence,
            cross_dataset_neighbors=[],
            reason=[f"behavior_mapped_to_family={family}"],
            next_route="explain",
        )

    @staticmethod
    def _family_from_behavior(event: CanonicalEvent) -> str | None:
        indicators = {str(item).strip().lower() for item in event.attack_indicators if str(item).strip()}
        tags = {str(item).strip().lower() for item in event.behavior_tags if str(item).strip()}
        signals = indicators | tags
        if any("ddos" in item or "volume_anomaly" in item for item in signals):
            return "ddos"
        if any("injection" in item or "web_surface_activity" in item for item in signals):
            return "injection"
        if any(
            "bruteforce" in item
            or "authentication_pressure" in item
            or "remote_access_exposure" in item
            for item in signals
        ):
            return "bruteforce"
        if any("scan" in item or "service_probe_pattern" in item for item in signals):
            return "scanning"
        if any("mitm" in item or "spoof" in item for item in signals):
            return "mitm"
        if any("botnet" in item for item in signals):
            return "botnet"
        if any("malware" in item or "suspicious_execution_context" in item for item in signals):
            return "malware"
        if any("exfiltration" in item for item in signals):
            return "exfiltration"
        if indicators:
            return "unknown_attack"
        return None


class LLMClassificationAgent:
    def __init__(
        self,
        model: str = "gemma3:12b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float | None = None,
        provider: str | None = None,
        fallback_classifier: RuleBasedClassifier | None = None,
    ):
        provider_name = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
        self.model_name = f"llm_classifier::{provider_name}::{model}"
        if provider_name == "openrouter":
            self.agent = OpenRouterChatAgent(
                model=model,
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                timeout_seconds=timeout_seconds,
            )
        elif provider_name == "transformers":
            self.agent = TransformersChatAgent(
                model=model,
                adapter_path=os.getenv("CLASSIFIER_LORA_ADAPTER") or os.getenv("TRANSFORMERS_ADAPTER_PATH"),
                timeout_seconds=timeout_seconds,
            )
        else:
            self.agent = OllamaChatAgent(model=model, base_url=base_url, timeout_seconds=timeout_seconds)
        self.fallback_classifier = fallback_classifier or RuleBasedClassifier()

    def classify(self, event: CanonicalEvent, detection: DetectionOutput) -> ClassificationOutput:
        fallback = self.fallback_classifier.classify(event, detection)
        return fallback.model_copy(
            update={
                "reason": ["llm_classifier_requires_async", *fallback.reason],
            }
        )

    async def classify_async(self, event: CanonicalEvent, detection: DetectionOutput) -> ClassificationOutput:
        try:
            payload = await self.agent.invoke_json(
                system_prompt=CLASSIFIER_SYSTEM,
                user_payload=classification_input(event, detection),
                json_schema=CLASSIFICATION_SCHEMA,
            )
            return normalize_classification_payload(payload, event, detection, self.model_name)
        except Exception as exc:
            fallback = self.fallback_classifier.classify(event, detection)
            return fallback.model_copy(
                update={
                    "reason": [
                        f"llm_classifier_failed={type(exc).__name__}",
                        *fallback.reason,
                    ],
                }
            )
