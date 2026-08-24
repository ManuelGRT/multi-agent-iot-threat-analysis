from __future__ import annotations

import math
import re
from typing import Any

from src.contracts.canonical import CanonicalEvent


class BehaviorAbstractionAgent:
    """Derives dataset-agnostic behavioral signals from a canonical event."""

    version = "behavior_abstraction_v1"

    def abstract(self, event: CanonicalEvent) -> CanonicalEvent:
        behavior_tags = set(event.behavior_tags or [])
        attack_indicators = set(event.attack_indicators or [])
        uncertainty = set(event.uncertainty or [])

        profile = event.schema_profile or "unknown"
        behavior_tags.add(f"signal.{profile}")

        numeric = self._numeric_values(event)
        groups = event.feature_groups or {}
        service = event.service_context or {}
        host = event.host_context or {}
        telemetry = event.telemetry_context or {}
        text = self._semantic_blob(event)

        self._add_network_behavior(event, service, behavior_tags, attack_indicators)
        self._add_host_behavior(host, groups, text, behavior_tags, attack_indicators)
        self._add_telemetry_behavior(telemetry, groups, behavior_tags, attack_indicators)
        self._add_textual_behavior(text, behavior_tags, attack_indicators)
        self._add_numeric_behavior(numeric, behavior_tags, attack_indicators)
        self._add_uncertainty(event, groups, uncertainty)

        asset_context = event.asset_context or self._asset_context(event, groups)
        anomaly_summary = event.anomaly_summary or self._summary(
            profile=profile,
            asset_context=asset_context,
            behavior_tags=behavior_tags,
            attack_indicators=attack_indicators,
            uncertainty=uncertainty,
        )
        semantic_text = event.semantic_text
        if anomaly_summary and anomaly_summary not in semantic_text:
            semantic_text = f"{semantic_text} | behavior::{anomaly_summary}"

        return event.model_copy(update={
            "behavior_tags": sorted(behavior_tags),
            "attack_indicators": sorted(attack_indicators),
            "asset_context": asset_context,
            "uncertainty": sorted(uncertainty),
            "anomaly_summary": anomaly_summary,
            "semantic_text": semantic_text,
        })

    def _add_network_behavior(
        self,
        event: CanonicalEvent,
        service: dict[str, Any],
        behavior_tags: set[str],
        attack_indicators: set[str],
    ) -> None:
        if event.src_ip or event.dst_ip or event.src_port or event.dst_port:
            behavior_tags.add("communication_observed")
        if event.transport_proto:
            behavior_tags.add(f"transport.{_token(event.transport_proto)}")
        if event.traffic_direction:
            behavior_tags.add(f"direction.{_token(event.traffic_direction)}")

        dst_service = service.get("dst_common_service")
        if dst_service:
            token = _token(dst_service)
            behavior_tags.add(f"service.{token}")
            if token in {"http", "http_alt", "https"}:
                behavior_tags.add("web_service_activity")
            if token in {"telnet", "ssh", "ftp"}:
                behavior_tags.add("remote_access_service")
            if token == "dns":
                behavior_tags.add("name_resolution_activity")

        packet_count = _number(event.packet_count)
        byte_count = _number(event.byte_count)
        duration_ms = _number(event.duration_ms)
        if packet_count is not None:
            if packet_count >= 1000:
                behavior_tags.add("high_packet_volume")
                attack_indicators.add("volume_anomaly")
            elif packet_count <= 3:
                behavior_tags.add("sparse_packet_exchange")
        if byte_count is not None:
            if byte_count >= 1_000_000:
                behavior_tags.add("high_byte_volume")
                attack_indicators.add("volume_anomaly")
            elif byte_count <= 128:
                behavior_tags.add("low_byte_exchange")
        if duration_ms is not None:
            if duration_ms <= 100:
                behavior_tags.add("short_burst")
            elif duration_ms >= 60_000:
                behavior_tags.add("long_session")

        if event.dst_port in {21, 22, 23, 53, 80, 443, 8080} and packet_count is not None and packet_count <= 5:
            attack_indicators.add("service_probe_pattern")
        if event.dst_port in {21, 22, 23, 3389, 5900}:
            attack_indicators.add("remote_access_exposure")
        if event.dst_port in {80, 443, 8080}:
            attack_indicators.add("web_surface_activity")

    def _add_host_behavior(
        self,
        host: dict[str, Any],
        groups: dict[str, list[str]],
        text: str,
        behavior_tags: set[str],
        attack_indicators: set[str],
    ) -> None:
        if host.get("host_field_count") or groups.get("host_metrics"):
            behavior_tags.add("endpoint_activity")
        if host.get("has_process_signal"):
            behavior_tags.add("process_activity")
            attack_indicators.add("process_signal")
        if host.get("has_resource_signal"):
            behavior_tags.add("resource_usage_signal")
        if any(token in text for token in ("powershell", "cmd", "shell", "script", "binary", "exe")):
            attack_indicators.add("suspicious_execution_context")
        if any(token in text for token in ("login", "password", "credential", "auth", "failed")):
            attack_indicators.add("authentication_pressure")

    def _add_telemetry_behavior(
        self,
        telemetry: dict[str, Any],
        groups: dict[str, list[str]],
        behavior_tags: set[str],
        attack_indicators: set[str],
    ) -> None:
        if telemetry.get("telemetry_field_count") or groups.get("iot_telemetry"):
            behavior_tags.add("sensor_state_observed")
        if telemetry.get("has_sensor_state"):
            behavior_tags.add("sensor_status_signal")
        telemetry_range = _number(telemetry.get("telemetry_range"))
        telemetry_max = _number(telemetry.get("telemetry_max"))
        if telemetry_range is not None and telemetry_range > 20:
            behavior_tags.add("sensor_value_shift")
            attack_indicators.add("telemetry_anomaly")
        if telemetry_max is not None and abs(telemetry_max) > 1000:
            behavior_tags.add("extreme_sensor_value")
            attack_indicators.add("telemetry_anomaly")

    def _add_textual_behavior(
        self,
        text: str,
        behavior_tags: set[str],
        attack_indicators: set[str],
    ) -> None:
        keyword_map = {
            "scan_pattern": ("scan", "probe", "recon", "fingerprint", "enumerat"),
            "ddos_pattern": ("ddos", "dos", "flood", "syn"),
            "bruteforce_pattern": ("brute", "password", "credential", "login", "auth"),
            "injection_pattern": ("injection", "sql", "xss", "command injection"),
            "malware_pattern": ("malware", "ransom", "backdoor", "trojan", "worm", "botnet"),
            "mitm_pattern": ("mitm", "spoof", "arp"),
            "exfiltration_pattern": ("exfil", "data theft", "leak"),
        }
        for indicator, keywords in keyword_map.items():
            if any(keyword in text for keyword in keywords):
                attack_indicators.add(indicator)
                behavior_tags.add(f"keyword.{indicator}")

    def _add_numeric_behavior(
        self,
        numeric: list[float],
        behavior_tags: set[str],
        attack_indicators: set[str],
    ) -> None:
        if not numeric:
            return
        behavior_tags.add("numeric_signal")
        if len(numeric) >= 10:
            behavior_tags.add("wide_numeric_profile")
        values = [abs(value) for value in numeric if math.isfinite(value)]
        if not values:
            return
        nonzero = [value for value in values if value > 0]
        zero_ratio = 1.0 - (len(nonzero) / len(values))
        if zero_ratio > 0.7:
            behavior_tags.add("sparse_numeric_profile")
        if max(values) >= 1_000_000:
            behavior_tags.add("large_magnitude_signal")
            attack_indicators.add("magnitude_anomaly")

    def _add_uncertainty(
        self,
        event: CanonicalEvent,
        groups: dict[str, list[str]],
        uncertainty: set[str],
    ) -> None:
        if event.schema_profile in {None, "unknown"}:
            uncertainty.add("unknown_schema_profile")
        if not event.ts:
            uncertainty.add("missing_time")
        if not event.src_ip and not event.dst_ip and event.modality == "network_flow":
            uncertainty.add("missing_network_endpoints")
        if not event.transport_proto and event.modality == "network_flow":
            uncertainty.add("missing_transport_protocol")
        if not groups:
            uncertainty.add("no_feature_groups")
        if event.mapping_confidence < 0.5:
            uncertainty.add("low_mapping_confidence")

    def _asset_context(self, event: CanonicalEvent, groups: dict[str, list[str]]) -> str:
        profile = event.schema_profile or "unknown"
        if profile == "host_metrics" or groups.get("host_metrics"):
            return "endpoint"
        if profile == "iot_telemetry" or groups.get("iot_telemetry"):
            return "iot_device"
        if profile in {"network_flow", "network_packet"}:
            if event.dst_port or event.app_proto:
                return "network_service"
            return "network"
        if profile == "alert_text":
            return "alert_stream"
        return "unknown_asset"

    def _numeric_values(self, event: CanonicalEvent) -> list[float]:
        values: list[float] = []
        for value in [
            event.src_port,
            event.dst_port,
            event.packet_count,
            event.byte_count,
            event.duration_ms,
            *(event.telemetry or {}).values(),
            *(event.host or {}).values(),
            *(event.service_context or {}).values(),
            *(event.host_context or {}).values(),
            *(event.telemetry_context or {}).values(),
        ]:
            number = _number(value)
            if number is not None:
                values.append(number)
        return values

    def _semantic_blob(self, event: CanonicalEvent) -> str:
        parts = [
            event.semantic_text,
            event.anomaly_summary,
            " ".join(event.evidence_fields or []),
            " ".join(event.behavior_tags or []),
            " ".join(event.attack_indicators or []),
        ]
        return " ".join(str(part or "") for part in parts).lower()

    def _summary(
        self,
        profile: str,
        asset_context: str,
        behavior_tags: set[str],
        attack_indicators: set[str],
        uncertainty: set[str],
    ) -> str:
        tag_text = ",".join(sorted(behavior_tags)[:8])
        indicator_text = ",".join(sorted(attack_indicators)[:6]) or "none"
        uncertainty_text = ",".join(sorted(uncertainty)[:5]) or "none"
        return (
            f"profile={profile}; asset={asset_context}; "
            f"behaviors={tag_text}; indicators={indicator_text}; uncertainty={uncertainty_text}"
        )


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "-"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    return number if math.isfinite(number) else None


def _token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"
