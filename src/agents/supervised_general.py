from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import Any

import joblib

from src.contracts.agents import ClassificationOutput, DetectionOutput
from src.contracts.canonical import CanonicalEvent


GENERAL_FAMILIES = [
    "benign",
    "ddos",
    "scanning",
    "botnet",
    "bruteforce",
    "injection",
    "malware",
    "exfiltration",
    "mitm",
    "unknown_attack",
]


class GeneralizedFeatureStandardizer:
    """Builds dataset-agnostic features from a canonical event."""

    version = "generalized_feature_standardizer_v3"

    def __init__(self, include_origin: bool = False, feature_set: str = "full"):
        self.include_origin = include_origin
        self.feature_set = feature_set

    def event_to_features(self, event: CanonicalEvent) -> dict[str, Any]:
        features: dict[str, Any] = {
            "modality": event.modality,
            "schema_profile": schema_profile(event),
            "has_src_ip": int(bool(event.src_ip)),
            "has_dst_ip": int(bool(event.dst_ip)),
            "has_transport_proto": int(bool(event.transport_proto)),
            "mapping_confidence": event.mapping_confidence,
            "missing_field_count": len(event.missing_fields or []),
            "evidence_field_count": len(event.evidence_fields or []),
            "behavior_tag_count": len(event.behavior_tags or []),
            "attack_indicator_count": len(event.attack_indicators or []),
            "uncertainty_count": len(event.uncertainty or []),
        }
        self._add_context_features(features, event)
        if self.include_origin:
            self._add_origin_features(features, event)
        if self.feature_set == "abstract":
            self._add_abstract_numeric_features(features, event)
            self._add_semantic_hints(features, event)
            return features

        self._add_numeric(features, "src_port", event.src_port)
        self._add_numeric(features, "dst_port", event.dst_port)
        self._add_numeric(features, "packet_count", event.packet_count)
        self._add_numeric(features, "byte_count", event.byte_count)
        self._add_numeric(features, "duration_ms", event.duration_ms)

        if event.transport_proto:
            features["transport_proto"] = _normalize_token(event.transport_proto)
        if event.app_proto:
            features["app_proto"] = _normalize_token(event.app_proto)
        self._add_port_buckets(features, event.src_port, "src")
        self._add_port_buckets(features, event.dst_port, "dst")

        raw_values: dict[str, Any] = {}
        raw_values.update(event.telemetry or {})
        raw_values.update({f"host.{key}": value for key, value in (event.host or {}).items()})
        group_counts: dict[str, int] = {}
        group_numeric_counts: dict[str, int] = {}
        group_abs_log_sum: dict[str, float] = {}
        group_numeric_values: dict[str, list[float]] = {}
        for key, value in raw_values.items():
            group = feature_group(str(key))
            group_counts[group] = group_counts.get(group, 0) + 1
            features[f"has_group.{group}"] = 1
            numeric = _coerce_number(value)
            if numeric is not None:
                group_numeric_counts[group] = group_numeric_counts.get(group, 0) + 1
                group_abs_log_sum[group] = group_abs_log_sum.get(group, 0.0) + math.log1p(abs(numeric))
                group_numeric_values.setdefault(group, []).append(numeric)
                continue
            token = _compact_categorical_value(value)
            if token and group in {"protocol", "state_flags", "application_service", "textual_alert_context"}:
                features[f"cat.{group}.{token}"] = 1

        for group, count in group_counts.items():
            features[f"group_count.{group}"] = count
        for group, count in group_numeric_counts.items():
            features[f"group_numeric_count.{group}"] = count
            features[f"group_abs_log_mean.{group}"] = group_abs_log_sum[group] / max(count, 1)
        for group, values in group_numeric_values.items():
            self._add_group_numeric_stats(features, group, values)

        self._add_semantic_hints(features, event)
        return features

    @staticmethod
    def _add_abstract_numeric_features(features: dict[str, Any], event: CanonicalEvent) -> None:
        for name, value in (
            ("packet_count", event.packet_count),
            ("byte_count", event.byte_count),
            ("duration_ms", event.duration_ms),
        ):
            number = _coerce_number(value)
            if number is not None:
                features[f"abstract.{name}.present"] = 1
                features[f"abstract.{name}.log1p_abs"] = math.log1p(abs(number))
        if event.dst_port is not None:
            port = _coerce_number(event.dst_port)
            if port is not None:
                features[f"abstract.dst_port_bucket.{_port_bucket(int(port))}"] = 1

    @staticmethod
    def _add_semantic_hints(features: dict[str, Any], event: CanonicalEvent) -> None:
        semantic = event.semantic_text or ""
        if event.anomaly_summary:
            semantic = f"{semantic} {event.anomaly_summary}"
        for family, keywords in FAMILY_HINTS.items():
            if any(keyword in semantic.lower() for keyword in keywords):
                features[f"semantic_hint.{family}"] = 1

    @staticmethod
    def _add_context_features(features: dict[str, Any], event: CanonicalEvent) -> None:
        profile = schema_profile(event)
        features[f"schema_profile.{profile}"] = 1
        if event.traffic_direction:
            features[f"traffic_direction.{_normalize_token(event.traffic_direction)}"] = 1
        if event.asset_context:
            features[f"asset_context.{_normalize_token(event.asset_context)}"] = 1

        for tag in event.behavior_tags or []:
            features[f"behavior.{_normalize_token(tag)}"] = 1
        for indicator in event.attack_indicators or []:
            features[f"attack_indicator.{_normalize_token(indicator)}"] = 1
        for item in event.uncertainty or []:
            features[f"uncertainty.{_normalize_token(item)}"] = 1

        for group, columns in (event.feature_groups or {}).items():
            group_token = _normalize_token(group)
            features[f"context_group.{group_token}"] = 1
            features[f"context_group_count.{group_token}"] = len(columns or [])

        for field in event.evidence_fields or []:
            group = feature_group(str(field))
            features[f"evidence_group.{group}"] = 1

        for prefix, context in (
            ("service", event.service_context or {}),
            ("host", event.host_context or {}),
            ("telemetry", event.telemetry_context or {}),
        ):
            for key, value in context.items():
                feature_key = f"context.{prefix}.{_normalize_token(key)}"
                numeric = _coerce_number(value)
                if numeric is not None:
                    features[feature_key] = numeric
                    features[f"{feature_key}.log1p_abs"] = math.log1p(abs(numeric))
                    continue
                token = _compact_categorical_value(value)
                if token:
                    features[f"{feature_key}.{token}"] = 1

    @staticmethod
    def _add_group_numeric_stats(features: dict[str, Any], group: str, values: list[float]) -> None:
        if not values:
            return
        count = len(values)
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / count
        features[f"group_min.{group}"] = min(values)
        features[f"group_max.{group}"] = max(values)
        features[f"group_mean.{group}"] = mean
        features[f"group_std.{group}"] = math.sqrt(variance)
        features[f"group_range.{group}"] = max(values) - min(values)
        features[f"group_zero_ratio.{group}"] = sum(1 for value in values if value == 0) / count
        features[f"group_positive_ratio.{group}"] = sum(1 for value in values if value > 0) / count

    @staticmethod
    def _add_origin_features(features: dict[str, Any], event: CanonicalEvent) -> None:
        provenance = event.provenance
        origin = event.origin or {}
        dataset = _normalize_token(origin.get("source_name") or (provenance.dataset if provenance else "unknown"))
        source_file = _normalize_token(Path(provenance.source_file or "").parent.name if provenance else "")
        features["origin.dataset"] = dataset
        if source_file:
            features["origin.source_dir"] = source_file
        features[f"origin_schema.{schema_profile(event)}"] = 1

    @staticmethod
    def _add_numeric(features: dict[str, Any], name: str, value: Any) -> None:
        number = _coerce_number(value)
        if number is None:
            return
        features[f"{name}.present"] = 1
        features[f"{name}.log1p_abs"] = math.log1p(abs(number))
        if name.endswith("_port"):
            features[f"{name}.raw"] = int(number)

    @staticmethod
    def _add_port_buckets(features: dict[str, Any], value: Any, prefix: str) -> None:
        port = _coerce_number(value)
        if port is None:
            return
        port_int = int(port)
        if port_int in COMMON_PORTS:
            features[f"{prefix}_port.common.{COMMON_PORTS[port_int]}"] = 1
        elif port_int < 1024:
            features[f"{prefix}_port.bucket.system"] = 1
        elif port_int < 49152:
            features[f"{prefix}_port.bucket.registered"] = 1
        else:
            features[f"{prefix}_port.bucket.ephemeral"] = 1


@dataclass
class GeneralSupervisedModelBundle:
    detector_pipeline: Any
    classifier_pipeline: Any
    detector_threshold: float = 0.5
    detector_abstain_margin: float = 0.0
    classifier_abstain_threshold: float = 0.45
    include_origin: bool = False
    feature_set: str = "full"
    standardizer_version: str = GeneralizedFeatureStandardizer.version
    model_name: str = "generalized_supervised_multiagent_v1"


def save_general_bundle(bundle: GeneralSupervisedModelBundle, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_general_bundle(path: str | Path) -> GeneralSupervisedModelBundle:
    return joblib.load(path)


class GeneralSupervisedDetector:
    def __init__(self, bundle: GeneralSupervisedModelBundle):
        self.bundle = bundle
        self.standardizer = GeneralizedFeatureStandardizer(
            include_origin=getattr(bundle, "include_origin", False),
            feature_set=getattr(bundle, "feature_set", "full"),
        )
        self.model_name = f"general_detector::{bundle.model_name}"

    def detect(self, event: CanonicalEvent) -> DetectionOutput:
        features = self.standardizer.event_to_features(event)
        probability = _positive_probability(self.bundle.detector_pipeline, features)
        threshold = self.bundle.detector_threshold
        abstain_margin = max(0.0, float(getattr(self.bundle, "detector_abstain_margin", 0.0) or 0.0))
        if abstain_margin and abs(probability - threshold) <= abstain_margin:
            return DetectionOutput(
                event_id=event.event_id,
                is_malicious=False,
                probability=probability,
                evidence=[
                    f"general_supervised_probability={probability:.4f}",
                    f"threshold={threshold:.4f}",
                    f"abstain_margin={abstain_margin:.4f}",
                    f"feature_count={len(features)}",
                ],
                model_name=self.model_name,
                next_route="judge",
                abstain=True,
            )
        is_malicious = probability >= self.bundle.detector_threshold
        return DetectionOutput(
            event_id=event.event_id,
            is_malicious=is_malicious,
            probability=probability,
            evidence=[
                f"general_supervised_probability={probability:.4f}",
                f"threshold={threshold:.4f}",
                f"feature_count={len(features)}",
            ],
            model_name=self.model_name,
            next_route="classify" if is_malicious else "end",
            abstain=False,
        )


class GeneralSupervisedClassifier:
    def __init__(self, bundle: GeneralSupervisedModelBundle):
        self.bundle = bundle
        self.standardizer = GeneralizedFeatureStandardizer(
            include_origin=getattr(bundle, "include_origin", False),
            feature_set=getattr(bundle, "feature_set", "full"),
        )
        self.model_name = f"general_classifier::{bundle.model_name}"

    def classify(self, event: CanonicalEvent, detection: DetectionOutput) -> ClassificationOutput:
        if not detection.is_malicious:
            return ClassificationOutput(
                event_id=event.event_id,
                attack_family=None,
                attack_subtype=None,
                confidence=max(0.0, min(1.0, 1.0 - detection.probability)),
                reason=["general_detector_marked_benign", f"general_classifier_model={self.model_name}"],
                next_route="end",
            )
        features = self.standardizer.event_to_features(event)
        predicted_family = str(self.bundle.classifier_pipeline.predict([features])[0])
        confidence = _predicted_class_probability(self.bundle.classifier_pipeline, features, predicted_family)
        if predicted_family == "benign":
            return ClassificationOutput(
                event_id=event.event_id,
                attack_family=None,
                attack_subtype=None,
                confidence=confidence,
                reason=["general_classifier_predicted_benign", f"general_classifier_model={self.model_name}"],
                next_route="end",
            )
        return ClassificationOutput(
            event_id=event.event_id,
            attack_family=predicted_family,
            attack_subtype=predicted_family,
            confidence=confidence,
            cross_dataset_neighbors=[],
            reason=[
                f"general_attack_family={predicted_family}",
                f"general_classifier_probability={confidence:.4f}",
                f"feature_count={len(features)}",
                f"general_classifier_model={self.model_name}",
            ],
            next_route="explain" if confidence >= getattr(self.bundle, "classifier_abstain_threshold", 0.45) else "judge",
        )


COMMON_PORTS = {
    20: "ftp_data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "dns",
    80: "http",
    110: "pop3",
    123: "ntp",
    143: "imap",
    443: "https",
    1883: "mqtt",
    5683: "coap",
    8080: "http_alt",
}


FAMILY_HINTS = {
    "ddos": ("ddos", "dos", "flood", "syn", "udp flood", "tcp flood"),
    "scanning": ("scan", "recon", "probe", "enumerat"),
    "botnet": ("botnet", "mirai", "gafgyt", "torii", "c&c", "c2"),
    "bruteforce": ("brute", "password", "login", "credential"),
    "injection": ("injection", "sql", "xss", "command injection"),
    "malware": ("malware", "ransom", "backdoor", "trojan", "worm"),
    "exfiltration": ("exfil", "data theft", "leak"),
    "mitm": ("mitm", "man in the middle", "spoof"),
}


def feature_group(feature: str) -> str:
    raw = feature.lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", raw) if token]
    token_set = set(tokens)
    name = " ".join(tokens)
    compact = "".join(tokens)
    if any(token in name for token in ("temperature", "thermostat", "sensor", "humidity", "pressure", "water", "weather")):
        return "iot_telemetry"
    if any(token in name for token in ("byte", "bytes", "payload", "tcp len", "frame len", "sbytes", "dbytes")):
        return "bytes"
    if any(token in name for token in ("pkt", "pkts", "packet")):
        return "packets"
    if (
        any(token in token_set for token in ("srcport", "sport"))
        or any(token in compact for token in ("srcport", "sourceport", "idorigp"))
        or ({"src", "port"} <= token_set)
        or ({"source", "port"} <= token_set)
        or ({"orig", "p"} <= token_set)
    ):
        return "source_port"
    if (
        any(token in token_set for token in ("dstport", "dport"))
        or any(token in compact for token in ("dstport", "destport", "destinationport", "idrespp"))
        or ({"dst", "port"} <= token_set)
        or ({"dest", "port"} <= token_set)
        or ({"destination", "port"} <= token_set)
        or ({"resp", "p"} <= token_set)
    ):
        return "destination_port"
    if any(token in name for token in ("src", "source", "orig", "saddr", "id orig")):
        return "source_address"
    if any(token in name for token in ("dst", "dest", "resp", "daddr", "id resp")):
        return "destination_address"
    if any(token in name for token in ("proto", "protocol")):
        return "protocol"
    if any(token in name for token in ("duration", "dur")):
        return "duration"
    if any(token in name for token in ("status", "state", "flag", "flgs", "history", "conn", "ack", "seq")):
        return "state_flags"
    if any(token in name for token in ("rate", "mean", "stddev", "sum", "min", "max", "magnitude", "radius", "covariance")):
        return "flow_statistics"
    if any(token in name for token in ("cpu", "processor", "process", "memory", "disk", "thread", "handle", "pid", "cmd")):
        return "host_metrics"
    if any(token in name for token in ("service", "http", "dns", "mqtt", "modbus", "ssl", "tls")):
        return "application_service"
    if any(token in name for token in ("message", "description", "rule", "alert", "raw", "log")):
        return "textual_alert_context"
    return "other"


def schema_profile(event: CanonicalEvent) -> str:
    if event.schema_profile:
        return event.schema_profile
    if event.modality == "host_log":
        return "host_metrics"
    if event.modality == "telemetry":
        return "iot_telemetry"
    if event.modality == "alert":
        return "alert_text"
    keys = {str(key).lower() for key in (event.telemetry or {})}
    if any(key.startswith(("tcp.", "udp.", "icmp.", "mqtt.", "http.", "arp.")) for key in keys):
        return "network_packet"
    if {"src_port", "dst_port"} & keys or any("state" in key or "proto" in key for key in keys):
        return "network_flow"
    return event.modality


def _positive_probability(pipeline: Any, features: dict[str, Any]) -> float:
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([features])[0]
        classes = _pipeline_classes(pipeline)
        if True in classes:
            return float(probabilities[classes.index(True)])
        if 1 in classes:
            return float(probabilities[classes.index(1)])
        if "malicious" in classes:
            return float(probabilities[classes.index("malicious")])
        return float(max(probabilities))
    prediction = pipeline.predict([features])[0]
    return 1.0 if bool(prediction) else 0.0


def _predicted_class_probability(pipeline: Any, features: dict[str, Any], predicted: str) -> float:
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([features])[0]
        classes = [str(item) for item in _pipeline_classes(pipeline)]
        if predicted in classes:
            return float(probabilities[classes.index(predicted)])
        return float(max(probabilities))
    return 1.0


def _pipeline_classes(pipeline: Any) -> list[Any]:
    if hasattr(pipeline, "classes_"):
        return list(getattr(pipeline, "classes_"))
    named_steps = getattr(pipeline, "named_steps", {})
    model = named_steps.get("model") if isinstance(named_steps, dict) else None
    if model is not None and hasattr(model, "classes_"):
        return list(getattr(model, "classes_"))
    return []


def _coerce_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return int(value) if isinstance(value, bool) else None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null", "-"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    if not math.isfinite(number) or abs(number) > 1_000_000_000_000:
        return None
    return number


def _normalize_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:40] or "unknown"


def _port_bucket(port: int) -> str:
    if port < 1024:
        return "system"
    if port < 49152:
        return "registered"
    return "ephemeral"


def _compact_categorical_value(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "null", "-"}:
        return None
    if len(text) > 80 or re.search(r"\d+\.\d+\.\d+\.\d+", text):
        return None
    return _normalize_token(text)
