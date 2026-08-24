from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

import joblib

from src.agents.predictive_sanitization import is_predictive_target_field, scrub_predictive_payload
from src.contracts.agents import ClassificationOutput, DetectionOutput
from src.contracts.canonical import CanonicalEvent


EDGE_CLASSES = [
    "Normal",
    "DDoS_UDP",
    "DDoS_ICMP",
    "SQL_injection",
    "Password",
    "Vulnerability_scanner",
    "DDoS_TCP",
    "DDoS_HTTP",
    "Uploading",
    "Backdoor",
    "Port_Scanning",
    "XSS",
    "Ransomware",
    "MITM",
    "Fingerprinting",
]


EDGE_TO_FAMILY = {
    "Normal": None,
    "DDoS_UDP": "ddos",
    "DDoS_ICMP": "ddos",
    "DDoS_TCP": "ddos",
    "DDoS_HTTP": "ddos",
    "SQL_injection": "injection",
    "XSS": "injection",
    "Uploading": "injection",
    "Password": "bruteforce",
    "Vulnerability_scanner": "scanning",
    "Port_Scanning": "scanning",
    "Fingerprinting": "scanning",
    "Backdoor": "malware",
    "Ransomware": "malware",
    "MITM": "mitm",
}


LABEL_COLUMNS = {
    "Attack_label",
    "Attack_type",
    "label",
    "type",
    "category",
    "class",
    "target",
    "detailed-label",
    "detailed_label",
}


class TechnicalFeatureStandardizer:
    """Extracts label-free technical features from raw rows or canonical events."""

    version = "technical_feature_standardizer_v2"

    def row_to_features(self, row: dict[str, Any]) -> dict[str, Any]:
        features: dict[str, Any] = {}
        label_names = {name.lower() for name in LABEL_COLUMNS}
        for key, value in row.items():
            key_text = str(key).strip()
            if (
                not key_text
                or key_text in LABEL_COLUMNS
                or key_text.lower() in label_names
                or is_predictive_target_field(key_text)
            ):
                continue
            if self._is_empty(value):
                continue
            features[key_text] = self._coerce_feature_value(value)
        return features

    def event_to_features(self, event: CanonicalEvent) -> dict[str, Any]:
        features = self.row_to_features(event.telemetry or {})
        for key, value in self.row_to_features(event.host or {}).items():
            features[f"host.{key}"] = value
        canonical_values = {
            "canonical.modality": event.modality,
            "canonical.schema_profile": event.schema_profile,
            "canonical.src_ip": event.src_ip,
            "canonical.dst_ip": event.dst_ip,
            "canonical.src_port": event.src_port,
            "canonical.dst_port": event.dst_port,
            "canonical.transport_proto": event.transport_proto,
            "canonical.app_proto": event.app_proto,
            "canonical.packet_count": event.packet_count,
            "canonical.byte_count": event.byte_count,
            "canonical.duration_ms": event.duration_ms,
            "canonical.severity": event.severity,
            "canonical.traffic_direction": event.traffic_direction,
            "canonical.mapping_confidence": event.mapping_confidence,
            "canonical.missing_fields_count": len(event.missing_fields or []),
            "canonical.evidence_fields_count": len(event.evidence_fields or []),
        }
        for key, value in canonical_values.items():
            if not self._is_empty(value):
                features[key] = self._coerce_feature_value(value)
        self._add_context_features(features, event)
        return features

    def _add_context_features(self, features: dict[str, Any], event: CanonicalEvent) -> None:
        """Preserve label-free structure produced by the LLM standardizer."""
        for group, columns in scrub_predictive_payload(event.feature_groups or {}).items():
            if is_predictive_target_field(group):
                continue
            group_token = self._token(group)
            if not group_token:
                continue
            safe_columns = [
                column
                for column in columns or []
                if not self._is_empty(column) and not is_predictive_target_field(column)
            ]
            features[f"canonical.feature_group.{group_token}"] = 1
            features[f"canonical.feature_group_count.{group_token}"] = len(safe_columns)
            for column in safe_columns[:40]:
                column_token = self._token(column)
                if column_token:
                    features[f"canonical.feature_group_column.{group_token}.{column_token}"] = 1

        for field in scrub_predictive_payload(event.evidence_fields or []):
            if self._is_empty(field) or is_predictive_target_field(field):
                continue
            field_token = self._token(field)
            if field_token:
                features[f"canonical.evidence_field.{field_token}"] = 1

        for prefix, context in (
            ("service", event.service_context or {}),
            ("host_context", event.host_context or {}),
            ("telemetry_context", event.telemetry_context or {}),
        ):
            self._add_flat_context(features, prefix, scrub_predictive_payload(context))

    def _add_flat_context(self, features: dict[str, Any], prefix: str, values: Any) -> None:
        if not isinstance(values, dict):
            return
        for key, value in values.items():
            if self._is_empty(value) or is_predictive_target_field(key):
                continue
            key_token = self._token(key)
            if not key_token:
                continue
            feature_key = f"canonical.{prefix}.{key_token}"
            coerced = self._coerce_feature_value(value)
            if isinstance(coerced, str):
                value_token = self._compact_categorical_value(coerced)
                if value_token:
                    features[f"{feature_key}.{value_token}"] = 1
                continue
            features[feature_key] = coerced

    @staticmethod
    def _is_empty(value: Any) -> bool:
        if value is None:
            return True
        text = str(value).strip()
        return text == "" or text.lower() in {"nan", "none", "null", "-"}

    @staticmethod
    def _coerce_feature_value(value: Any) -> Any:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return value
        text = str(value).strip()
        try:
            number = float(text)
        except ValueError:
            return text
        if not math.isfinite(number) or abs(number) > 1_000_000_000_000:
            return text
        if number.is_integer():
            return int(number)
        return number

    @staticmethod
    def _token(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        return text[:80]

    @classmethod
    def _compact_categorical_value(cls, value: Any) -> str | None:
        text = str(value or "").strip().lower()
        if cls._is_empty(text) or len(text) > 80 or re.search(r"\d+\.\d+\.\d+\.\d+", text):
            return None
        return cls._token(text) or None


@dataclass
class EdgeSupervisedModelBundle:
    detector_pipeline: Any
    classifier_pipeline: Any
    detector_threshold: float = 0.5
    domain_guard: bool = True
    standardizer_version: str = TechnicalFeatureStandardizer.version
    model_name: str = "edgeiiot_supervised_multiagent_v1"


def save_edge_bundle(bundle: EdgeSupervisedModelBundle, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_edge_bundle(path: str | Path) -> EdgeSupervisedModelBundle:
    return joblib.load(path)


class EdgeSupervisedDetector:
    def __init__(self, bundle: EdgeSupervisedModelBundle):
        self.bundle = bundle
        self.standardizer = TechnicalFeatureStandardizer()
        self.model_name = f"edge_detector::{bundle.model_name}"

    def detect(self, event: CanonicalEvent) -> DetectionOutput:
        features = self.standardizer.event_to_features(event)
        if getattr(self.bundle, "domain_guard", True) and not _is_edge_compatible(event, features):
            return DetectionOutput(
                event_id=event.event_id,
                is_malicious=False,
                probability=0.0,
                evidence=[
                    "edge_domain_guard=out_of_domain",
                    f"feature_count={len(features)}",
                    "specialized_edge_detector_abstained",
                ],
                model_name=self.model_name,
                next_route="judge",
                abstain=True,
            )
        probability = _positive_probability(self.bundle.detector_pipeline, features)
        is_malicious = probability >= self.bundle.detector_threshold
        return DetectionOutput(
            event_id=event.event_id,
            is_malicious=is_malicious,
            probability=probability,
            evidence=[
                f"edge_supervised_probability={probability:.4f}",
                f"threshold={self.bundle.detector_threshold:.4f}",
                f"feature_count={len(features)}",
            ],
            model_name=self.model_name,
            next_route="classify" if is_malicious else "end",
            abstain=False,
        )


class EdgeSupervisedClassifier:
    def __init__(self, bundle: EdgeSupervisedModelBundle):
        self.bundle = bundle
        self.standardizer = TechnicalFeatureStandardizer()
        self.model_name = f"edge_classifier::{bundle.model_name}"

    def classify(self, event: CanonicalEvent, detection: DetectionOutput) -> ClassificationOutput:
        if detection.abstain:
            return ClassificationOutput(
                event_id=event.event_id,
                attack_family=None,
                attack_subtype=None,
                confidence=0.0,
                reason=["edge_detector_abstained", f"edge_classifier_model={self.model_name}"],
                next_route="judge",
            )
        if not detection.is_malicious:
            return ClassificationOutput(
                event_id=event.event_id,
                attack_family=None,
                attack_subtype=None,
                confidence=max(0.0, min(1.0, 1.0 - detection.probability)),
                reason=["edge_detector_marked_benign", f"edge_classifier_model={self.model_name}"],
                next_route="end",
            )
        features = self.standardizer.event_to_features(event)
        predicted_type = str(self.bundle.classifier_pipeline.predict([features])[0])
        confidence = _predicted_class_probability(self.bundle.classifier_pipeline, features, predicted_type)
        family = EDGE_TO_FAMILY.get(predicted_type, "unknown_attack")
        return ClassificationOutput(
            event_id=event.event_id,
            attack_family=family,
            attack_subtype=predicted_type,
            confidence=confidence,
            cross_dataset_neighbors=[],
            reason=[
                f"edge_attack_type={predicted_type}",
                f"edge_classifier_probability={confidence:.4f}",
                f"feature_count={len(features)}",
                f"edge_classifier_model={self.model_name}",
            ],
            next_route="explain" if family is not None else "end",
        )


def _positive_probability(pipeline: Any, features: dict[str, Any]) -> float:
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([features])[0]
        classes = _pipeline_classes(pipeline)
        if True in classes:
            return float(probabilities[classes.index(True)])
        if 1 in classes:
            return float(probabilities[classes.index(1)])
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


def _is_edge_compatible(event: CanonicalEvent, features: dict[str, Any]) -> bool:
    provenance = event.provenance
    source_text = " ".join(
        str(value).lower()
        for value in (
            event.event_id,
            provenance.dataset if provenance else "",
            provenance.source_file if provenance else "",
        )
        if value
    )
    if "edge" in source_text or "edgeiiot" in source_text or "iiotset" in source_text:
        return True
    edge_feature_prefixes = (
        "arp.",
        "http.",
        "icmp.",
        "ip.",
        "mqtt.",
        "tcp.",
        "udp.",
    )
    edge_like_features = sum(
        1
        for key in features
        if str(key).lower().startswith(edge_feature_prefixes)
    )
    return edge_like_features >= 8
