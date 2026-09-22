"""Deteccion no mutante de campos objetivo en contratos operacionales."""
from __future__ import annotations

import re
from typing import Any


TARGET_FIELD_NAMES = frozenset(
    {
        "attack",
        "attack_cat",
        "attack_category",
        "attack_family",
        "attack_label",
        "attack_subtype",
        "attack_type",
        "category",
        "class",
        "classification",
        "class_id",
        "detailed_label",
        "det_label",
        "family",
        "family_hint",
        "ground_truth",
        "groundtruth",
        "is_attack",
        "label",
        "label_hint",
        "label_id",
        "label_raw",
        "malicious",
        "malware",
        "outcome",
        "prediction",
        "predicted_class",
        "risk_class",
        "risk_family",
        "risk_label",
        "risk_type",
        "subcategory",
        "subtype",
        "target",
        "target_class",
        "target_value",
        "threat",
        "threat_family",
        "threat_label",
        "threat_type",
        "type_attack",
        "y",
        "y_true",
    }
)

PREDICTIVE_TARGET_VALUES = frozenset(
    {
        "attack",
        "backdoor",
        "benign",
        "botnet",
        "brute_force",
        "bruteforce",
        "command_and_control",
        "c2",
        "c_c",
        "c_c_filedownload",
        "c_c_torii",
        "ddos",
        "ddos_http",
        "ddos_icmp",
        "ddos_tcp",
        "ddos_udp",
        "dos",
        "exfiltration",
        "fingerprinting",
        "filedownload",
        "injection",
        "malicious",
        "malware",
        "man_in_the_middle",
        "mirai",
        "mitm",
        "part_of_a_horizontal_port_scan",
        "partofahorizontalportscan",
        "password",
        "port_scanning",
        "ransomware",
        "reconnaissance",
        "scanning",
        "sql_injection",
        "theft",
        "unknown_attack",
        "uploading",
        "vulnerability_scanner",
        "xss",
    }
)

_TARGET_KEY_PATTERN = (
    r"(?:attack(?:[-_](?:cat|category|family|label|subtype|type))?|type[-_]attack|"
    r"detailed[-_]label|det_label|label(?:[-_](?:raw|id))?|target(?:[-_](?:class|value))?|"
    r"ground[-_]?truth|category|subcategory|class(?:[-_]id)?|family|subtype|outcome|"
    r"(?:class|family|label|category)[-_]hint|classification|prediction|predicted[-_]class|"
    r"risk[-_](?:class|family|label|type)|"
    r"y(?:[-_]true)?|is[-_]attack|malicious|malware|"
    r"threat(?:[-_](?:family|label|type))?|type)"
)
_TARGET_ASSIGNMENT_RE = re.compile(
    rf"(?i)(?<![A-Za-z0-9_.-])(?:[\"']?{_TARGET_KEY_PATTERN}[\"']?)\s*[:=]\s*"
)

ALLOWED_CANONICAL_TARGET_LIKE_PATHS = frozenset(
    {
        "attack_indicators",
        "service_context.protocol_family",
        "service_context.dns.query_class",
        "telemetry_context.dns.query_class",
        "service_context.dns.query.type",
        "service_context.mqtt.message.type",
        "telemetry_context.mqtt_message.type",
        "service_context.icmp.type",
        "service_context.dns.type",
        "service_context.dns.qry.type",
        "telemetry_context.iot_sensor.type",
        "telemetry_context.dns.query.type",
        "telemetry_context.sensor_data.type",
        "host_context.interface.type",
        "telemetry_context.geolocation.type",
        "service_context.process.type",
        "feature_groups.other.execution_class",
        "host_context.metric_category",
    }
)


def is_predictive_target_field(key: Any) -> bool:
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(key or "").strip().casefold()
    ).strip("_")
    tokens = {token for token in normalized.split("_") if token}
    return (
        normalized in TARGET_FIELD_NAMES
        or "attack" in tokens
        or "threat" in tokens
        or "malware" in tokens
        or normalized.startswith("attack_")
        or normalized.endswith(
            ("_attack", "_label", "_category", "_subcategory", "_class", "_family", "_target")
        )
    )


def is_predictive_target_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = re.sub(
        r"[^a-z0-9]+", "_", value.strip().casefold()
    ).strip("_")
    return normalized in PREDICTIVE_TARGET_VALUES


def contains_predictive_target_text(text: str | None) -> bool:
    return bool(_TARGET_ASSIGNMENT_RE.search(text or ""))


def is_allowed_canonical_target_path(path: str) -> bool:
    return path.casefold() in ALLOWED_CANONICAL_TARGET_LIKE_PATHS
