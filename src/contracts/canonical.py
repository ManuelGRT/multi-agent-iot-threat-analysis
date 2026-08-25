# src/contracts/canonical.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field

Modality = Literal["network_flow", "telemetry", "host_log", "alert", "pcap_ref"]
Severity = Literal["low", "medium", "high", "critical"]
SchemaProfile = Literal[
    "network_flow",
    "network_packet",
    "host_metrics",
    "iot_telemetry",
    "alert_text",
    "pcap_ref",
    "unknown",
]

class Provenance(BaseModel):
    dataset: str
    source_file: str | None = None
    row_id: str | int | None = None
    split: Literal["train", "val", "test", "stream"] = "train"
    parser_version: str = "0.1.0"
    ingestion_ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CanonicalEvent(BaseModel):
    event_id: str
    modality: Modality
    ts: datetime | None = None
    src_ip: str | None = None
    dst_ip: str | None = None
    src_port: int | None = None
    dst_port: int | None = None
    transport_proto: str | None = None
    app_proto: str | None = None

    packet_count: int | None = None
    byte_count: int | None = None
    duration_ms: float | None = None

    telemetry: dict[str, float | int | str] = Field(default_factory=dict)
    host: dict[str, Any] = Field(default_factory=dict)

    label_raw: str | None = None
    attack_family: str | None = None
    attack_subtype: str | None = None
    severity: Severity | None = None
    schema_profile: SchemaProfile | None = None
    origin: dict[str, Any] = Field(default_factory=dict)
    feature_groups: dict[str, list[str]] = Field(default_factory=dict)
    evidence_fields: list[str] = Field(default_factory=list)
    traffic_direction: str | None = None
    service_context: dict[str, Any] = Field(default_factory=dict)
    host_context: dict[str, Any] = Field(default_factory=dict)
    telemetry_context: dict[str, Any] = Field(default_factory=dict)
    anomaly_summary: str | None = None
    behavior_tags: list[str] = Field(default_factory=list)
    attack_indicators: list[str] = Field(default_factory=list)
    asset_context: str | None = None
    uncertainty: list[str] = Field(default_factory=list)

    semantic_text: str
    provenance: Provenance
    mapping_confidence: float = Field(ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)

    @property
    def is_labeled_malicious(self) -> bool | None:
        if self.label_raw is None:
            return None
        normalized = self.label_raw.strip().lower()
        if normalized in {"benign", "normal", "0", "false", "none", "-"}:
            return False
        return True
