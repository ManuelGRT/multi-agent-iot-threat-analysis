# src/adapters/generic.py
from __future__ import annotations

from datetime import datetime
import math
import re
from typing import Any

from src.adapters.base import BaseAdapter
from src.agents.predictive_sanitization import is_predictive_target_field
from src.contracts.canonical import CanonicalEvent, Provenance


class GenericAdapter(BaseAdapter):
    dataset_name = "generic"

    def adapt(self, row: dict[str, Any], source_file: str, row_id: int | str) -> CanonicalEvent:
        raw_text = self.first_present(row, "raw_text", "text", "message", "log")
        mapped = {
            "event_id": f"generic::{source_file}::{row_id}",
            "modality": self._infer_modality(row),
            "ts": self._timestamp(row),
            "src_ip": self._endpoint(row, "src_ip", "source_ip", "src", "saddr", "id.orig_h", "ip.src_host"),
            "dst_ip": self._endpoint(row, "dst_ip", "destination_ip", "dst", "daddr", "id.resp_h", "ip.dst_host"),
            "src_port": self.to_int(self.first_present(row, "src_port", "source_port", "sport", "id.orig_p", "tcp.srcport", "udp.srcport")),
            "dst_port": self.to_int(self.first_present(row, "dst_port", "destination_port", "dport", "id.resp_p", "tcp.dstport", "udp.dstport", "udp.port")),
            "transport_proto": self._transport_proto(row),
            "app_proto": self.first_present(row, "service", "app_proto", "application_protocol"),
            "packet_count": self._sum_ints(row, "packet_count", "packets", "pkts", "orig_pkts", "src_pkts", "dst_pkts"),
            "byte_count": self._sum_ints(row, "byte_count", "bytes", "orig_ip_bytes", "src_bytes", "dst_bytes", "frame.len"),
            "duration_ms": self._duration_ms(row),
            "telemetry": self._telemetry(row),
            "label_raw": self._label(row),
        }
        semantic_text = str(raw_text) if raw_text else self.build_semantic_text(row, mapped)
        provenance = Provenance(dataset=str(row.get("dataset") or self.dataset_name), source_file=source_file, row_id=row_id)
        mapped["semantic_text"] = semantic_text
        mapped["provenance"] = provenance
        mapped["schema_profile"] = self._schema_profile(row, mapped)
        mapped["origin"] = {
            "source_name": provenance.dataset,
            "source_file": source_file,
            "row_id": row_id,
            "schema_profile": mapped["schema_profile"],
        }
        mapped["feature_groups"] = self._feature_groups(row)
        mapped["evidence_fields"] = self._evidence_fields(row, mapped)
        mapped["traffic_direction"] = self._traffic_direction(mapped)
        mapped["service_context"] = self._service_context(row, mapped)
        mapped["host_context"] = self._host_context(row)
        mapped["telemetry_context"] = self._telemetry_context(row)
        mapped["anomaly_summary"] = self._anomaly_summary(mapped)
        mapped["mapping_confidence"] = self.compute_mapping_confidence(row, mapped)
        mapped["missing_fields"] = self.missing_fields(mapped, ("src_ip", "dst_ip", "transport_proto", "label_raw"))
        return CanonicalEvent(**mapped)

    def _endpoint(self, row: dict[str, Any], *names: str) -> str | None:
        value = self.first_present(row, *names)
        if value in (None, ""):
            return None
        text = str(value).strip()
        if text.lower() in {"nan", "none", "null", "-", "0", "0.0"}:
            return None
        return text

    def _timestamp(self, row: dict[str, Any]) -> str | None:
        direct = self.first_present(row, "ts", "timestamp", "datetime")
        if direct is not None:
            return self._normalize_timestamp(str(direct))
        frame_time = self.to_float(self.first_present(row, "frame.time"))
        if frame_time is not None and frame_time > 1_000_000_000:
            return datetime.fromtimestamp(frame_time).isoformat()
        date = self.first_present(row, "date")
        time = self.first_present(row, "time")
        if date is not None and time is not None:
            return self._normalize_timestamp(f"{date} {str(time).strip()}")
        return self._normalize_timestamp(str(date or time)) if date or time else None

    def _normalize_timestamp(self, value: str) -> str | None:
        return self.to_iso_timestamp(value)

    def _transport_proto(self, row: dict[str, Any]) -> Any:
        direct = self.first_present(row, "proto", "protocol", "transport_proto", "ip.proto")
        if direct not in (None, "", "0", "0.0", "-"):
            return direct
        if self._nonzero(row, "tcp.srcport", "tcp.dstport", "tcp.len", "tcp.ack"):
            return "tcp"
        if self._nonzero(row, "udp.srcport", "udp.dstport", "udp.port", "udp.stream", "udp.time_delta"):
            return "udp"
        if self._nonzero(row, "icmp.checksum", "icmp.seq_le"):
            return "icmp"
        return None

    def _label(self, row: dict[str, Any]) -> Any:
        structured = self._structured_label(row)
        if structured is not None:
            return structured
        label = self.first_present(row, "label", "Label", "target")
        semantic = self.first_present(row, "type", "attack_type", "Attack_type", "category", "subcategory", "class", "detailed-label", "det_label", "attack")
        if str(label).strip().lower() in {"1", "true", "malicious", "attack"} and semantic not in (None, "", "-"):
            return semantic
        if str(label).strip().lower() in {"0", "false", "normal", "benign"}:
            return "normal"
        return semantic or label

    def _structured_label(self, row: dict[str, Any]) -> str | None:
        label = self.first_present(row, "label", "Label")
        detailed = self.first_present(row, "detailed-label", "det_label")
        if str(label).strip().lower() in {"1", "true", "malicious", "attack"} and detailed not in (None, "", "-"):
            return str(detailed)

        category = self.first_present(row, "category")
        subcategory = self.first_present(row, "subcategory")
        if category not in (None, "", "-"):
            category_text = str(category).strip()
            if category_text.lower() == "normal":
                return "normal"
            parts = [category_text]
            if subcategory not in (None, "", "-") and str(subcategory).strip().lower() != "normal":
                parts.append(str(subcategory).strip())
            return " ".join(parts)
        return None

    def _infer_modality(self, row: dict[str, Any]) -> str:
        keys = {str(key).lower() for key in row}
        telemetry_indicators = (
            "current_temperature",
            "humidity",
            "pressure",
            "sensor",
            "temperature",
            "thermostat",
            "water",
            "weather",
        )
        if {"sensor", "value"} <= keys or any(indicator in key for key in keys for indicator in telemetry_indicators):
            return "telemetry"
        host_indicators = (
            "process",
            "pid",
            "cmd",
            "memory",
            "logicaldisk",
            "processor",
            "vsize",
            "rsize",
            "majflt",
            "minflt",
            "log",
        )
        if any(indicator in key for key in keys for indicator in host_indicators):
            return "host_log"
        return "network_flow"

    def _duration_ms(self, row: dict[str, Any]) -> float | None:
        milliseconds = self.to_float(self.first_present(row, "duration_ms", "duration_milliseconds"))
        if milliseconds is not None:
            return milliseconds
        seconds = self.to_float(self.first_present(row, "duration", "dur"))
        return seconds * 1000 if seconds is not None else None

    def _sum_ints(self, row: dict[str, Any], *names: str) -> int | None:
        values = [self.to_int(row.get(name)) for name in names if row.get(name) not in (None, "")]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return sum(values)

    def _nonzero(self, row: dict[str, Any], *names: str) -> bool:
        for name in names:
            value = self.to_float(row.get(name))
            if value not in (None, 0.0):
                return True
        return False

    def compute_mapping_confidence(self, row: dict[str, Any], mapped: dict[str, Any]) -> float:
        required = ["modality", "semantic_text", "provenance"]
        present_required = sum(1 for key in required if mapped.get(key) is not None)
        base_score = present_required / len(required)
        modality = str(mapped.get("modality") or "").strip()
        telemetry = mapped.get("telemetry") or {}

        if modality == "network_flow":
            core = ["src_ip", "dst_ip", "transport_proto"]
            present_core = sum(1 for key in core if mapped.get(key) not in (None, "", "unknown"))
            structural_score = present_core / len(core)
            if telemetry:
                structural_score = max(structural_score, min(1.0, 0.35 + len(telemetry) / 20))
        elif modality in {"host_log", "telemetry", "alert"}:
            structural_score = min(1.0, len(telemetry) / 4)
            if mapped.get("ts") is not None:
                structural_score = max(structural_score, 0.65)
            if mapped.get("label_raw") not in (None, "", "unknown"):
                structural_score = max(structural_score, 0.75)
        else:
            structural_score = min(1.0, len(telemetry) / 6)

        return round(0.35 * base_score + 0.65 * structural_score, 3)

    def _schema_profile(self, row: dict[str, Any], mapped: dict[str, Any]) -> str:
        modality = str(mapped.get("modality") or "").strip()
        keys = [str(key).lower() for key in row]
        if modality == "host_log":
            return "host_metrics"
        if modality == "telemetry":
            return "iot_telemetry"
        if modality == "alert":
            return "alert_text"
        packet_prefixes = ("arp.", "eth.", "frame.", "http.", "icmp.", "ip.", "mqtt.", "tcp.", "udp.")
        if sum(1 for key in keys if key.startswith(packet_prefixes)) >= 3:
            return "network_packet"
        if any(any(token in key for token in ("alert", "message", "description", "rule", "signature")) for key in keys):
            return "alert_text"
        if modality == "pcap_ref":
            return "pcap_ref"
        if modality == "network_flow":
            return "network_flow"
        return "unknown"

    def _feature_groups(self, row: dict[str, Any]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for key, value in row.items():
            if self._is_label_key(key) or self._is_blank_value(value):
                continue
            group = self._generic_group(str(key))
            groups.setdefault(group, []).append(str(key))
        return {
            group: sorted(columns)[:40]
            for group, columns in sorted(groups.items())
            if columns
        }

    def _evidence_fields(self, row: dict[str, Any], mapped: dict[str, Any]) -> list[str]:
        selected = []
        for key, value in row.items():
            key_text = str(key)
            if self._is_label_key(key_text) or self._is_blank_value(value):
                continue
            group = self._generic_group(key_text)
            if group != "other":
                selected.append(key_text)
        canonical_present = [
            key for key in (
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
                "transport_proto",
                "app_proto",
                "packet_count",
                "byte_count",
                "duration_ms",
            )
            if mapped.get(key) not in (None, "", "unknown")
        ]
        return sorted({*selected[:60], *canonical_present})

    def _traffic_direction(self, mapped: dict[str, Any]) -> str | None:
        src = mapped.get("src_ip")
        dst = mapped.get("dst_ip")
        if not src and not dst:
            return None
        if self._is_private_endpoint(src) and not self._is_private_endpoint(dst):
            return "internal_to_external"
        if not self._is_private_endpoint(src) and self._is_private_endpoint(dst):
            return "external_to_internal"
        if self._is_private_endpoint(src) and self._is_private_endpoint(dst):
            return "internal_to_internal"
        return "external_or_unknown"

    def _service_context(self, row: dict[str, Any], mapped: dict[str, Any]) -> dict[str, Any]:
        context: dict[str, Any] = {}
        dst_port = mapped.get("dst_port")
        src_port = mapped.get("src_port")
        context["has_transport"] = bool(mapped.get("transport_proto"))
        context["has_application"] = bool(mapped.get("app_proto"))
        if dst_port is not None:
            context["dst_port_bucket"] = self._port_bucket(dst_port)
            context["dst_common_service"] = COMMON_PORTS.get(int(dst_port))
        if src_port is not None:
            context["src_port_bucket"] = self._port_bucket(src_port)
        context["protocol_fields"] = len(self._feature_groups(row).get("protocol", []))
        context["volume_fields"] = len(self._feature_groups(row).get("network_volume", []))
        return {key: value for key, value in context.items() if value not in (None, "", [])}

    def _host_context(self, row: dict[str, Any]) -> dict[str, Any]:
        host_keys = [
            str(key) for key in row
            if self._generic_group(str(key)) == "host_metrics" and not self._is_blank_value(row.get(key))
        ]
        numeric = [self.to_float(row.get(key)) for key in host_keys]
        numeric_values = [value for value in numeric if value is not None and math.isfinite(value)]
        return {
            "host_field_count": len(host_keys),
            "host_numeric_count": len(numeric_values),
            "has_process_signal": any("process" in key.lower() or "pid" in key.lower() for key in host_keys),
            "has_resource_signal": any(
                token in key.lower()
                for key in host_keys
                for token in ("cpu", "processor", "memory", "disk", "thread", "handle")
            ),
        }

    def _telemetry_context(self, row: dict[str, Any]) -> dict[str, Any]:
        telemetry_keys = [
            str(key) for key in row
            if self._generic_group(str(key)) == "iot_telemetry" and not self._is_blank_value(row.get(key))
        ]
        numeric_values = [
            value for value in (self.to_float(row.get(key)) for key in telemetry_keys)
            if value is not None and math.isfinite(value)
        ]
        context: dict[str, Any] = {
            "telemetry_field_count": len(telemetry_keys),
            "telemetry_numeric_count": len(numeric_values),
            "has_sensor_state": any("status" in key.lower() or "state" in key.lower() for key in telemetry_keys),
        }
        if numeric_values:
            context.update({
                "telemetry_min": min(numeric_values),
                "telemetry_max": max(numeric_values),
                "telemetry_mean": sum(numeric_values) / len(numeric_values),
                "telemetry_range": max(numeric_values) - min(numeric_values),
            })
        return context

    def _anomaly_summary(self, mapped: dict[str, Any]) -> str:
        parts = [
            f"schema_profile={mapped.get('schema_profile') or 'unknown'}",
            f"traffic_direction={mapped.get('traffic_direction') or 'unknown'}",
        ]
        service = mapped.get("service_context") or {}
        if service.get("dst_common_service"):
            parts.append(f"service={service['dst_common_service']}")
        if mapped.get("packet_count") is not None:
            parts.append(f"packets={mapped['packet_count']}")
        if mapped.get("byte_count") is not None:
            parts.append(f"bytes={mapped['byte_count']}")
        return "; ".join(parts)

    def _generic_group(self, feature: str) -> str:
        raw = feature.lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", raw) if token]
        token_set = set(tokens)
        compact = "".join(tokens)
        name = " ".join(tokens)
        if any(token in name for token in ("temperature", "thermostat", "sensor", "humidity", "pressure", "water", "weather")):
            return "iot_telemetry"
        if any(token in name for token in ("cpu", "processor", "process", "memory", "disk", "thread", "handle", "pid", "cmd")):
            return "host_metrics"
        if any(token in name for token in ("message", "description", "rule", "alert", "raw", "log", "signature")):
            return "textual_alert_context"
        if any(token in name for token in ("byte", "payload", "tcp len", "frame len", "sbytes", "dbytes", "pkt", "packet", "rate")):
            return "network_volume"
        if any(token in name for token in ("proto", "protocol", "service", "http", "dns", "mqtt", "modbus", "ssl", "tls")):
            return "protocol"
        if (
            any(token in compact for token in ("srcport", "sourceport", "dstport", "destport", "destinationport"))
            or ({"src", "port"} <= token_set)
            or ({"dst", "port"} <= token_set)
            or ({"source", "port"} <= token_set)
            or ({"destination", "port"} <= token_set)
        ):
            return "ports"
        if any(token in name for token in ("src", "source", "orig", "saddr", "dst", "dest", "resp", "daddr")):
            return "network_endpoint"
        if any(token in name for token in ("duration", "dur", "time", "timestamp", "date")):
            return "timing"
        if any(token in name for token in ("status", "state", "flag", "flgs", "history", "conn", "ack", "seq")):
            return "state_flags"
        if any(token in name for token in ("mean", "stddev", "sum", "min", "max", "magnitude", "radius", "covariance")):
            return "statistics"
        return "other"

    def _port_bucket(self, value: Any) -> str:
        port = self.to_int(value)
        if port is None:
            return "unknown"
        if port < 1024:
            return "system"
        if port < 49152:
            return "registered"
        return "ephemeral"

    def _is_private_endpoint(self, value: Any) -> bool:
        text = str(value or "").strip()
        return (
            text.startswith("10.")
            or text.startswith("192.168.")
            or any(text.startswith(f"172.{index}.") for index in range(16, 32))
        )

    def _is_label_key(self, key: Any) -> bool:
        return is_predictive_target_field(key)

    def _is_blank_value(self, value: Any) -> bool:
        if value is None:
            return True
        text = str(value).strip().lower()
        return text in {"", "-", "nan", "none", "null"}

    def _telemetry(self, row: dict[str, Any]) -> dict[str, float | int | str]:
        excluded = {
            "Attack_label",
            "label",
            "Label",
            "attack",
            "attack_label",
            "Attack_type",
            "attack_type",
            "src_ip",
            "dst_ip",
            "source_ip",
            "destination_ip",
            "src_port",
            "dst_port",
            "proto",
            "protocol",
            "target",
            "type",
            "category",
            "subcategory",
            "class",
            "detailed-label",
            "detailed_label",
            "det_label",
        }
        return {
            str(key): value
            for key, value in row.items()
            if key not in excluded
            and not is_predictive_target_field(key)
            and isinstance(value, (int, float, str))
        }


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
