# src/adapters/iot23.py
from __future__ import annotations

from src.adapters.base import BaseAdapter
from src.contracts.canonical import CanonicalEvent, Provenance

class IoT23Adapter(BaseAdapter):
    dataset_name = "IoT-23"

    def adapt(self, row: dict, source_file: str, row_id: int | str) -> CanonicalEvent:
        label = self._label(row)
        mapped = {
            "event_id": f"iot23::{source_file}::{row_id}",
            "modality": "network_flow",
            "src_ip": self.first_present(row, "id.orig_h", "src_ip", "srcip"),
            "dst_ip": self.first_present(row, "id.resp_h", "dst_ip", "dstip"),
            "src_port": self.to_int(self.first_present(row, "id.orig_p", "src_port", "sport")),
            "dst_port": self.to_int(self.first_present(row, "id.resp_p", "dst_port", "dport")),
            "transport_proto": self.first_present(row, "proto", "protocol", "transport_proto"),
            "duration_ms": self._duration_ms(row),
            "packet_count": self.to_int(self.first_present(row, "orig_pkts", "packet_count", "pkts")),
            "byte_count": self.to_int(self.first_present(row, "orig_ip_bytes", "byte_count", "bytes")),
            "label_raw": label,
        }
        semantic_text = self.build_semantic_text(row, mapped)
        provenance = Provenance(dataset=self.dataset_name, source_file=source_file, row_id=row_id)
        mapped["semantic_text"] = semantic_text
        mapped["provenance"] = provenance
        mapped["mapping_confidence"] = self.compute_mapping_confidence(row, mapped)
        mapped["missing_fields"] = self.missing_fields(mapped, ("src_ip", "dst_ip", "transport_proto", "label_raw"))
        return CanonicalEvent(**mapped)

    def _label(self, row: dict) -> str | None:
        label = self.first_present(row, "label", "attack")
        detailed = self.first_present(row, "detailed-label", "det_label")
        if str(label).strip().lower() in {"malicious", "attack", "1", "true"} and detailed not in (None, "", "-"):
            return str(detailed)
        return label or detailed

    def _duration_ms(self, row: dict) -> float | None:
        value = self.first_present(row, "duration_ms")
        if value is not None:
            return self.to_float(value)
        seconds = self.to_float(self.first_present(row, "duration", "dur"))
        return seconds * 1000 if seconds is not None else None
