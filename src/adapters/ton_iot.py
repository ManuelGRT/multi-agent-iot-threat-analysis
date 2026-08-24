# src/adapters/ton_iot.py
from __future__ import annotations

from src.adapters.base import BaseAdapter
from src.contracts.canonical import CanonicalEvent, Provenance

class TONIoTAdapter(BaseAdapter):
    dataset_name = "TON_IoT"

    def adapt(self, row: dict, source_file: str, row_id: int | str) -> CanonicalEvent:
        modality = row.get("modality") or ("telemetry" if "sensor" in source_file.lower() else "network_flow")
        telemetry = {}
        if modality == "telemetry":
            telemetry = {
                k: v for k, v in row.items()
                if k not in {"label", "attack", "ts"} and isinstance(v, (int, float, str))
            }

        mapped = {
            "event_id": f"ton_iot::{source_file}::{row_id}",
            "modality": modality,
            "ts": self.to_iso_timestamp(row.get("ts")),
            "src_ip": self.first_present(row, "src_ip", "srcip", "src"),
            "dst_ip": self.first_present(row, "dst_ip", "dstip", "dst"),
            "src_port": self.to_int(self.first_present(row, "src_port", "sport", "srcport")),
            "dst_port": self.to_int(self.first_present(row, "dst_port", "dport", "dstport")),
            "transport_proto": self.first_present(row, "proto", "protocol", "transport_proto"),
            "duration_ms": self.to_float(self.first_present(row, "duration_ms", "dur", "duration")),
            "packet_count": self.to_int(self.first_present(row, "packet_count", "pkts", "packets")),
            "byte_count": self.to_int(self.first_present(row, "byte_count", "bytes")),
            "telemetry": telemetry,
            "label_raw": row.get("label") or row.get("attack"),
        }
        semantic_text = self.build_semantic_text(row, mapped)
        provenance = Provenance(dataset=self.dataset_name, source_file=source_file, row_id=row_id)
        mapped["semantic_text"] = semantic_text
        mapped["provenance"] = provenance
        mapped["mapping_confidence"] = self.compute_mapping_confidence(row, mapped)
        mapped["missing_fields"] = self.missing_fields(mapped, ("label_raw",))
        return CanonicalEvent(**mapped)
