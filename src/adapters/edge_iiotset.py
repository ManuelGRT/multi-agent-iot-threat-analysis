# src/adapters/edge_iiotset.py
from __future__ import annotations

from src.adapters.base import BaseAdapter
from src.contracts.canonical import CanonicalEvent, Provenance


class EdgeIIoTSetAdapter(BaseAdapter):
    dataset_name = "Edge-IIoTset"

    def adapt(self, row: dict, source_file: str, row_id: int | str) -> CanonicalEvent:
        mapped = {
            "event_id": f"edge_iiotset::{source_file}::{row_id}",
            "modality": "network_flow",
            "ts": self.to_iso_timestamp(
                self.first_present(row, "frame.time", "timestamp", "ts")
            ),
            "src_ip": self.first_present(row, "ip.src_host", "src_ip", "srcip"),
            "dst_ip": self.first_present(row, "ip.dst_host", "dst_ip", "dstip"),
            "src_port": self.to_int(self.first_present(row, "tcp.srcport", "udp.srcport", "src_port")),
            "dst_port": self.to_int(self.first_present(row, "tcp.dstport", "udp.dstport", "dst_port")),
            "transport_proto": self.first_present(row, "ip.proto", "protocol", "proto"),
            "duration_ms": self.to_float(self.first_present(row, "duration_ms", "frame.time_delta")),
            "packet_count": self.to_int(self.first_present(row, "packet_count", "pkts")),
            "byte_count": self.to_int(self.first_present(row, "frame.len", "byte_count", "bytes")),
            "label_raw": self.first_present(row, "Attack_type", "attack", "label"),
        }
        semantic_text = self.build_semantic_text(row, mapped)
        provenance = Provenance(dataset=self.dataset_name, source_file=source_file, row_id=row_id)
        mapped["semantic_text"] = semantic_text
        mapped["provenance"] = provenance
        mapped["mapping_confidence"] = self.compute_mapping_confidence(row, mapped)
        mapped["missing_fields"] = self.missing_fields(
            mapped, ("ts", "src_ip", "dst_ip", "transport_proto", "label_raw")
        )
        return CanonicalEvent(**mapped)
