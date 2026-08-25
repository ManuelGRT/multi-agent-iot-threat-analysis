# src/adapters/bot_iot.py
from __future__ import annotations

from src.adapters.base import BaseAdapter
from src.contracts.canonical import CanonicalEvent, Provenance


class BotIoTAdapter(BaseAdapter):
    dataset_name = "Bot-IoT"

    def adapt(self, row: dict, source_file: str, row_id: int | str) -> CanonicalEvent:
        attack = self.first_present(row, "attack", "category", "subcategory")
        label = self.first_present(row, "label", "Label")
        mapped = {
            "event_id": f"bot_iot::{source_file}::{row_id}",
            "modality": "network_flow",
            "ts": self.to_iso_timestamp(
                self.first_present(row, "stime", "ltime", "ts", "timestamp")
            ),
            "src_ip": self.first_present(row, "saddr", "src_ip", "srcip"),
            "dst_ip": self.first_present(row, "daddr", "dst_ip", "dstip"),
            "src_port": self.to_int(self.first_present(row, "sport", "src_port")),
            "dst_port": self.to_int(self.first_present(row, "dport", "dst_port")),
            "transport_proto": self.first_present(row, "proto", "protocol"),
            "duration_ms": self._duration_ms(row),
            "packet_count": self.to_int(self.first_present(row, "pkts", "spkts", "packet_count")),
            "byte_count": self.to_int(self.first_present(row, "bytes", "sbytes", "byte_count")),
            "label_raw": attack or label,
        }
        semantic_text = self.build_semantic_text(row, mapped)
        provenance = Provenance(dataset=self.dataset_name, source_file=source_file, row_id=row_id)
        mapped["semantic_text"] = semantic_text
        mapped["provenance"] = provenance
        mapped["mapping_confidence"] = self.compute_mapping_confidence(row, mapped)
        mapped["missing_fields"] = self.missing_fields(mapped, ("src_ip", "dst_ip", "transport_proto", "label_raw"))
        return CanonicalEvent(**mapped)

    def _duration_ms(self, row: dict) -> float | None:
        milliseconds = self.to_float(self.first_present(row, "duration_ms"))
        if milliseconds is not None:
            return milliseconds
        seconds = self.to_float(self.first_present(row, "dur", "duration"))
        return seconds * 1000 if seconds is not None else None
