# src/adapters/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import re
from typing import Any

from src.eval.predictive_sanitization import is_predictive_target_field
from src.contracts.canonical import CanonicalEvent

class BaseAdapter(ABC):
    dataset_name: str

    @abstractmethod
    def adapt(self, row: dict[str, Any], source_file: str, row_id: int | str) -> CanonicalEvent:
        ...

    def build_semantic_text(self, row: dict[str, Any], mapped: dict[str, Any]) -> str:
        parts = [
            f"dataset={self.dataset_name}",
            f"transport_proto={mapped.get('transport_proto')}",
            f"src_ip={mapped.get('src_ip')}",
            f"dst_ip={mapped.get('dst_ip')}",
            f"src_port={mapped.get('src_port')}",
            f"dst_port={mapped.get('dst_port')}",
            f"duration_ms={mapped.get('duration_ms')}",
            f"packet_count={mapped.get('packet_count')}",
            f"byte_count={mapped.get('byte_count')}",
        ]
        for k, v in row.items():
            if k not in mapped and v is not None and not is_predictive_target_field(k):
                parts.append(f"raw::{k}={v}")
        return " | ".join(map(str, parts))

    def compute_mapping_confidence(self, row: dict[str, Any], mapped: dict[str, Any]) -> float:
        required = ["modality", "semantic_text", "provenance"]
        core = ["src_ip", "dst_ip", "transport_proto", "label_raw"]
        present_required = sum(1 for k in required if mapped.get(k) is not None)
        present_core = sum(1 for k in core if mapped.get(k) not in (None, "", "unknown"))
        structural_score = present_core / len(core)
        return round(0.4 * (present_required / len(required)) + 0.6 * structural_score, 3)

    def missing_fields(self, mapped: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
        return [field for field in fields if mapped.get(field) in (None, "", "unknown")]

    @staticmethod
    def first_present(row: dict[str, Any], *names: str) -> Any:
        for name in names:
            value = row.get(name)
            if value not in (None, ""):
                return value
        return None

    @staticmethod
    def to_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def to_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def to_iso_timestamp(value: Any) -> str | None:
        """Normaliza timestamps conocidos y descarta valores no verificables.

        Edge-IIoTset contiene valores ``AAAA HH:MM:SS.fraccion`` sin mes ni
        dia. Se conserva la politica historica del parser LLM (1 de enero) y
        se limita la fraccion a los seis digitos admitidos por ``datetime``.
        Un adapter de fallback nunca debe romper toda la campana por un campo
        temporal incompleto: la ausencia se representa con ``None``.
        """
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (int, float)):
            if value <= 1_000_000_000:
                return None
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
            except (OverflowError, OSError, ValueError):
                return None

        text = re.sub(r"\s+", " ", str(value)).strip()
        if text.lower() in {"", "-", "nan", "none", "null"}:
            return None
        try:
            numeric = float(text)
        except ValueError:
            numeric = None
        if numeric is not None:
            if numeric <= 1_000_000_000:
                return None
            try:
                return datetime.fromtimestamp(numeric, tz=timezone.utc).isoformat()
            except (OverflowError, OSError, ValueError):
                return None

        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
        except ValueError:
            pass

        for fmt in (
            "%d-%b-%y %H:%M:%S",
            "%d-%b-%Y %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%y-%m-%dT%H:%M:%SZ",
            "%y-%m-%dT%H:%M:%S%z",
            "%y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(text, fmt).isoformat()
            except ValueError:
                continue

        partial = re.fullmatch(
            r"(?P<year>\d{4})[\s-]+(?P<time>\d{1,2}:\d{2}:\d{2})(?:\.(?P<fraction>\d+))?",
            text,
        )
        if partial is None:
            return None
        fraction = partial.group("fraction")
        suffix = f".{fraction[:6]}" if fraction else ""
        try:
            parsed = datetime.fromisoformat(
                f"{partial.group('year')}-01-01T{partial.group('time')}{suffix}"
            )
        except ValueError:
            return None
        return parsed.isoformat()
