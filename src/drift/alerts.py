# src/drift/alerts.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


AlertLevel = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class DriftAlert:
    monitor: str
    level: AlertLevel
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def alert_from_mmd_result(result: dict[str, Any]) -> DriftAlert | None:
    data = result.get("data", {})
    if not data.get("is_drift"):
        return None
    distance = data.get("distance")
    threshold = data.get("threshold")
    return DriftAlert(
        monitor="mmd",
        level="warning",
        message=f"Embedding drift detected: distance={distance}, threshold={threshold}",
        payload=result,
    )
