# src/orchestration/checkpoints.py
from __future__ import annotations

from typing import Any


def build_checkpointer(kind: str = "memory", **kwargs: Any) -> Any:
    if kind == "none":
        return None
    if kind != "memory":
        raise ValueError("Only the memory checkpointer is implemented in this smoke-test version")

    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:
        return None
    return MemorySaver(**kwargs)
