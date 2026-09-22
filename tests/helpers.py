"""Construcciones compartidas por la suite minima de siete pruebas."""
from __future__ import annotations

import copy
from typing import Any

from src.contracts.attack_taxonomy import MULTIDATASET_TAXONOMY_VERSION
from src.mcp.client import MCPToolClient
from src.mcp.standardization_cache import bind_event_identity
from src.mcp.threat_catalog import (
    THREAT_INTEL_CATALOG_VERSION,
    expected_catalog_contract,
)


CANONICAL_EVENT: dict[str, Any] = {
    "event_id": "evt-minimal-suite",
    "modality": "network_flow",
    "src_ip": "192.168.100.5",
    "dst_ip": "192.168.100.3",
    "src_port": 45312,
    "dst_port": 80,
    "transport_proto": "tcp",
    "packet_count": 120,
    "byte_count": 4096,
    "duration_ms": 1500.0,
    "schema_profile": "network_flow",
    "semantic_text": "tcp flow with high packet rate",
    "provenance": {"dataset": "edge_iiotset", "row_id": 1},
    "mapping_confidence": 0.95,
}


class StubClient:
    """Cliente MCP con respuestas sustituibles y registro de llamadas."""

    def __init__(self, overrides: dict[tuple[str, str], Any] | None = None):
        self.overrides = overrides or {}
        self.real = MCPToolClient(mode="inprocess")
        self.calls: list[tuple[str, str]] = []
        self.call_arguments: list[tuple[str, str, dict[str, Any]]] = []

    def call(self, server: str, tool: str, **arguments: Any) -> dict[str, Any]:
        self.calls.append((server, tool))
        self.call_arguments.append((server, tool, copy.deepcopy(arguments)))
        key = (server, tool)
        if key in self.overrides:
            value = self.overrides[key]
            result = value(**arguments) if callable(value) else copy.deepcopy(value)
            return result
        if key == ("case_memory", "lookup_standardization_cache"):
            return {"ok": True, "hit": False, "cache_available": True}
        if key == ("case_memory", "store_standardization_cache"):
            return {"ok": True, "stored": True, "status": "inserted"}
        if key == ("case_memory", "evict_standardization_cache"):
            return {"ok": True, "deleted": True}
        return self.real.call(server, tool, **arguments)


def standardization_success(
    *,
    dataset: str = "iot23",
    source_file: str = "events.csv",
    row_id: int = 1,
    split: str = "test",
    selected_columns: list[str] | None = None,
    mapping_confidence: float = 0.95,
) -> dict[str, Any]:
    """Respuesta Mistral valida para una fila cruda del caso indicado."""

    canonical = bind_event_identity(
        {**CANONICAL_EVENT, "mapping_confidence": mapping_confidence},
        dataset=dataset,
        source_file=source_file,
        row_id=row_id,
        split=split,
        parser_version="llm-0.2.0",
    )
    return {
        "ok": True,
        "canonical_event": canonical,
        "from_cache": False,
        "source": "llm",
        "provider": "mistral",
        "model": "mistral-small-2603",
        "mapping_confidence": mapping_confidence,
        "selected_columns": list(selected_columns or ["proto"]),
        "notes": ["parsed_by_llm", "source:mistral_live"],
        "cache_content_hash": None,
        "cache_pipeline_hash": None,
    }


def detection_success(probability: float) -> dict[str, Any]:
    return {
        "ok": True,
        "is_malicious": probability >= 0.5,
        "probability": probability,
        "model_name": "xgboost_detection_final",
    }


def classification_success(
    attack_type: str = "DDoS_TCP",
    confidence: float = 0.95,
) -> dict[str, Any]:
    alternatives = [
        candidate
        for candidate in ("DDoS_UDP", "XSS", "DoS", "Command_and_Control")
        if candidate != attack_type
    ][:2]
    remainder = max(0.0, 1.0 - confidence)
    return {
        "ok": True,
        "attack_type": attack_type,
        "confidence": confidence,
        "decision_threshold": 0.65,
        "top_scores": {
            attack_type: confidence,
            alternatives[0]: remainder * 0.6,
            alternatives[1]: remainder * 0.4,
        },
        "model_task": "attack_type",
        "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        "model_name": "xgboost_classification_final",
    }


def malicious_state(
    attack_type: str = "DDoS_TCP",
    confidence: float = 0.95,
) -> dict[str, Any]:
    classification = classification_success(attack_type, confidence)
    classification.pop("ok")
    return {
        "canonical_event": copy.deepcopy(CANONICAL_EVENT),
        "ingest_output": {"mapping_confidence": 0.95},
        "detection_output": {
            "is_malicious": True,
            "probability": 0.97,
            "abstain": False,
        },
        "classification_output": classification,
        "trace": [],
    }


def catalog_explanation(attack_type: str = "DDoS_TCP") -> dict[str, Any]:
    expected = expected_catalog_contract(attack_type)
    mitigation_items = [
        {
            "base_id": base_id,
            "text": item["text"],
            "phase": item["phase"],
            "source": "catalog",
            "base": None,
            "context_trusted": False,
        }
        for base_id, item in enumerate(expected["items"], start=1)
    ]
    return {
        "risk_summary": f"Mitigacion catalogada para {attack_type}",
        "attack_type": attack_type,
        "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        "catalog_scope": "attack_type",
        "catalog_version": THREAT_INTEL_CATALOG_VERSION,
        "catalog_taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        "catalog_compatible_taxonomy_versions": list(
            expected["compatible_taxonomy_versions"]
        ),
        "reference_quality": expected["reference_quality"],
        "mitigation_items": mitigation_items,
        "mitigations": [item["text"] for item in mitigation_items],
        "references": expected["references"],
        "source": "catalog",
        "has_llm_suggested": False,
        "first_five_catalog_anchored": False,
        "requires_human_review": False,
        "review_reasons": [],
        "confidence": 0.95,
    }


def complete_malicious_state(
    attack_type: str = "DDoS_TCP",
    confidence: float = 0.95,
) -> dict[str, Any]:
    state = malicious_state(attack_type, confidence)
    state["explanation_output"] = catalog_explanation(attack_type)
    return state
