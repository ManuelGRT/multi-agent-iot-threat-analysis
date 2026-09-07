# src/contracts/agents.py
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

class IngestOutput(BaseModel):
    event_id: str
    normalized: bool
    mapping_confidence: float = Field(ge=0.0, le=1.0)
    schema_version: str
    modality: str
    notes: list[str] = Field(default_factory=list)
    abstain: bool = False
    requires_human_review: bool = False
    failure_code: str | None = None
    failure_reason: str | None = None

class DetectionOutput(BaseModel):
    event_id: str
    is_malicious: bool
    probability: float = Field(ge=0.0, le=1.0)
    evidence: list[str]
    model_name: str
    next_route: Literal["classify", "judge", "end"]
    abstain: bool = False

class ClassificationOutput(BaseModel):
    event_id: str
    attack_type: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    decision_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    model_name: str | None = None
    model_task: Literal["attack_type"] = "attack_type"
    taxonomy_version: str | None = None
    top_scores: dict[str, float] = Field(default_factory=dict)
    cross_dataset_neighbors: list[str] = Field(default_factory=list)
    reason: list[str] = Field(default_factory=list)
    next_route: Literal["explain", "judge", "end"]

class ExplanationOutput(BaseModel):
    event_id: str
    risk_summary: str
    mitigations: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    attack_type: str | None = None
    taxonomy_version: str | None = None
    catalog_scope: Literal["attack_type"] | None = None
    catalog_version: str | None = None
    catalog_taxonomy_version: str | None = None
    catalog_compatible_taxonomy_versions: list[str] = Field(default_factory=list)
    reference_quality: dict[str, str] = Field(default_factory=dict)
    requires_human_review: bool = False
    next_route: Literal["judge", "end"]
    model_name: str | None = None

class JudgeOutput(BaseModel):
    event_id: str
    approved: bool
    final_label: str | None = None
    final_confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    action: Literal["approve", "rework", "human_interrupt", "reject"]
