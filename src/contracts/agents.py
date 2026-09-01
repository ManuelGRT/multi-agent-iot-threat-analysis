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
    attack_family: str | None = None
    attack_subtype: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    cross_dataset_neighbors: list[str] = Field(default_factory=list)
    reason: list[str] = Field(default_factory=list)
    next_route: Literal["explain", "judge", "end"]

class ExplanationOutput(BaseModel):
    event_id: str
    risk_summary: str
    mitigations: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
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
