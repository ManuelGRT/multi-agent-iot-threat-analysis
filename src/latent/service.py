# src/latent/service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from src.contracts.canonical import CanonicalEvent
from src.latent.embed_api import OllamaEmbedClient

@dataclass
class LatentRecord:
    event_id: str
    embedding_model: str
    embedding_dim: int
    embedding: list[float]
    semantic_text: str
    mapping_confidence: float
    dataset: str
    modality: str
    provenance: dict[str, Any]
    features_present: list[str]
    latent_version: str = "0.1.0"

class LatentService:
    def __init__(self, embed_client: OllamaEmbedClient):
        self.embed_client = embed_client

    async def build_record(self, event: CanonicalEvent) -> LatentRecord:
        [vec] = await self.embed_client.embed([event.semantic_text])
        features_present = [
            name for name, value in event.model_dump().items()
            if value not in (None, "", [], {}, "unknown")
        ]
        return LatentRecord(
            event_id=event.event_id,
            embedding_model=self.embed_client.model,
            embedding_dim=len(vec),
            embedding=vec,
            semantic_text=event.semantic_text,
            mapping_confidence=event.mapping_confidence,
            dataset=event.provenance.dataset,
            modality=event.modality,
            provenance=event.provenance.model_dump(),
            features_present=features_present,
        )
