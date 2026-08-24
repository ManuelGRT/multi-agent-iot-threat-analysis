# src/latent/embed_api.py
from __future__ import annotations

import httpx
from typing import Sequence

class OllamaEmbedClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "embeddinggemma"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": list(texts)}
            )
            r.raise_for_status()
            data = r.json()
            return data["embeddings"]
