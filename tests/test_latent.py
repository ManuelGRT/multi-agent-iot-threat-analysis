import pytest

from src.adapters.iot23 import IoT23Adapter
from src.latent.service import LatentService


class FakeEmbedClient:
    model = "fake-embedder"

    async def embed(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]


@pytest.mark.asyncio
async def test_latent_service_builds_record_without_ollama():
    event = IoT23Adapter().adapt(
        {
            "id.orig_h": "10.0.0.2",
            "id.resp_h": "10.0.0.3",
            "proto": "tcp",
            "label": "Benign",
        },
        source_file="conn.log",
        row_id=1,
    )

    record = await LatentService(FakeEmbedClient()).build_record(event)

    assert record.event_id == event.event_id
    assert record.embedding_model == "fake-embedder"
    assert record.embedding_dim == 3
    assert record.dataset == "IoT-23"
