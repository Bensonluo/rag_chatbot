"""Vector and keyword adapters must not erase contradictory stored identities."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.retrieval.qdrant_client import QdrantClient
from app.services.retrieval.vector_base import VectorSearchRequest


@pytest.mark.parametrize(
    "payload",
    [
        {"document_id": "policy", "metadata": {"document_id": "other", "version_id": "v1"}},
        {"document_id": "policy", "metadata": {"chunk_id": "other", "version_id": "v1"}},
        {"document_id": True, "metadata": {"version_id": "v1"}},
        {"document_id": "policy", "metadata": ["malformed"]},
        {"document_id": "policy", "version_id": "v2", "metadata": {"version_id": "v1"}},
    ],
)
async def test_conflicting_raw_identity_stays_unusable_through_vector_and_bm25_conversion(
    payload, monkeypatch
):
    point = SimpleNamespace(id="chunk", score=1, payload={"content": "policy text", **payload})
    sdk = Mock(
        search=AsyncMock(return_value=[point]), scroll=AsyncMock(return_value=([point], None))
    )
    client = QdrantClient(url="unused", collection_name="policies", client=sdk)
    monkeypatch.setattr(client, "_ensure_collection", AsyncMock())
    monkeypatch.setattr(client, "_generate_embedding", AsyncMock(return_value=[1.0, 0.0]))
    result = (await client.search(VectorSearchRequest(query="policy", top_k=1)))[0]
    chunk = (await client.list_chunks())[0]
    assert result.metadata is not None
    assert result.metadata["provenance_incomplete"] is True
    assert chunk["metadata"]["provenance_incomplete"] is True
