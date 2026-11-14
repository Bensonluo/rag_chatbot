"""Tests for Qdrant vector client"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestQdrantClient:
    """Test Qdrant client implementation"""

    async def test_replace_removes_old_chunks_and_retry_keeps_new_version(self):
        import uuid

        sdk = pytest.importorskip("qdrant_client")
        from qdrant_client import models

        from app.services.retrieval.qdrant_client import QdrantClient

        raw = sdk.AsyncQdrantClient(location=":memory:")
        await raw.create_collection(
            "test", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
        )
        client = QdrantClient(url="unused", collection_name="test", client=raw)
        old_ids, new_ids = [str(uuid.uuid4()), str(uuid.uuid4())], [str(uuid.uuid4())]

        def payloads(ids: list[str]) -> list[dict[str, str]]:
            return [
                {"document_id": "doc", "chunk_id": chunk_id, "content": chunk_id}
                for chunk_id in ids
            ]

        try:
            await client.add(old_ids, [[1.0, 0.0]] * 2, payloads(old_ids))
            await client.replace_document("doc", new_ids, [[0.0, 1.0]], payloads(new_ids))
            await client.replace_document("doc", new_ids, [[0.0, 1.0]], payloads(new_ids))
            assert [chunk["id"] for chunk in await client.list_chunks()] == new_ids
        finally:
            await raw.close()

    async def test_failed_replacement_upsert_never_deletes_previous_version(self):
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import VectorClientError

        raw = Mock(upsert=AsyncMock(side_effect=RuntimeError("offline")), delete=AsyncMock())
        client = QdrantClient(url="unused", collection_name="test", client=raw)
        with pytest.raises(VectorClientError, match="Failed to add points"):
            await client.replace_document(
                "doc", ["chunk"], [[1.0]], [{"document_id": "doc", "content": "new"}]
            )
        raw.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_it_for_first_use(self):
        """A fresh Compose Qdrant instance should work without manual setup."""
        from app.services.retrieval.qdrant_client import QdrantClient

        class FakeDistance:
            COSINE = "Cosine"

        class FakeVectorParams:
            def __init__(self, **values):
                self.__dict__.update(values)

        models = {
            "Distance": FakeDistance,
            "VectorParams": FakeVectorParams,
        }
        mock_client = Mock()
        mock_client.collection_exists = AsyncMock(return_value=False)
        mock_client.create_collection = AsyncMock(return_value=True)
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="documents",
            client=mock_client,
            embedding_service=Mock(dimensions=1024),
        )
        client._client_injected = False

        with patch.object(
            client,
            "_qdrant_model",
            side_effect=lambda name: models[name],
        ):
            await client._ensure_collection()

        mock_client.create_collection.assert_awaited_once()
        vector_config = mock_client.create_collection.await_args.kwargs["vectors_config"]
        assert vector_config.size == 1024
        assert vector_config.distance == "Cosine"

    @pytest.mark.asyncio
    async def test_ensure_collection_falls_back_when_exists_endpoint_missing(self):
        """Qdrant servers < 1.8 answer 404 for GET /collections/{name}/exists;
        ensure must fall back to the collection listing instead of crashing
        before create_collection ever runs."""
        from app.services.retrieval.qdrant_client import QdrantClient

        class FakeDistance:
            COSINE = "Cosine"

        class FakeVectorParams:
            def __init__(self, **values):
                self.__dict__.update(values)

        models = {
            "Distance": FakeDistance,
            "VectorParams": FakeVectorParams,
        }
        mock_client = Mock()
        mock_client.collection_exists = AsyncMock(
            side_effect=RuntimeError("Unexpected Response: 404 (Not Found)")
        )
        mock_client.get_collections = AsyncMock(return_value=Mock(collections=[]))
        mock_client.create_collection = AsyncMock(return_value=True)
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="documents",
            client=mock_client,
            embedding_service=Mock(dimensions=512),
        )
        client._client_injected = False

        with patch.object(
            client,
            "_qdrant_model",
            side_effect=lambda name: models[name],
        ):
            await client._ensure_collection()

        mock_client.get_collections.assert_awaited_once()
        mock_client.create_collection.assert_awaited_once()
        vector_config = mock_client.create_collection.await_args.kwargs["vectors_config"]
        assert vector_config.size == 512

    @pytest.mark.asyncio
    async def test_ensure_collection_fallback_skips_create_when_present(self):
        """When the fallback listing finds the collection, creation is skipped."""
        from app.services.retrieval.qdrant_client import QdrantClient

        mock_client = Mock()
        mock_client.collection_exists = AsyncMock(
            side_effect=RuntimeError("Unexpected Response: 404 (Not Found)")
        )
        # NOTE: Mock(name=...) is a constructor-reserved kwarg; assign the
        # attribute directly so `collection.name` reads the plain string.
        listed = Mock()
        listed.name = "documents"
        mock_client.get_collections = AsyncMock(return_value=Mock(collections=[listed]))
        mock_client.create_collection = AsyncMock(return_value=True)
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="documents",
            client=mock_client,
            embedding_service=Mock(dimensions=512),
        )
        client._client_injected = False

        await client._ensure_collection()

        mock_client.get_collections.assert_awaited_once()
        mock_client.create_collection.assert_not_awaited()

    def test_client_initialization(self):
        """Test Qdrant client initialization"""
        # Arrange & Act
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import VectorClient

        mock_qdrant = Mock()
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            api_key=None,
            client=mock_qdrant,
        )

        # Assert
        assert client.collection_name == "test_collection"
        assert isinstance(client, VectorClient)

    def test_client_initialization_with_api_key(self):
        """Test client initialization with API key"""
        # Arrange & Act
        from app.services.retrieval.qdrant_client import QdrantClient

        mock_qdrant = Mock()
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            api_key="test_api_key",
            client=mock_qdrant,
        )

        # Assert
        assert client.api_key == "test_api_key"

    @pytest.mark.asyncio
    async def test_add_documents(self):
        """Test adding documents to Qdrant"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import Document

        mock_client = Mock()
        mock_client.upsert = AsyncMock(return_value=Mock(status="completed"))

        client = QdrantClient(
            url="http://localhost:6333", collection_name="test_collection", client=mock_client
        )

        documents = [
            Document(
                id="doc1",
                content="Test content 1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"category": "test"},
            ),
            Document(id="doc2", content="Test content 2", embedding=[0.4, 0.5, 0.6]),
        ]

        # Act
        result = await client.add_documents(documents)

        # Assert
        assert len(result) == 2
        assert result == ["doc1", "doc2"]
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_search(self):
        """Test searching documents in Qdrant"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import VectorSearchRequest

        mock_client = Mock()
        mock_search_result = Mock(
            id="chunk1",
            payload={
                "document_id": "doc1",
                "chunk_id": "chunk1",
                "content": "Test content",
                "metadata": {"category": "test", "chunk_id": "chunk1"},
            },
            score=0.95,
        )
        mock_client.query_points = AsyncMock(
            return_value=SimpleNamespace(points=[mock_search_result])
        )

        client = QdrantClient(
            url="http://localhost:6333", collection_name="test_collection", client=mock_client
        )

        # Mock embedding generation
        with patch.object(client, "_generate_embedding", return_value=[0.1, 0.2, 0.3]):
            request = VectorSearchRequest(query="test query", top_k=5)

            # Act
            results = await client.search(request)

            # Assert
            assert len(results) == 1
            assert results[0].document_id == "doc1"
            assert results[0].score == 0.95
            assert results[0].content == "Test content"
            assert results[0].metadata == {
                "category": "test",
                "chunk_id": "chunk1",
            }

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Test searching with metadata filters"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import VectorSearchRequest

        mock_client = Mock()
        mock_client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

        client = QdrantClient(
            url="http://localhost:6333", collection_name="test_collection", client=mock_client
        )

        with patch.object(client, "_generate_embedding", return_value=[0.1, 0.2, 0.3]):
            request = VectorSearchRequest(
                query="test query", top_k=10, filters={"category": "tech"}
            )

            # Act
            await client.search(request)

            # Assert - Verify filters were passed
            mock_client.query_points.assert_called_once()
            call_args = mock_client.query_points.call_args
            assert call_args is not None

    @pytest.mark.asyncio
    async def test_delete_by_filter_keeps_root_payload_keys(self):
        """Document deletion filters target root payload fields and the
        qdrant-client 1.9 delete() signature (points_selector)."""
        from qdrant_client.models import Filter

        from app.services.retrieval.qdrant_client import QdrantClient

        mock_client = Mock()
        mock_client.count = AsyncMock(return_value=Mock(count=3))
        mock_client.delete = AsyncMock(return_value=Mock(status="completed"))

        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            client=mock_client,
        )

        real_filter = Filter(must=[])
        with patch.object(
            client,
            "_build_filter",
            return_value=real_filter,
        ) as build_filter:
            deleted = await client.delete_by_filter({"document_id": "doc-demo"})

        build_filter.assert_called_once_with({"document_id": "doc-demo"})
        mock_client.count.assert_awaited_once()
        mock_client.delete.assert_awaited_once()
        # qdrant-client 1.9: delete() takes points_selector, not query_filter —
        # the old kwarg name TypeError'd at runtime (mocks hid it).
        kwargs = mock_client.delete.await_args.kwargs
        assert "query_filter" not in kwargs
        assert kwargs["collection_name"] == "test_collection"
        assert kwargs["points_selector"].filter == real_filter
        assert deleted == 3

    @pytest.mark.asyncio
    async def test_delete_documents(self):
        """Test deleting documents from Qdrant"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient

        mock_client = Mock()
        mock_client.delete = AsyncMock(return_value=Mock(status="completed"))

        client = QdrantClient(
            url="http://localhost:6333", collection_name="test_collection", client=mock_client
        )

        # Act
        await client.delete(["doc1", "doc2"])

        # Assert
        mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_document(self):
        """Test updating a document in Qdrant"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import Document

        mock_client = Mock()
        mock_client.upsert = AsyncMock(return_value=Mock(status="completed"))

        client = QdrantClient(
            url="http://localhost:6333", collection_name="test_collection", client=mock_client
        )

        document = Document(
            id="doc1",
            content="Updated content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"updated": True},
        )

        # Act
        await client.update(document)

        # Assert
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document(self):
        """Test getting a document by ID"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient

        mock_client = Mock()
        mock_retrieve = Mock(
            id="doc1",
            payload={"content": "Test content", "metadata": {"category": "test"}},
            vector=[0.1, 0.2, 0.3],
        )
        mock_client.retrieve = AsyncMock(return_value=[mock_retrieve])

        client = QdrantClient(
            url="http://localhost:6333", collection_name="test_collection", client=mock_client
        )

        # Act
        document = await client.get_document("doc1")

        # Assert
        assert document is not None
        assert document.id == "doc1"
        assert document.content == "Test content"
        assert document.embedding == [0.1, 0.2, 0.3]
        assert document.metadata == {"category": "test"}

    @pytest.mark.asyncio
    async def test_get_document_not_found(self):
        """Test getting a non-existent document"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient

        mock_client = Mock()
        mock_client.retrieve = AsyncMock(return_value=[])

        client = QdrantClient(
            url="http://localhost:6333", collection_name="test_collection", client=mock_client
        )

        # Act
        document = await client.get_document("nonexistent")

        # Assert
        assert document is None

    @pytest.mark.asyncio
    async def test_add_documents_without_embeddings(self):
        """Test adding documents without embeddings (should generate them)"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import Document

        mock_client = Mock()
        mock_client.upsert = AsyncMock(return_value=Mock(status="completed"))

        client = QdrantClient(
            url="http://localhost:6333", collection_name="test_collection", client=mock_client
        )

        # Mock embedding generation
        with patch.object(client, "_generate_embedding", return_value=[0.1, 0.2, 0.3]):
            documents = [
                Document(id="doc1", content="Test content")  # No embedding
            ]

            # Act
            result = await client.add_documents(documents)

            # Assert
            assert len(result) == 1
            mock_client.upsert.assert_called_once()


class TestListChunks:
    """Scroll-based corpus listing — the BM25 warmup source."""

    @pytest.mark.asyncio
    async def test_scrolls_all_pages_and_maps_payloads(self):
        from app.services.retrieval.qdrant_client import QdrantClient

        class _Point:
            def __init__(self, point_id: str, payload: dict[str, Any]) -> None:
                self.id = point_id
                self.payload = payload

        page_one = (
            [
                _Point("c1", {"content": "iPhone 退款 政策", "metadata": {"product": "iPhone 13"}}),
                _Point("c2", {"content": "MacBook 退货 政策", "metadata": {"product": "MacBook"}}),
            ],
            "next-offset",
        )
        page_two = ([_Point("c3", {"content": "保修 条款"})], None)

        mock_client = Mock()
        mock_client.scroll = AsyncMock(side_effect=[page_one, page_two])
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="documents",
            client=mock_client,
            embedding_service=Mock(dimensions=1024),
        )

        chunks = await client.list_chunks()

        assert chunks == [
            {
                "id": "c1",
                "content": "iPhone 退款 政策",
                "metadata": {
                    "product": "iPhone 13",
                    "chunk_id": "c1",
                    "document_id": "c1",
                },
            },
            {
                "id": "c2",
                "content": "MacBook 退货 政策",
                "metadata": {
                    "product": "MacBook",
                    "chunk_id": "c2",
                    "document_id": "c2",
                },
            },
            {
                "id": "c3",
                "content": "保修 条款",
                "metadata": {
                    "chunk_id": "c3",
                    "document_id": "c3",
                },
            },
        ]
        assert mock_client.scroll.await_count == 2
        second_call = mock_client.scroll.await_args_list[1]
        assert second_call.kwargs.get("offset") == "next-offset"

    @pytest.mark.asyncio
    async def test_empty_collection_returns_empty_list(self):
        from app.services.retrieval.qdrant_client import QdrantClient

        mock_client = Mock()
        mock_client.scroll = AsyncMock(return_value=([], None))
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="documents",
            client=mock_client,
            embedding_service=Mock(dimensions=1024),
        )

        assert await client.list_chunks() == []

    @pytest.mark.asyncio
    async def test_rejects_truncated_snapshot_at_max_chunks(self):
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import VectorClientError

        class _Point:
            def __init__(self, point_id: str, payload: dict[str, Any]) -> None:
                self.id = point_id
                self.payload = payload

        pages = [([_Point(f"c{i}", {"content": f"doc {i}"})], f"off-{i}") for i in range(5)]

        mock_client = Mock()
        mock_client.scroll = AsyncMock(side_effect=pages)
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="documents",
            client=mock_client,
            embedding_service=Mock(dimensions=1024),
        )

        with pytest.raises(VectorClientError, match="incomplete index"):
            await client.list_chunks(max_chunks=3)


class TestEnsureCollectionDimensionMismatch:
    """Provider switches change embedding dimensions; a stale collection
    must not poison every search with opaque 400s (live finding
    2026-11-12: a 512-dim bge-small collection vs 1024-dim embedding-2
    queries — every vector leg failed until the collection was manually
    recreated). Empty collections self-heal; non-empty ones fail fast
    with an actionable message."""

    def _make_client(self, dims: int) -> tuple[Any, Any]:
        from app.services.retrieval.qdrant_client import QdrantClient

        class FakeDistance:
            COSINE = "Cosine"

        class FakeVectorParams:
            def __init__(self, **values):
                self.__dict__.update(values)

        models = {"Distance": FakeDistance, "VectorParams": FakeVectorParams}
        mock_client = Mock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_client.delete_collection = AsyncMock(return_value=True)
        mock_client.create_collection = AsyncMock(return_value=True)
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="documents",
            client=mock_client,
            embedding_service=Mock(dimensions=dims),
        )
        client._client_injected = False
        return client, models

    @staticmethod
    def _collection_info(size: int, points: int) -> Mock:
        info = Mock()
        info.config.params.vectors.size = size
        info.points_count = points
        return info

    @pytest.mark.asyncio
    async def test_mismatched_empty_collection_is_recreated_at_new_dims(self):
        client, models = self._make_client(dims=1024)
        client.client.get_collection = AsyncMock(
            return_value=self._collection_info(size=512, points=0)
        )
        with patch.object(client, "_qdrant_model", side_effect=lambda name: models[name]):
            await client._ensure_collection()

        client.client.delete_collection.assert_awaited_once_with("documents")
        client.client.create_collection.assert_awaited_once()
        vector_config = client.client.create_collection.await_args.kwargs["vectors_config"]
        assert vector_config.size == 1024

    @pytest.mark.asyncio
    async def test_mismatched_nonempty_collection_raises_actionable_error(self):
        from app.services.retrieval.vector_base import VectorClientError

        client, models = self._make_client(dims=1024)
        client.client.get_collection = AsyncMock(
            return_value=self._collection_info(size=512, points=7)
        )
        with (
            patch.object(client, "_qdrant_model", side_effect=lambda name: models[name]),
            pytest.raises(VectorClientError, match="512.*1024|1024.*512"),
        ):
            await client._ensure_collection()

        client.client.delete_collection.assert_not_awaited()
        client.client.create_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_matching_collection_is_left_alone(self):
        client, models = self._make_client(dims=1024)
        client.client.get_collection = AsyncMock(
            return_value=self._collection_info(size=1024, points=5)
        )
        with patch.object(client, "_qdrant_model", side_effect=lambda name: models[name]):
            await client._ensure_collection()

        client.client.delete_collection.assert_not_awaited()
        client.client.create_collection.assert_not_awaited()


class _FakeQdrant404(Exception):
    """Mimics qdrant-client's UnexpectedResponse for a missing collection."""

    status_code = 404


class TestVanishedCollectionReconciliation:
    """The process-lifetime ``_collection_ready`` latch must not pin a stale
    state after the collection vanishes mid-lifetime (Qdrant volume wipe /
    manual delete while the api keeps running). Search and upsert get one
    reconciliation: reset the latch, re-ensure, retry once — the historical
    behavior 404'd every call until an api restart (ops finding
    2026-11-12, tick27 forensics)."""

    def _make_client(self) -> tuple[Any, Any, Any]:
        from app.services.retrieval.qdrant_client import QdrantClient

        class FakeDistance:
            COSINE = "Cosine"

        class FakeVectorParams:
            def __init__(self, **values):
                self.__dict__.update(values)

        class FakePointStruct:
            def __init__(self, **values):
                self.__dict__.update(values)

        models = {
            "Distance": FakeDistance,
            "VectorParams": FakeVectorParams,
            "PointStruct": FakePointStruct,
        }
        mock_client = Mock()
        mock_client.collection_exists = AsyncMock(return_value=False)
        mock_client.create_collection = AsyncMock(return_value=True)
        embedding = Mock(dimensions=1024)
        embedding.embed_single = AsyncMock(return_value=[0.1] * 1024)
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="documents",
            client=mock_client,
            embedding_service=embedding,
        )
        client._client_injected = False
        client._collection_ready = True  # simulate a long-running process
        return client, mock_client, models

    @pytest.mark.asyncio
    async def test_search_reconciles_once_when_collection_vanishes(self):
        from app.services.retrieval.vector_base import VectorSearchRequest

        client, mock_client, models = self._make_client()
        mock_client.query_points = AsyncMock(
            side_effect=[
                _FakeQdrant404("Collection `documents` doesn't exist!"),
                SimpleNamespace(points=[]),
            ]
        )
        with patch.object(client, "_qdrant_model", side_effect=lambda name: models[name]):
            results = await client.search(VectorSearchRequest("refund policy", top_k=3))

        assert results == []
        assert mock_client.query_points.await_count == 2
        mock_client.create_collection.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_does_not_retry_on_other_errors(self):
        from app.services.retrieval.vector_base import VectorClientError, VectorSearchRequest

        client, mock_client, models = self._make_client()
        mock_client.query_points = AsyncMock(side_effect=RuntimeError("connection refused"))
        with (
            patch.object(client, "_qdrant_model", side_effect=lambda name: models[name]),
            pytest.raises(VectorClientError),
        ):
            await client.search(VectorSearchRequest("refund policy", top_k=3))

        assert mock_client.query_points.await_count == 1
        mock_client.create_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_add_documents_reconciles_once_when_collection_vanishes(self):
        from app.services.retrieval.vector_base import Document

        client, mock_client, models = self._make_client()
        mock_client.upsert = AsyncMock(
            side_effect=[_FakeQdrant404("Collection `documents` doesn't exist!"), None]
        )
        with patch.object(client, "_qdrant_model", side_effect=lambda name: models[name]):
            ids = await client.add_documents(
                [Document(id="doc-1", content="hello world", metadata={})]
            )

        assert ids == ["doc-1"]
        assert mock_client.upsert.await_count == 2
        mock_client.create_collection.assert_awaited_once()


class TestListChunksEnsuresCollection:
    """The BM25 warmup scroll (boot + periodic refresh) reads the whole
    collection. Unlike search/upsert it never ensured the collection
    first, so a fresh stack logged a misleading 404 "vector-only"
    warmup skip at boot instead of creating the collection (finding
    2026-11-13, tick30 forensics)."""

    @pytest.mark.asyncio
    async def test_list_chunks_creates_missing_collection(self):
        from app.services.retrieval.qdrant_client import QdrantClient

        class FakeDistance:
            COSINE = "Cosine"

        class FakeVectorParams:
            def __init__(self, **values):
                self.__dict__.update(values)

        models = {"Distance": FakeDistance, "VectorParams": FakeVectorParams}
        mock_client = Mock()
        mock_client.collection_exists = AsyncMock(return_value=False)
        mock_client.create_collection = AsyncMock(return_value=True)
        mock_client.scroll = AsyncMock(return_value=([], None))
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="documents",
            client=mock_client,
            embedding_service=Mock(dimensions=1024),
        )
        client._client_injected = False

        with patch.object(client, "_qdrant_model", side_effect=lambda name: models[name]):
            chunks = await client.list_chunks()

        assert chunks == []
        mock_client.create_collection.assert_awaited_once()
        mock_client.scroll.assert_awaited_once()


class TestSelfHealMetrics:
    """Self-heals are warn-logged; logs need someone reading them. A
    Prometheus counter with a ``reason`` label makes frequency alertable
    (sustained growth = something upstream churns the collection)."""

    @staticmethod
    def _count(reason: str) -> float:
        from prometheus_client import REGISTRY

        return (
            REGISTRY.get_sample_value("retrieval_collection_self_heals_total", {"reason": reason})
            or 0.0
        )

    @pytest.mark.asyncio
    async def test_dim_drift_recreate_increments_counter(self):
        before = self._count("dim_drift")
        before_vanished = self._count("vanished")
        client, models = TestEnsureCollectionDimensionMismatch()._make_client(dims=1024)
        client.client.get_collection = AsyncMock(
            return_value=TestEnsureCollectionDimensionMismatch()._collection_info(
                size=512, points=0
            )
        )
        with patch.object(client, "_qdrant_model", side_effect=lambda name: models[name]):
            await client._ensure_collection()

        assert self._count("dim_drift") == before + 1
        assert self._count("vanished") == before_vanished  # untouched

    @pytest.mark.asyncio
    async def test_vanished_reconcile_increments_counter(self):
        from app.services.retrieval.vector_base import VectorSearchRequest

        before = self._count("vanished")
        client, mock_client, models = TestVanishedCollectionReconciliation()._make_client()
        mock_client.query_points = AsyncMock(
            side_effect=[
                _FakeQdrant404("Collection `documents` doesn't exist!"),
                SimpleNamespace(points=[]),
            ]
        )
        with patch.object(client, "_qdrant_model", side_effect=lambda name: models[name]):
            await client.search(VectorSearchRequest("refund", top_k=3))

        assert self._count("vanished") == before + 1
