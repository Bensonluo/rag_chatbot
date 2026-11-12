"""
Qdrant vector database client implementation.

Provides async client for Qdrant vector database operations.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

from app.services.retrieval.metrics import RETRIEVAL_COLLECTION_SELF_HEALS
from app.services.retrieval.vector_base import (
    Document,
    SearchResult,
    VectorClient,
    VectorClientError,
    VectorSearchRequest,
)

if TYPE_CHECKING:
    from app.services.embeddings.base import EmbeddingServiceBase

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _retrieval_metadata(payload: dict[str, Any], point_id: object) -> dict[str, Any]:
    """Preserve unusable-identity evidence before adapting storage payloads."""
    raw = payload.get("metadata")
    metadata = dict(raw) if isinstance(raw, dict) else {}
    invalid = raw is not None and not isinstance(raw, dict)
    document_id = payload.get("document_id")
    if document_id is not None:
        invalid |= not isinstance(document_id, str) or not document_id.strip()
        invalid |= (
            metadata.get("document_id") is not None and metadata["document_id"] != document_id
        )
    chunk_id = str(point_id)
    invalid |= metadata.get("chunk_id") is not None and metadata["chunk_id"] != chunk_id
    invalid |= payload.get("chunk_id") is not None and payload["chunk_id"] != chunk_id
    if payload.get("version_id") is not None:
        invalid |= metadata.get("version_id") != payload["version_id"]
    if invalid:
        metadata["provenance_incomplete"] = True
    metadata["chunk_id"] = chunk_id
    return metadata


class _InjectedClientModel:
    """Minimal SDK model used only with an injected test/demo client."""

    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)


class QdrantClient(VectorClient):
    """
    Qdrant vector database client.

    Provides async operations for document storage and retrieval
    using Qdrant vector database.
    """

    def __init__(
        self,
        url: str,
        collection_name: str,
        api_key: str | None = None,
        client: object | None = None,
        embedding_service: EmbeddingServiceBase | None = None,
    ) -> None:
        """
        Initialize Qdrant client.

        Args:
            url: Qdrant server URL
            collection_name: Collection name
            api_key: Optional API key for authentication
            client: Optional pre-configured Qdrant client (for testing)
            embedding_service: Optional embedding service for generating embeddings
        """
        self.url = url
        self.collection_name = collection_name
        self.api_key = api_key
        self.embedding_service = embedding_service
        # Injected test doubles and the SDK client expose different
        # surfaces; dynamic dispatch is the contract here.
        self.client: Any
        self._client_injected = client is not None
        self._collection_ready = False
        self._collection_lock = asyncio.Lock()

        # Use provided client or create new one
        if client:
            self.client = client
        else:
            try:
                from qdrant_client import AsyncQdrantClient

                self.client = AsyncQdrantClient(
                    url=url,
                    api_key=api_key,
                )
            except ImportError as e:
                raise ImportError(
                    "qdrant-client is not installed. Install it with: pip install qdrant-client"
                ) from e

    async def add_documents(
        self,
        documents: list[Document],
    ) -> list[str]:
        """
        Add documents to Qdrant collection.

        Args:
            documents: List of documents to add

        Returns:
            List[str]: List of document IDs

        Raises:
            VectorClientError: If operation fails
        """
        try:
            return await self._add_documents_once(documents)
        except VectorClientError as e:
            if not self._is_collection_missing(e):
                raise
            return await self._reconcile_and_retry(
                operation=lambda: self._add_documents_once(documents),
                failure_label="Failed to add documents",
                details={"document_count": len(documents)},
            )

    async def _add_documents_once(
        self,
        documents: list[Document],
    ) -> list[str]:
        try:
            await self._ensure_collection()
            PointStruct = self._qdrant_model("PointStruct")

            points = []
            document_ids = []

            for doc in documents:
                # Generate embedding if not provided
                embedding = doc.embedding
                if embedding is None:
                    embedding = await self._generate_embedding(doc.content)

                # Create Qdrant point
                point = PointStruct(
                    id=doc.id,
                    vector=embedding,
                    payload={
                        "document_id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata or {},
                    },
                )
                points.append(point)
                document_ids.append(doc.id)

            # Batch upsert
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            return document_ids

        except Exception as e:
            raise VectorClientError(
                f"Failed to add documents: {str(e)}", details={"document_count": len(documents)}
            ) from e

    async def search(
        self,
        request: VectorSearchRequest,
    ) -> list[SearchResult]:
        """
        Search for similar documents in Qdrant.

        Args:
            request: Search request with query and parameters

        Returns:
            List[SearchResult]: List of search results sorted by score

        Raises:
            VectorClientError: If operation fails
        """
        try:
            return await self._search_once(request)
        except VectorClientError as e:
            if not self._is_collection_missing(e):
                raise
            # The collection vanished after the process-lifetime latch
            # closed (Qdrant volume wipe / manual delete while the api
            # process keeps running): reconcile once, then retry.
            return await self._reconcile_and_retry(
                operation=lambda: self._search_once(request),
                failure_label="Search failed",
                details={"query": request.query},
            )

    async def _search_once(
        self,
        request: VectorSearchRequest,
    ) -> list[SearchResult]:
        try:
            await self._ensure_collection()
            # Generate embedding for query
            query_embedding = await self._generate_embedding(request.query)

            # Build search filters if provided
            search_filter = None
            if request.filters:
                search_filter = self._build_filter(
                    request.filters,
                    metadata_prefix=True,
                )

            # Search in Qdrant
            response = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=request.top_k,
                query_filter=search_filter,
            )

            # Convert to SearchResult objects
            results = []
            for hit in response:
                payload = hit.payload or {}
                result = SearchResult(
                    document_id=str(payload.get("document_id", hit.id)),
                    content=payload.get("content", ""),
                    score=float(hit.score),
                    metadata=_retrieval_metadata(payload, hit.id),
                )
                results.append(result)

            return results

        except Exception as e:
            raise VectorClientError(
                f"Search failed: {str(e)}", details={"query": request.query}
            ) from e

    async def delete(
        self,
        document_ids: list[str],
    ) -> None:
        """
        Delete documents from Qdrant.

        Args:
            document_ids: List of document IDs to delete

        Raises:
            VectorClientError: If operation fails
        """
        try:
            PointIdsList = self._qdrant_model("PointIdsList")

            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(
                    points=document_ids,
                ),
            )

        except Exception as e:
            raise VectorClientError(
                f"Failed to delete documents: {str(e)}", details={"document_ids": document_ids}
            ) from e

    async def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """
        Add points with pre-computed vectors to Qdrant.

        This is useful when you want to batch generate embeddings
        before inserting, rather than generating them one by one.

        Args:
            ids: List of point IDs
            vectors: List of embedding vectors
            payloads: List of payload dictionaries

        Raises:
            VectorClientError: If operation fails
        """
        try:
            await self._ensure_collection()
            PointStruct = self._qdrant_model("PointStruct")

            if not (len(ids) == len(vectors) == len(payloads)):
                raise VectorClientError(
                    "ids, vectors, and payloads must have the same length",
                    details={
                        "len_ids": len(ids),
                        "len_vectors": len(vectors),
                        "len_payloads": len(payloads),
                    },
                )

            points = []
            # Parallel arrays: a length mismatch would silently
            # truncate the upsert — make it loud instead.
            for point_id, vector, payload in zip(ids, vectors, payloads, strict=True):
                point = PointStruct(id=point_id, vector=vector, payload=payload)
                points.append(point)

            # Batch upsert
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        except Exception as e:
            raise VectorClientError(
                f"Failed to add points: {str(e)}", details={"point_count": len(ids)}
            ) from e

    async def replace_document(
        self,
        document_id: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Publish new chunks before removing previous chunks of a document.

        This is not a cross-request transaction: callers must serialize
        concurrent replacements of the same document. An embedding/upsert
        failure leaves the old version intact. Retrying with the same chunk
        IDs is safe, including after a cleanup failure.
        """
        if not ids or any(payload.get("document_id") != document_id for payload in payloads):
            raise VectorClientError("Replacement requires chunks belonging to the document")
        await self.add(ids=ids, vectors=vectors, payloads=payloads)
        Filter = self._qdrant_model("Filter")
        HasIdCondition = self._qdrant_model("HasIdCondition")
        FilterSelector = self._qdrant_model("FilterSelector")
        document_filter = cast("Any", self._build_filter({"document_id": document_id}))
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=document_filter.must,
                        must_not=[HasIdCondition(has_id=ids)],
                    )
                ),
                wait=True,
            )
        except Exception as e:
            raise VectorClientError(
                "New document chunks were stored but old chunk cleanup failed",
                details={"document_id": document_id},
            ) from e

    async def delete_by_filter(self, filter: dict[str, Any]) -> int:
        """
        Delete points matching a filter.

        Args:
            filter: Filter dictionary following Qdrant filter syntax

        Returns:
            int: Number of deleted points

        Raises:
            VectorClientError: If operation fails
        """
        try:
            # Build Qdrant filter
            qdrant_filter = self._build_filter(filter)

            # Count first because Qdrant's delete response only reports the
            # operation status, not how many points matched.
            count_result = await self.client.count(
                collection_name=self.collection_name,
                count_filter=qdrant_filter,
                exact=True,
            )

            # Delete with filter. qdrant-client 1.x's delete() takes the
            # filter wrapped in points_selector (FilterSelector) — passing
            # a query_filter kwarg raises TypeError at runtime.
            FilterSelector = self._qdrant_model("FilterSelector")
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=qdrant_filter),
                wait=True,
            )

            return int(count_result.count)

        except Exception as e:
            raise VectorClientError(
                f"Failed to delete by filter: {str(e)}", details={"filter": filter}
            ) from e

    async def update(
        self,
        document: Document,
    ) -> None:
        """
        Update a document in Qdrant.

        Args:
            document: Document with updated content

        Raises:
            VectorClientError: If operation fails
        """
        try:
            await self._ensure_collection()
            PointStruct = self._qdrant_model("PointStruct")

            # Generate embedding if not provided
            embedding = document.embedding
            if embedding is None:
                embedding = await self._generate_embedding(document.content)

            point = PointStruct(
                id=document.id,
                vector=embedding,
                payload={
                    "document_id": document.id,
                    "content": document.content,
                    "metadata": document.metadata or {},
                },
            )

            await self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

        except Exception as e:
            raise VectorClientError(
                f"Failed to update document: {str(e)}", details={"document_id": document.id}
            ) from e

    async def get_document(
        self,
        document_id: str,
    ) -> Document | None:
        """
        Get a document by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document | None: Document if found, None otherwise
        """
        try:
            response = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[document_id],
            )

            if not response:
                return None

            # Get first (and only) result
            hit = response[0]

            return Document(
                id=str(hit.id),
                content=hit.payload.get("content", ""),
                embedding=hit.vector,
                metadata=hit.payload.get("metadata"),
            )

        except Exception as e:
            raise VectorClientError(
                f"Failed to get document: {str(e)}", details={"document_id": document_id}
            ) from e

    async def list_chunks(self, max_chunks: int | None = None) -> list[dict[str, Any]]:
        """
        Scroll the whole collection and return raw chunk payloads.

        The BM25 warmup feed: pulls {id, content, metadata} for every
        point so the keyword leg can index the live corpus. Errors
        propagate to the caller, which decides how to degrade.

        Args:
            max_chunks: Optional memory guard; exceeding it raises instead
                of silently publishing an incomplete keyword index.

        Returns:
            list[dict[str, Any]]: Chunk dicts with id/content/metadata
        """
        chunks: list[dict[str, Any]] = []
        offset: Any = None
        if max_chunks is not None and max_chunks < 1:
            raise ValueError("max_chunks must be positive")
        # The scroll path needs the collection to exist just like search
        # and upsert: without this, a fresh stack's boot warmup logged a
        # misleading 404 "vector-only" skip instead of creating it.
        await self._ensure_collection()
        while True:
            points, offset = await self.client.scroll(
                collection_name=self.collection_name,
                limit=256 if max_chunks is None else min(256, max_chunks - len(chunks)),
                offset=offset,
                with_payload=True,
            )
            for point in points:
                payload = point.payload or {}
                chunks.append(
                    {
                        "id": str(point.id),
                        "content": payload.get("content", ""),
                        "metadata": {
                            **_retrieval_metadata(payload, point.id),
                            "document_id": str(payload.get("document_id", point.id)),
                        },
                    }
                )
            if offset is None:
                break
            if max_chunks is not None and len(chunks) >= max_chunks:
                raise VectorClientError(
                    "Keyword snapshot exceeds max_chunks; refusing an incomplete index"
                )
        return chunks

    async def _generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text using the configured embedding service.

        Args:
            text: Text to embed

        Returns:
            List[float]: Vector embedding

        Raises:
            VectorClientError: If no embedding service configured or generation fails
        """
        if self.embedding_service is None:
            raise VectorClientError(
                "No embedding service configured. "
                "Please provide an embedding_service when initializing QdrantClient."
            )

        try:
            # Use the embedding service

            # Generate embedding for single text
            result = await self.embedding_service.embed_single(text)

            return result if isinstance(result, list) else result.tolist()

        except Exception as e:
            raise VectorClientError(
                f"Failed to generate embedding: {str(e)}", details={"text_length": len(text)}
            ) from e

    def _build_filter(
        self,
        filters: dict[str, Any],
        *,
        metadata_prefix: bool = False,
    ) -> object:
        """
        Build Qdrant filter from dict.

        Args:
            filters: Filter dictionary

        Returns:
            Qdrant Filter object
        """
        FieldCondition = self._qdrant_model("FieldCondition")
        MatchValue = self._qdrant_model("MatchValue")

        conditions = []
        for key, value in filters.items():
            payload_key = f"metadata.{key}" if metadata_prefix else key
            condition = FieldCondition(
                key=payload_key,
                match=MatchValue(value=value),
            )
            conditions.append(condition)

        Filter = self._qdrant_model("Filter")
        return Filter(must=conditions)

    def _qdrant_model(self, name: str) -> Any:
        """Load a Qdrant SDK model, with a lightweight injected-client fallback."""
        try:
            from qdrant_client import models

            return getattr(models, name)
        except (ImportError, AttributeError) as e:
            if self._client_injected:
                return _InjectedClientModel
            raise ImportError("qdrant-client is required for vector database operations") from e

    async def _ensure_collection(self) -> None:
        """Create the configured collection lazily on a fresh demo stack."""
        if self._client_injected or self._collection_ready:
            return

        async with self._collection_lock:
            if self._collection_ready:
                return

            async def _list_contains_collection() -> bool:
                collections = await self.client.get_collections()
                return any(
                    collection.name == self.collection_name
                    for collection in collections.collections
                )

            collection_exists = getattr(self.client, "collection_exists", None)
            if collection_exists is None:
                exists = await _list_contains_collection()
            else:
                try:
                    exists = await collection_exists(self.collection_name)
                except Exception:
                    # Qdrant servers older than 1.8 answer 404 for the
                    # /collections/{name}/exists endpoint. The collection
                    # listing is authoritative on every server version.
                    exists = await _list_contains_collection()

            if exists:
                await self._verify_or_repair_collection_dims()
            else:
                await self._create_collection_at_provider_dims()

            self._collection_ready = True

    async def _create_collection_at_provider_dims(self) -> None:
        """Create the collection sized to the configured embedding provider."""
        Distance = self._qdrant_model("Distance")
        VectorParams = self._qdrant_model("VectorParams")
        vector_size = int(getattr(self.embedding_service, "dimensions", 1024))
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    @staticmethod
    def _is_collection_missing(error: BaseException) -> bool:
        """True when the exception chain indicates the collection is gone.

        Qdrant answers 404 with ``Collection `name` doesn't exist`` for
        search and upsert against a deleted collection. The client wraps
        raw SDK errors into VectorClientError, so both the status code
        (when present) and the message are checked along the cause chain.
        """
        cause: BaseException | None = error
        while cause is not None:
            if getattr(cause, "status_code", None) == 404:
                return True
            text = str(cause)
            if "doesn't exist" in text or "Not found" in text:
                return True
            cause = cause.__cause__ or cause.__context__
        return False

    async def _reconcile_and_retry(
        self,
        operation: Callable[[], Awaitable[_T]],
        failure_label: str,
        details: dict[str, Any],
    ) -> _T:
        """Re-ensure a vanished collection and retry the operation once.

        The process-lifetime ``_collection_ready`` latch exists to keep the
        hot path free of existence checks; its flip side is that a
        collection deleted mid-lifetime (volume wipe, operator error) would
        404 every call until an api restart. One reconciliation pass keeps
        that trade-off honest: reset the latch, re-run ensure (recreating
        an empty collection if needed), retry the operation a single time.
        A second failure propagates untouched.
        """
        logger.warning(
            "Collection '%s' missing after ensure (latch stale); reconciling and retrying once",
            self.collection_name,
        )
        RETRIEVAL_COLLECTION_SELF_HEALS.labels(reason="vanished").inc()
        self._collection_ready = False
        try:
            await self._ensure_collection()
            return await operation()
        except VectorClientError:
            raise
        except Exception as inner:  # pragma: no cover - defensive wrap parity
            raise VectorClientError(
                f"{failure_label} after collection reconciliation: {inner}",
                details=details,
            ) from inner

    async def _verify_or_repair_collection_dims(self) -> None:
        """Guard against embedding-dimension drift on an existing collection.

        Switching providers changes the query-vector dimension (live finding
        2026-11-12: a 512-dim bge-small-era collection vs 1024-dim embedding-2
        queries). Qdrant answers every search with an opaque 400 that only
        surfaces in tracebacks, so drift is checked here at ensure time:
        an empty collection is recreated at the provider's dims (zero data
        loss — nothing was indexed); a non-empty one fails fast with the
        collection name and both dimensions, telling the operator exactly
        what to do instead of poisoning every search.

        Named/multi-vector configs carry no scalar ``size``; those are left
        untouched for the search path to surface any genuine incompatibility.
        """
        try:
            info = await self.client.get_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - schema unreadable → search will error concretely
            logger.warning(
                "Could not read schema of collection '%s'; skipping dimension check",
                self.collection_name,
            )
            return

        vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
        existing_size = getattr(vectors, "size", None)
        if not isinstance(existing_size, int):
            return
        expected_size = int(getattr(self.embedding_service, "dimensions", 1024))
        if existing_size == expected_size:
            return

        points = getattr(info, "points_count", 0) or 0
        if points:
            raise VectorClientError(
                f"Collection '{self.collection_name}' stores {existing_size}-dim vectors "
                f"but the embedding provider produces {expected_size}-dim vectors. "
                f"Re-index required: either switch EMBEDDING_PROVIDER back to the "
                f"{existing_size}-dim model or delete the collection and re-ingest "
                f"to rebuild at {expected_size} dims."
            )

        # Empty collection: recreate at the provider's dims. The check-then-delete
        # window could race a concurrent first ingest at demo scale; the ingest
        # upsert would then fail loudly rather than corrupt silently.
        logger.warning(
            "Collection '%s' is empty with %d-dim vectors; recreating at %d dims "
            "to match the configured embedding provider",
            self.collection_name,
            existing_size,
            expected_size,
        )
        RETRIEVAL_COLLECTION_SELF_HEALS.labels(reason="dim_drift").inc()
        await self.client.delete_collection(self.collection_name)
        await self._create_collection_at_provider_dims()
