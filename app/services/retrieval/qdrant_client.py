"""
Qdrant vector database client implementation.

Provides async client for Qdrant vector database operations.
"""
import asyncio

from typing import Optional, List
from app.services.retrieval.vector_base import (
    VectorClient,
    Document,
    SearchResult,
    VectorSearchRequest,
    VectorClientError,
)


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
        api_key: Optional[str] = None,
        client: Optional[object] = None,
        embedding_service=None,
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
            except ImportError:
                raise ImportError(
                    "qdrant-client is not installed. "
                    "Install it with: pip install qdrant-client"
                )

    async def add_documents(
        self,
        documents: List[Document],
    ) -> List[str]:
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
                    }
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
                f"Failed to add documents: {str(e)}",
                details={"document_count": len(documents)}
            ) from e

    async def search(
        self,
        request: VectorSearchRequest,
    ) -> List[SearchResult]:
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
                    metadata=payload.get("metadata"),
                )
                results.append(result)

            return results

        except Exception as e:
            raise VectorClientError(
                f"Search failed: {str(e)}",
                details={"query": request.query}
            ) from e

    async def delete(
        self,
        document_ids: List[str],
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
                f"Failed to delete documents: {str(e)}",
                details={"document_ids": document_ids}
            ) from e

    async def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[dict],
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
                        "len_payloads": len(payloads)
                    }
                )

            points = []
            for point_id, vector, payload in zip(ids, vectors, payloads):
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                points.append(point)

            # Batch upsert
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

        except Exception as e:
            raise VectorClientError(
                f"Failed to add points: {str(e)}",
                details={"point_count": len(ids)}
            ) from e

    async def delete_by_filter(self, filter: dict) -> int:
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

            # Delete with filter
            await self.client.delete(
                collection_name=self.collection_name,
                query_filter=qdrant_filter
            )

            return int(count_result.count)

        except Exception as e:
            raise VectorClientError(
                f"Failed to delete by filter: {str(e)}",
                details={"filter": filter}
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
                }
            )

            await self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

        except Exception as e:
            raise VectorClientError(
                f"Failed to update document: {str(e)}",
                details={"document_id": document.id}
            ) from e

    async def get_document(
        self,
        document_id: str,
    ) -> Optional[Document]:
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
                f"Failed to get document: {str(e)}",
                details={"document_id": document_id}
            ) from e

    async def _generate_embedding(self, text: str) -> List[float]:
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
            from app.services.embeddings.base import EmbeddingResult

            # Generate embedding for single text
            result = await self.embedding_service.embed_single(text)

            return result if isinstance(result, list) else result.tolist()

        except Exception as e:
            raise VectorClientError(
                f"Failed to generate embedding: {str(e)}",
                details={"text_length": len(text)}
            ) from e

    def _build_filter(
        self,
        filters: dict,
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

    def _qdrant_model(self, name: str):
        """Load a Qdrant SDK model, with a lightweight injected-client fallback."""
        try:
            from qdrant_client import models

            return getattr(models, name)
        except (ImportError, AttributeError):
            if self._client_injected:
                return _InjectedClientModel
            raise ImportError(
                "qdrant-client is required for vector database operations"
            )

    async def _ensure_collection(self) -> None:
        """Create the configured collection lazily on a fresh demo stack."""
        if self._client_injected or self._collection_ready:
            return

        async with self._collection_lock:
            if self._collection_ready:
                return

            collection_exists = getattr(self.client, "collection_exists", None)
            if collection_exists is not None:
                exists = await collection_exists(self.collection_name)
            else:
                collections = await self.client.get_collections()
                exists = any(
                    collection.name == self.collection_name
                    for collection in collections.collections
                )

            if not exists:
                Distance = self._qdrant_model("Distance")
                VectorParams = self._qdrant_model("VectorParams")
                vector_size = int(
                    getattr(self.embedding_service, "dimensions", 1024)
                )
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )

            self._collection_ready = True
