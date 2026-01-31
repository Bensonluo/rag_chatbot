"""
Document ingestion service.

Handles document upload, processing, chunking, and storage in vector database.
"""
import uuid
from typing import List, Optional, Dict
from pathlib import Path

from app.services.documents.base import Document, DocumentChunk
from app.services.documents.chunking import (
    FixedSizeChunking,
    SemanticChunking,
    RecursiveCharacterChunking
)
from app.services.documents.preprocessing import DocumentPreprocessor
from app.services.embeddings import EmbeddingFactory
from app.services.retrieval.qdrant_client import QdrantClient
from app.core.exceptions import ValidationError, ExternalServiceError


class DocumentIngestionService:
    """
    Service for ingesting documents into the RAG system.

    Handles:
    - Document reading (PDF, TXT, MD)
    - Text preprocessing
    - Chunking
    - Embedding generation
    - Storage in vector database
    """

    CHUNKING_STRATEGIES = {
        "fixed": FixedSizeChunking,
        "semantic": SemanticChunking,
        "recursive": RecursiveCharacterChunking,
    }

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_provider: str = "local",
        chunking_strategy: str = "semantic",
        max_chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Initialize document ingestion service.

        Args:
            qdrant_client: Qdrant client for vector storage
            embedding_provider: Provider for embeddings (local, glm)
            chunking_strategy: Strategy for chunking (fixed, semantic, recursive)
            max_chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.qdrant_client = qdrant_client
        self.embedding_provider = embedding_provider
        self.chunking_strategy = chunking_strategy
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embedding service
        self.embedding_service = EmbeddingFactory.create(
            provider=embedding_provider
        )

        # Initialize chunking strategy
        if chunking_strategy not in self.CHUNKING_STRATEGIES:
            raise ValidationError(
                f"Unknown chunking strategy: {chunking_strategy}. "
                f"Available: {list(self.CHUNKING_STRATEGIES.keys())}"
            )
        self.chunking = self.CHUNKING_STRATEGIES[chunking_strategy]()

        # Initialize preprocessor
        self.preprocessor = DocumentPreprocessor()

    async def ingest_text(
        self,
        text: str,
        title: str,
        metadata: dict = None,
        document_id: str = None,
    ) -> dict:
        """
        Ingest a text document into the RAG system.

        Args:
            text: Document text content
            title: Document title
            metadata: Additional metadata
            document_id: Optional custom document ID

        Returns:
            dict: Ingestion results with document_id, chunks_count, etc.

        Raises:
            ValidationError: If input is invalid
            ExternalServiceError: If ingestion fails
        """
        # Validate input
        if not text or not text.strip():
            raise ValidationError("Document text cannot be empty")

        if not title:
            raise ValidationError("Document title is required")

        # Generate document ID if not provided
        if not document_id:
            document_id = str(uuid.uuid4())

        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata["title"] = title

        # Create document object
        document = Document(
            document_id=document_id,
            title=title,
            content=text,
            file_type="txt",
            metadata=metadata
        )

        # Preprocess text
        processed_text, enhanced_metadata = await self.preprocessor.process(
            text=text,
            metadata=metadata
        )
        document.content = processed_text
        document.doc_metadata.update(enhanced_metadata)

        # Chunk document
        chunks = await self.chunking.chunk(
            document=document,
            max_chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        if not chunks:
            raise ValidationError("Document chunking produced no chunks")

        # Generate embeddings for chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embedding_result = await self.embedding_service.embed(chunk_texts)

        # Store in Qdrant
        vectors = embedding_result.embeddings
        payloads = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "index": chunk.index,
                **chunk.metadata
            }
            for chunk in chunks
        ]

        # Insert into Qdrant
        await self.qdrant_client.add(
            ids=[chunk.chunk_id for chunk in chunks],
            vectors=vectors,
            payloads=payloads
        )

        return {
            "document_id": document_id,
            "title": title,
            "chunks_count": len(chunks),
            "total_tokens": embedding_result.tokens_used,
            "embedding_model": embedding_result.model,
            "chunking_strategy": self.chunking_strategy,
        }

    async def ingest_file(
        self,
        file_path: str,
        metadata: dict = None,
        document_id: str = None,
    ) -> dict:
        """
        Ingest a document from file into the RAG system.

        Args:
            file_path: Path to the file
            metadata: Additional metadata
            document_id: Optional custom document ID

        Returns:
            dict: Ingestion results

        Raises:
            ValidationError: If file is invalid or not supported
            ExternalServiceError: If ingestion fails
        """
        path = Path(file_path)

        # Validate file exists
        if not path.exists():
            raise ValidationError(f"File not found: {file_path}")

        # Get file extension
        file_type = path.suffix.lower().lstrip('.')

        # Read file based on type
        if file_type == "txt":
            text = self._read_txt(path)
        elif file_type == "md":
            text = self._read_txt(path)
        elif file_type == "pdf":
            text = await self._read_pdf(path)
        else:
            raise ValidationError(
                f"Unsupported file type: {file_type}. "
                f"Supported: txt, md, pdf"
            )

        # Use filename as title if not provided
        title = metadata.get("title") if metadata else None
        if not title:
            title = path.stem

        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata["file_name"] = path.name
        metadata["file_type"] = file_type
        metadata["file_path"] = str(path)

        # Ingest
        return await self.ingest_text(
            text=text,
            title=title,
            metadata=metadata,
            document_id=document_id
        )

    def _read_txt(self, path: Path) -> str:
        """Read text file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(path, 'r', encoding='latin-1') as f:
                return f.read()

    async def _read_pdf(self, path: Path) -> str:
        """Read PDF file."""
        try:
            import pypdf
            text = ""

            with open(path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"

            return text.strip()

        except ImportError:
            raise ExternalServiceError(
                service="PDF Reader",
                message="pypdf not installed. Run: pip install pypdf"
            )
        except Exception as e:
            raise ExternalServiceError(
                service="PDF Reader",
                message=f"Failed to read PDF: {str(e)}"
            )

    async def delete_document(self, document_id: str) -> dict:
        """
        Delete a document and all its chunks from the vector database.

        Args:
            document_id: ID of document to delete

        Returns:
            dict: Deletion results
        """
        # Find all chunks for this document
        # Note: This requires Qdrant to support filtering by document_id
        # For now, we'll use the client's delete method with filter

        deleted_count = await self.qdrant_client.delete_by_filter(
            filter={
                "must": [
                    {"key": "document_id", "match": {"value": document_id}}
                ]
            }
        )

        return {
            "document_id": document_id,
            "deleted_chunks": deleted_count
        }
