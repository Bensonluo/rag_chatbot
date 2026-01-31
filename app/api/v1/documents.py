"""
Document management API endpoints.

Provides REST API for document upload, search, and management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel, Field
from io import BytesIO

from app.services.documents.ingestion import DocumentIngestionService
from app.services.embeddings import EmbeddingFactory
from app.services.retrieval.qdrant_client import QdrantClient
from app.services.retrieval.factory import RetrievalFactory
from app.services.retrieval.vector_base import VectorSearchRequest
from app.api.deps import get_current_user, get_current_active_user
from app.models.database.user import User
from app.config.settings import get_settings


router = APIRouter(prefix="/documents", tags=["documents"])


# Request/Response Schemas
class DocumentUploadRequest(BaseModel):
    """Document upload request (for text-based upload)."""
    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    content: str = Field(..., min_length=1, description="Document text content")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")


class DocumentUploadResponse(BaseModel):
    """Document upload response."""
    document_id: str
    title: str
    chunks_count: int
    total_tokens: int
    embedding_model: str
    chunking_strategy: str
    message: str = "Document uploaded successfully"


class SearchRequest(BaseModel):
    """Search request."""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    filters: Optional[dict] = Field(default=None, description="Optional filters")


class SearchResult(BaseModel):
    """Single search result."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Optional[dict] = None


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    results: List[SearchResult]
    total_count: int


class DocumentInfo(BaseModel):
    """Document information."""
    document_id: str
    title: str
    chunks_count: int


class DeleteResponse(BaseModel):
    """Delete response."""
    document_id: str
    deleted_chunks: int
    message: str = "Document deleted successfully"


# Dependencies
async def get_ingestion_service() -> DocumentIngestionService:
    """
    Get document ingestion service instance.

    Creates service with embedding and vector DB configuration.
    """
    settings = get_settings()

    # Create embedding service
    embedding_service = EmbeddingFactory.create_from_settings()

    # Create Qdrant client
    qdrant_client = RetrievalFactory.create_vector_client(
        client_type="qdrant",
        url=settings.VECTOR_DB_URL,
        collection_name=settings.VECTOR_COLLECTION_NAME,
        api_key=settings.VECTOR_API_KEY,
        embedding_service=embedding_service,
    )

    # Create ingestion service
    return DocumentIngestionService(
        qdrant_client=qdrant_client,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        chunking_strategy="semantic",  # Default chunking strategy
    )


async def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance."""
    settings = get_settings()

    # Create embedding service
    embedding_service = EmbeddingFactory.create_from_settings()

    # Create Qdrant client
    return RetrievalFactory.create_vector_client(
        client_type="qdrant",
        url=settings.VECTOR_DB_URL,
        collection_name=settings.VECTOR_COLLECTION_NAME,
        api_key=settings.VECTOR_API_KEY,
        embedding_service=embedding_service,
    )


# Endpoints
@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload document text"
)
async def upload_document_text(
    request: DocumentUploadRequest,
    current_user: User = Depends(get_current_active_user),
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
):
    """
    Upload a document via text content.

    - **title**: Document title
    - **content**: Full text content of the document
    - **metadata**: Optional metadata (author, category, tags, etc.)

    Returns document ID and processing statistics.
    """
    try:
        # Add user info to metadata
        metadata = request.metadata or {}
        metadata["uploaded_by"] = current_user.id
        metadata["uploaded_by_email"] = current_user.email

        # Ingest document
        result = await ingestion_service.ingest_text(
            text=request.content,
            title=request.title,
            metadata=metadata
        )

        return DocumentUploadResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )


@router.post(
    "/upload/file",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload document file"
)
async def upload_document_file(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
):
    """
    Upload a document file (PDF, TXT, MD).

    Supported formats:
    - PDF (.pdf)
    - Text (.txt)
    - Markdown (.md)

    File is processed, chunked, and stored in vector database.
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )

        # Check file extension
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['pdf', 'txt', 'md']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: .{file_ext}. Supported: pdf, txt, md"
            )

        # Read file content
        content = await file.read()

        # Save to temp file for ingestion service
        import tempfile
        import os

        # Create temp file
        with tempfile.NamedTemporaryFile(
            mode='wb',
            delete=False,
            suffix=f'.{file_ext}'
        ) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Prepare metadata
            metadata = {
                "uploaded_by": current_user.id,
                "uploaded_by_email": current_user.email,
                "original_filename": file.filename,
            }

            if title:
                metadata["title"] = title

            # Ingest file
            result = await ingestion_service.ingest_file(
                file_path=temp_path,
                metadata=metadata
            )

            return DocumentUploadResponse(**result)

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search documents"
)
async def search_documents(
    request: SearchRequest,
    current_user: User = Depends(get_current_user),
    qdrant_client: QdrantClient = Depends(get_qdrant_client),
):
    """
    Search for relevant document chunks using semantic search.

    - **query**: Search query text
    - **top_k**: Number of results to return (1-20)
    - **filters**: Optional filters for metadata

    Returns ranked list of relevant document chunks.
    """
    try:
        # Create search request
        search_request = VectorSearchRequest(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )

        # Perform search
        results = await qdrant_client.search(search_request)

        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                chunk_id=result.metadata.get("chunk_id", "") if result.metadata else "",
                document_id=result.document_id,
                content=result.content,
                score=result.score,
                metadata=result.metadata
            ))

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_count=len(search_results)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    summary="Delete document"
)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
):
    """
    Delete a document and all its chunks from the vector database.

    This will remove all chunks associated with the document.
    """
    try:
        result = await ingestion_service.delete_document(document_id)

        return DeleteResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.get(
    "/health",
    summary="Document service health check"
)
async def health_check():
    """Check if document service is healthy."""
    return {
        "status": "healthy",
        "service": "document-management",
        "embedding_enabled": True,
        "vector_db_enabled": True
    }
