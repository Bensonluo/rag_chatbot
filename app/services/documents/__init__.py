"""
Document processing services package.

Exports all document-related components including chunking,
preprocessing, and ingestion.
"""

from app.services.documents.base import ChunkingStrategy, Document, DocumentChunk
from app.services.documents.chunking import (
    FixedSizeChunking,
    RecursiveCharacterChunking,
    SemanticChunking,
)
from app.services.documents.ingestion import DocumentIngestionService
from app.services.documents.preprocessing import DocumentPreprocessor, TextPreprocessor

__all__ = [
    # Base classes
    "Document",
    "DocumentChunk",
    "ChunkingStrategy",
    # Chunking strategies
    "FixedSizeChunking",
    "SemanticChunking",
    "RecursiveCharacterChunking",
    # Preprocessing
    "TextPreprocessor",
    "DocumentPreprocessor",
    # Ingestion
    "DocumentIngestionService",
]
