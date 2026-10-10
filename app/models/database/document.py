"""Document database model"""

from typing import Any

from sqlalchemy import JSON, Boolean, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database.base import Base, TimestampMixin


class Document(Base, TimestampMixin):
    """
    Document model for tracking ingested documents.

    This row owns the logical document and its active revision pointer.
    DocumentVersion retains original bytes and normalized text; searchable
    chunks and embeddings are stored in the external vector database.

    Attributes:
        id: Primary key
        external_doc_id: Document ID in vector service (must be unique)
        title: Document title
        source: Source file path or URL
        doc_type: Document type (pdf, txt, html, etc.)
        chunk_count: Number of chunks in vector database
        metadata: Additional metadata as JSON
        is_active: Whether document is active
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_doc_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )
    source: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    doc_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    doc_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    active_version_id: Mapped[str | None] = mapped_column(String(36), default=None)

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, external_id={self.external_doc_id}, title={self.title})>"


class DocumentVersion(Base, TimestampMixin):
    """An immutable source revision and its durable processing job.

    Sources live in Postgres so API/worker replicas share the same bytes.
    Large source and normalized-text columns are deferred on status/list reads.
    """

    __tablename__ = "document_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.external_doc_id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(500))
    filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(16))
    content_hash: Mapped[str] = mapped_column(String(64))
    revision: Mapped[int] = mapped_column(Integer)
    source_content: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True)
    normalized_text: Mapped[str | None] = mapped_column(Text, deferred=True)
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    pipeline_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
