"""Retain document sources, revisions, and durable ingestion jobs.

Revision ID: f9a2c7d6b410
Revises: c5f2a8e3d17b
"""

import sqlalchemy as sa
from alembic import op

revision = "f9a2c7d6b410"
down_revision = "c5f2a8e3d17b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Adoption also supports databases made by the current ORM create_all.
    # In SQL-output mode emit the complete migration without inspecting a DB.
    inspector = None if op.get_context().as_sql else sa.inspect(op.get_bind())
    columns = {c["name"] for c in inspector.get_columns("documents")} if inspector else set()
    if "active_version_id" not in columns:
        op.add_column("documents", sa.Column("active_version_id", sa.String(36), nullable=True))
    if inspector and inspector.has_table("document_versions"):
        return
    op.create_table(
        "document_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(255),
            sa.ForeignKey("documents.external_doc_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("file_type", sa.String(16), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("source_content", sa.LargeBinary(), nullable=True),
        sa.Column("normalized_text", sa.Text(), nullable=True),
        sa.Column("source_metadata", sa.JSON(), nullable=False),
        sa.Column("pipeline_config", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])
    op.create_index("ix_document_versions_status", "document_versions", ["status"])


def downgrade() -> None:
    op.drop_table("document_versions")
    op.drop_column("documents", "active_version_id")
