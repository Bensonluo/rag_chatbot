"""Add versioned commerce taxonomy, semantic, and coverage tables.

Revision ID: f3a8b2c91d04
Revises: e75c9d1064a3
"""

from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "f3a8b2c91d04"
down_revision = "e75c9d1064a3"
branch_labels = None
depends_on = None

TABLES = (
    "commerce_coverage_reviews",
    "commerce_coverage_events",
    "commerce_semantic_relations",
    "commerce_semantic_nodes",
    "commerce_external_category_mappings",
    "commerce_taxonomy_attributes",
    "commerce_taxonomy_categories",
    "commerce_taxonomy_versions",
)


def _timestamps() -> list[sa.Column[Any]]:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    ]


def _json_column(name: str) -> sa.Column[Any]:
    return sa.Column(name, sa.JSON(), nullable=False, default=dict)


def upgrade() -> None:
    inspector = None if op.get_context().as_sql else sa.inspect(op.get_bind())

    def create(name: str, columns: list[Any], indexes: tuple[str, ...] = ()) -> None:
        if inspector is not None and inspector.has_table(name):
            return
        op.create_table(name, *columns, *_timestamps())
        for column in indexes:
            op.create_index(f"ix_{name}_{column}", name, [column])

    if inspector is None or not inspector.has_table("commerce_taxonomy_versions"):
        op.create_table(
            "commerce_taxonomy_versions",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("status", sa.String(16), nullable=False, default="draft"),
            sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
            *_timestamps(),
            sa.CheckConstraint(
                "status IN ('draft','active','archived')",
                name="ck_taxonomy_version_status",
            ),
        )

    create(
        "commerce_taxonomy_categories",
        [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "version_id",
                sa.String(64),
                sa.ForeignKey("commerce_taxonomy_versions.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("category_key", sa.String(64), nullable=False),
            sa.Column("parent_key", sa.String(64), nullable=True),
            sa.Column("name", sa.String(128), nullable=False),
            sa.Column("description", sa.Text(), nullable=False, default=""),
            sa.Column("domain", sa.String(64), nullable=False, default="general"),
            sa.Column("provider_policy", sa.String(16), nullable=False, default="generated"),
            sa.Column("status", sa.String(16), nullable=False, default="active"),
            sa.Column("provider_aliases", sa.JSON(), nullable=False, default=list),
            sa.Column("base_price_minor", sa.Integer(), nullable=True),
            sa.Column("position", sa.Integer(), nullable=False, default=0),
            sa.UniqueConstraint("version_id", "category_key", name="uq_taxonomy_category_key"),
            sa.CheckConstraint(
                "provider_policy IN ('generated','external','unsupported')",
                name="ck_taxonomy_policy",
            ),
            sa.CheckConstraint(
                "status IN ('active','planned','deprecated')",
                name="ck_taxonomy_category_status",
            ),
        ],
        ("version_id",),
    )
    create(
        "commerce_taxonomy_attributes",
        [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "version_id",
                sa.String(64),
                sa.ForeignKey("commerce_taxonomy_versions.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("category_key", sa.String(64), nullable=False),
            sa.Column("name", sa.String(64), nullable=False),
            sa.Column("allowed_values", sa.JSON(), nullable=False, default=list),
            sa.Column("required", sa.Boolean(), nullable=False, default=False),
            sa.Column("position", sa.Integer(), nullable=False, default=0),
            sa.UniqueConstraint(
                "version_id", "category_key", "name", name="uq_taxonomy_attribute_key"
            ),
        ],
        ("version_id", "category_key"),
    )
    create(
        "commerce_external_category_mappings",
        [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "version_id",
                sa.String(64),
                sa.ForeignKey("commerce_taxonomy_versions.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("internal_category_key", sa.String(64), nullable=False),
            sa.Column("provider", sa.String(64), nullable=False),
            sa.Column("external_category_id", sa.String(128), nullable=False),
            sa.Column("external_path", sa.String(500), nullable=False, default=""),
            sa.Column("confidence", sa.Float(), nullable=False, default=0.0),
            sa.Column("status", sa.String(16), nullable=False, default="active"),
            sa.UniqueConstraint(
                "version_id",
                "provider",
                "external_category_id",
                name="uq_external_category_mapping",
            ),
            sa.CheckConstraint(
                "status IN ('active','planned','deprecated')",
                name="ck_external_mapping_status",
            ),
        ],
        ("version_id", "internal_category_key"),
    )
    create(
        "commerce_semantic_nodes",
        [
            sa.Column("id", sa.String(96), primary_key=True),
            sa.Column("node_type", sa.String(64), nullable=False),
            sa.Column("name", sa.String(128), nullable=False),
            sa.Column("description", sa.Text(), nullable=False, default=""),
            _json_column("properties"),
            sa.Column("source", sa.String(64), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False, default=0.0),
            sa.Column("status", sa.String(16), nullable=False, default="approved"),
            sa.Column("version", sa.String(64), nullable=False, default="semantic-v1"),
            sa.CheckConstraint(
                "status IN ('pending','approved','rejected')",
                name="ck_semantic_node_status",
            ),
        ],
        ("node_type", "status"),
    )
    create(
        "commerce_semantic_relations",
        [
            sa.Column("id", sa.String(128), primary_key=True),
            sa.Column("source_ref", sa.String(128), nullable=False),
            sa.Column("target_ref", sa.String(128), nullable=False),
            sa.Column("relation_type", sa.String(64), nullable=False),
            sa.Column("description", sa.Text(), nullable=False, default=""),
            _json_column("properties"),
            sa.Column("source", sa.String(64), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False, default=0.0),
            sa.Column("status", sa.String(16), nullable=False, default="approved"),
            sa.Column("version", sa.String(64), nullable=False, default="semantic-v1"),
            sa.CheckConstraint(
                "status IN ('pending','approved','rejected')",
                name="ck_semantic_relation_status",
            ),
        ],
        ("relation_type", "status"),
    )
    create(
        "commerce_coverage_events",
        [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("query", sa.String(500), nullable=False),
            sa.Column("normalized_query", sa.String(500), nullable=False),
            sa.Column("owner_hash", sa.String(64), nullable=False, default=""),
            sa.Column("matched_category_ids", sa.JSON(), nullable=False, default=list),
            sa.Column("nearest_category_ids", sa.JSON(), nullable=False, default=list),
            sa.Column("result_count", sa.Integer(), nullable=False, default=0),
            sa.Column("status", sa.String(16), nullable=False, default="zero_result"),
            sa.CheckConstraint(
                "status IN ('zero_result','low_result','wrong_result')",
                name="ck_coverage_status",
            ),
        ],
        ("normalized_query", "status"),
    )
    create(
        "commerce_coverage_reviews",
        [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "event_id",
                sa.Integer(),
                sa.ForeignKey("commerce_coverage_events.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("proposed_type", sa.String(16), nullable=False),
            _json_column("proposal_payload"),
            sa.Column("status", sa.String(16), nullable=False, default="pending"),
            sa.Column("reviewer", sa.String(128), nullable=True),
            sa.Column("review_note", sa.Text(), nullable=False, default=""),
            sa.CheckConstraint(
                "status IN ('pending','approved','rejected')",
                name="ck_coverage_review_status",
            ),
            sa.CheckConstraint(
                "proposed_type IN ('category','alias','use_case','relation')",
                name="ck_coverage_proposed_type",
            ),
        ],
        ("event_id", "status"),
    )


def downgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name != "postgresql":
            raise RuntimeError("Commerce taxonomy downgrade requires PostgreSQL for SQL mode")
        op.execute("LOCK TABLE " + ", ".join(TABLES) + " IN ACCESS EXCLUSIVE MODE")
        condition = " OR ".join(f"EXISTS (SELECT 1 FROM {name})" for name in TABLES)
        op.execute(
            "DO $$ BEGIN IF "
            + condition
            + " THEN RAISE EXCEPTION 'Commerce taxonomy data prevents lossless downgrade'; END IF; END $$"
        )
    else:
        db = op.get_bind()
        if context.dialect.name == "postgresql":
            op.execute("LOCK TABLE " + ", ".join(TABLES) + " IN ACCESS EXCLUSIVE MODE")
        elif context.dialect.name == "sqlite":
            op.execute("UPDATE commerce_taxonomy_versions SET status=status WHERE 0")
        else:
            raise RuntimeError("Commerce taxonomy downgrade supports PostgreSQL or SQLite only")
        if any(
            db.execute(sa.text(f"SELECT 1 FROM {name} LIMIT 1")).first() is not None
            for name in TABLES
        ):
            raise RuntimeError("Commerce taxonomy data prevents lossless downgrade")
    for table in TABLES:
        op.drop_table(table)
