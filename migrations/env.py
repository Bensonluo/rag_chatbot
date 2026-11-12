"""
Alembic environment configuration.

This file is used by Alembic to configure the migration environment.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config.settings import settings
from app.models.database.action_execution import ActionExecution  # noqa: F401

# Import models and settings
from app.models.database.base import Base
from app.models.database.commerce import (  # noqa: F401
    MockCommerceObject,
    MockCommerceReceipt,
    PurchaseAction,
    PurchaseConfirmation,
    PurchaseOperation,
    PurchaseSession,
)
from app.models.database.commerce_taxonomy import (  # noqa: F401
    CommerceCoverageEvent,
    CommerceCoverageReview,
    CommerceExternalCategoryMapping,
    CommerceSemanticNode,
    CommerceSemanticRelation,
    CommerceTaxonomyAttribute,
    CommerceTaxonomyCategory,
    CommerceTaxonomyVersion,
)
from app.models.database.commerce_watch import PriceWatch  # noqa: F401

# Import all models to ensure they're registered with SQLAlchemy
# (side-effect imports — removing them empties Base.metadata and would
# make autogenerate diff against an empty model set)
from app.models.database.document import Document  # noqa: F401
from app.models.database.guest import GuestVisitor  # noqa: F401
from app.models.database.handoff_message import HandoffMessage  # noqa: F401
from app.models.database.knowledge_gap import KnowledgeGapRecord  # noqa: F401
from app.models.database.knowledge_gap_resolution import KnowledgeGapResolution  # noqa: F401
from app.models.database.message import Message  # noqa: F401
from app.models.database.session import ChatSession  # noqa: F401
from app.models.database.ticket import HandoffTicket  # noqa: F401
from app.models.database.user import User  # noqa: F401
from app.models.database.user_fact import UserFactRecord  # noqa: F401
from app.models.database.visitor_preference import (  # noqa: F401
    VisitorPreference,
    VisitorPreferenceAudit,
)

# Alembic config object
config = context.config

# Interpret the config file for Python logging.
# disable_existing_loggers=False is required: fileConfig defaults it to True,
# which disables every logger created before this point for the rest of the
# process. env.py imports the app models above (their `logging.getLogger`
# calls run first), and under pytest every logger from earlier tests already
# exists — the default silently mutes them all, e.g. the shopping router's
# fallback warning vanishes from caplog in any suite that runs an alembic
# migration test first.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Set the SQLAlchemy URL from settings
# Convert async URL to sync URL for alembic migrations
database_url = settings.DATABASE_URL
if database_url.startswith("postgresql+asyncpg://"):
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql+pg8000://")
config.set_main_option("sqlalchemy.url", database_url)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        compare_server_default=True,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a
    connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
