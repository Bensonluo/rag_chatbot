#!/usr/bin/env python3
"""Database migration runner for container startup.

Runs Alembic migrations to head instead of ``Base.metadata.create_all``
so schema changes are versioned, reviewable, and safe to replay across
deploys and replicas.

Databases created by the legacy ``create_all`` path (tables present but
no ``alembic_version`` table) are stamped to head once on first boot of
this version, making the transition non-destructive.

Demo data seeding is deliberately NOT part of boot: creating
well-known credentials (demo@ragchatbot.com / demo123) in every
production deployment is a security liability. Local development can
opt in via ``scripts/seed_demo.py``.
"""

import asyncio
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config.logging import logger
from app.config.settings import settings

REPO_ROOT = Path(__file__).parent.parent


async def verify_database() -> bool:
    """Verify database is reachable."""
    engine = create_async_engine(settings.DATABASE_URL)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection verified")
        return True
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False
    finally:
        await engine.dispose()


async def _database_state() -> str:
    """Classify the database as 'empty', 'legacy', or 'versioned'."""
    engine = create_async_engine(settings.DATABASE_URL)
    try:
        async with engine.connect() as conn:
            table_names = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )
    finally:
        await engine.dispose()

    if "alembic_version" in table_names:
        return "versioned"
    # The dialogue checkpointer's tables are created by its own setup()
    # outside Alembic; they do not count as legacy ORM-managed schema.
    orm_tables = [t for t in table_names if not t.startswith("checkpoint")]
    return "legacy" if orm_tables else "empty"


def _run_migrations() -> None:
    """Run ``alembic upgrade head`` (sync engine via the alembic env)."""
    alembic_cfg = Config(str(REPO_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    command.upgrade(alembic_cfg, "head")


def _stamp_legacy_schema() -> None:
    """Mark a legacy create_all database as being at head, without DDL."""
    alembic_cfg = Config(str(REPO_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    command.stamp(alembic_cfg, "head")


async def main() -> None:
    """Wait for the database, then bring the schema to head."""
    logger.info("Starting database initialization...")

    # Wait for database to be ready
    max_retries = 30
    for i in range(max_retries):
        if await verify_database():
            break
        logger.info(f"Database not ready, retrying... ({i + 1}/{max_retries})")
        await asyncio.sleep(2)
    else:
        logger.error("Database connection failed after max retries")
        sys.exit(1)

    state = await _database_state()
    if state == "legacy":
        # Pre-existing create_all deployment: adopting Alembic must not
        # re-run DDL that already ran. Stamp, then no-op upgrade.
        logger.info(
            "Legacy schema detected (tables without alembic_version); "
            "stamping to head without running DDL"
        )
        _stamp_legacy_schema()
    else:
        logger.info("Running migrations to head (%s database)", state)
        _run_migrations()

    logger.info("Database initialization complete")


if __name__ == "__main__":
    asyncio.run(main())
