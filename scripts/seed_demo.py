#!/usr/bin/env python3
"""Opt-in demo data seeding for local development.

Deliberately NOT run at container boot: creating well-known credentials
(demo@ragchatbot.com / demo123) in every deployment is a security
liability. Run this by hand when you want the demo user/session:

    python scripts/seed_demo.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config.logging import logger
from app.config.settings import settings
from app.core.security import hash_password
from app.models.database.session import ChatSession
from app.models.database.user import User

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)


async def create_demo_session() -> None:
    """Create the demo user (if missing) and demo session id=1."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session_maker() as session:
        result = await session.execute(text("SELECT id FROM chat_sessions WHERE id = 1"))
        if result.first() is not None:
            logger.info("Demo session already exists")
            return

        user_result = await session.execute(
            text("SELECT id FROM users WHERE email = 'demo@ragchatbot.com'")
        )
        row = user_result.first()
        if row is None:
            demo_user = User(
                email="demo@ragchatbot.com",
                full_name="Demo User",
                hashed_password=hash_password("demo123"),
                is_active=True,
                is_admin=False,
            )
            session.add(demo_user)
            await session.flush()
            demo_user_id = demo_user.id
            logger.info("Demo user created")
        else:
            demo_user_id = row[0]
            logger.info("Demo user already exists")

        session.add(
            ChatSession(
                id=1,
                user_id=demo_user_id,
                title="Demo Chat Session",
                memory_type="sliding_window",
                context_window=10,
            )
        )
        await session.commit()
        logger.info("Demo session created with ID=1")


async def main() -> None:
    """Create demo data if missing."""
    await create_demo_session()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
