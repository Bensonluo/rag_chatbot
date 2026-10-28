"""Shared fixtures for the commerce provider unit/contract suite."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.commerce_provider.api import create_app
from app.commerce_provider.catalog import VirtualCatalog
from app.commerce_provider.core import CommerceProviderCore
from app.commerce_provider.faults import FaultController
from app.commerce_provider.settings import CommerceProviderSettings
from app.commerce_provider.storage import Database

AUTH_HEADERS = {"Authorization": "Bearer test-token"}


class FixedClock:
    """Deterministic wall clock the core can be driven through."""

    def __init__(self) -> None:
        self.now = datetime(2026, 10, 29, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now = self.now + timedelta(seconds=seconds)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock()


@pytest.fixture
async def core(tmp_path: Path, clock: FixedClock) -> CommerceProviderCore:
    database = Database(str(tmp_path / "provider.sqlite3"))
    await database.startup()
    return CommerceProviderCore(
        database,
        VirtualCatalog(),
        clock=clock,
        quote_ttl_seconds=300,
        faults=FaultController(database, enabled=True),
    )


@pytest.fixture
def app(tmp_path: Path):
    return create_app(
        CommerceProviderSettings(
            database_path=str(tmp_path / "provider.sqlite3"),
            auth_token="test-token",
            faults_enabled=True,
        )
    )


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
