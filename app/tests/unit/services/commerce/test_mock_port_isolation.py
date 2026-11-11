"""Mock provider facts must never land in the main application database.

Production wires MockCommercePort to the app's PostgreSQL session maker (the
default DATABASE_URL under COMMERCE_PROVIDER_MODE=mock). The port then
transparently keeps every provider fact — seeded stock, carts, orders,
idempotency receipts — in its own SQLite store at COMMERCE_MOCK_DATABASE_PATH,
so the emulated provider never writes commerce_mock_* tables into the main DB.

Makers bound to SQLite (every existing commerce test fixture shares an
in-memory or file-backed sqlite DB) keep using exactly the maker they were
handed: the 180+ existing commerce tests are untouched by this isolation.
"""

from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.settings import settings
from app.models.database.base import Base
from app.models.database.commerce import MockCommerceObject
from app.models.database.user import User
from app.models.schemas.commerce import CartLineSelection
from app.services.commerce.mock_port import MockCommercePort

# A URL that can build an engine but never actually connects anywhere.
_MAIN_DB_URL = "postgresql+asyncpg://isolation:pw@invalid.invalid/app"


@pytest.fixture
def mock_store_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "commerce_mock.sqlite3"
    monkeypatch.setattr(settings, "COMMERCE_MOCK_DATABASE_PATH", str(path))
    return path


def _main_maker() -> tuple[async_sessionmaker[AsyncSession], list[int]]:
    engine = create_async_engine(_MAIN_DB_URL)
    connections: list[int] = []

    @event.listens_for(engine.sync_engine, "connect")
    def connected(*_args) -> None:  # pragma: no cover - must never fire
        connections.append(1)

    return async_sessionmaker(engine, expire_on_commit=False), connections


def _sqlite_scalar(path: Path, statement: str) -> int:
    sync = create_engine(f"sqlite:///{path}")
    try:
        with sync.connect() as conn:
            value = conn.execute(text(statement)).scalar()
            assert value is not None, statement
            return int(value)
    finally:
        sync.dispose()


def test_default_mock_store_path_is_a_local_sqlite_file() -> None:
    assert settings.model_fields["COMMERCE_MOCK_DATABASE_PATH"].default == "commerce_mock.sqlite3"


async def test_non_sqlite_main_db_never_receives_catalog_facts(mock_store_path: Path) -> None:
    maker, connections = _main_maker()
    port = MockCommercePort(maker)
    page = await port.search_products(1, "降噪耳机")
    assert page.items, "the deterministic catalog must seed the private store"

    assert mock_store_path.exists()
    assert _sqlite_scalar(mock_store_path, "select count(*) from commerce_mock_objects") > 0
    assert connections == []  # the main (postgres) engine is never touched
    await port.aclose()


async def test_cart_and_receipt_facts_stay_in_the_private_store(mock_store_path: Path) -> None:
    maker, connections = _main_maker()
    port = MockCommercePort(maker)
    page = await port.search_products(1, "降噪耳机")
    sku = page.items[0].sku_id
    snapshot = await port.update_cart_lines(
        1, None, None, [CartLineSelection(sku_id=sku, quantity=1)], request_key="iso-cart-1"
    )
    assert snapshot.lines[0].sku_id == sku
    assert _sqlite_scalar(mock_store_path, "select count(*) from commerce_mock_receipts") >= 1
    non_stock = "select count(*) from commerce_mock_objects where kind != 'stock'"
    assert _sqlite_scalar(mock_store_path, non_stock) >= 1
    assert connections == []
    await port.aclose()


async def test_private_store_file_survives_port_reconstruction(mock_store_path: Path) -> None:
    maker, _connections = _main_maker()
    first = MockCommercePort(maker)
    page = await first.search_products(1, "降噪耳机")
    cart = await first.update_cart_lines(
        1,
        None,
        None,
        [CartLineSelection(sku_id=page.items[0].sku_id, quantity=1)],
        request_key="iso-cart-2",
    )
    await first.aclose()

    # A fresh port over the same main DB reuses the same private file: the
    # cart written by the previous instance is still there (no re-seed wipe).
    second = MockCommercePort(maker)
    snapshot = await second.get_cart(1, cart.cart_id)
    assert snapshot.lines and snapshot.lines[0].sku_id == page.items[0].sku_id
    await second.aclose()


async def test_sqlite_bound_maker_keeps_the_shared_sessions(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/main.sqlite")
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with maker() as db:
        db.add(User(id=1, email="iso@unit.invalid", hashed_password="!"))
        await db.commit()

    store = tmp_path / "commerce_mock.sqlite3"
    port = MockCommercePort(maker)
    assert port.sessions is maker  # unchanged: no private store is built
    await port.search_products(1, "降噪耳机")
    assert not store.exists()  # no side file appears

    async with maker() as db:  # facts live in the shared sqlite DB
        rows = (
            await db.scalars(select(MockCommerceObject).where(MockCommerceObject.kind == "stock"))
        ).all()
        assert rows
    await port.aclose()  # no-op on the shared-sessions path
    await engine.dispose()
