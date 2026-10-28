"""PurchaseService orchestration over the real external HTTP provider app.

This is the composition seam test: no local mock provider participates, every
commerce fact crosses the documented HTTP wire through ``ExternalCommercePort``,
while confirmation/version/idempotency safety remains in ``PurchaseService``.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.commerce_provider.api import create_app
from app.commerce_provider.settings import CommerceProviderSettings
from app.models.database.base import Base
from app.models.database.commerce import MockCommerceObject, PurchaseOperation
from app.models.database.user import User
from app.models.schemas.commerce import CartBlock, CartLineSelection, CatalogBlock, CommerceView
from app.services.commerce.external_port import ExternalCommercePort
from app.services.commerce.service import PurchaseService


def _catalog(view: CommerceView) -> CatalogBlock:
    for block in view.blocks:
        if isinstance(block, CatalogBlock):
            return block
    raise AssertionError("search did not return a catalog block")


def _cart(view: CommerceView) -> CartBlock:
    for block in view.blocks:
        if isinstance(block, CartBlock):
            return block
    raise AssertionError("cart update did not return a cart block")


async def _operation_counts(maker: async_sessionmaker[AsyncSession]) -> dict[str, int]:
    rows: dict[str, int] = {}
    async with maker() as db:
        result = await db.execute(
            select(PurchaseOperation.kind, func.count()).group_by(PurchaseOperation.kind)
        )
        for kind, count in result:
            rows[str(kind)] = int(count)
        local_mock_count = await db.scalar(select(func.count()).select_from(MockCommerceObject))
    assert local_mock_count == 0
    return rows


@pytest.fixture
async def domain(maker: async_sessionmaker[AsyncSession]) -> None:
    async with maker() as db:
        db.add_all(
            [
                User(id=41, email="external-commerce@unit.invalid", hashed_password="!"),
                User(
                    id=42,
                    email="external-commerce-recovery@unit.invalid",
                    hashed_password="!",
                ),
            ]
        )
        await db.commit()


@pytest.fixture
async def maker(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("external-provider composition test must not open sockets")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/domain.sqlite")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection: Any, _: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield session_maker
    await engine.dispose()


async def test_purchase_safety_runs_over_external_http_provider_facts(
    tmp_path: Any,
    maker: async_sessionmaker[AsyncSession],
    domain: None,
) -> None:
    provider = create_app(
        CommerceProviderSettings(
            host="unused",
            port=1,
            database_path=str(tmp_path / "provider.sqlite3"),
            auth_token="",
            quote_ttl_seconds=300,
            faults_enabled=False,
        )
    )

    async with (
        provider.router.lifespan_context(provider),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=provider),
            base_url="http://commerce-provider.internal",
        ) as client,
    ):
        port = ExternalCommercePort(
            "http://commerce-provider.internal", token="", timeout_seconds=5, client=client
        )
        service = PurchaseService(maker, port)

        started = await service.start(41)
        catalog = _catalog(
            await service.search(41, started.purchase_id, "显示器 黑色", page_size=5)
        )
        assert len(catalog.items) >= 2
        first, second = catalog.items[:2]
        selections = [
            CartLineSelection(sku_id=first.sku_id.lower(), quantity=2),
            CartLineSelection(sku_id=second.sku_id, quantity=1),
        ]
        expected_selections = [
            (first.sku_id, 2),
            (second.sku_id, 1),
        ]

        cart_view = await service.update_cart_lines(
            41, started.purchase_id, selections, started.version
        )
        cart = _cart(cart_view)
        assert cart_view.stage == "cart"
        assert [(line.sku_id, line.quantity) for line in cart.cart.lines] == expected_selections

        quote_view = await service.prepare_checkout(41, started.purchase_id, cart_view.version)
        assert quote_view.stage == "quoted"
        order_view = await service.commit_checkout(
            41, started.purchase_id, expected_version=quote_view.version
        )
        assert order_view.stage == "awaiting_payment"
        paid_view = await service.mock_pay(
            41, started.purchase_id, expected_version=order_view.version
        )
        assert paid_view.stage == "paid"

    counts = await _operation_counts(maker)
    assert counts == {
        "update_cart": 1,
        "quote": 1,
        "order": 1,
        "payment": 1,
        "mock_pay": 1,
    }


async def test_external_unknown_order_reconciles_from_provider_receipt(
    tmp_path: Any,
    maker: async_sessionmaker[AsyncSession],
    domain: None,
) -> None:
    provider = create_app(
        CommerceProviderSettings(
            host="unused",
            port=1,
            database_path=str(tmp_path / "provider-recovery.sqlite3"),
            auth_token="recovery-token",
            quote_ttl_seconds=300,
            faults_enabled=True,
        )
    )

    async with (
        provider.router.lifespan_context(provider),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=provider),
            base_url="http://commerce-provider.internal",
        ) as client,
    ):
        port = ExternalCommercePort(
            "http://commerce-provider.internal",
            token="recovery-token",
            timeout_seconds=5,
            client=client,
        )
        service = PurchaseService(maker, port)
        started = await service.start(42)
        catalog = _catalog(await service.search(42, started.purchase_id, "显示器", page_size=1))
        assert catalog.items
        cart_view = await service.update_cart_lines(
            42,
            started.purchase_id,
            [CartLineSelection(sku_id=catalog.items[0].sku_id, quantity=1)],
            started.version,
        )
        quote_view = await service.prepare_checkout(42, started.purchase_id, cart_view.version)

        planned = await client.post(
            "/testing/faults",
            headers={"Authorization": "Bearer recovery-token"},
            json={"kind": "create_order", "mode": "unknown_after_commit"},
        )
        assert planned.status_code == 200
        unknown = await service.commit_checkout(
            42, started.purchase_id, expected_version=quote_view.version
        )
        assert unknown.stage == "unknown"

        recovered = await service.status(42, started.purchase_id)
        assert recovered.stage == "awaiting_payment"

    counts = await _operation_counts(maker)
    assert counts == {"update_cart": 1, "quote": 1, "order": 1}
