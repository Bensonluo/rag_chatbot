"""Internal FastAPI facade over ``CommerceProviderCore``.

Wire contract (verbatim from ``app/services/commerce/external_port.py``):
all business paths live under ``/commerce``, bodies are JSON, ``owner_user_id``
is always explicit, and success bodies validate exactly against the shared
models in ``app/models/schemas/commerce.py``. Authentication uses
``Authorization: Bearer <token>`` when a token is configured (empty token
disables the check); ``/health`` and ``/ready`` stay open for probes.

Errors: 4xx bodies ``{"error": {"code": str, "message": str}}`` are definitive
refusals — HTTP 409 specifically for a ``request_key`` replayed with different
arguments — while 5xx answers mean "outcome unconfirmed, reconcile via the
original request key".
"""

from __future__ import annotations

import hmac
import json
import logging
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse

from app.commerce_provider.catalog import VirtualCatalog
from app.commerce_provider.core import CommerceProviderCore
from app.commerce_provider.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderNotFoundError,
    ProviderRejectedError,
)
from app.commerce_provider.faults import FaultController
from app.commerce_provider.schemas import (
    CatalogSearchRequest,
    CreateOrderRequest,
    CreatePaymentRequest,
    FaultPlanRequest,
    MintCategoryRequest,
    MintCategoryResult,
    OwnerParam,
    PrepareCheckoutRequest,
    SandboxPayRequest,
    SettlePaymentRequest,
    UpdateCartLinesRequest,
)
from app.commerce_provider.settings import DEFAULT_AUTH_TOKEN, CommerceProviderSettings
from app.commerce_provider.storage import Database
from app.models.schemas.commerce import (
    CartSnapshot,
    CatalogItem,
    CatalogPage,
    CatalogTaxonomy,
    OrderSnapshot,
    PaymentSnapshot,
    QuoteSnapshot,
)
from app.models.schemas.commerce_taxonomy import RegistryTaxonomy


def _settings(request: Request) -> CommerceProviderSettings:
    settings: CommerceProviderSettings = request.app.state.commerce_settings
    return settings


def get_core(request: Request) -> CommerceProviderCore:
    core: CommerceProviderCore = request.app.state.commerce_core
    return core


def require_bearer(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> None:
    """Enforce the internal Bearer token when one is configured."""

    expected = _settings(request).auth_token
    if not expected:
        return
    scheme, _, credential = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(credential, expected):
        raise ProviderAuthError("缺少或无效的内部服务令牌")


def get_faults(request: Request) -> FaultController:
    controller: FaultController = request.app.state.commerce_faults
    if not _settings(request).faults_enabled:
        raise ProviderNotFoundError("未找到该路径")
    return controller


def _load_taxonomy(path: str) -> RegistryTaxonomy | None:
    if not path:
        return None
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        return RegistryTaxonomy.model_validate(payload)
    except Exception as exc:
        raise RuntimeError(f"Invalid commerce taxonomy snapshot: {path}") from exc


MAX_CATEGORY_FILTER_IDS = 32
MAX_CATEGORY_ID_LENGTH = 64


def _parse_category_ids(raw: str | None) -> set[str] | None:
    """Parse a comma-separated ``category_ids`` query filter.

    Returns ``None`` when the parameter is absent (full taxonomy). The filter
    is bounded (32 ids × 64 chars) so one request can never demand an
    unbounded detail pull; malformed values are definitive refusals.
    """

    if raw is None:
        return None
    ids = {part.strip() for part in raw.split(",")}
    ids.discard("")
    if not ids:
        raise ProviderRejectedError("category_ids 参数无效")
    if len(ids) > MAX_CATEGORY_FILTER_IDS:
        raise ProviderRejectedError("category_ids 数量超出上限")
    if any(len(item) > MAX_CATEGORY_ID_LENGTH for item in ids):
        raise ProviderRejectedError("category_ids 内容无效")
    return ids


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = _settings_from_app(app)
    database = Database(settings.database_path)
    await database.startup()
    faults = FaultController(database, enabled=settings.faults_enabled)
    app.state.commerce_database = database
    app.state.commerce_faults = faults
    core = CommerceProviderCore(
        database,
        VirtualCatalog(_load_taxonomy(settings.taxonomy_path)),
        quote_ttl_seconds=settings.quote_ttl_seconds,
        faults=faults,
        dynamic_category_limit=settings.dynamic_category_limit,
    )
    reloaded = await core.load_dynamic_categories()
    if reloaded:
        logging.getLogger("app.commerce_provider").info(
            "Reloaded %d dynamic categories from storage", reloaded
        )
    app.state.commerce_core = core
    yield


def _settings_from_app(app: FastAPI) -> CommerceProviderSettings:
    settings: CommerceProviderSettings = app.state.commerce_settings
    return settings


def create_app(settings: CommerceProviderSettings | None = None) -> FastAPI:
    """Build the provider service; state other than SQLite lives in-memory."""

    resolved = settings if settings is not None else CommerceProviderSettings()
    app = FastAPI(
        title="Internal Virtual Commerce Provider",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.commerce_settings = resolved

    @app.exception_handler(ProviderError)
    async def provider_error_handler(_: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "commerce-provider"}

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        database: Database = request.app.state.commerce_database
        try:
            await database.read(lambda connection: connection.execute("SELECT 1").fetchone())
        except sqlite3.Error:
            return JSONResponse(
                status_code=503,
                content={"error": {"code": "not_ready", "message": "数据库不可用"}},
            )
        return JSONResponse(content={"status": "ready", "database": "ok"})

    protected = [
        Depends(require_bearer),
    ]

    @app.post("/commerce/search", response_model=CatalogPage, dependencies=protected)
    async def search_catalog(
        body: CatalogSearchRequest,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
    ) -> CatalogPage:
        return await core.search_products(
            body.owner_user_id,
            body.query,
            filters=body.filters,
            sort=body.sort,
            page_size=body.page_size,
            page_token=body.page_token,
        )

    @app.get("/commerce/categories", response_model=CatalogTaxonomy, dependencies=protected)
    async def catalog_categories(
        owner_user_id: OwnerParam,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
        category_ids: str | None = None,
    ) -> CatalogTaxonomy:
        """Read-only authoritative category taxonomy for open-language discovery.

        ``category_ids`` is an optional comma-separated filter: when present,
        only those categories are published (up to 32 ids, each 1-64 chars).
        """

        parsed = _parse_category_ids(category_ids)
        return await core.catalog_categories(owner_user_id, category_ids=parsed)

    @app.post(
        "/commerce/categories/dynamic",
        response_model=MintCategoryResult,
        dependencies=protected,
    )
    async def mint_dynamic_category(
        body: MintCategoryRequest,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
    ) -> MintCategoryResult:
        """Mint a runtime dynamic category (idempotent by ``request_key``).

        Returns ``status="existing"`` with the twin category when the
        normalized name already exists, ``status="minted"`` when a new
        category was created and registered. The category is immediately
        searchable and purchasable in either case.
        """

        return MintCategoryResult.model_validate(
            await core.propose_category(
                body.owner_user_id,
                request_key=body.request_key,
                name=body.name,
                description=body.description,
                domain=body.domain,
                attributes=[(item.name, list(item.allowed_values)) for item in body.attributes],
                variant_key=body.variant_key,
                variant_values=list(body.variant_values),
                base_price_minor=body.base_price_minor,
                source_query=body.source_query,
            )
        )

    @app.get("/commerce/products/{sku_id}", response_model=CatalogItem, dependencies=protected)
    async def get_product(
        sku_id: str,
        owner_user_id: OwnerParam,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
    ) -> CatalogItem:
        return await core.get_product(owner_user_id, sku_id)

    @app.post("/commerce/cart/update", response_model=CartSnapshot, dependencies=protected)
    async def update_cart(
        body: UpdateCartLinesRequest,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
    ) -> CartSnapshot:
        return await core.update_cart_lines(
            body.owner_user_id,
            body.cart_id,
            body.expected_cart_version,
            body.lines,
            body.request_key,
        )

    @app.get("/commerce/carts/{cart_id}", response_model=CartSnapshot, dependencies=protected)
    async def get_cart(
        cart_id: str,
        owner_user_id: OwnerParam,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
    ) -> CartSnapshot:
        return await core.get_cart(owner_user_id, cart_id)

    @app.post("/commerce/checkout", response_model=QuoteSnapshot, dependencies=protected)
    async def checkout(
        body: PrepareCheckoutRequest,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
    ) -> QuoteSnapshot:
        return await core.prepare_checkout(
            body.owner_user_id,
            body.cart_id,
            body.expected_cart_version,
            body.request_key,
        )

    @app.post("/commerce/orders", response_model=OrderSnapshot, dependencies=protected)
    async def create_order(
        body: CreateOrderRequest,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
    ) -> OrderSnapshot:
        return await core.create_order(body.owner_user_id, body.quote_id, body.request_key)

    @app.get("/commerce/orders/{order_id}", response_model=OrderSnapshot, dependencies=protected)
    async def get_order(
        order_id: str,
        owner_user_id: OwnerParam,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
    ) -> OrderSnapshot:
        return await core.get_order(owner_user_id, order_id)

    @app.post("/commerce/payments", response_model=PaymentSnapshot, dependencies=protected)
    async def create_payment(
        body: CreatePaymentRequest,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
    ) -> PaymentSnapshot:
        return await core.create_payment(body.owner_user_id, body.order_id, body.request_key)

    @app.post(
        "/commerce/payments/{payment_id}/sandbox-pay",
        response_model=PaymentSnapshot,
        dependencies=protected,
    )
    async def sandbox_pay(
        payment_id: str,
        body: SandboxPayRequest,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
    ) -> PaymentSnapshot:
        return await core.mock_pay(body.owner_user_id, payment_id, body.request_key)

    @app.get(
        "/commerce/payments/{payment_id}",
        response_model=PaymentSnapshot,
        dependencies=protected,
    )
    async def get_payment(
        payment_id: str,
        owner_user_id: OwnerParam,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
    ) -> PaymentSnapshot:
        return await core.get_status(owner_user_id, payment_id)

    @app.get("/commerce/operations/{request_key}", dependencies=protected)
    async def lookup_operation(
        request_key: str,
        owner_user_id: OwnerParam,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
    ) -> JSONResponse:
        operation = await core.lookup_operation(owner_user_id, request_key)
        if operation is None:
            return JSONResponse(content=None)
        return JSONResponse(content=operation)

    # --- test-only fault injection (hidden unless enabled) -------------------
    fault_guard = [Depends(require_bearer), Depends(get_faults)]

    @app.post("/testing/faults", dependencies=fault_guard)
    async def plan_fault(
        body: FaultPlanRequest,
        faults: Annotated[FaultController, Depends(get_faults)],
    ) -> dict[str, bool]:
        await faults.plan(kind=body.kind, mode=body.mode, target=body.target, times=body.times)
        return {"planned": True}

    @app.get("/testing/faults", dependencies=fault_guard)
    async def list_faults(
        faults: Annotated[FaultController, Depends(get_faults)],
    ) -> dict[str, Any]:
        return {"rules": await faults.list_rules()}

    @app.delete("/testing/faults", dependencies=fault_guard)
    async def clear_faults(
        faults: Annotated[FaultController, Depends(get_faults)],
    ) -> dict[str, int]:
        return {"cleared": await faults.clear()}

    @app.post(
        "/testing/payments/settle",
        response_model=PaymentSnapshot,
        dependencies=fault_guard,
    )
    async def settle_payment(
        body: SettlePaymentRequest,
        core: Annotated[CommerceProviderCore, Depends(get_core)],
    ) -> PaymentSnapshot:
        return await core.settle_payment(body.owner_user_id, body.payment_id, paid=body.paid)

    if resolved.auth_token == DEFAULT_AUTH_TOKEN:
        # Keep the default loud: it exists only for local development.
        logging.getLogger("app.commerce_provider").warning(
            "Commerce provider is using the default dev token; set "
            "COMMERCE_PROVIDER_AUTH_TOKEN in real deployments."
        )

    return app


__all__ = ["create_app"]
