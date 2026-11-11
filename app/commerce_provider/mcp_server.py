"""MCP (Model Context Protocol) server over the commerce provider core.

Exposes the same authoritative commerce facts as the internal HTTP facade in
``app.commerce_provider.api``, but through the MCP tool protocol so any MCP
host (or the chat application's agent loop) can drive search and purchase
flows without speaking the private REST dialect. The domain core in
``app.commerce_provider.core`` is shared verbatim — this module is a thin
typed facade, exactly like the HTTP layer, so both surfaces stay
semantically identical by construction.

Design decisions (verified against the ``mcp`` Python SDK 2.3.0 and the MCP
specification 2025-11-25 elicitation page on 2026-11-11):

- Read tools (search/get) answer directly; write tools (cart/quote/order/
  payment) carry a ``Resolve``-annotated confirmation parameter whose
  resolver returns ``Elicit(...)``. The SDK carries the question over
  whichever connection era the client speaks (a live ``elicitation/create``
  request on <=2025-11-25 sessions, a multi-round-trip ``InputRequiredResult``
  on 2026-07-28), so one tool body serves both. Decline or cancel aborts the
  mutation before the core is touched.
- Write tools derive their ``request_key`` structurally from the logical
  operation (owner + target + version + content digest) — never a random
  UUID — so an identical retry replays the original result through the
  core's idempotency receipts instead of double-charging.
- Authentication reuses the ``COMMERCE_PROVIDER_AUTH_TOKEN`` Bearer token
  via the SDK's ``TokenVerifier`` hook: missing or wrong tokens get HTTP
  401 with ``WWW-Authenticate`` before any tool logic runs. An empty token
  disables the check, matching the HTTP facade's semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import (
    AcceptedElicitation,
    Context,
    Elicit,
    ElicitationResult,
    Resolve,
)
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import AnyHttpUrl, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.commerce_provider.catalog import CommerceSort, VirtualCatalog
from app.commerce_provider.core import MAX_CART_LINES, CommerceProviderCore
from app.commerce_provider.errors import ProviderError, ProviderUnknownError
from app.commerce_provider.faults import FaultController
from app.commerce_provider.settings import DEFAULT_AUTH_TOKEN
from app.commerce_provider.storage import Database
from app.models.schemas.commerce import (
    CartLineSelection,
    CartSnapshot,
    CatalogFilters,
    CatalogItem,
    CatalogPage,
    CatalogTaxonomy,
    OrderSnapshot,
    PaymentSnapshot,
    QuoteSnapshot,
)
from app.models.schemas.commerce_taxonomy import RegistryTaxonomy

logger = logging.getLogger("app.commerce_provider.mcp")

MAX_MCP_PAGE_SIZE = 5
"""MCP tools cap result pages at 5 items: agent contexts stay small; the
HTTP facade keeps the wider 1-50 range for the application's own calls."""

MAX_CATEGORY_FILTER_IDS = 32
MAX_CATEGORY_ID_LENGTH = 64
MAX_REQUEST_KEY_LENGTH = 128

# Annotated argument aliases. They must stay module-level: the SDK evaluates
# string annotations (``from __future__ import annotations``) against module
# globals, so function-local aliases would fail signature evaluation.
OwnerParam = Annotated[int, Field(ge=1)]
IdParam = Annotated[str, Field(min_length=1, max_length=64)]
QueryParam = Annotated[str, Field(max_length=500)]


class CommerceMcpSettings(BaseSettings):
    """Configuration for the commerce MCP server.

    Shares the ``COMMERCE_PROVIDER_`` prefix with the HTTP provider so one
    ``COMMERCE_PROVIDER_AUTH_TOKEN`` (and one SQLite database file, by
    default) serves both surfaces; the MCP-specific knobs live under
    ``COMMERCE_PROVIDER_MCP_*``. Unknown sibling variables are ignored on
    purpose — the HTTP provider's settings are validated by its own class.
    """

    model_config = SettingsConfigDict(
        env_prefix="COMMERCE_PROVIDER_", extra="ignore", env_file_encoding="utf-8"
    )

    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(default=8072, ge=1, le=65535)
    mcp_path: str = "/mcp"
    database_path: str = "commerce_provider.sqlite3"
    taxonomy_path: str = ""
    auth_token: str = DEFAULT_AUTH_TOKEN
    quote_ttl_seconds: int = Field(default=300, ge=1)
    faults_enabled: bool = False
    dynamic_category_limit: int = Field(default=200, ge=0, le=1_000)


# --- structural idempotency keys -------------------------------------------


def _digest(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _canonical_lines(lines: Sequence[CartLineSelection]) -> list[dict[str, int | str]]:
    return sorted(
        ({"sku_id": line.sku_id, "quantity": line.quantity} for line in lines),
        key=lambda item: str(item["sku_id"]),
    )


def cart_request_key(
    owner_user_id: int,
    cart_id: str | None,
    expected_cart_version: int | None,
    lines: Sequence[CartLineSelection],
) -> str:
    """Key for one logical cart mutation: same intent -> same key -> replay."""

    key = (
        f"mcp:cart:{owner_user_id}:{cart_id or 'new'}"
        f":v{expected_cart_version or 0}:{_digest(_canonical_lines(lines))}"
    )
    return key[:MAX_REQUEST_KEY_LENGTH]


def quote_request_key(owner_user_id: int, cart_id: str, expected_cart_version: int) -> str:
    return f"mcp:quote:{owner_user_id}:{cart_id}:v{expected_cart_version}"[:MAX_REQUEST_KEY_LENGTH]


def order_request_key(owner_user_id: int, quote_id: str) -> str:
    return f"mcp:order:{owner_user_id}:{quote_id}"[:MAX_REQUEST_KEY_LENGTH]


def payment_init_request_key(owner_user_id: int, order_id: str) -> str:
    return f"mcp:pay-init:{owner_user_id}:{order_id}"[:MAX_REQUEST_KEY_LENGTH]


def payment_request_key(owner_user_id: int, payment_id: str) -> str:
    return f"mcp:pay:{owner_user_id}:{payment_id}"[:MAX_REQUEST_KEY_LENGTH]


# --- confirmation model ------------------------------------------------------


class WriteConfirmation(BaseModel):
    """What the client's user answers when a write tool asks permission."""

    confirmed: bool


def _ask(message: str) -> Callable[[], Elicit[WriteConfirmation]]:
    def resolver() -> Elicit[WriteConfirmation]:
        return Elicit(message, WriteConfirmation)

    return resolver


def _accepted(confirmation: ElicitationResult[WriteConfirmation]) -> bool:
    return isinstance(confirmation, AcceptedElicitation) and confirmation.data.confirmed


def _refuse(confirmation: ElicitationResult[WriteConfirmation]) -> ToolError:
    action = confirmation.action
    return ToolError(f"user_declined: 用户拒绝了该写操作（{action}），未执行任何变更")


def _provider_error(exc: ProviderError) -> ToolError:
    if isinstance(exc, ProviderUnknownError):
        return ToolError(
            f"[{exc.code}] 操作结果未确认：请用完全相同的参数重试同一工具，"
            "幂等键会回放原结果，不会重复扣款"
        )
    return ToolError(f"[{exc.code}] {exc.message}")


# --- authentication ----------------------------------------------------------


class StaticTokenVerifier:
    """Verify the shared internal Bearer token (``TokenVerifier`` protocol)."""

    def __init__(self, expected: str) -> None:
        self._expected = expected

    async def verify_token(self, token: str) -> AccessToken | None:
        if hmac.compare_digest(token, self._expected):
            return AccessToken(token=token, client_id="commerce-mcp-internal", scopes=[])
        return None


# --- composite view -----------------------------------------------------------


class CatalogDiscovery(BaseModel):
    """One-call orientation view: authoritative categories plus first products."""

    query: str
    categories: CatalogTaxonomy
    products: CatalogPage


def _load_taxonomy(path: str) -> RegistryTaxonomy | None:
    if not path:
        return None
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        return RegistryTaxonomy.model_validate(payload)
    except Exception as exc:  # pragma: no cover - startup-time guard
        raise RuntimeError(f"Invalid commerce taxonomy snapshot: {path}") from exc


# --- server construction ------------------------------------------------------


def create_mcp_server(
    settings: CommerceMcpSettings | None = None,
    *,
    core: CommerceProviderCore | None = None,
    faults: FaultController | None = None,
) -> MCPServer[dict[str, Any]]:
    """Build the commerce MCP server.

    ``core``/``faults`` inject an already-started domain core (tests, or an
    embedding process); when omitted the server opens its own SQLite database
    on lifespan startup, mirroring the HTTP facade.
    """

    resolved = settings if settings is not None else CommerceMcpSettings()

    @asynccontextmanager
    async def lifespan(_: MCPServer[dict[str, Any]]) -> AsyncGenerator[dict[str, Any], None]:
        if core is not None:
            yield {"core": core}
            return
        database = Database(resolved.database_path)
        await database.startup()
        controller = (
            faults
            if faults is not None
            else FaultController(database, enabled=resolved.faults_enabled)
        )
        built = CommerceProviderCore(
            database,
            VirtualCatalog(_load_taxonomy(resolved.taxonomy_path)),
            quote_ttl_seconds=resolved.quote_ttl_seconds,
            faults=controller,
            dynamic_category_limit=resolved.dynamic_category_limit,
        )
        reloaded = await built.load_dynamic_categories()
        if reloaded:
            logger.info("Reloaded %d dynamic categories from storage", reloaded)
        yield {"core": built}

    verifier = StaticTokenVerifier(resolved.auth_token) if resolved.auth_token else None
    base_url = f"http://{resolved.mcp_host}:{resolved.mcp_port}/"
    mcp: MCPServer[dict[str, Any]] = MCPServer(
        "commerce-provider",
        title="Internal Virtual Commerce Provider (MCP)",
        instructions=(
            "内部虚拟电商服务：只读工具直接调用；写工具（update_cart / prepare_quote / "
            "create_order / sandbox_pay）会先向用户请求确认，拒绝即中止。所有写操作幂等，"
            "结果未确认时用相同参数重试即可回放原结果。"
        ),
        lifespan=lifespan,
        token_verifier=verifier,
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(base_url),
            resource_server_url=AnyHttpUrl(base_url),
            # The static verifier authenticates the exact shared token; there
            # is no audience/resource claim to cross-check.
            validate_token_resource=False,
        )
        if verifier
        else None,
    )

    def _core_from(ctx: Context) -> CommerceProviderCore:
        core_: CommerceProviderCore = ctx.request_context.lifespan_context["core"]
        return core_

    def _category_ids(raw: list[str] | None) -> set[str] | None:
        if raw is None:
            return None
        ids = {item.strip() for item in raw}
        ids.discard("")
        if not ids:
            raise ToolError("[invalid_request] category_ids 参数无效")
        if len(ids) > MAX_CATEGORY_FILTER_IDS:
            raise ToolError("[invalid_request] category_ids 数量超出上限")
        if any(len(item) > MAX_CATEGORY_ID_LENGTH for item in ids):
            raise ToolError("[invalid_request] category_ids 内容无效")
        return ids

    # --- read tools -----------------------------------------------------------

    @mcp.tool()
    async def search_products(
        query: QueryParam,
        owner_user_id: OwnerParam,
        ctx: Context,
        page_size: Annotated[int, Field(ge=1, le=MAX_MCP_PAGE_SIZE)] = MAX_MCP_PAGE_SIZE,
        page_token: str | None = None,
        sort: CommerceSort = "relevance",
        filters: CatalogFilters | None = None,
    ) -> CatalogPage:
        """按开放语言搜索商品目录，返回一页商品（最多 5 条）。"""
        try:
            return await _core_from(ctx).search_products(
                owner_user_id,
                query,
                filters=filters,
                sort=sort,
                page_size=page_size,
                page_token=page_token,
            )
        except ProviderError as exc:
            raise _provider_error(exc) from exc

    @mcp.tool()
    async def get_product(sku_id: IdParam, owner_user_id: OwnerParam, ctx: Context) -> CatalogItem:
        """读取单个 SKU 的权威商品事实（价格、属性、库存）。"""
        try:
            return await _core_from(ctx).get_product(owner_user_id, sku_id)
        except ProviderError as exc:
            raise _provider_error(exc) from exc

    @mcp.tool()
    async def get_categories(
        owner_user_id: OwnerParam,
        ctx: Context,
        category_ids: list[str] | None = None,
    ) -> CatalogTaxonomy:
        """读取权威类目体系（id/名称/属性与可选值），可按 id 过滤。"""
        try:
            return await _core_from(ctx).catalog_categories(
                owner_user_id, category_ids=_category_ids(category_ids)
            )
        except ProviderError as exc:
            raise _provider_error(exc) from exc

    @mcp.tool()
    async def discover_catalog(
        query: QueryParam,
        owner_user_id: OwnerParam,
        ctx: Context,
    ) -> CatalogDiscovery:
        """一次调用完成开场定位：完整权威类目 + 该查询的前 3 条商品。"""
        core_ = _core_from(ctx)
        try:
            categories = await core_.catalog_categories(owner_user_id)
            products = await core_.search_products(
                owner_user_id, query, page_size=3, sort="relevance"
            )
        except ProviderError as exc:
            raise _provider_error(exc) from exc
        return CatalogDiscovery(query=query, categories=categories, products=products)

    # --- write tools (confirmation + structural idempotency) ------------------

    @mcp.tool()
    async def update_cart(
        owner_user_id: OwnerParam,
        lines: Annotated[list[CartLineSelection], Field(min_length=0, max_length=MAX_CART_LINES)],
        ctx: Context,
        cart_id: IdParam | None = None,
        expected_cart_version: Annotated[int, Field(ge=1)] | None = None,
        confirmation: Annotated[
            ElicitationResult[WriteConfirmation], Resolve(_ask("确认提交购物车变更？"))
        ] = None,  # type: ignore[assignment]
    ) -> CartSnapshot:
        """整体替换购物车明细（写操作，需用户确认；同参数重试幂等回放）。"""
        if not _accepted(confirmation):
            raise _refuse(confirmation)
        try:
            return await _core_from(ctx).update_cart_lines(
                owner_user_id,
                cart_id,
                expected_cart_version,
                list(lines),
                cart_request_key(owner_user_id, cart_id, expected_cart_version, lines),
            )
        except ProviderError as exc:
            raise _provider_error(exc) from exc

    @mcp.tool()
    async def prepare_quote(
        owner_user_id: OwnerParam,
        cart_id: IdParam,
        expected_cart_version: Annotated[int, Field(ge=1)],
        ctx: Context,
        confirmation: Annotated[
            ElicitationResult[WriteConfirmation], Resolve(_ask("确认结算当前购物车并生成报价？"))
        ] = None,  # type: ignore[assignment]
    ) -> QuoteSnapshot:
        """对购物车当前版本生成报价（写操作，需用户确认；报价有有效期）。"""
        if not _accepted(confirmation):
            raise _refuse(confirmation)
        try:
            return await _core_from(ctx).prepare_checkout(
                owner_user_id,
                cart_id,
                expected_cart_version,
                quote_request_key(owner_user_id, cart_id, expected_cart_version),
            )
        except ProviderError as exc:
            raise _provider_error(exc) from exc

    @mcp.tool()
    async def create_order(
        owner_user_id: OwnerParam,
        quote_id: IdParam,
        ctx: Context,
        confirmation: Annotated[
            ElicitationResult[WriteConfirmation], Resolve(_ask("确认依据该报价创建订单？"))
        ] = None,  # type: ignore[assignment]
    ) -> OrderSnapshot:
        """将报价落为订单（写操作，需用户确认；同报价重试幂等回放）。"""
        if not _accepted(confirmation):
            raise _refuse(confirmation)
        try:
            return await _core_from(ctx).create_order(
                owner_user_id, quote_id, order_request_key(owner_user_id, quote_id)
            )
        except ProviderError as exc:
            raise _provider_error(exc) from exc

    @mcp.tool()
    async def sandbox_pay(
        owner_user_id: OwnerParam,
        ctx: Context,
        payment_id: IdParam | None = None,
        order_id: IdParam | None = None,
        confirmation: Annotated[
            ElicitationResult[WriteConfirmation], Resolve(_ask("确认在沙箱中支付该订单？"))
        ] = None,  # type: ignore[assignment]
    ) -> PaymentSnapshot:
        """沙箱支付（写操作，需用户确认）。

        传 ``payment_id`` 支付既有支付单；传 ``order_id`` 时先为订单开支付单
        再支付（两步各自幂等）。两个参数必须恰好提供一个。
        """
        if not _accepted(confirmation):
            raise _refuse(confirmation)
        if (payment_id is None) == (order_id is None):
            raise ToolError("[invalid_request] payment_id 与 order_id 必须恰好提供一个")
        core_ = _core_from(ctx)
        try:
            target = payment_id
            if target is None and order_id is not None:
                created = await core_.create_payment(
                    owner_user_id, order_id, payment_init_request_key(owner_user_id, order_id)
                )
                target = created.payment_id
            assert target is not None  # narrowed above
            return await core_.mock_pay(
                owner_user_id, target, payment_request_key(owner_user_id, target)
            )
        except ProviderError as exc:
            raise _provider_error(exc) from exc

    if resolved.auth_token == DEFAULT_AUTH_TOKEN:
        logger.warning(
            "Commerce MCP server is using the default dev token; set "
            "COMMERCE_PROVIDER_AUTH_TOKEN in real deployments."
        )

    return mcp


def main(argv: list[str] | None = None) -> None:
    """Entry point: ``python -m app.commerce_provider.mcp_server``."""

    parser = argparse.ArgumentParser(
        prog="python -m app.commerce_provider.mcp_server",
        description="Run the commerce provider MCP server (Streamable HTTP)",
    )
    parser.add_argument(
        "--host", dest="mcp_host", default=None, help="bind address (default 127.0.0.1)"
    )
    parser.add_argument(
        "--port", dest="mcp_port", type=int, default=None, help="bind port (default 8072)"
    )
    parser.add_argument(
        "--path", dest="mcp_path", default=None, help="endpoint path (default /mcp)"
    )
    parser.add_argument(
        "--database", dest="database_path", default=None, help="SQLite database file path"
    )
    parser.add_argument(
        "--token", dest="auth_token", default=None, help="Bearer token (empty disables auth)"
    )
    args = parser.parse_args(argv)

    settings = CommerceMcpSettings()
    updates: dict[str, object] = {
        key: value
        for key, value in {
            "mcp_host": args.mcp_host,
            "mcp_port": args.mcp_port,
            "mcp_path": args.mcp_path,
            "database_path": args.database_path,
            "auth_token": args.auth_token,
        }.items()
        if value is not None
    }
    if updates:
        settings = settings.model_copy(update=updates)

    server = create_mcp_server(settings)
    server.run(
        transport="streamable-http",
        host=settings.mcp_host,
        port=settings.mcp_port,
        streamable_http_path=settings.mcp_path,
    )


if __name__ == "__main__":
    main()
