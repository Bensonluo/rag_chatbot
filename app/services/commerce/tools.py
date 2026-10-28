"""Request-local shopping tools executed by the bounded commerce LLM Agent.

The model owns semantic planning. Handlers own principal binding and typed,
non-semantic transaction checks; there is no phrase-matching authorization.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import select

from app.core.exceptions import NotFoundError, ValidationError
from app.models.database.commerce import PurchaseSession
from app.models.schemas.commerce import CommerceView
from app.services.commerce.service import PurchaseService
from app.services.dialogue.tools import ToolDefinition, ToolRegistry


async def current_purchase(service: PurchaseService, owner: int) -> CommerceView | None:
    active = await service.active(owner)
    if active is not None:
        return active
    async with service.sessions() as db:
        identifier = await db.scalar(
            select(PurchaseSession.id)
            .where(PurchaseSession.owner_user_id == owner)
            .order_by(PurchaseSession.created_at.desc(), PurchaseSession.id.desc())
            .limit(1)
        )
    return None if identifier is None else await service.status(owner, identifier)


def create_commerce_tools(
    service: PurchaseService,
    owner: int,
    views: list[CommerceView],
    current: CommerceView | None,
) -> ToolRegistry:
    """Bind one verified principal in a local closure, never a model argument."""
    if type(owner) is not int or owner <= 0:
        raise ValidationError("A verified visitor identity is required")
    registry = ToolRegistry()

    async def target(args: dict[str, Any], *, create: bool = False) -> CommerceView:
        identifier = args.get("purchase_id")
        if identifier is not None:
            if not isinstance(identifier, str):
                raise ValidationError("Invalid purchase identifier")
            return await service.status(owner, identifier)
        result = await (service.start(owner) if create else current_purchase(service, owner))
        if result is None:
            raise NotFoundError("Purchase")
        return result

    async def search(args: dict[str, Any]) -> CommerceView:
        query = args.get("query", "")
        if not isinstance(query, str) or len(query) > 500:
            raise ValidationError("Search query must be a bounded string")
        view = await service.start(owner)
        return await service.search(owner, view.purchase_id, query)

    async def product(args: dict[str, Any]) -> CommerceView:
        view = await target(args)
        sku = args.get("sku_id")
        if not isinstance(sku, str):
            raise ValidationError("Choose a SKU from the displayed catalog")
        return await service.get_product(owner, view.purchase_id, sku)

    async def cart(args: dict[str, Any]) -> CommerceView:
        view = await target(args)
        sku, quantity, version = (
            args.get("sku_id"),
            args.get("quantity"),
            args.get("expected_version"),
        )
        if not isinstance(sku, str) or type(quantity) is not int or type(version) is not int:
            raise ValidationError("SKU, integer quantity and purchase version are required")
        return await service.set_cart_line(owner, view.purchase_id, sku, quantity, version)

    async def quote(args: dict[str, Any]) -> CommerceView:
        view = await target(args)
        version = args.get("expected_version")
        if type(version) is not int:
            raise ValidationError("A purchase version is required")
        return await service.prepare_checkout(owner, view.purchase_id, version)

    async def status(args: dict[str, Any]) -> CommerceView:
        return await target(args)

    async def commit(args: dict[str, Any]) -> CommerceView:
        del args
        if current is None or current.stage != "quoted":
            raise ValidationError("A current quoted purchase is required before ordering")
        return await service.commit_checkout(owner, current.purchase_id, current.version)

    async def pay(args: dict[str, Any]) -> CommerceView:
        del args
        if current is None or current.stage != "awaiting_payment":
            raise ValidationError("An unpaid mock order is required before payment")
        return await service.mock_pay(owner, current.purchase_id, current.version)

    definitions: list[
        tuple[
            str, str, dict[str, Any], list[str], Callable[[dict[str, Any]], Awaitable[CommerceView]]
        ]
    ] = [
        (
            "search_products",
            "Search the mock product catalog without placing an order.",
            {"query": {"type": "string", "maxLength": 500}},
            ["query"],
            search,
        ),
        (
            "get_product",
            "Read a displayed mock SKU.",
            {"purchase_id": {"type": "string"}, "sku_id": {"type": "string"}},
            ["sku_id"],
            product,
        ),
        (
            "set_cart_line",
            "Set one SKU and quantity only when the customer clearly selected them.",
            {
                "purchase_id": {"type": "string"},
                "sku_id": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 1, "maximum": 10},
                "expected_version": {"type": "integer", "minimum": 0},
            },
            ["sku_id", "quantity", "expected_version"],
            cart,
        ),
        (
            "prepare_checkout",
            "Prepare a quote for review; this never confirms or pays an order.",
            {
                "purchase_id": {"type": "string"},
                "expected_version": {"type": "integer", "minimum": 0},
            },
            ["expected_version"],
            quote,
        ),
        (
            "purchase_status",
            "Read the customer's current or most recent mock purchase.",
            {"purchase_id": {"type": "string"}},
            [],
            status,
        ),
    ]
    if current is not None and current.stage == "quoted":
        definitions.append(
            (
                "commit_checkout",
                "Create the currently displayed mock order only when the latest user message explicitly and unambiguously confirms this quote.",
                {},
                [],
                commit,
            )
        )
    if current is not None and current.stage == "awaiting_payment":
        definitions.append(
            (
                "mock_pay",
                "Settle the currently displayed mock payment only when the latest user message explicitly and unambiguously confirms payment.",
                {},
                [],
                pay,
            )
        )
    for name, description, properties, required, handler in definitions:

        async def execute(
            args: dict[str, Any],
            call: Callable[[dict[str, Any]], Awaitable[CommerceView]] = handler,
        ) -> dict[str, Any]:
            # The closure is the authority even if a model supplies user_id.
            result = await call(args)
            views.append(result)
            return {"commerce_ui": result.model_dump(mode="json"), "mode": "mock"}

        registry.register(
            ToolDefinition(
                name="commerce_" + name,
                intent="commerce:" + name,
                description=description,
                required_slots=required,
                handler=execute,
                parameters_schema={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            )
        )
    return registry
