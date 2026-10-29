"""Request-local shopping tools executed by the bounded commerce LLM Agent.

The model owns semantic planning. Handlers own principal binding and typed,
non-semantic transaction checks; there is no phrase-matching authorization.
Tool schemas expose structured search filters/sort/paging and a full multi-line
cart payload, so the provider stays the only source of prices, names and limits.
"""

import math
from collections.abc import Awaitable, Callable
from typing import Any, cast

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select

from app.core.exceptions import NotFoundError, ValidationError
from app.models.database.commerce import PurchaseSession
from app.models.schemas.commerce import (
    CartLineSelection,
    CatalogFilters,
    CommerceView,
)
from app.services.commerce.port import CommerceSort
from app.services.commerce.service import PurchaseService
from app.services.dialogue.tools import ToolDefinition, ToolRegistry

_SORTS = ("relevance", "price_asc", "price_desc", "newest")
_PAGE_SIZE_MAX = 50


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


def _price_minor(value: Any, label: str) -> int | None:
    """Convert a yuan amount from the model into typed minor units."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValidationError(f"{label} must be a non-negative number of yuan")
    return round(float(value) * 100)


def _catalog_filters(args: dict[str, Any]) -> CatalogFilters | None:
    minimum, maximum, in_stock = args.get("min_price"), args.get("max_price"), args.get("in_stock")
    raw_attributes = args.get("attributes")
    if minimum is None and maximum is None and in_stock is None and raw_attributes is None:
        return None
    min_minor, max_minor = (
        _price_minor(minimum, "Minimum price"),
        _price_minor(maximum, "Maximum price"),
    )
    if min_minor is not None and max_minor is not None and min_minor > max_minor:
        raise ValidationError("Minimum price must not exceed maximum price")
    attributes: dict[str, list[str]] = {}
    if raw_attributes is not None:
        if not isinstance(raw_attributes, dict) or len(raw_attributes) > 8:
            raise ValidationError("Catalog attribute filters must be an object with at most 8 keys")
        for key, allowed in raw_attributes.items():
            if not isinstance(key, str) or not key or not isinstance(allowed, list):
                raise ValidationError("Each catalog attribute filter needs a name and string array")
            if (
                not allowed
                or len(allowed) > 8
                or not all(isinstance(value, str) for value in allowed)
            ):
                raise ValidationError("Each catalog attribute filter needs 1 to 8 string values")
            attributes[key] = allowed
    try:
        return CatalogFilters(
            **({"min_price_minor": min_minor} if min_minor is not None else {}),
            **({"max_price_minor": max_minor} if max_minor is not None else {}),
            **({"in_stock": in_stock} if in_stock is not None else {}),
            **({"attributes": attributes} if attributes else {}),
        )
    except PydanticValidationError as exc:
        raise ValidationError(
            "Catalog filters must be prices, a boolean stock flag, and attributes"
        ) from exc


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
        sort = args.get("sort", "relevance")
        if sort not in _SORTS:
            raise ValidationError("Catalog sort must be one of: " + ", ".join(_SORTS))
        page_size = args.get("page_size", 10)
        if type(page_size) is not int or not 1 <= page_size <= _PAGE_SIZE_MAX:
            raise ValidationError(f"Page size must be an integer from 1 to {_PAGE_SIZE_MAX}")
        page_token = args.get("page_token")
        if page_token is not None and (not isinstance(page_token, str) or not page_token):
            raise ValidationError("Page token must be a non-empty cursor string")
        view = await service.start(owner)
        return await service.search(
            owner,
            view.purchase_id,
            query,
            filters=_catalog_filters(args),
            sort=cast(CommerceSort, sort),
            page_size=page_size,
            page_token=page_token,
        )

    async def product(args: dict[str, Any]) -> CommerceView:
        view = await target(args)
        sku = args.get("sku_id")
        if not isinstance(sku, str):
            raise ValidationError("Choose a SKU from the displayed catalog")
        return await service.get_product(owner, view.purchase_id, sku)

    async def cart(args: dict[str, Any]) -> CommerceView:
        view = await target(args)
        version = args.get("expected_version")
        if type(version) is not int:
            raise ValidationError("A purchase version is required")
        raw_lines = args.get("lines")
        if not isinstance(raw_lines, list) or not raw_lines:
            raise ValidationError("A non-empty cart lines array is required")
        lines: list[CartLineSelection] = []
        for entry in raw_lines:
            if not isinstance(entry, dict):
                raise ValidationError("Each cart line must be an object with sku_id and quantity")
            sku = entry.get("sku_id")
            quantity = entry.get("quantity")
            if not isinstance(sku, str) or type(quantity) is not int:
                raise ValidationError(
                    "Each cart line must name one displayed sku_id and a positive integer quantity"
                )
            lines.append(CartLineSelection(sku_id=sku, quantity=quantity))
        return await service.update_cart_lines(owner, view.purchase_id, lines, version)

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
            "Search the mock catalog with a short product query plus structured filters, sort and "
            "paging; never places an order. If the result is empty, reinterpret the customer's "
            "need as a broader standard product-category query and search again before asking a "
            "focused clarification. Invite clarification when the reference or budget is ambiguous.",
            {
                "query": {"type": "string", "maxLength": 500},
                "min_price": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Lower price bound in yuan",
                },
                "max_price": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Upper price bound in yuan",
                },
                "in_stock": {"type": "boolean"},
                "attributes": {
                    "type": "object",
                    "maxProperties": 8,
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 8,
                    },
                    "description": "Exact catalog attribute filters from displayed items",
                },
                "sort": {"type": "string", "enum": list(_SORTS)},
                "page_size": {"type": "integer", "minimum": 1, "maximum": _PAGE_SIZE_MAX},
                "page_token": {
                    "type": "string",
                    "description": "Opaque cursor from a previous result page",
                },
            },
            ["query"],
            search,
        ),
        (
            "get_product",
            "Read one displayed mock SKU.",
            {"purchase_id": {"type": "string"}, "sku_id": {"type": "string"}},
            ["sku_id"],
            product,
        ),
        (
            "update_cart_lines",
            "Replace the whole mock cart with the exact list of SKU quantities the customer "
            "clearly chose; lines absent from the list are removed and the provider re-checks "
            "stock and per-SKU purchase limits. Invite clarification when a product reference, "
            "variant or quantity is ambiguous instead of guessing.",
            {
                "purchase_id": {"type": "string"},
                "lines": {
                    "type": "array",
                    "maxItems": 50,
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku_id": {"type": "string", "maxLength": 64},
                            "quantity": {"type": "integer", "minimum": 1},
                        },
                        "required": ["sku_id", "quantity"],
                        "additionalProperties": False,
                    },
                },
                "expected_version": {"type": "integer", "minimum": 0},
            },
            ["lines", "expected_version"],
            cart,
        ),
        (
            "prepare_checkout",
            "Prepare a quote for the current cart; this never confirms or pays an order.",
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
