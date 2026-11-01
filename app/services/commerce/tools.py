"""Request-local shopping tools executed by the bounded commerce LLM Agent.

The model owns semantic planning. Handlers own principal binding and typed,
non-semantic transaction checks; there is no phrase-matching authorization.
Tool schemas expose structured search filters/sort/paging and a full multi-line
cart payload, so the provider stays the only source of prices, names and limits.
"""

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select

from app.core.exceptions import NotFoundError, ValidationError
from app.models.database.commerce import PurchaseSession
from app.models.schemas.commerce import (
    CartLineSelection,
    CatalogBlock,
    CatalogFilters,
    CatalogItem,
    CommerceView,
)
from app.services.commerce.coverage import CommerceCoverageRecorder
from app.services.commerce.graph_service import CommerceGraphService
from app.services.commerce.port import CommerceSort
from app.services.commerce.service import PurchaseService
from app.services.dialogue.tools import ToolDefinition, ToolRegistry
from app.services.observability.trace_events import emit_trace

_SORTS = ("relevance", "price_asc", "price_desc", "newest")
logger = logging.getLogger(__name__)
# One guest-facing turn keeps at most five typed candidates, even when the
# model searches several provider categories. Provider paging remains larger;
# this is the Agent presentation contract, not a provider catalog limit.
_AGENT_PAGE_SIZE_MAX = 5
_ZERO_RESULT_CATEGORY_SEARCH_LIMIT = 2
_PAGE_SIZE_MAX = 50
_VISIBLE_CANDIDATES_MAX = 5


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
    commerce_graph: CommerceGraphService | None = None,
    coverage_recorder: CommerceCoverageRecorder | None = None,
) -> ToolRegistry:
    """Bind one verified principal in a local closure, never a model argument."""

    if type(owner) is not int or owner <= 0:
        raise ValidationError("A verified visitor identity is required")
    registry = ToolRegistry()
    catalog_batches: dict[str, list[CatalogItem]] = {}
    catalog_total_estimates: dict[str, int] = {}
    seen_catalog_skus: set[str] = set()
    search_batches = 0
    zero_result_queries: set[str] = set()
    # Minted this turn: their names may always be (re-)searched even after
    # the zero-result budget is spent — presenting their SKUs is the whole
    # point of minting.
    minted_category_names: set[str] = set()
    graph_upsert_tasks: set[Any] = set()

    def balanced_catalog_candidates() -> list[CatalogItem]:
        """Round-robin provider rankings across distinct search queries."""

        selected: list[CatalogItem] = []
        depth = 0
        while len(selected) < _VISIBLE_CANDIDATES_MAX:
            added = False
            for batch in catalog_batches.values():
                if depth >= len(batch):
                    continue
                selected.append(batch[depth])
                added = True
                if len(selected) == _VISIBLE_CANDIDATES_MAX:
                    break
            if not added:
                break
            depth += 1
        return selected

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
        nonlocal catalog_total_estimates, search_batches, zero_result_queries
        query = args.get("query", "")
        if not isinstance(query, str) or len(query) > 500:
            raise ValidationError("Search query must be a bounded string")
        sort = args.get("sort", "relevance")
        if sort not in _SORTS:
            raise ValidationError("Catalog sort must be one of: " + ", ".join(_SORTS))
        requested_page_size = args.get("page_size", _AGENT_PAGE_SIZE_MAX)
        if type(requested_page_size) is not int or not 1 <= requested_page_size <= _PAGE_SIZE_MAX:
            raise ValidationError(f"Page size must be an integer from 1 to {_PAGE_SIZE_MAX}")
        page_size = min(requested_page_size, _AGENT_PAGE_SIZE_MAX)
        page_token = args.get("page_token")
        if page_token is not None and (not isinstance(page_token, str) or not page_token):
            raise ValidationError("Page token must be a non-empty cursor string")
        normalized_query = " ".join(query.strip().lower().split())
        if (
            page_token is None
            and len(zero_result_queries) >= _ZERO_RESULT_CATEGORY_SEARCH_LIMIT
            and normalized_query not in zero_result_queries
            and normalized_query not in minted_category_names
        ):
            raise ValidationError(
                "Zero-result category searches are exhausted for this turn: "
                "ask one focused clarification or record the unsupported domain"
            )
        view = await service.start(owner)
        result = await service.search(
            owner,
            view.purchase_id,
            query,
            filters=_catalog_filters(args),
            sort=cast(CommerceSort, sort),
            page_size=page_size,
            page_token=page_token,
        )
        result_count = sum(
            len(block.items) for block in result.blocks if isinstance(block, CatalogBlock)
        )
        if page_token is None and result_count == 0 and normalized_query:
            zero_result_queries.add(normalized_query)
        if coverage_recorder is not None and page_token is None:
            try:
                await coverage_recorder.record_search(
                    query=query,
                    result_count=result_count,
                    owner=owner,
                )
            except Exception as exc:  # pragma: no cover - fail-open telemetry
                logger.warning("Commerce coverage recording failed: %s", exc)
        search_batches += 1
        batch: list[CatalogItem] = []
        for block in result.blocks:
            if not isinstance(block, CatalogBlock):
                continue
            catalog_total_estimates[query] = max(
                catalog_total_estimates.get(query, 0),
                block.total_estimate,
                len(block.items),
            )
            for item in block.items:
                if item.sku_id in seen_catalog_skus:
                    continue
                seen_catalog_skus.add(item.sku_id)
                batch.append(item)
        # Exact query identity groups provider pages back into one ordered
        # category batch. Distinct queries still round-robin below.
        catalog_batches.setdefault(query, []).extend(batch)
        visible_candidates = balanced_catalog_candidates()
        if search_batches == 1:
            return result
        # A single open need can span several standard categories. The model
        # chooses the categories; this presentation-only round-robin keeps the
        # guest-facing typed set bounded while preserving category diversity.
        # Per-query page tokens cannot be combined, so a merged page has none.
        return result.model_copy(
            update={
                "blocks": [
                    CatalogBlock(
                        items=visible_candidates,
                        total_estimate=max(
                            sum(catalog_total_estimates.values()), len(visible_candidates)
                        ),
                    ),
                    *(block for block in result.blocks if not isinstance(block, CatalogBlock)),
                ]
            }
        )

    async def discover(args: dict[str, Any]) -> dict[str, Any]:
        # Read-only provider facts for the model's semantic mapping; no
        # purchase session is started and no view is staged or mutated.
        # Two levels keep the payload need-sized: a compact id/name index
        # first, then a bounded detail pull only for chosen categories.
        raw_ids = args.get("category_ids")
        if raw_ids is None:
            taxonomy = await service.discover_catalog(owner)
            return {
                "catalog_category_index": [
                    {"category_id": category.category_id, "name": category.name}
                    for category in taxonomy.categories
                ],
                "category_count": len(taxonomy.categories),
                "detail_hint": (
                    "Pass category_ids (1-16 ids from this index) to pull full category "
                    "detail: description and attribute vocabularies with allowed values."
                ),
                "mode": "mock",
            }
        if (
            not isinstance(raw_ids, list)
            or not 1 <= len(raw_ids) <= 16
            or any(
                not isinstance(item, str) or not item.strip() or len(item) > 64 for item in raw_ids
            )
        ):
            raise ValidationError(
                "category_ids must be a list of 1 to 16 non-empty category id strings"
            )
        taxonomy = await service.discover_catalog(
            owner, category_ids={item.strip() for item in raw_ids}
        )
        return {
            "catalog_categories": taxonomy.model_dump(mode="json"),
            "mode": "mock",
        }

    async def graph_context(args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query", "")
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValidationError("Graph context query must be a bounded non-empty string")
        top_k = args.get("top_k", 3)
        if type(top_k) is not int or not 1 <= top_k <= 5:
            raise ValidationError("Graph context top_k must be an integer from 1 to 5")
        assert commerce_graph is not None
        facts = await commerce_graph.search(query, top_k=top_k)
        return {
            "commerce_graph": {
                "facts": [fact.__dict__ for fact in facts],
                "semantic_only": True,
                "authority": "semantic-context",
            },
            "mode": "semantic-context",
        }

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
            "paging; never places an order. For an unknown brand/product name, possible typo or "
            "reference whose use category is unclear, ask one focused clarification before "
            "searching. For vague, novel or open-ended needs, and whenever a search returns "
            "nothing, call discover_catalog first and use its standard category names as "
            "queries. At most two distinct standard categories may return zero results in one "
            "turn; then stop searching and either ask one focused clarification or record the "
            "unsupported domain. Do not translate a whole-basket budget into arbitrary "
            "per-category price caps: search each category without max_price unless the user "
            "supplied that cap, then control the total while composing bundles. A zero after "
            "filters means retry without those non-user filters before declaring a category "
            "unavailable.",
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
                    "description": (
                        "Upper price bound in yuan; use only for an explicit "
                        "user per-item/category cap, never to allocate a whole-basket budget"
                    ),
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
                "page_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _AGENT_PAGE_SIZE_MAX,
                    "default": 3,
                    "description": "Candidates per category page; the turn displays at most five total",
                },
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

    async def propose_category(args: dict[str, Any]) -> dict[str, Any]:
        nonlocal minted_category_names
        name = args.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 32:
            raise ValidationError("Category name must be 1 to 32 characters")
        description = args.get("description", "")
        if not isinstance(description, str) or len(description) > 200:
            raise ValidationError("Category description must be at most 200 characters")
        domain = args.get("domain")
        if not isinstance(domain, str) or not domain.strip() or len(domain) > 32:
            raise ValidationError("A domain from the fixed domain frame is required")
        raw_attributes = args.get("attributes")
        if not isinstance(raw_attributes, list) or not 1 <= len(raw_attributes) <= 5:
            raise ValidationError("Category attributes must be a list of 1 to 5 axes")
        attributes: list[tuple[str, list[str]]] = []
        for entry in raw_attributes:
            if not isinstance(entry, dict):
                raise ValidationError("Each category attribute needs a name and values")
            attribute_name = entry.get("name")
            values = entry.get("values")
            if (
                not isinstance(attribute_name, str)
                or not attribute_name.strip()
                or len(attribute_name) > 16
            ):
                raise ValidationError("Each attribute name must be 1 to 16 characters")
            if (
                not isinstance(values, list)
                or not 1 <= len(values) <= 4
                or not all(isinstance(value, str) and value.strip() for value in values)
            ):
                raise ValidationError(f"Attribute {attribute_name} needs 1 to 4 non-empty values")
            attributes.append((attribute_name.strip(), [value.strip() for value in values]))
        variant_key = args.get("variant_key")
        if not isinstance(variant_key, str) or not variant_key.strip() or len(variant_key) > 16:
            raise ValidationError("Variant axis name must be 1 to 16 characters")
        variant_values = args.get("variant_values")
        if (
            not isinstance(variant_values, list)
            or len(variant_values) != 4
            or not all(isinstance(value, str) and value.strip() for value in variant_values)
        ):
            raise ValidationError("The variant axis must provide exactly 4 non-empty values")
        base_price_minor = _price_minor(args.get("base_price_yuan"), "Base price")
        if base_price_minor is None or not 100 <= base_price_minor <= 1_000_000:
            raise ValidationError("Base price must be between 1 and 10000 yuan")
        source_query = args.get("source_query", "")
        if not isinstance(source_query, str):
            source_query = ""
        raw_styles = args.get("brand_styles")
        brand_styles: list[dict[str, str]] | None = None
        if raw_styles is not None:
            # Shape gate here; shelf-quality rules (brand count, dedupe,
            # blocklist) live in the provider's deterministic validator.
            if not isinstance(raw_styles, list) or not all(
                isinstance(entry, dict)
                and isinstance(entry.get("brand"), str)
                and isinstance(entry.get("series"), str)
                and entry["brand"].strip()
                and entry["series"].strip()
                for entry in raw_styles
            ):
                raise ValidationError(
                    "brand_styles must be a list of {brand, series} string pairs"
                )
            brand_styles = [
                {"brand": entry["brand"].strip()[:16], "series": entry["series"].strip()[:16]}
                for entry in raw_styles
            ]

        outcome = await service.propose_category(
            owner,
            name=name.strip(),
            description=description.strip(),
            domain=domain.strip(),
            attributes=attributes,
            variant_key=variant_key.strip(),
            variant_values=[value.strip() for value in variant_values],
            base_price_minor=base_price_minor,
            source_query=source_query,
            brand_styles=brand_styles,
        )
        normalized = " ".join(outcome.category.name.strip().lower().split())
        minted_category_names.add(normalized)
        if outcome.status == "minted" and commerce_graph is not None:

            async def traced_upsert() -> None:
                # Fire-and-forget semantic patch: purchases never depend on
                # the graph; the next full rebuild remains the convergence
                # baseline. The task inherits the request's trace sink via
                # asyncio context copy, so the background work stays visible
                # in the live waterfall instead of hiding inside whatever
                # span runs next. Emission is never-fail and a no-op
                # without an active sink (startup, non-stream paths).
                started = time.perf_counter()
                emit_trace("cs.tool", status="start", tool="commerce_graph_upsert", background=True)
                ok = await commerce_graph.upsert_category(outcome.category)
                emit_trace(
                    "cs.tool",
                    status="end",
                    ms=(time.perf_counter() - started) * 1000,
                    tool="commerce_graph_upsert",
                    ok=ok,
                    background=True,
                )

            upsert_task = asyncio.create_task(traced_upsert())
            graph_upsert_tasks.add(upsert_task)
            upsert_task.add_done_callback(graph_upsert_tasks.discard)
        return {
            "mint_result": outcome.model_dump(mode="json"),
            "next_step": (
                "Immediately call commerce_search_products with the returned "
                "category name to present its purchasable SKUs."
            ),
            "mode": "mock",
        }

    # Discovery is pure provider facts: it stages no purchase view, so it is
    # registered outside the CommerceView-producing loop.
    registry.register(
        ToolDefinition(
            name="commerce_discover_catalog",
            intent="commerce:discover_catalog",
            description=(
                "Discover the provider's authoritative product categories in two steps. "
                "Without arguments, get the compact index of every category id and "
                "standard name; use it before searching when the customer's need is "
                "vague, novel or open-ended, when a search returned no results, or "
                "whenever the standard category names are unknown. Then pass "
                "category_ids (1-16 ids from the index) to pull full category detail: "
                "descriptions and attribute vocabularies with allowed values for "
                "building typed search filters. Read-only: never changes cart, order "
                "or payment state. Decide yourself how an open need maps to standard "
                "category names. If no related category exists, say so or ask one "
                "focused question rather than searching. At most two chosen categories "
                "may return zero results before the turn must clarify or record the "
                "unsupported domain."
            ),
            required_slots=[],
            handler=discover,
            parameters_schema={
                "type": "object",
                "properties": {
                    "category_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 64},
                        "minItems": 1,
                        "maxItems": 16,
                        "description": (
                            "Category ids from the compact index; omit to list all "
                            "categories. Each id pull returns full attribute detail."
                        ),
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        )
    )

    registry.register(
        ToolDefinition(
            name="commerce_propose_category",
            intent="commerce:propose_category",
            description=(
                "Mint one new dynamic product category after discovery and "
                "bounded searches prove the catalog lacks an everyday physical "
                "good the customer clearly wants. Propose the definition: a "
                "concise standard category name, a one-sentence description, "
                "one domain from the fixed frame announced in the system "
                "prompt (use the general fallback domain only when nothing "
                "fits), 1 to 5 attribute axes with up to 4 values each, a "
                "variant axis with exactly 4 values, and a realistic base "
                "price in yuan. The provider deduplicates by normalized name "
                "(an existing twin is returned unchanged), enforces the "
                "blocklist and shape rules, and may refuse when the dynamic "
                "cap is reached. Never propose prescription drugs, weapons, "
                "adult goods, virtual services, accounts, living animals or "
                "gambling; refuse those needs yourself instead. After minting "
                "(or an existing result), immediately re-search the returned "
                "category name to present its purchasable SKUs. Catalog facts "
                "only: no cart, order or payment state changes."
            ),
            required_slots=[
                "name",
                "description",
                "domain",
                "attributes",
                "variant_key",
                "variant_values",
                "base_price_yuan",
            ],
            handler=propose_category,
            parameters_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 32,
                        "description": "Standard category name, e.g. 糖果巧克力",
                    },
                    "description": {
                        "type": "string",
                        "maxLength": 200,
                        "description": "One sentence describing what this category covers",
                    },
                    "domain": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 32,
                        "description": "One domain from the fixed frame in the system prompt",
                    },
                    "attributes": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "minLength": 1, "maxLength": 16},
                                "values": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 4,
                                    "items": {"type": "string", "minLength": 1, "maxLength": 24},
                                },
                            },
                            "required": ["name", "values"],
                            "additionalProperties": False,
                        },
                        "description": "Filterable attribute axes, e.g. 类型: [黑巧, 牛奶, 白巧, 夹心]",
                    },
                    "variant_key": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 16,
                        "description": "Variant axis name, e.g. 规格",
                    },
                    "variant_values": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "string", "minLength": 1, "maxLength": 24},
                        "description": "Exactly 4 variant values, e.g. [45g, 100g, 250g, 礼盒]",
                    },
                    "base_price_yuan": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 10000,
                        "description": "Realistic entry price in yuan for this category",
                    },
                    "brand_styles": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "properties": {
                                "brand": {"type": "string", "minLength": 1, "maxLength": 16},
                                "series": {"type": "string", "minLength": 1, "maxLength": 16},
                            },
                            "required": ["brand", "series"],
                            "additionalProperties": False,
                        },
                        "description": (
                            "Invented-but-shelf-realistic naming matrix for this "
                            "category: 4-8 {brand, series} pairs covering 2-4 brands "
                            "(interleave brands so one page mixes them). Brand/series "
                            "names must FIT the category (食用油→金穗坊/澄香/鲁源坊, "
                            "小榨原香/家庭装), be fictional — never real trademarks "
                            "(金龙鱼/鲁花/苹果/华为 are forbidden) — and avoid generic "
                            "series codes like Astro/Nova/Pro."
                        ),
                    },
                    "source_query": {
                        "type": "string",
                        "maxLength": 200,
                        "description": "The customer's original words that motivated this mint",
                    },
                },
                "required": [
                    "name",
                    "description",
                    "domain",
                    "attributes",
                    "variant_key",
                    "variant_values",
                    "base_price_yuan",
                ],
                "additionalProperties": False,
            },
        )
    )

    if commerce_graph is not None:
        # Product graph context is semantic only: it stages no purchase view
        # and never replaces provider SKU, price, stock or limit facts.
        registry.register(
            ToolDefinition(
                name="commerce_graph_context",
                intent="commerce:graph_context",
                description=(
                    "Search the existing product knowledge graph for semantic category "
                    "relationships and attribute vocabulary. Read-only semantic context: "
                    "use it to understand use cases, alternatives, compatibility and "
                    "setup bundles, then verify every SKU, price, stock and limit with "
                    "commerce provider tools. Never present graph output as availability."
                ),
                required_slots=["query"],
                handler=graph_context,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 500},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
        )
    return registry
