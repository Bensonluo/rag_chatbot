"""HTTP adapter for the internal commerce provider; facts arrive only as typed JSON.

The provider is a separate internal HTTP service that owns every commerce
fact. This adapter is a thin typed boundary: it forwards structured arguments
(never user language), decodes responses into the shared Pydantic models from
``app/models/schemas/commerce.py``, and never substitutes local mock data when
the provider is missing, slow, or broken.

Wire contract — the provider service implements exactly these paths. All
bodies are JSON; ``owner_user_id`` is always explicit; ``base_url`` may carry a
path prefix. Optional keys are omitted when unset:

  POST {base}/commerce/search
    {"owner_user_id": int, "query": str, "sort": "relevance"|"price_asc"|
     "price_desc"|"newest", "filters"?: CatalogFilters, "page_size": int,
     "page_token"?: str}
    -> 200 CatalogPage
  GET  {base}/commerce/categories?owner_user_id=int
    -> 200 CatalogTaxonomy   (read-only authoritative category facts:
    category id/name/description plus attribute names and allowed values;
    never user-language alias lists)
  GET  {base}/commerce/products/{sku_id}?owner_user_id=int -> 200 CatalogItem
  POST {base}/commerce/cart/update
    {"owner_user_id": int, "cart_id"?: str, "expected_cart_version"?: int,
     "lines": [CartLineSelection], "request_key": str}
    -> 200 CartSnapshot          (creates the cart or replaces all its lines)
  GET  {base}/commerce/carts/{cart_id}?owner_user_id=int -> 200 CartSnapshot
  POST {base}/commerce/checkout
    {"owner_user_id": int, "cart_id": str, "expected_cart_version": int,
     "request_key": str}
    -> 200 QuoteSnapshot
  POST {base}/commerce/orders
    {"owner_user_id": int, "quote_id": str, "request_key": str}
    -> 200 OrderSnapshot
  GET  {base}/commerce/orders/{order_id}?owner_user_id=int -> 200 OrderSnapshot
  POST {base}/commerce/payments
    {"owner_user_id": int, "order_id": str, "request_key": str}
    -> 200 PaymentSnapshot
  POST {base}/commerce/payments/{payment_id}/sandbox-pay
    {"owner_user_id": int, "request_key": str} -> 200 PaymentSnapshot
  GET  {base}/commerce/payments/{payment_id}?owner_user_id=int
    -> 200 PaymentSnapshot
  GET  {base}/commerce/operations/{request_key}?owner_user_id=int
    -> 200 {"kind": str, "status": "completed"|"failed", "result": {...}} | null
       (the adapter normalizes ``result`` to ``data`` for PurchaseService)

``request_key`` is the idempotency key for every mutating endpoint: replaying
it with identical arguments must return the recorded receipt; replaying it with
different arguments must answer HTTP 409.

Errors: a 4xx body ``{"error": {"code": str, "message": str}}`` (or a flat
``message``/``detail`` string) is a definitive refusal — ProviderRejected.
Timeouts, connection failures, 5xx responses, and bodies that do not decode
into the expected model raise ProviderUnknown: the call may have committed, so
callers reconcile through the original request key instead of retrying blind.
"""

from typing import Any, TypeVar, cast
from urllib.parse import quote as url_quote

import httpx
from pydantic import BaseModel, ValidationError

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
from app.services.commerce.port import (
    CommerceSort,
    ProviderOperationReceipt,
    ProviderRejected,
    ProviderUnknown,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

__all__ = ["ExternalCommercePort"]


class ExternalCommercePort:
    """Typed HTTP client for the internal commerce provider (no local fallback)."""

    mode: str = "external"

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        parsed = httpx.URL(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("base_url must be an absolute http(s) URL")
        self._base_url = str(parsed).rstrip("/")
        self._token = token
        self._timeout = httpx.Timeout(timeout_seconds)
        self._client = client
        self._owns_client = client is None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying client when this adapter created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
            self._owns_client = False

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        headers = {"accept": "application/json"}
        if self._token:
            headers["authorization"] = f"Bearer {self._token}"
        if json_body is not None:
            headers["content-type"] = "application/json"
        client = self._http()
        request = client.build_request(
            method,
            f"{self._base_url}{path}",
            params=params,
            json=json_body,
            headers=headers,
            timeout=self._timeout,
        )
        try:
            response = await client.send(request)
        except httpx.TimeoutException as exc:
            raise ProviderUnknown("外部商城响应超时，结果未确认，请按原请求核实") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnknown("外部商城暂不可用，结果未确认，请按原请求核实") from exc
        if response.status_code >= 500:
            raise ProviderUnknown(
                f"外部商城内部错误（HTTP {response.status_code}），结果未确认，请按原请求核实"
            )
        if response.status_code in {408, 429}:
            raise ProviderUnknown(
                f"外部商城请求未确认（HTTP {response.status_code}），请按原请求核实"
            )
        if response.status_code >= 400:
            raise ProviderRejected(_refusal_message(response))
        if not 200 <= response.status_code < 300:
            raise ProviderUnknown(f"外部商城返回异常状态（HTTP {response.status_code}）")
        return response

    async def _call(
        self,
        method: str,
        path: str,
        *,
        model: type[ModelT],
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> ModelT:
        response = await self._request(method, path, params=params, json_body=json_body)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderUnknown("外部商城返回内容不可解析，结果未确认") from exc
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise ProviderUnknown("外部商城返回内容不符合类型约定，结果未确认") from exc

    async def _get(
        self, path_template: str, identifier: str, owner_user_id: int, model: type[ModelT]
    ) -> ModelT:
        return await self._call(
            "GET",
            path_template.format(url_quote(identifier, safe="")),
            model=model,
            params={"owner_user_id": owner_user_id},
        )

    async def search_products(
        self,
        owner_user_id: int,
        query: str,
        *,
        filters: CatalogFilters | None = None,
        sort: CommerceSort = "relevance",
        page_size: int = 10,
        page_token: str | None = None,
    ) -> CatalogPage:
        json_body: dict[str, Any] = {
            "owner_user_id": owner_user_id,
            "query": query,
            "sort": sort,
            "page_size": page_size,
        }
        if filters is not None:
            json_body["filters"] = filters.model_dump(mode="json")
        if page_token is not None:
            json_body["page_token"] = page_token
        return await self._call("POST", "/commerce/search", model=CatalogPage, json_body=json_body)

    async def catalog_categories(self, owner_user_id: int) -> CatalogTaxonomy:
        """Read the provider's authoritative category taxonomy (read-only)."""

        return await self._call(
            "GET",
            "/commerce/categories",
            model=CatalogTaxonomy,
            params={"owner_user_id": owner_user_id},
        )

    async def get_product(self, owner_user_id: int, sku_id: str) -> CatalogItem:
        return await self._get("/commerce/products/{}", sku_id, owner_user_id, CatalogItem)

    async def update_cart_lines(
        self,
        owner_user_id: int,
        cart_id: str | None,
        expected_cart_version: int | None,
        lines: list[CartLineSelection],
        request_key: str,
    ) -> CartSnapshot:
        json_body: dict[str, Any] = {
            "owner_user_id": owner_user_id,
            "lines": [line.model_dump(mode="json") for line in lines],
            "request_key": request_key,
        }
        if cart_id is not None:
            json_body["cart_id"] = cart_id
        if expected_cart_version is not None:
            json_body["expected_cart_version"] = expected_cart_version
        return await self._call(
            "POST", "/commerce/cart/update", model=CartSnapshot, json_body=json_body
        )

    async def get_cart(self, owner_user_id: int, cart_id: str) -> CartSnapshot:
        return await self._get("/commerce/carts/{}", cart_id, owner_user_id, CartSnapshot)

    async def prepare_checkout(
        self,
        owner_user_id: int,
        cart_id: str,
        expected_cart_version: int,
        request_key: str,
    ) -> QuoteSnapshot:
        return await self._call(
            "POST",
            "/commerce/checkout",
            model=QuoteSnapshot,
            json_body={
                "owner_user_id": owner_user_id,
                "cart_id": cart_id,
                "expected_cart_version": expected_cart_version,
                "request_key": request_key,
            },
        )

    async def create_order(
        self, owner_user_id: int, quote_id: str, request_key: str
    ) -> OrderSnapshot:
        return await self._call(
            "POST",
            "/commerce/orders",
            model=OrderSnapshot,
            json_body={
                "owner_user_id": owner_user_id,
                "quote_id": quote_id,
                "request_key": request_key,
            },
        )

    async def create_payment(
        self, owner_user_id: int, order_id: str, request_key: str
    ) -> PaymentSnapshot:
        return await self._call(
            "POST",
            "/commerce/payments",
            model=PaymentSnapshot,
            json_body={
                "owner_user_id": owner_user_id,
                "order_id": order_id,
                "request_key": request_key,
            },
        )

    async def mock_pay(
        self, owner_user_id: int, payment_id: str, request_key: str
    ) -> PaymentSnapshot:
        return await self._call(
            "POST",
            f"/commerce/payments/{url_quote(payment_id, safe='')}/sandbox-pay",
            model=PaymentSnapshot,
            json_body={"owner_user_id": owner_user_id, "request_key": request_key},
        )

    async def get_order(self, owner_user_id: int, order_id: str) -> OrderSnapshot:
        return await self._get("/commerce/orders/{}", order_id, owner_user_id, OrderSnapshot)

    async def get_status(self, owner_user_id: int, payment_id: str) -> PaymentSnapshot:
        return await self._get("/commerce/payments/{}", payment_id, owner_user_id, PaymentSnapshot)

    async def lookup_operation(
        self, owner_user_id: int, request_key: str
    ) -> ProviderOperationReceipt | None:
        response = await self._request(
            "GET",
            f"/commerce/operations/{url_quote(request_key, safe='')}",
            params={"owner_user_id": owner_user_id},
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderUnknown("外部商城返回内容不可解析，结果未确认") from exc
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ProviderUnknown("外部商城操作回执不符合类型约定，结果未确认")
        kind = payload.get("kind")
        status = payload.get("status")
        result = payload.get("result")
        if (
            not isinstance(kind, str)
            or not isinstance(status, str)
            or status not in {"completed", "failed", "unknown"}
            or not isinstance(result, dict)
        ):
            raise ProviderUnknown("外部商城操作回执缺少完整事实，结果未确认")
        # Wire format is ``result``; PurchaseService's reconciliation contract
        # uses ``data`` so mock and external ports cannot drift apart.
        return cast(ProviderOperationReceipt, {"kind": kind, "status": status, "data": result})


def _refusal_message(response: httpx.Response) -> str:
    try:
        payload: Any = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        for key in ("message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    return f"外部商城拒绝请求（HTTP {response.status_code}）"
