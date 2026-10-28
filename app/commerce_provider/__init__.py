"""Standalone internal virtual-commerce provider service.

The package is self-contained and owns every commerce fact:

- ``catalog``  — deterministic virtual catalog over a stable 100,000,000-SKU
  address space (pure functions, never enumerated).
- ``storage``  — durable SQLite persistence (stdlib ``sqlite3`` via
  ``asyncio.to_thread``, WAL mode, explicit transactions).
- ``core``     — reusable business core: carts, quotes, orders, payments,
  stock deltas, and idempotency receipts keyed by ``request_key``.
- ``api``      — typed internal FastAPI facade (``/commerce/*`` wire contract,
  ``Authorization: Bearer`` when configured, test-only fault injection).
- ``__main__`` — ``python -m app.commerce_provider`` service entrypoint.

It deliberately holds no user/chat/domain foreign keys and never imports from
``app.services.commerce``; the only shared surface is the Pydantic models in
``app.models.schemas.commerce``.
"""

from __future__ import annotations

from app.commerce_provider.api import create_app
from app.commerce_provider.catalog import CommerceSort, VirtualCatalog
from app.commerce_provider.core import CommerceProviderCore
from app.commerce_provider.errors import (
    ProviderAuthError,
    ProviderConflictError,
    ProviderError,
    ProviderNotFoundError,
    ProviderRejectedError,
    ProviderUnknownError,
)
from app.commerce_provider.settings import CommerceProviderSettings
from app.commerce_provider.storage import Database

__all__ = [
    "CommerceProviderCore",
    "CommerceProviderSettings",
    "CommerceSort",
    "Database",
    "ProviderAuthError",
    "ProviderConflictError",
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderRejectedError",
    "ProviderUnknownError",
    "VirtualCatalog",
    "create_app",
]
