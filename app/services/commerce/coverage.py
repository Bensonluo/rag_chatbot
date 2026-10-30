"""Coverage recording for shopping searches.

Recording is fail-open telemetry: search behavior never depends on the
registry write succeeding, and provider facts remain the only SKU authority.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from app.services.commerce.taxonomy import CommerceTaxonomyRegistry


class CommerceCoverageRecorder(Protocol):
    """Record low-result shopping searches for taxonomy review."""

    async def record_search(
        self,
        *,
        query: str,
        result_count: int,
        owner: int,
    ) -> None:
        """Record one search outcome without raising into the shopping path."""
        ...


class SqlCommerceCoverageRecorder:
    """Database-backed recorder with exact category matching."""

    def __init__(self, registry: CommerceTaxonomyRegistry) -> None:
        self._registry = registry

    async def record_search(self, *, query: str, result_count: int, owner: int) -> None:
        matched: list[str] = []
        try:
            taxonomy = await self._registry.load_active()
            normalized = " ".join(query.strip().lower().split())
            for category in taxonomy.categories:
                aliases = {
                    category.name.lower(),
                    *(alias.lower() for alias in category.provider_aliases),
                }
                if normalized and normalized in aliases:
                    matched.append(category.category_id)
        except Exception:
            matched = []
        owner_hash = hashlib.sha256(str(owner).encode("ascii")).hexdigest()[:24]
        await self._registry.record_coverage(
            query=query,
            result_count=result_count,
            matched_category_ids=matched,
            owner_hash=owner_hash,
        )
