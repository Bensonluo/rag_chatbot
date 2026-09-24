"""Per-user TTL cache for cross-session user-facts recall (Phase B2 read side).

At 800K-1M daily requests the facts provider is consulted on every
generation call. Personalization facts are slow-stale — extraction runs
every USER_FACT_EXTRACTION_INTERVAL turns — so a short per-user TTL
trades bounded staleness for the bulk of those store hits. Cache
discipline follows the read-path research: per-user keying isolates
context (AWS namespacing guidance); failures are never cached so a
store outage cannot be frozen as empty facts; successful empty lists
are cached (the dominant new-user case).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable

from app.services.chat.metrics import USER_FACTS_CACHE_HITS, USER_FACTS_CACHE_MISSES

FactsProvider = Callable[[int], Awaitable[list[str]]]

# (stored-at monotonic time, facts). Empty lists are valid entries.
_Entry = tuple[float, list[str]]

_DEFAULT_MAXSIZE = 4096


def cached_user_facts_provider(
    provider: FactsProvider,
    *,
    ttl_seconds: float,
    maxsize: int = _DEFAULT_MAXSIZE,
) -> FactsProvider:
    """Wrap a user-facts provider with a per-user TTL + LRU cache.

    Worker-local by design: the cache only short-circuits reads this
    process would have made anyway, and each worker stays within its own
    memory bound. No lock — a single event loop interleaves awaits, so
    at worst two concurrent recalls for one user both fetch (harmless
    duplicate read, last write wins).

    Args:
        provider: Async store callable returning the user's facts.
        ttl_seconds: Entry lifetime. ``<= 0`` disables caching entirely
            (the provider is returned unwrapped).
        maxsize: Upper bound on distinct users kept per worker; the
            least-recently-used user is evicted first.

    Returns:
        A drop-in provider with the same ``Callable[[int], Awaitable[list[str]]]``
        shape. Store failures propagate uncached — the recall wrapper in
        the dialogue nodes already catches them (never-fail-chat), and a
        transient outage must not be pinned for a whole TTL window.
    """
    if ttl_seconds <= 0:
        return provider

    cache: OrderedDict[int, _Entry] = OrderedDict()

    async def _cached(user_id: int) -> list[str]:
        now = time.monotonic()
        entry = cache.get(user_id)
        if entry is not None and now - entry[0] < ttl_seconds:
            cache.move_to_end(user_id)  # LRU touch: recent users stay hot
            USER_FACTS_CACHE_HITS.inc()
            return entry[1]

        USER_FACTS_CACHE_MISSES.inc()
        facts = await provider(user_id)  # raises propagate, never cached
        cache[user_id] = (time.monotonic(), facts)
        cache.move_to_end(user_id)  # refresh moves an expired entry to hot
        while len(cache) > maxsize:
            cache.popitem(last=False)
        return facts

    return _cached
