"""Periodic one-shot KPI refresh: the north-star rate becomes scrape-able.

The session-level one-shot rate lives in Postgres (a repo aggregate),
so Prometheus cannot see it until something bridges the two. This
module runs the query on a refresh loop and sets process-level Gauges
— same doctrine as the BM25 refresher it mirrors: every replica reads
the same DB rows so their gauges agree, and a DB outage leaves the
last value in place (fail-open, logged) instead of scraping the
metric to zero.

Undefined-rate note: when the window has no served sessions the rate
is None and the gauge is deliberately left untouched — publishing 0
there would read as "zero one-shot resolution" (an outage-shaped
lie) when the truth is "no traffic to measure".
"""

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from prometheus_client import REGISTRY, Gauge, Histogram
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.repositories.ticket_repository import TicketRepository

logger = logging.getLogger(__name__)

# Metrics belong to the process (same pattern as retrieval/metrics.py)
# to avoid duplicate registration across app factories and test clients.
ONE_SHOT_RATE = Gauge(
    "chat_one_shot_rate",
    "Session-level one-shot resolution rate over the stats window "
    "(served sessions with no handoff ticket / served sessions)",
    registry=REGISTRY,
)
SESSIONS_SERVED = Gauge(
    "chat_sessions_served",
    "Sessions served (>=1 assistant message) over the stats window",
    registry=REGISTRY,
)
SESSIONS_ESCALATED = Gauge(
    "chat_sessions_escalated",
    "Served sessions with a handoff ticket over the stats window",
    registry=REGISTRY,
)
ONE_SHOT_REFRESH_SECONDS = Histogram(
    "one_shot_refresh_seconds",
    "Duration of one one-shot stats refresh, by outcome",
    labelnames=["outcome"],
    registry=REGISTRY,
)

_refresh_task: asyncio.Task[None] | None = None


async def refresh_one_shot_stats(
    session_maker: async_sessionmaker[AsyncSession],
    window_days: int = 7,
) -> dict[str, Any] | None:
    """Run the one-shot aggregate and publish it to the Gauges.

    Fail-open: a DB error is observed (histogram outcome=error), the
    Gauges keep their last values, and None is returned — availability
    of the metric over strictness.
    """
    started = time.monotonic()
    try:
        since = datetime.now(UTC) - timedelta(days=window_days)
        async with session_maker() as db:
            stats = await TicketRepository(db).get_one_shot_stats(since=since)
    except Exception:
        ONE_SHOT_REFRESH_SECONDS.labels(outcome="error").observe(time.monotonic() - started)
        logger.exception("One-shot stats refresh failed; gauges keep last values")
        return None

    ONE_SHOT_REFRESH_SECONDS.labels(outcome="ok").observe(time.monotonic() - started)
    SESSIONS_SERVED.set(stats["sessions_served"])
    SESSIONS_ESCALATED.set(stats["sessions_escalated"])
    if stats["one_shot_rate"] is not None:
        ONE_SHOT_RATE.set(stats["one_shot_rate"])
    return stats


async def run_one_shot_refresher(
    session_maker: async_sessionmaker[AsyncSession],
    interval_seconds: float,
    window_days: int = 7,
) -> None:
    """Refresh loop: sleep one interval, aggregate, repeat until cancelled.

    Sleeps first so the first refresh lands a full interval after
    boot, staggered from the request-serving startup path.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        await refresh_one_shot_stats(session_maker=session_maker, window_days=window_days)


def start_one_shot_refresher(
    session_maker: async_sessionmaker[AsyncSession],
    interval_seconds: float,
    window_days: int = 7,
) -> asyncio.Task[None] | None:
    """Spawn the refresher task (no-op if one is already running).

    Must be called from a running event loop (service init).
    """
    global _refresh_task
    if _refresh_task is not None and not _refresh_task.done():
        return _refresh_task
    _refresh_task = asyncio.create_task(
        run_one_shot_refresher(
            session_maker=session_maker,
            interval_seconds=interval_seconds,
            window_days=window_days,
        )
    )
    return _refresh_task


def stop_one_shot_refresher() -> None:
    """Cancel and clear the refresher task (safe when not running)."""
    global _refresh_task
    if _refresh_task is not None and not _refresh_task.done():
        _refresh_task.cancel()
    _refresh_task = None
