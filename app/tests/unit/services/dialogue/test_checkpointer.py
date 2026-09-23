"""Shared dialogue checkpointer lifecycle and graph wiring.

At 800K-1M daily requests the API runs as multiple replicas behind a
load balancer; a process-local MemorySaver loses dialogue state (intent,
slots, staged confirmations) whenever a user's consecutive turns land on
different replicas. These tests pin the manager contract: postgres mode
opens/setups/closes correctly, failures degrade to MemorySaver, and the
compiled graph actually uses the injected checkpointer.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import langgraph.checkpoint.postgres.aio  # noqa: F401 (resolves patch target)
from langgraph.checkpoint.memory import MemorySaver

from app.services.dialogue.checkpointer import (
    DialogueCheckpointerManager,
    normalize_db_url,
)
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import create_default_tool_registry


def _postgres_cm(saver: MagicMock) -> MagicMock:
    """A context manager mimicking AsyncPostgresSaver.from_conn_string."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=saver)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ── URL normalization ──────────────────────────────────────────────────────


class TestNormalizeDbUrl:
    def test_strips_asyncpg_driver_suffix(self):
        assert (
            normalize_db_url("postgresql+asyncpg://u:p@h:5432/db") == "postgresql://u:p@h:5432/db"
        )

    def test_plain_url_unchanged(self):
        assert normalize_db_url("postgresql://u:p@h:5432/db") == "postgresql://u:p@h:5432/db"

    def test_only_first_occurrence_replaced(self):
        url = "postgresql+asyncpg://u:p@h/db?opts=postgresql+asyncpg://x"
        assert normalize_db_url(url).startswith("postgresql://u:p@h/db?")


# ── Manager lifecycle ──────────────────────────────────────────────────────


class TestManagerLifecycle:
    async def test_memory_mode_returns_memory_saver(self):
        manager = DialogueCheckpointerManager("memory", db_url=None)
        saver = await manager.start()
        assert isinstance(saver, MemorySaver)
        # stop() must be a no-op without an open connection pool
        await manager.stop()

    async def test_postgres_mode_opens_sets_up_and_closes(self):
        saver = MagicMock()
        saver.setup = AsyncMock()
        cm = _postgres_cm(saver)

        with patch(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.from_conn_string",
            return_value=cm,
        ) as from_conn:
            manager = DialogueCheckpointerManager(
                "postgres", db_url="postgresql+asyncpg://u:p@h:5432/db"
            )
            started = await manager.start()

            assert started is saver
            from_conn.assert_called_once_with("postgresql://u:p@h:5432/db")
            cm.__aenter__.assert_awaited_once()
            saver.setup.assert_awaited_once()

            await manager.stop()
            cm.__aexit__.assert_awaited_once()

    async def test_postgres_failure_degrades_to_memory_saver(self):
        with patch(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.from_conn_string",
            side_effect=RuntimeError("db unreachable"),
        ):
            manager = DialogueCheckpointerManager("postgres", db_url="postgresql://h/db")
            started = await manager.start()

        assert isinstance(started, MemorySaver)
        await manager.stop()  # nothing open — must not raise

    async def test_postgres_without_url_degrades_to_memory_saver(self):
        manager = DialogueCheckpointerManager("postgres", db_url=None)
        started = await manager.start()
        assert isinstance(started, MemorySaver)

    async def test_stop_swallows_close_errors(self):
        saver = MagicMock()
        saver.setup = AsyncMock()
        cm = _postgres_cm(saver)
        cm.__aexit__ = AsyncMock(side_effect=RuntimeError("pool closed twice"))

        with patch(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.from_conn_string",
            return_value=cm,
        ):
            manager = DialogueCheckpointerManager("postgres", db_url="postgresql://h/db")
            await manager.start()
            await manager.stop()  # must not raise
            # Second stop is a no-op, not a second __aexit__ call
            await manager.stop()
            assert cm.__aexit__.await_count == 1


# ── Graph wiring ───────────────────────────────────────────────────────────


class TestGraphCheckpointerInjection:
    def _build(self, checkpointer):
        detector = MagicMock()
        detector.detect_with_confidence = AsyncMock()
        return build_dialogue_graph(
            intent_detector=detector,
            slot_filler=None,
            tool_registry=create_default_tool_registry(),
            retrieval_pipeline={},
            llm_service=None,
            checkpointer=checkpointer,
        )

    def test_injected_checkpointer_is_used_by_compiled_graph(self):
        # compile() type-checks the saver, so prove injection with a real
        # instance: the default path would create a different one.
        shared = MemorySaver()
        compiled = self._build(shared)
        assert compiled.checkpointer is shared

    def test_default_is_process_local_memory_saver(self):
        compiled = self._build(None)
        assert isinstance(compiled.checkpointer, MemorySaver)
        assert self._build(None).checkpointer is not compiled.checkpointer
