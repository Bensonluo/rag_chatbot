"""Chat stream 时效 metrics (GB/T 47746 响应速度 dimension, AI side).

TTFT (time to first token) is the canonical perceived-latency metric for
streaming CS bots — industry targets sub-1s, and support-bot drop-off
rises ~7-10% per extra second of waiting (NVIDIA/Anyscale TTFT
definitions; tianpan.co drop-off data). GB/T 47746—2026 lists 响应速度
as a quantitative evaluation dimension. These tests pin that the stream
coroutine records TTFT to first *content* (heartbeats don't count),
total duration, and the outcome of every exit path.
"""

import asyncio
from unittest.mock import Mock

from app.services.chat.chat_service import ChatService


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value(name, labels)
    return value or 0.0


class TestStreamLatencyMetrics:
    async def test_completed_stream_records_ttft_duration_and_outcome(self):
        async def mock_ainvoke(state, config):
            queue = config["configurable"]["stream_queue"]
            queue.put_nowait("您好")
            queue.put_nowait("，请问有什么可以帮您")
            return {"response": "完整回复"}

        graph = Mock()
        graph.ainvoke = mock_ainvoke
        service = ChatService(graph=graph)

        ttft_before = _sample("chat_first_token_seconds_count")
        dur_before = _sample("chat_stream_duration_seconds_count")
        done_before = _sample("chat_stream_outcomes_total", {"outcome": "completed"})

        chunks = [c async for c in service.process_message_stream(1, "你好", 1)]

        assert chunks == ["您好", "，请问有什么可以帮您"]
        assert _sample("chat_first_token_seconds_count") == ttft_before + 1
        assert _sample("chat_stream_duration_seconds_count") == dur_before + 1
        assert _sample("chat_stream_outcomes_total", {"outcome": "completed"}) == done_before + 1

    async def test_heartbeat_is_not_a_first_token(self):
        """Keepalives keep proxies open but the user has seen no text —
        TTFT must measure time to content, not to the first yield."""

        async def mock_ainvoke(state, config):
            await asyncio.sleep(0.12)
            config["configurable"]["stream_queue"].put_nowait("迟到的内容")
            return {"response": "迟到的内容"}

        graph = Mock()
        graph.ainvoke = mock_ainvoke
        service = ChatService(graph=graph)

        ttft_before = _sample("chat_first_token_seconds_count")
        from app.services.chat.chat_service import HEARTBEAT

        chunks = [
            c async for c in service.process_message_stream(1, "你好", 1, heartbeat_seconds=0.03)
        ]

        assert HEARTBEAT in chunks  # keepalive fired while the LLM worked
        assert _sample("chat_first_token_seconds_count") == ttft_before + 1  # once, at content

    async def test_graph_exception_counts_error_outcome_no_ttft(self):
        async def mock_ainvoke(state, config):
            config["configurable"]["stream_queue"].put_nowait(None)
            raise RuntimeError("graph blew up")

        graph = Mock()
        graph.ainvoke = mock_ainvoke
        service = ChatService(graph=graph)

        ttft_before = _sample("chat_first_token_seconds_count")
        err_before = _sample("chat_stream_outcomes_total", {"outcome": "graph_error"})

        chunks = [
            c async for c in service.process_message_stream(1, "你好", 1, stream_max_seconds=5.0)
        ]

        assert chunks[-1] == "[[STREAM_ERROR]]"
        assert _sample("chat_first_token_seconds_count") == ttft_before  # nothing reached the user
        assert _sample("chat_stream_outcomes_total", {"outcome": "graph_error"}) == err_before + 1

    async def test_budget_exhaustion_counts_budget_outcome(self):
        async def mock_ainvoke(state, config):
            await asyncio.sleep(5.0)
            return {"response": "never"}

        graph = Mock()
        graph.ainvoke = mock_ainvoke
        service = ChatService(graph=graph)

        budget_before = _sample("chat_stream_outcomes_total", {"outcome": "budget_exhausted"})
        ttft_before = _sample("chat_first_token_seconds_count")

        chunks = [
            c async for c in service.process_message_stream(1, "你好", 1, stream_max_seconds=0.05)
        ]

        assert chunks == ["[[STREAM_ERROR]]"]
        assert (
            _sample("chat_stream_outcomes_total", {"outcome": "budget_exhausted"})
            == budget_before + 1
        )
        assert _sample("chat_first_token_seconds_count") == ttft_before

    async def test_client_disconnect_counts_disconnect_outcome(self):
        async def mock_ainvoke(state, config):
            config["configurable"]["stream_queue"].put_nowait("部分内容")
            await asyncio.sleep(5.0)  # still generating when the client leaves
            return {"response": "never finished"}

        graph = Mock()
        graph.ainvoke = mock_ainvoke
        service = ChatService(graph=graph)

        disc_before = _sample("chat_stream_outcomes_total", {"outcome": "client_disconnect"})
        ttft_before = _sample("chat_first_token_seconds_count")

        agen = service.process_message_stream(1, "你好", 1, stream_max_seconds=30.0)
        first = await agen.__anext__()
        assert first == "部分内容"
        await agen.aclose()

        assert _sample("chat_first_token_seconds_count") == ttft_before + 1
        assert (
            _sample("chat_stream_outcomes_total", {"outcome": "client_disconnect"})
            == disc_before + 1
        )
