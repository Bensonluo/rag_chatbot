"""True token streaming contract: queue bridge, heartbeat, disconnect, dedup.

At 800K-1M daily requests the SSE endpoint must stream tokens as they
arrive (users abandon slow chat windows), survive client disconnects
without leaking graph tasks, and keep proxies from reaping idle
connections. These tests pin the queue-bridge contract end to end:
nodes push tokens/template responses onto the per-request queue, the
service forwards them with heartbeat sentinels, and a disconnect cancels
the graph task while persisting the partial turn.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, Mock

from langchain_core.runnables import RunnableConfig

from app.services.chat.chat_service import HEARTBEAT, STREAM_ERROR, ChatService
from app.services.dialogue.nodes import NodeFactory
from app.services.dialogue.state import DialogueState
from app.services.dialogue.tools import ToolRegistry
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase

_FALLBACK = "抱歉，生成回复时出现错误，请稍后重试。"


class _StreamLLM(LLMServiceBase):
    """Fake LLM whose generate_stream yields tokens then optionally raises."""

    def __init__(self, tokens: list[str], raise_after: int | None = None) -> None:
        super().__init__(api_key="fake", model="fake")
        self._tokens = tokens
        self._raise_after = raise_after

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="".join(self._tokens), model="fake")

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        for i, token in enumerate(self._tokens):
            if self._raise_after is not None and i >= self._raise_after:
                raise RuntimeError("mid-stream drop")
            yield token

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages)


def _drain(queue: asyncio.Queue[str]) -> list[str]:
    items: list[str] = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def _config_with_queue() -> tuple[RunnableConfig, asyncio.Queue[str]]:
    queue: asyncio.Queue[str] = asyncio.Queue()
    config: RunnableConfig = {"configurable": {"thread_id": "t", "stream_queue": queue}}
    return config, queue


def _factory(
    llm: LLMServiceBase | None = None,
    tool_registry: ToolRegistry | None = None,
) -> NodeFactory:
    return NodeFactory(
        intent_detector=Mock(),
        slot_filler=None,
        tool_registry=tool_registry or Mock(),
        llm_service=llm,
    )


class TestChatServiceStreamBridge:
    async def test_tokens_forwarded_in_order(self):
        class _Graph:
            async def ainvoke(
                self, state: dict[str, Any], config: dict[str, Any]
            ) -> dict[str, Any]:
                queue = config["configurable"]["stream_queue"]
                queue.put_nowait("您")
                queue.put_nowait("好")
                return {"response": "您好"}

        service = ChatService(graph=_Graph())
        chunks = [c async for c in service.process_message_stream(1, "你好", 1)]
        assert chunks == ["您", "好"]

    async def test_heartbeat_when_queue_idle(self):
        class _SlowGraph:
            async def ainvoke(
                self, state: dict[str, Any], config: dict[str, Any]
            ) -> dict[str, Any]:
                queue = config["configurable"]["stream_queue"]
                queue.put_nowait("首块")
                await asyncio.sleep(0.1)
                return {}

        service = ChatService(graph=_SlowGraph())
        chunks = [
            c async for c in service.process_message_stream(1, "你好", 1, heartbeat_seconds=0.01)
        ]
        assert chunks[0] == "首块"
        assert HEARTBEAT in chunks  # keepalive arrived during the idle window

    async def test_disconnect_persists_partial_and_cancels_graph(self):
        class _HangingGraph:
            def __init__(self) -> None:
                self.cancelled = False

            async def ainvoke(
                self, state: dict[str, Any], config: dict[str, Any]
            ) -> dict[str, Any]:
                queue = config["configurable"]["stream_queue"]
                queue.put_nowait("部分回答")
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
                queue.put_nowait("永不到达")
                return {}

        graph = _HangingGraph()
        persister = AsyncMock()
        service = ChatService(graph=graph, persister=persister)
        stream = service.process_message_stream(1, "查订单", 1)
        assert await stream.__anext__() == "部分回答"
        await stream.aclose()  # client disconnect

        assert graph.cancelled is True  # background task did not leak
        persister.persist_turn.assert_awaited_once()
        assert persister.persist_turn.await_args.kwargs["response"] == "部分回答"

    async def test_graph_exception_yields_error_sentinel_with_partial_persisted(self):
        class _BoomGraph:
            async def ainvoke(
                self, state: dict[str, Any], config: dict[str, Any]
            ) -> dict[str, Any]:
                config["configurable"]["stream_queue"].put_nowait("半")
                raise RuntimeError("graph blew up")

        persister = AsyncMock()
        service = ChatService(graph=_BoomGraph(), persister=persister)
        chunks = [chunk async for chunk in service.process_message_stream(1, "m", 1)]
        assert chunks == ["半", STREAM_ERROR]
        assert persister.persist_turn.await_args.kwargs["response"] == "半"

    async def test_empty_stream_persists_nothing(self):
        class _QuietGraph:
            async def ainvoke(
                self, state: dict[str, Any], config: dict[str, Any]
            ) -> dict[str, Any]:  # noqa: ARG002
                return {}

        persister = AsyncMock()
        service = ChatService(graph=_QuietGraph(), persister=persister)
        chunks = [c async for c in service.process_message_stream(1, "m", 1)]
        assert chunks == []
        persister.persist_turn.assert_not_awaited()


class TestNodeStreaming:
    async def test_llm_tokens_stream_without_full_text_duplicate(self):
        config, queue = _config_with_queue()
        factory = _factory(llm=_StreamLLM(["您", "好", "呀"]))
        state: DialogueState = {
            "message": "你好",
            "session_id": 1,
            "user_id": 1,
            "intent": "chitchat",
        }

        updates = await factory.generate_response_node(state, config)

        assert updates["response"] == "您好呀"
        assert _drain(queue) == ["您", "好", "呀"]  # tokens, and nothing more
        assert config["configurable"]["streamed_response"] is True

    async def test_template_response_pushed_whole(self):
        config, queue = _config_with_queue()
        factory = _factory(llm=_StreamLLM(["unused"]))
        state: DialogueState = {
            "message": "退款",
            "intent": "refund",
            "slot_prompt": "请提供订单号",
        }

        updates = await factory.generate_response_node(state, config)

        assert updates["response"] == "请提供订单号"
        assert _drain(queue) == ["请提供订单号"]

    async def test_confirm_question_pushed_from_execute_tool(self):
        config, queue = _config_with_queue()
        tool = Mock(intent="refund", description="退款", requires_confirmation=True)
        registry = Mock()
        registry.get_tool_for_intent.return_value = tool
        factory = _factory(tool_registry=registry)
        state: DialogueState = {
            "message": "退款",
            "intent": "refund",
            "filled_slots": {"order_id": "A1"},
            "user_id": 1,
        }

        updates = await factory.execute_tool_node(state, config)

        pushed = _drain(queue)
        assert len(pushed) == 1
        assert "A1" in pushed[0]
        assert "确认" in pushed[0]
        assert updates["pending_confirmation"] == {
            "intent": "refund",
            "args": {"order_id": "A1"},
        }

    async def test_midstream_failure_appends_fallback_without_duplication(self):
        config, queue = _config_with_queue()
        factory = _factory(llm=_StreamLLM(["部分", "回答"], raise_after=1))
        state: DialogueState = {
            "message": "你好",
            "session_id": 1,
            "user_id": 1,
            "intent": "chitchat",
        }

        updates = await factory.generate_response_node(state, config)

        pushed = _drain(queue)
        assert pushed == ["部分", _FALLBACK]  # visible apology, no full-text replay
        assert updates["response"] == "部分" + _FALLBACK

    async def test_sync_path_untouched_without_queue(self):
        # No stream queue in config: nodes must behave exactly as before.
        factory = _factory(llm=_StreamLLM(["整", "段"]))
        state: DialogueState = {
            "message": "你好",
            "session_id": 1,
            "user_id": 1,
            "intent": "chitchat",
        }

        updates = await factory.generate_response_node(state, None)

        assert updates["response"] == "整段"
