"""L1 semantic cache graph wiring: the direct tier replays near-duplicates.

The cache-layering plan's L1 sits directly under L0 exact replay. Wired
placement is ``direct_response_node`` — after intent detection, inside
the small-talk tier it serves — so the embedding lookup is paid only on
direct-tier turns instead of taxing every L0-miss turn. Pins the node
contract: a hit skips the LLM entirely and re-runs the deterministic
freshness insurance (claim gate + output guardrail); a miss generates
as before and writes back only for anonymous, stateless turns.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

from prometheus_client import REGISTRY

from app.services.chat.answer_cache import CachedAnswer
from app.services.dialogue.nodes import NodeFactory
from app.services.llm.base import LLMResponse


class StubSemanticCache:
    """Recording stand-in for the L1 service; hit/miss via ``hit``."""

    def __init__(self, hit: CachedAnswer | None) -> None:
        self.hit = hit
        self.gets: list[str] = []
        self.puts: list[tuple[str, CachedAnswer]] = []

    async def get(self, message: str) -> CachedAnswer | None:
        self.gets.append(message)
        return self.hit

    async def put(self, message: str, answer: CachedAnswer) -> None:
        self.puts.append((message, answer))


def _make_factory(semantic_cache: Any, llm: Any) -> NodeFactory:
    return NodeFactory(
        intent_detector=Mock(),
        slot_filler=None,
        tool_registry=Mock(),
        llm_service=llm,
        semantic_cache=semantic_cache,
    )


def _stub_llm(content: str = "你好！请问有什么可以帮您？") -> Any:
    llm = Mock()
    llm.generate = AsyncMock(return_value=LLMResponse(content=content, model="fake"))
    return llm


def _layer_value(layer: str) -> float:
    value = REGISTRY.get_sample_value("chat_funnel_layers_total", {"layer": layer})
    return value if value is not None else 0.0


def _state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {"message": "你好呀", "session_id": 1, "intent": "chitchat"}
    state.update(overrides)
    return state


class TestSemanticCacheHit:
    async def test_hit_replays_without_llm(self):
        cached = CachedAnswer(response="缓存的话术", sources=[], intent="chitchat")
        stub = StubSemanticCache(hit=cached)
        llm = _stub_llm()
        factory = _make_factory(stub, llm)

        updates = await factory.direct_response_node(_state(), None)

        assert updates["response"] == "缓存的话术"
        assert stub.gets == ["你好呀"]
        llm.generate.assert_not_awaited()
        assert stub.puts == []  # a hit never rewrites the same entry

    async def test_hit_records_l1_not_direct_layer(self):
        stub = StubSemanticCache(hit=CachedAnswer(response="x", sources=[], intent="chitchat"))
        factory = _make_factory(stub, _stub_llm())

        before_l1 = _layer_value("l1_semantic")
        before_direct = _layer_value("direct")
        await factory.direct_response_node(_state(), None)

        assert _layer_value("l1_semantic") == before_l1 + 1.0
        assert _layer_value("direct") == before_direct

    async def test_hit_still_serves_with_pending_confirmation(self):
        # Direct-tier answers are stateless by construction, so a cached
        # replay is safe mid-flow — unlike L0, which gates on in-flight
        # state because grounded answers there can embed task context.
        stub = StubSemanticCache(hit=CachedAnswer(response="hi", sources=[], intent="greeting"))
        factory = _make_factory(stub, _stub_llm())

        updates = await factory.direct_response_node(
            _state(pending_confirmation={"intent": "refund", "args": {}}), None
        )
        assert updates["response"] == "hi"

    async def test_hit_blocked_by_output_guardrail_replaced(self):
        cached = CachedAnswer(response="联系我的私聊账号", sources=[], intent="chitchat")
        guardrail = Mock()
        guardrail.check_output.return_value = Mock(was_blocked=True, sanitized_content="")
        factory = NodeFactory(
            intent_detector=Mock(),
            slot_filler=None,
            tool_registry=Mock(),
            llm_service=_stub_llm(),
            guardrail_service=guardrail,
            semantic_cache=StubSemanticCache(hit=cached),
        )

        updates = await factory.direct_response_node(_state(), None)
        assert updates["response"] == "抱歉，该回复未能通过安全检查，请重新提问。"


class TestSemanticCacheMiss:
    async def test_miss_generates_and_records_direct(self):
        stub = StubSemanticCache(hit=None)
        llm = _stub_llm("生成的问候")
        factory = _make_factory(stub, llm)

        before = _layer_value("direct")
        updates = await factory.direct_response_node(_state(), None)

        assert updates["response"] == "生成的问候"
        llm.generate.assert_awaited_once()
        assert _layer_value("direct") == before + 1.0

    async def test_miss_writes_anonymous_allowlisted_turn(self):
        stub = StubSemanticCache(hit=None)
        factory = _make_factory(stub, _stub_llm("早上好"))

        await factory.direct_response_node(_state(), None)

        assert len(stub.puts) == 1
        message, answer = stub.puts[0]
        assert message == "你好呀"
        assert answer.response == "早上好"
        assert answer.intent == "chitchat"

    async def test_identified_user_never_written(self):
        # Personalized turns must never be cached: a hit replays the
        # stored text verbatim at any later visitor (L0 doctrine).
        stub = StubSemanticCache(hit=None)
        factory = _make_factory(stub, _stub_llm())

        await factory.direct_response_node(_state(user_id=7), None)
        assert stub.puts == []

    async def test_pending_confirmation_never_written(self):
        stub = StubSemanticCache(hit=None)
        factory = _make_factory(stub, _stub_llm())

        await factory.direct_response_node(
            _state(pending_confirmation={"intent": "refund", "args": {}}), None
        )
        assert stub.puts == []

    async def test_empty_response_never_written(self):
        stub = StubSemanticCache(hit=None)
        factory = _make_factory(stub, _stub_llm(""))

        await factory.direct_response_node(_state(), None)
        assert stub.puts == []


class TestSemanticCacheUnwired:
    async def test_none_service_is_pure_passthrough(self):
        llm = _stub_llm("原路返回")
        factory = _make_factory(None, llm)

        before = _layer_value("direct")
        updates = await factory.direct_response_node(_state(), None)

        assert updates["response"] == "原路返回"
        assert _layer_value("direct") == before + 1.0
