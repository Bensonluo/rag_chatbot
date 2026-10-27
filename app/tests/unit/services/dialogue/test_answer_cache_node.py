"""L0 answer-cache lookup node: hit/miss routing and serve-time re-gating.

Pins the node contract from docs/cache-layering-plan.md:
- a miss (or unwired cache, blocked turn, in-flight slot state) leaves
  the pipeline routing exactly as before;
- a hit ends the turn with the replayed answer, re-running the
  deterministic claim gate and output guardrail first — the cached
  text is never trusted blindly;
- the hit response reaches the stream queue like any terminal node.
"""

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.dialogue.nodes import NodeFactory
from app.services.dialogue.state import DialogueState


class _FakeCache:
    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.looked_up: list[str] = []

    async def get(self, message: str) -> Any:
        self.looked_up.append(message)
        return self.answer


def _cached_answer() -> Any:
    from app.services.chat.answer_cache import CachedAnswer

    return CachedAnswer(
        response="签收后7天内可申请无理由退货。",
        sources=["doc-refund"],
        intent="question",
    )


def _make_factory(
    cache: Any,
    guardrail: Any = None,
) -> NodeFactory:
    return NodeFactory(
        intent_detector=Mock(),
        slot_filler=None,
        tool_registry=Mock(),
        guardrail_service=guardrail,
        answer_cache=cache,
    )


def _queue_config() -> tuple[Any, asyncio.Queue[Any]]:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    return {"configurable": {"stream_queue": queue}}, queue


@pytest.fixture(autouse=True)
def _claim_gate_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The claim gate has its own test file; keep this one about routing."""
    from app.config.settings import settings

    monkeypatch.setattr(settings, "FACT_CLAIM_CHECK_ENABLED", False)


class TestAnswerCacheLookupNode:
    async def test_miss_routes_into_normal_pipeline(self):
        cache = _FakeCache(None)
        factory = _make_factory(cache)

        updates = await factory.answer_cache_lookup_node({"message": "退货政策"})

        assert updates == {"route_after_cache": "miss"}
        assert cache.looked_up == ["退货政策"]

    async def test_unwired_cache_is_pure_passthrough(self):
        factory = _make_factory(None)

        updates = await factory.answer_cache_lookup_node({"message": "退货政策"})

        assert updates == {"route_after_cache": "miss"}

    async def test_blocked_turn_never_looks_up(self):
        cache = _FakeCache(_cached_answer())
        factory = _make_factory(cache)

        updates = await factory.answer_cache_lookup_node(
            {"message": "诈骗话术", "blocked": True, "response": "已拦截"}
        )

        assert updates == {"route_after_cache": "miss"}
        assert cache.looked_up == []

    async def test_pending_slot_flow_skips_lookup(self):
        cache = _FakeCache(_cached_answer())
        factory = _make_factory(cache)

        updates = await factory.answer_cache_lookup_node(
            {"message": "12345", "intent": "refund", "pending_slots": ["order_id"]}
        )

        assert updates == {"route_after_cache": "miss", "shared_cache_blocked": True}
        assert cache.looked_up == []

    async def test_hit_ends_turn_with_replayed_answer(self):
        factory = _make_factory(_FakeCache(_cached_answer()))
        config, queue = _queue_config()

        updates = await factory.answer_cache_lookup_node({"message": "退货政策是什么"}, config)

        assert updates["route_after_cache"] == "hit"
        assert updates["response"] == "签收后7天内可申请无理由退货。"
        assert updates["sources"] == ["doc-refund"]
        assert updates["intent"] == "question"
        # Terminal-node contract: the full text reaches the stream queue.
        assert queue.get_nowait() == "签收后7天内可申请无理由退货。"

    async def test_hit_output_guardrail_still_runs(self):
        from app.services.guardrails.base import GuardrailResult

        blocked_guardrail = Mock()
        blocked_guardrail.check_output = Mock(
            return_value=GuardrailResult(action="block", violations=["x"])
        )
        factory = _make_factory(_FakeCache(_cached_answer()), guardrail=blocked_guardrail)

        updates = await factory.answer_cache_lookup_node({"message": "退货政策是什么"})

        assert "安全检查" in updates["response"]

    async def test_hit_with_staged_confirmation_skips(self):
        cache = _FakeCache(_cached_answer())
        factory = _make_factory(cache)

        updates = await factory.answer_cache_lookup_node(
            {"message": "退货", "pending_confirmation": {"intent": "refund", "args": {}}}
        )

        assert updates == {"route_after_cache": "miss", "shared_cache_blocked": True}
        assert cache.looked_up == []


class TestRouteAfterCache:
    def test_hit_state_routes_to_end(self):
        assert NodeFactory.route_after_cache({"route_after_cache": "hit"}) == "hit"

    def test_miss_delegates_to_intent_skip_logic(self):
        # Slot-answer shape → the skip branch (collect_slots), not full detect.
        state: DialogueState = {
            "route_after_cache": "miss",
            "intent": "refund",
            "pending_slots": ["x"],
            "message": "12345",
        }
        assert NodeFactory.route_after_cache(state) == "skip"

    def test_plain_message_routes_full(self):
        state: DialogueState = {"route_after_cache": "miss", "message": "退货政策是什么"}
        assert NodeFactory.route_after_cache(state) == "full"


@pytest.mark.parametrize(
    "extra",
    [
        {"user_id": 42},
        {"filled_slots": {"order_id": "ORD1001"}},
        {"state_stack": [{"intent": "refund"}]},
        {"task_executed": True},
        {"pending_refund_review": {"trigger": "refund_threshold"}},
        {"shared_cache_blocked": True},
    ],
)
async def test_l0_replay_never_overrides_current_context(extra: DialogueState) -> None:
    cache = _FakeCache(_cached_answer())
    factory = _make_factory(cache)
    updates = await factory.answer_cache_lookup_node({"message": "coverage?", **extra})
    assert updates["route_after_cache"] == "miss"
    assert updates["shared_cache_blocked"] is True
    assert cache.looked_up == []


@pytest.mark.parametrize("unavailable", [False, True], ids=["existing-history", "unknown-history"])
async def test_l0_context_read_must_prove_history_empty(unavailable: bool) -> None:
    from app.services.llm.base import LLMMessage

    cache = _FakeCache(_cached_answer())
    factory = _make_factory(cache)
    history = AsyncMock(
        side_effect=RuntimeError("isolated history unavailable") if unavailable else None,
        return_value=[LLMMessage(role="user", content="My plan is Premium")],
    )
    factory._history_provider = history
    updates = await factory.answer_cache_lookup_node(
        {"message": "coverage?", "session_id": 7, "turn_id": "context-turn"}
    )
    assert updates["route_after_cache"] == "miss"
    assert updates["shared_cache_blocked"] is True
    assert cache.looked_up == []
    history.assert_awaited_once_with(7)


async def test_l0_history_failure_stays_blocked_if_generation_recovers_empty() -> None:
    from app.services.llm.base import LLMResponse

    cache = _FakeCache(_cached_answer())
    factory = _make_factory(cache)
    factory._history_provider = AsyncMock(side_effect=[RuntimeError("isolated outage"), []])
    factory._llm_service = Mock()
    factory._llm_service.generate = AsyncMock(
        return_value=LLMResponse(content="fresh", model="local")
    )
    semantic = Mock()
    semantic.get = AsyncMock(return_value=_cached_answer())
    semantic.put = AsyncMock()
    semantic.capture_epoch = AsyncMock(return_value="test-epoch")
    factory._semantic_cache = semantic
    state: DialogueState = {"message": "coverage?", "session_id": 7, "turn_id": "sticky-turn"}
    updates = await factory.answer_cache_lookup_node(state)
    state = cast(DialogueState, {**state, **updates})
    response = await factory.direct_response_node(state, {"configurable": {}})
    assert response["response"] == "fresh"
    assert response["shared_cache_blocked"] is True
    assert factory._history_provider.await_count == 2
    semantic.get.assert_not_awaited()
    semantic.put.assert_not_awaited()
