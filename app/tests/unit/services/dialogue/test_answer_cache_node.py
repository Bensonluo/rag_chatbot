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
from typing import Any
from unittest.mock import Mock

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

        assert updates == {"route_after_cache": "miss"}
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

        assert updates == {"route_after_cache": "miss"}
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
