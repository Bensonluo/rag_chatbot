"""L0 answer-cache service and write-path contract.

Pins the three invariants from docs/cache-layering-plan.md:
- fail-open — a Redis outage is a miss / skipped write, never a raise;
- epoch-scoped — a KB mutation rotates the key namespace wholesale;
- stateless-turns-only — ChatService writes only anonymous, grounded,
  stateless turns (nothing personalized can ever be replayed).
"""

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.chat.answer_cache import (
    AnswerCacheService,
    CachedAnswer,
    normalize_message,
)
from app.services.chat.chat_service import ChatService

if TYPE_CHECKING:
    from redis.asyncio import Redis


class _FakeRedis:
    """Async get/set surface the AnswerCacheService touches."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.fail = False

    async def get(self, key: str) -> Any:
        if self.fail:
            raise ConnectionError("redis down")
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int = 0) -> Any:
        if self.fail:
            raise ConnectionError("redis down")
        self.store[key] = value
        self.ttl = ex
        return True


def _make_service(**kwargs: Any) -> tuple[AnswerCacheService, _FakeRedis]:
    svc = AnswerCacheService(redis_url="redis://localhost:6379/0", ttl_seconds=60, **kwargs)
    fake = _FakeRedis()
    svc._redis = cast("Redis[str]", fake)
    return svc, fake


@pytest.fixture(autouse=True)
def _hermetic_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Key derivation must not depend on a live Redis epoch in unit tests."""
    monkeypatch.setattr(
        "app.services.chat.answer_cache.get_kb_epoch",
        AsyncMock(return_value="epoch-test"),
    )


class TestNormalizeMessage:
    def test_whitespace_runs_collapse_and_case_folds(self):
        assert normalize_message("  RETURN   policy ") == "return policy"

    def test_trim_and_casefold(self):
        assert normalize_message("退货政策") == normalize_message("退货政策")


class TestAnswerCacheService:
    async def test_put_then_get_roundtrip(self):
        svc, _ = _make_service()
        await svc.put(
            "退货政策是什么",
            response="7天无理由退货",
            sources=["doc-1"],
            intent="question",
        )
        hit = await svc.get("退货政策是什么")
        assert hit is not None
        assert hit.response == "7天无理由退货"
        assert hit.sources == ["doc-1"]
        assert hit.intent == "question"

    async def test_normalized_variants_share_one_entry(self):
        svc, _ = _make_service()
        await svc.put("Return  POLICY", response="ok", sources=["d"], intent="question")
        assert await svc.get("return policy") is not None

    async def test_unknown_message_misses(self):
        svc, _ = _make_service()
        assert await svc.get("没缓存过的问题") is None

    async def test_epoch_rotation_invalidates_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        svc, _ = _make_service()
        await svc.put("q", response="a", sources=["d"], intent="question")
        monkeypatch.setattr(
            "app.services.chat.answer_cache.get_kb_epoch", AsyncMock(return_value="new-epoch")
        )
        assert await svc.get("q") is None

    async def test_redis_outage_get_is_a_miss_not_raise(self):
        svc, fake = _make_service()
        fake.fail = True
        assert await svc.get("q") is None

    async def test_redis_outage_put_does_not_raise(self):
        svc, fake = _make_service()
        fake.fail = True
        await svc.put("q", response="a", sources=["d"], intent="question")

    async def test_oversized_response_is_not_stored(self):
        svc, _ = _make_service(max_response_chars=10)
        await svc.put("q", response="x" * 11, sources=["d"], intent="question")
        assert await svc.get("q") is None

    async def test_unsourced_response_is_not_stored(self):
        svc, _ = _make_service()
        await svc.put("q", response="a", sources=[], intent="question")
        assert await svc.get("q") is None


class _RecordingCache(AnswerCacheService):
    """ChatService write-path spy: records eligible puts."""

    def __init__(self) -> None:
        super().__init__(redis_url="redis://localhost:6379/0", ttl_seconds=60)
        self.puts: list[dict[str, Any]] = []

    async def put(self, message: str, *, response: str, sources: list[str], intent: str) -> None:
        self.puts.append(
            {"message": message, "response": response, "sources": sources, "intent": intent}
        )

    async def get(self, message: str) -> CachedAnswer | None:  # pragma: no cover
        return None


def _graph_returning(result: dict[str, Any]) -> Mock:
    graph = Mock()
    graph.ainvoke = AsyncMock(return_value=result)
    return graph


class TestChatServiceCacheWritePath:
    async def test_eligible_anonymous_grounded_turn_is_cached(self):
        cache = _RecordingCache()
        service = ChatService(
            graph=_graph_returning(
                {
                    "message": "退货政策是什么",
                    "response": "7天无理由退货",
                    "sources": ["doc-1"],
                    "intent": "question",
                }
            ),
            answer_cache=cache,
        )
        await service.process_message(session_id=1, message="退货政策是什么", user_id=0)
        assert len(cache.puts) == 1
        assert cache.puts[0]["sources"] == ["doc-1"]

    async def test_write_key_uses_sanitized_message(self):
        """The read site looks up with post-guardrail text; writes must match."""
        cache = _RecordingCache()
        service = ChatService(
            graph=_graph_returning(
                {
                    "message": "我的手机号是[REDACTED]",
                    "response": "好的",
                    "sources": ["doc-1"],
                    "intent": "question",
                }
            ),
            answer_cache=cache,
        )
        await service.process_message(session_id=1, message="我的手机号是13800000000", user_id=0)
        assert cache.puts[0]["message"] == "我的手机号是[REDACTED]"

    async def test_identified_user_never_cached(self):
        cache = _RecordingCache()
        service = ChatService(
            graph=_graph_returning({"response": "a", "sources": ["d"], "intent": "question"}),
            answer_cache=cache,
        )
        await service.process_message(session_id=1, message="q", user_id=42)
        assert cache.puts == []

    @pytest.mark.parametrize(
        "extra",
        [
            {"pending_slots": ["order_id"]},
            {"pending_confirmation": {"intent": "refund", "args": {}}},
            {"executed_tools": [{"tool": "refund"}]},
            {"blocked": True},
            {"sources": []},
            {"response": ""},
        ],
    )
    async def test_ineligible_turns_are_never_cached(self, extra: dict[str, Any]) -> None:
        cache = _RecordingCache()
        base = {"message": "q", "response": "a", "sources": ["d"], "intent": "question"}
        service = ChatService(graph=_graph_returning({**base, **extra}), answer_cache=cache)
        await service.process_message(session_id=1, message="q", user_id=0)
        assert cache.puts == []

    async def test_completed_stream_caches(self):
        cache = _RecordingCache()

        async def fake_ainvoke(_state: Any, config: Any) -> dict[str, Any]:
            # Mirror a terminal node: content reaches the consumer via the
            # stream queue in the invoke config, not the return value.
            config["configurable"]["stream_queue"].put_nowait("7天无理由退货")
            return {
                "message": "退货政策是什么",
                "response": "7天无理由退货",
                "sources": ["doc-1"],
                "intent": "question",
            }

        graph = Mock()
        graph.ainvoke = fake_ainvoke
        service = ChatService(graph=graph, answer_cache=cache)
        chunks = [
            item
            async for item in service.process_message_stream(
                session_id=1, message="退货政策是什么", user_id=0
            )
        ]
        assert any(isinstance(c, str) and c for c in chunks)
        assert len(cache.puts) == 1
