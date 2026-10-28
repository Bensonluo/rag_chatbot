"""Publication checks preserve source-free smalltalk, without excusing policy.

The shared fixture supplies a real SQLite-backed PublicationGuard and blocks
all socket connections. SemanticCacheService and compiled graph are real;
only Redis transport, embeddings and model generation are local substitutes.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.services.chat.answer_cache import CachedAnswer
from app.services.chat.chat_service import HEARTBEAT, STREAM_ERROR, ChatService
from app.services.chat.semantic_cache import SemanticCacheService
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import ToolRegistry
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMMessage, LLMResponse
from app.services.retrieval.publication import PublicationGuard
from app.tests.evaluation.test_publication_visibility import _Case, _ContextRecorder
from app.tests.evaluation.test_publication_visibility import case as case
from app.tests.unit.services.chat.test_cache_eviction import _FakeRedis

_GREETING = "Hello, how can I help?"


class _GreetingProvider(_ContextRecorder):
    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.prompts.append([message.content for message in messages])
        return LLMResponse(content=_GREETING, model=self.model)


def _chat(case: _Case) -> tuple[ChatService, SemanticCacheService, _GreetingProvider]:
    async def epoch() -> str:
        return "publication-smalltalk-evaluation"

    cache = SemanticCacheService(
        embedding_service=case.embeddings,
        redis_client=_FakeRedis(),
        epoch_provider=epoch,
        settings=SimpleNamespace(
            SEMANTIC_CACHE_ENABLED=True,
            SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.5,
            SEMANTIC_CACHE_MAX_ENTRIES=20,
            SEMANTIC_CACHE_TTL_SECONDS=60,
            # Include policy to exercise the publication consumer's own
            # decision even if an operator widens the cache allowlist.
            SEMANTIC_CACHE_SERVED_INTENTS="greeting,chitchat,policy",
        ),
    )
    provider = _GreetingProvider()
    graph = build_dialogue_graph(
        intent_detector=RuleBasedIntentDetector(),
        slot_filler=None,
        tool_registry=ToolRegistry(),
        retrieval_pipeline={"publication_guard": PublicationGuard(case.maker)},
        llm_service=provider,
        semantic_cache=cache,
        checkpointer=MemorySaver(),
    )
    return ChatService(graph=graph), cache, provider


async def _ask(chat: ChatService, session_id: int, transport: str) -> str:
    if transport == "normal":
        return (await chat.process_message(session_id, "Hello", 0)).content
    chunks = [chunk async for chunk in chat.process_message_stream(session_id, "Hello", 0)]
    assert STREAM_ERROR not in chunks
    return "".join(chunk for chunk in chunks if isinstance(chunk, str) and chunk != HEARTBEAT)


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_source_free_greeting_replays_without_a_second_model_call(
    case: _Case, transport: str
) -> None:
    chat, cache, provider = _chat(case)
    assert await _ask(chat, 1, transport) == _GREETING
    entry = await cache.get("Hello")
    assert entry is not None
    assert entry.intent == "greeting"
    assert entry.sources == []
    assert entry.citations == []
    assert len(provider.prompts) == 1
    assert await _ask(chat, 2, transport) == _GREETING
    assert len(provider.prompts) == 1


async def test_empty_policy_evidence_cannot_use_the_smalltalk_exception(case: _Case) -> None:
    chat, cache, provider = _chat(case)
    await cache.put(
        "Hello",
        CachedAnswer(response="UNPUBLISHED_POLICY_BODY", sources=[], intent="policy"),
    )
    assert await cache.get("Hello") is not None
    assert await _ask(chat, 1, "normal") == _GREETING
    assert len(provider.prompts) == 1
