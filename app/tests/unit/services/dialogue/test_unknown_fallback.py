"""Deterministic retrieval fallback for question-shaped UNKNOWN turns.

Deferred finding from the routing golden-set eval (862a777): an unknown
classification is not necessarily smalltalk — 「量子纠缠是什么」 fell
through route_intent_node's default branch to the chitchat LLM leg,
answering a knowledge question with an ungrounded free generation and
recording no knowledge gap. The fix is a controlled-node fallback: a
deterministic question-shape check routes the turn into retrieval,
where an honest KB miss lands the no-evidence copy and the gap recorder
sees the turn.

Contracts pinned here: the routing decision boundary (interrogative
tokens zh, word-bounded en interrogatives, terminal ？/?), statements
and keyword-classified smalltalk stay direct, and the helper keeps the
283064d lesson — an ASCII interrogative inside a longer word ("what" in
"whatever") is not a question.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from app.services.dialogue.nodes import NodeFactory, _looks_like_knowledge_question


def _factory() -> NodeFactory:
    return NodeFactory(intent_detector=Mock(), slot_filler=None, tool_registry=Mock())


async def _route(intent: str, message: str) -> dict[str, Any]:
    return await _factory().route_intent_node({"intent": intent, "message": message})


class TestUnknownQuestionFallback:
    async def test_unknown_zh_knowledge_question_routes_to_rag(self) -> None:
        assert await _route("unknown", "量子纠缠是什么") == {"route": "rag"}

    async def test_unknown_en_interrogative_routes_to_rag(self) -> None:
        assert await _route("unknown", "what is quantum entanglement") == {"route": "rag"}

    async def test_unknown_terminal_question_mark_routes_to_rag(self) -> None:
        # No interrogative token — the trailing ? alone marks the shape.
        assert await _route("unknown", "Do you offer gift wrapping?") == {"route": "rag"}

    async def test_unknown_zh_statement_stays_direct(self) -> None:
        assert await _route("unknown", "量子纠缠很神奇啊") == {"route": "direct"}

    async def test_unknown_en_statement_stays_direct(self) -> None:
        assert await _route("unknown", "nothing matches here at all") == {"route": "direct"}

    async def test_unknown_whitespace_message_stays_direct(self) -> None:
        assert await _route("unknown", "   ") == {"route": "direct"}

    async def test_classified_smalltalk_never_takes_the_fallback(self) -> None:
        # Keyword-bearing smalltalk never reaches the unknown residue:
        # 在吗 is a GREETING keyword, the joke ask is CHITCHAT.
        assert await _route("greeting", "在吗？") == {"route": "direct"}
        assert await _route("chitchat", "讲个笑话") == {"route": "direct"}


class TestQuestionShapeBoundary:
    def test_en_interrogative_matches(self) -> None:
        assert _looks_like_knowledge_question("how does one entangle qubits")

    def test_en_interrogative_inside_longer_word_does_not_match(self) -> None:
        # 283064d at helper level: \b word boundaries, not substrings.
        assert not _looks_like_knowledge_question("whatever you say")
        assert not _looks_like_knowledge_question("nowhere to be found")

    def test_zh_interrogative_tokens_match(self) -> None:
        assert _looks_like_knowledge_question("介绍一下你们的会员体系")
        assert _looks_like_knowledge_question("这两者有什么区别")

    def test_terminal_fullwidth_question_mark_matches(self) -> None:
        assert _looks_like_knowledge_question("支持货到付款吗？")

    def test_plain_statement_does_not_match(self) -> None:
        assert not _looks_like_knowledge_question("今天心情不太好")

    def test_empty_text_does_not_match(self) -> None:
        assert not _looks_like_knowledge_question("  ")
