"""Agent knowledge tool: deep-funnel task loops gain KB grounding.

The agent registry's six tools are all transactional (orders, refunds,
returns, complaints) — zero knowledge access. Mid-task policy questions
(「退货超过 7 天还能退吗」 while a return is being staged) were answered
from model priors: a documented hallucination surface the claim gate
only catches after the fact. This tool gives the loop the same hybrid
search the RAG leg uses, read-only and fail-open — an outage degrades
to "tell the user KB search is unavailable", never a dead loop.

These tests pin: the tool is agent-only (its registry key is not a
dialogue intent, so the slot pipeline can never route to it), results
carry grounded content, and failures surface as structured messages
the model can act on instead of raw exceptions.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

from app.models.enums.intent import TASK_INTENTS
from app.services.dialogue.tools import (
    ToolRegistry,
    create_default_tool_registry,
    create_knowledge_tool,
)


class FakeHit:
    """SearchResult-shaped object as hybrid_search returns."""

    def __init__(self, content: str, score: float) -> None:
        self.content = content
        self.score = score
        self.document_id = f"doc-{score}"
        self.metadata = None


def _hybrid(hits: list[FakeHit] | Exception) -> Mock:
    hybrid = Mock()
    if isinstance(hits, Exception):
        hybrid.search = AsyncMock(side_effect=hits)
    else:
        hybrid.search = AsyncMock(return_value=hits)
    return hybrid


async def _execute(hybrid: Mock, args: dict[str, Any]) -> dict[str, Any]:
    registry = ToolRegistry()
    registry.register(create_knowledge_tool(hybrid))
    result = await registry.execute("knowledge_search", args)
    return {"success": result.success, "data": result.data}


class TestKnowledgeTool:
    def test_tool_is_agent_only_and_exported(self) -> None:
        tool = create_knowledge_tool(_hybrid([]))
        registry = ToolRegistry()
        registry.register(tool)

        schemas = registry.to_function_schemas()

        assert tool.name == "search_knowledge_base"
        assert any(s["function"]["name"] == tool.name for s in schemas)
        # Registry key only — the slot pipeline routes on dialogue
        # intents, so a non-intent key keeps this tool agent-loop-only
        # (same doctrine as get_recent_orders).
        assert tool.intent not in TASK_INTENTS

    async def test_handler_returns_grounded_results(self) -> None:
        hybrid = _hybrid([FakeHit("退货政策：7天内可退…", 0.92), FakeHit("运费规则…", 0.61)])

        outcome = await _execute(hybrid, {"query": "退货政策"})

        assert outcome["success"] is True
        assert outcome["data"]["count"] == 2
        assert outcome["data"]["results"][0]["content"] == "退货政策：7天内可退…"
        assert outcome["data"]["results"][0]["score"] == 0.92
        # One hybrid search per call, top_k=3.
        request = hybrid.search.await_args.args[0]
        assert request.query == "退货政策"
        assert request.top_k == 3

    async def test_empty_query_short_circuits_without_search(self) -> None:
        hybrid = _hybrid([FakeHit("x", 1.0)])

        outcome = await _execute(hybrid, {"query": "   "})

        assert outcome["data"]["results"] == []
        hybrid.search.assert_not_awaited()

    async def test_search_failure_fails_open_with_structured_message(self) -> None:
        hybrid = _hybrid(RuntimeError("qdrant down"))

        outcome = await _execute(hybrid, {"query": "保修条款"})

        # Fail-open: the registry reports success with an empty result
        # set and a model-actionable message — the loop keeps serving.
        assert outcome["success"] is True
        assert outcome["data"]["results"] == []
        assert "不可用" in outcome["data"]["message"]

    def test_default_registry_stays_transactional_until_wired(self) -> None:
        """The knowledge tool is a factory-wired upgrade, not part of the
        default registry — tests that build the default registry keep
        their tool surface unchanged."""
        registry = create_default_tool_registry()
        assert registry.get_tool_by_name("search_knowledge_base") is None
