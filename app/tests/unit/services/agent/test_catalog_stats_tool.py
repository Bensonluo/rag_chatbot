"""Catalog grounding tool: the agent must state real store numbers.

Regression frame (2026-10-28 demo): asked how many categories the store
carries, the agent answered「160多」from priors — the actual count is 151.
Prompt rule 11 (prefer vague wording over invented numbers) was the
stopgap; ``get_catalog_stats`` is the source of truth that rule now
points at.
"""

import json
from typing import Any

from app.services.agent import AgentService
from app.services.dialogue.tools import (
    create_default_tool_registry,
    mock_catalog_stats,
)
from app.services.llm.base import LLMResponse


class ScriptedLLM:
    """Fake LLM replaying queued responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)

    async def generate_with_tools(self, messages: Any, tools: Any) -> LLMResponse:
        return self.responses.pop(0)


def _tool_call(name: str, args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content="",
        model="fake",
        tool_calls=[{"id": "c1", "name": name, "arguments": json.dumps(args)}],
    )


def _final_answer(text: str) -> LLMResponse:
    return LLMResponse(content=text, model="fake")


class TestCatalogStatsHandler:
    def test_numbers_match_provider_constants(self):
        stats = mock_catalog_stats({})
        assert stats["category_count"] == 151
        assert stats["total_skus"] == 154_624
        assert stats["brand_count"] == 32
        assert stats["skus_per_category"] == 1_024
        assert stats["models_per_brand"] == 8
        assert stats["variants_per_model"] == 4

    def test_domain_breakdown_complete_and_deterministic(self):
        first = mock_catalog_stats({})
        second = mock_catalog_stats({})
        assert first == second  # pure function of the taxonomy
        domains = first["domains"]
        assert len(domains) == first["domain_count"] == 18
        # The breakdown must add back up to the headline count.
        assert sum(d["category_count"] for d in domains) == first["category_count"]
        for entry in domains:
            assert entry["domain"]
            assert entry["category_count"] >= 1
            assert entry["example"]  # zh display name, present per domain


class TestCatalogStatsRegistration:
    def test_registered_readonly_no_confirmation(self):
        registry = create_default_tool_registry()
        tool = registry.get_tool_by_name("get_catalog_stats")
        assert tool is not None
        assert tool.requires_confirmation is False
        assert tool.required_slots == []
        assert tool.authorize is None  # public store facts: no ownership check

    async def test_registry_executes_and_returns_stats(self):
        registry = create_default_tool_registry()
        result = await registry.execute("catalog_stats", {}, user_id=None)
        assert result.success is True
        assert result.data["category_count"] == 151
        assert result.data["total_skus"] == 154_624


class TestCatalogStatsInAgentLoop:
    async def test_store_overview_question_grounded_by_tool(self):
        # The 2026-10-28 shape: an overview question must come back with
        # the real numbers, not a prior-invented count.
        service = AgentService(
            llm_service=ScriptedLLM(
                [
                    _tool_call("get_catalog_stats", {}),
                    _final_answer("我们目前有 151 个商品类目、超过 15 万件商品。"),
                ]
            ),
            tool_registry=create_default_tool_registry(),
            max_steps=3,
        )
        result = await service.run("你们店有多少类目？")
        assert result.executed_tools == ["get_catalog_stats"]
        assert result.outcome == "answered"
        assert "151" in result.response

    async def test_system_prompt_names_the_tool(self):
        # Rule 11 must route store-overview numbers to the tool.
        from app.services.agent.service import SYSTEM_PROMPT

        assert "get_catalog_stats" in SYSTEM_PROMPT
