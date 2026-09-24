"""Handoff summarizer: best-effort LLM case summary, never blocks."""

import asyncio
from types import SimpleNamespace
from typing import Any

from app.services.handoff.summarizer import summarize_for_handoff


def _msg(role: str, content: str) -> Any:
    return SimpleNamespace(role=role, content=content)


class _FakeLLM:
    def __init__(self, content: str = "用户要求查询订单 12345 物流。", delay: float = 0.0) -> None:
        self.content = content
        self.delay = delay
        self.calls = 0

    async def generate(self, messages: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return SimpleNamespace(content=self.content)


class _BrokenLLM:
    async def generate(self, messages: Any, **kwargs: Any) -> Any:
        raise RuntimeError("llm down")


class TestSummarizeForHandoff:
    async def test_returns_summary_text_on_success(self):
        messages = [
            _msg("user", "我的订单 12345 迟迟未到"),
            _msg("assistant", "已为您查询物流，显示运输中"),
        ]

        summary = await summarize_for_handoff(messages, _FakeLLM("物流投诉case"))

        assert summary == "物流投诉case"

    async def test_llm_failure_returns_none(self):
        summary = await summarize_for_handoff([_msg("user", "退款")], _BrokenLLM())
        assert summary is None

    async def test_timeout_returns_none(self):
        summary = await summarize_for_handoff(
            [_msg("user", "退款")], _FakeLLM(delay=0.2), timeout_seconds=0.01
        )
        assert summary is None

    async def test_empty_messages_skips_llm(self):
        llm = _FakeLLM()
        assert await summarize_for_handoff([], llm) is None
        assert llm.calls == 0
