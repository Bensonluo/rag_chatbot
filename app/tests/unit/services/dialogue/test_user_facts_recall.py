"""User-facts recall injection (Phase B2 — read side).

The write side (#65) extracts durable user facts post-durability; this
seam makes them visible where responses are generated. Industry
baseline (mem0 session-start recall, 阿里小蜜用户画像注入): the user's
long-term facts enter the prompt as bounded system context on both
generation surfaces — the slot-pipeline generators and the agent loop
(context_note) — best-effort, so a memory read can never fail a chat.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

from app.services.agent.service import AgentResult
from app.services.dialogue.nodes import NodeFactory


def _llm() -> Mock:
    llm = Mock()
    llm.generate = AsyncMock(return_value=Mock(content="好的"))
    return llm


def _factory(
    llm: Any,
    facts_provider: Any = None,
    agent_service: Any = None,
) -> NodeFactory:
    return NodeFactory(
        intent_detector=Mock(),
        slot_filler=None,
        tool_registry=Mock(),
        llm_service=llm,
        user_facts_provider=facts_provider,
        agent_service=agent_service,
    )


async def _system_content(llm: Mock, factory: NodeFactory, state: dict[str, Any]) -> str:
    await factory._call_llm("您好", state=state)  # type: ignore[arg-type]
    messages = llm.generate.await_args.args[0]
    return str(messages[0].content)


class TestGenerationPathInjection:
    async def test_facts_appended_to_persona(self):
        llm = _llm()
        provider = AsyncMock(return_value=["用户是 PLUS 会员", "偏好上午配送"])
        factory = _factory(llm, facts_provider=provider)

        content = await _system_content(llm, factory, {"session_id": 1, "user_id": 7})

        provider.assert_awaited_once_with(7)
        assert "用户是 PLUS 会员" in content
        assert "偏好上午配送" in content
        # Persona stays intact — facts extend it, they never replace it.
        assert "客服" in content

    async def test_empty_facts_leave_persona_unchanged(self):
        from app.services.llm.prompt_templates import PromptTemplates

        llm = _llm()
        provider = AsyncMock(return_value=[])
        factory = _factory(llm, facts_provider=provider)

        content = await _system_content(llm, factory, {"session_id": 1, "user_id": 7})

        assert content == PromptTemplates.get_cs_system_prompt()

    async def test_provider_failure_degrades_to_persona(self):
        llm = _llm()
        provider = AsyncMock(side_effect=RuntimeError("memory backend down"))
        factory = _factory(llm, facts_provider=provider)

        content = await _system_content(llm, factory, {"session_id": 1, "user_id": 7})

        assert "用户是" not in content  # no facts block, generation continues
        llm.generate.assert_awaited_once()

    async def test_missing_user_id_skips_provider(self):
        llm = _llm()
        provider = AsyncMock(return_value=["用户是 PLUS 会员"])
        factory = _factory(llm, facts_provider=provider)

        await _system_content(llm, factory, {"session_id": 1})

        provider.assert_not_awaited()

    async def test_no_provider_behaves_as_before(self):
        from app.services.llm.prompt_templates import PromptTemplates

        llm = _llm()
        factory = _factory(llm, facts_provider=None)

        content = await _system_content(llm, factory, {"session_id": 1, "user_id": 7})

        assert content == PromptTemplates.get_cs_system_prompt()


class TestAgentPathInjection:
    @staticmethod
    def _agent(response: str) -> Mock:
        agent = Mock()
        agent.run = AsyncMock(return_value=AgentResult(response=response))
        return agent

    async def test_facts_join_slots_in_context_note(self):
        provider = AsyncMock(return_value=["用户是 PLUS 会员"])
        agent = self._agent("已处理")
        factory = _factory(_llm(), facts_provider=provider, agent_service=agent)

        await factory.handle_agent_node(
            {
                "message": "退款",
                "session_id": 1,
                "user_id": 7,
                "filled_slots": {"order_id": "ORD1001"},
            }
        )

        note = agent.run.await_args.kwargs["context_note"]
        assert "order_id=ORD1001" in note
        assert "用户是 PLUS 会员" in note

    async def test_facts_alone_carry_the_note(self):
        provider = AsyncMock(return_value=["偏好上午配送"])
        agent = self._agent("已处理")
        factory = _factory(_llm(), facts_provider=provider, agent_service=agent)

        await factory.handle_agent_node({"message": "查订单", "session_id": 1, "user_id": 7})

        note = agent.run.await_args.kwargs["context_note"]
        assert "偏好上午配送" in note

    async def test_provider_failure_keeps_agent_running(self):
        provider = AsyncMock(side_effect=RuntimeError("down"))
        agent = self._agent("已处理")
        factory = _factory(_llm(), facts_provider=provider, agent_service=agent)

        updates = await factory.handle_agent_node(
            {"message": "查订单", "session_id": 1, "user_id": 7}
        )

        agent.run.assert_awaited_once()
        assert updates["response"] == "已处理"
