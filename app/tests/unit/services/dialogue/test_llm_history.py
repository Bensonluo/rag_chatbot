"""Primary-pipeline LLM calls carry bounded conversation history.

Follow-up questions ("那运费谁出？") dominate real CS conversations; a
generation path that sees only the current message cannot resolve them.
Every node-level generator prepends the session's recent turns via the
history provider.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

from app.services.dialogue.state import DialogueState


def _capturing_llm() -> Any:
    """LLM mock that records the message list handed to generate()."""
    llm = Mock()
    llm.generate = AsyncMock(return_value=Mock(content="好的"))
    return llm


def _sent_messages(llm: Any) -> list[Any]:
    return list(llm.generate.await_args.args[0])


def _factory(llm: Any, history_provider: Any = None, system_prompt: Any = None) -> Any:
    from app.services.dialogue.nodes import NodeFactory

    return NodeFactory(
        intent_detector=Mock(),
        slot_filler=None,
        tool_registry=Mock(),
        retrieval_pipeline={},
        llm_service=llm,
        guardrail_service=None,
        graph_retrieval_service=None,
        history_provider=history_provider,
        system_prompt=system_prompt,
    )


def _turns(*pairs: tuple[str, str]) -> list[Any]:
    from app.services.llm.base import LLMMessage

    return [LLMMessage(role=role, content=content) for role, content in pairs]


class TestDirectPathHistory:
    async def test_direct_response_prepends_history(self):
        llm = _capturing_llm()

        async def provider(session_id: int) -> list[Any]:
            assert session_id == 7
            return _turns(
                ("user", "昨天买的手机想退货"),
                ("assistant", "好的，已受理退货"),
            )

        factory = _factory(llm, history_provider=provider)
        state: DialogueState = {"message": "那运费谁出？", "session_id": 7}

        await factory.direct_response_node(state)

        messages = _sent_messages(llm)
        assert messages[0].role == "system"
        assert [(m.role, m.content) for m in messages[1:]] == [
            ("user", "昨天买的手机想退货"),
            ("assistant", "好的，已受理退货"),
            ("user", "那运费谁出？"),
        ]


class TestRagPathHistory:
    async def test_rag_generation_keeps_history_before_context(self):
        llm = _capturing_llm()

        async def provider(session_id: int) -> list[Any]:
            return _turns(("user", "iPhone 退货怎么办理"))

        factory = _factory(llm, history_provider=provider)
        state: DialogueState = {"session_id": 3}

        await factory._generate_with_rag(
            intent="question",
            message="那运费谁出？",
            retrieved_docs=[{"content": "退货包邮，运费商家承担"}],
            state=state,
        )

        messages = _sent_messages(llm)
        assert [m.role for m in messages] == ["system", "user", "user"]
        assert messages[1].content == "iPhone 退货怎么办理"
        assert "退货包邮" in messages[2].content
        assert "那运费谁出？" in messages[2].content


class TestHistoryBoundedAndSafe:
    async def test_history_is_capped_by_the_provider_tail(self):
        llm = _capturing_llm()

        async def provider(session_id: int) -> list[Any]:
            return _turns(*[(("user", f"第{i}条")) for i in range(1, 9)])

        factory = _factory(llm, history_provider=provider)
        state: DialogueState = {"message": "当前", "session_id": 1}

        await factory.direct_response_node(state)

        messages = _sent_messages(llm)
        # Provider already bounds; the node must not re-expand it.
        # (+1 for the persona system message.)
        assert len(messages) == 10

    async def test_provider_failure_degrades_to_current_message_only(self):
        llm = _capturing_llm()

        async def provider(session_id: int) -> list[Any]:
            raise RuntimeError("history backend down")

        factory = _factory(llm, history_provider=provider)
        state: DialogueState = {"message": "当前", "session_id": 1}

        await factory.direct_response_node(state)

        messages = _sent_messages(llm)
        assert messages[0].role == "system"
        assert [(m.role, m.content) for m in messages[1:]] == [("user", "当前")]

    async def test_no_provider_keeps_single_message(self):
        llm = _capturing_llm()
        factory = _factory(llm)
        state: DialogueState = {"message": "你好", "session_id": 1}

        await factory.direct_response_node(state)

        messages = _sent_messages(llm)
        assert messages[0].role == "system"
        assert [(m.role, m.content) for m in messages[1:]] == [("user", "你好")]


class TestSystemPersona:
    async def test_default_cs_persona_prepended(self):
        llm = _capturing_llm()
        factory = _factory(llm)  # system_prompt=None → built-in persona
        state: DialogueState = {"message": "退货政策是什么", "session_id": 1}

        await factory.direct_response_node(state)

        messages = _sent_messages(llm)
        assert messages[0].role == "system"
        assert "客服" in messages[0].content
        assert messages[-1].role == "user"

    async def test_custom_system_prompt_override(self):
        llm = _capturing_llm()
        factory = _factory(llm, system_prompt="测试人设：只谈物流")
        state: DialogueState = {"message": "你好", "session_id": 1}

        await factory.direct_response_node(state)

        assert _sent_messages(llm)[0].content == "测试人设：只谈物流"

    async def test_system_message_precedes_history(self):
        llm = _capturing_llm()

        async def provider(session_id: int) -> list[Any]:
            return _turns(
                ("user", "昨天买的手机想退货"),
                ("assistant", "好的，已受理退货"),
            )

        factory = _factory(llm, history_provider=provider)
        state: DialogueState = {"message": "那运费谁出？", "session_id": 7}

        await factory.direct_response_node(state)

        roles = [m.role for m in _sent_messages(llm)]
        assert roles == ["system", "user", "assistant", "user"]
