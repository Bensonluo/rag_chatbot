"""Compiled graph → ChatService → durable messages agree on handoff outcomes."""

from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.ticket import HandoffTicket
from app.models.enums.intent import Intent
from app.models.enums.message import MessageRole
from app.services.agent.service import AgentService
from app.services.chat.chat_service import ChatService
from app.services.chat.persistence import ChatMessagePersister
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import ToolRegistry, create_escalate_tool
from app.services.handoff.service import HandoffService
from app.services.intent.base import IntentResult


@pytest.fixture
async def handoff_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("agent_path", [False, True])
@pytest.mark.parametrize("available", [False, True])
async def test_handoff_visible_reply_matches_durable_outcome(
    handoff_db, stream, agent_path, available
):
    def unavailable_db():
        raise RuntimeError("test handoff database unavailable")

    handoff = HandoffService(handoff_db if available else unavailable_db)
    registry = ToolRegistry()
    registry.register(create_escalate_tool(handoff))
    llm = Mock()
    llm.generate_with_tools = AsyncMock(
        return_value=Mock(
            content="",
            tool_calls=[
                {"id": "escalate-1", "name": "escalate_to_human", "arguments": "{}"},
            ],
        )
    )
    detector = Mock()
    detector.detect_with_confidence = AsyncMock(
        return_value=IntentResult(
            intent=Intent.REFUND if agent_path else Intent.HANDOFF,
            confidence=1.0,
        )
    )
    chat = ChatService(
        graph=build_dialogue_graph(
            intent_detector=detector,
            slot_filler=None,
            tool_registry=registry,
            retrieval_pipeline={},
            llm_service=None,
            handoff_service=handoff,
            agent_service=AgentService(llm, registry) if agent_path else None,
        ),
        persister=ChatMessagePersister(handoff_db),
    )
    message = "Help me with this order" if agent_path else "human agent"
    if stream:
        chunks = [
            chunk
            async for chunk in chat.process_message_stream(1, message, 0)
            if isinstance(chunk, str)
        ]
        reply = "".join(chunks)
    else:
        response = await chat.process_message(1, message, 0)
        reply = response.content

    async with handoff_db() as session:
        assistant = (
            await session.execute(select(Message).where(Message.role == MessageRole.ASSISTANT))
        ).scalar_one()
        tickets = (await session.execute(select(HandoffTicket))).scalars().all()

    assert assistant.content == reply
    metadata = assistant.message_metadata
    assert metadata is not None
    assert metadata["handoff_outcome"] == ("queued" if available else "failed")
    if available:
        assert len(tickets) == 1
        assert metadata["handoff_ticket_id"] == tickets[0].id
        assert f"#{tickets[0].id}" in reply
        assert "position in the queue: 1" in reply
    else:
        assert tickets == []
        assert "handoff_ticket_id" not in metadata
        assert "couldn't confirm" in reply
        assert "try again" in reply
        assert "please hold on" not in reply
    if agent_path:
        llm.generate_with_tools.assert_awaited_once()
        assert metadata["executed_tools"][0]["ok"] is available
    else:
        llm.generate_with_tools.assert_not_awaited()
