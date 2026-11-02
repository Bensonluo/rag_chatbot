"""Deterministic collection-state projection for agent-served task turns.

The agent lane owns HOW required info is asked (natural multi-point
questions, free ordering), but WHAT a complete task requires stays the
shared ``INTENT_SLOT_SCHEMAS`` vocabulary — the same model the slot
pipeline reads. Every successful agent turn for a task intent projects
the deterministic collection state into the graph state:

- regex extraction over the turn's message merged with prior fills
  (zero LLM, admission-safe — ``canonical_only`` patterns gate the
  values), then ``get_missing_slots`` over the merge;
- the public metadata contract (``pending_slots`` / ``filled_slots``)
  holds no matter which lane served the turn — the live probe pins it,
  and it broke live (2026-11-02: agent-first routing left task turns
  with empty metadata while the probe still asserted the old shape);
- the fallback slot pipeline resumes from the same state if the agent
  lane dies mid-collection (one contract, two implementations);
- the collection CLOSES only when the intent's own business tool ran
  (read-only helpers like get_recent_orders keep it open) or an
  irreversible action is staged awaiting confirmation.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.services.agent.service import AgentResult
from app.services.dialogue.nodes import NodeFactory
from app.services.dialogue.state import DialogueState


def _factory(
    agent_result: AgentResult, business_tool: str | None = None
) -> tuple[NodeFactory, Mock]:
    agent = Mock()
    agent.run = AsyncMock(return_value=agent_result)
    registry = Mock()
    registry.get_tool_for_intent = Mock(
        return_value=SimpleNamespace(name=business_tool) if business_tool else None
    )
    factory = NodeFactory(
        intent_detector=Mock(),
        slot_filler=Mock(),
        tool_registry=registry,
        retrieval_pipeline={},
        llm_service=None,
        guardrail_service=None,
        graph_retrieval_service=None,
        agent_service=agent,
    )
    return factory, agent


class TestAgentCollectionProjection:
    async def test_complaint_opener_projects_missing_required_slots(self):
        """我要投诉 on the agent lane must still report what a complete
        complaint needs — the probe's turn-0 contract, restored."""
        factory, _ = _factory(AgentResult(response="请问您要投诉哪个方面？"))
        state: DialogueState = {
            "message": "我要投诉",
            "session_id": 1,
            "intent": "complaint",
        }

        updates = await factory.handle_agent_node(state)

        assert updates["filled_slots"] == {}
        assert updates["pending_slots"] == ["category", "description"]

    async def test_extractable_values_merge_into_projection(self):
        """An order id present in the message lands in the projection;
        only the free-form reason stays pending (regex cannot see it —
        the projection under-reports honestly until the tool runs)."""
        factory, _ = _factory(AgentResult(response="好的，请告诉我退款原因。"))
        state: DialogueState = {
            "message": "我要退款，订单号 ORD1001",
            "session_id": 1,
            "intent": "refund",
        }

        updates = await factory.handle_agent_node(state)

        assert updates["filled_slots"] == {"order_id": "ORD1001"}
        assert updates["pending_slots"] == ["reason"]

    async def test_prior_fills_carry_into_projection(self):
        """Values extracted on earlier agent turns persist through the
        checkpointer; the projection accumulates instead of resetting."""
        factory, _ = _factory(AgentResult(response="请再补充一些细节。"))
        state: DialogueState = {
            "message": "就这样",
            "session_id": 1,
            "intent": "complaint",
            "filled_slots": {"category": "商品质量"},
        }

        updates = await factory.handle_agent_node(state)

        assert updates["filled_slots"] == {"category": "商品质量"}
        assert updates["pending_slots"] == ["description"]

    async def test_business_tool_execution_closes_collection(self):
        """The intent's OWN business tool ran ⇒ the task consumed its
        args: pending empties (collection closed, the next 我要投诉
        starts clean) while filled keeps the regex-visible record —
        the authoritative args live in the executed_tools audit trail
        and the pre-existing task_executed flag blocks stale replay."""
        factory, _ = _factory(
            AgentResult(
                response="投诉工单已提交：CP123456",
                tool_trace=[{"tool": "submit_complaint", "ok": True}],
            ),
            business_tool="submit_complaint",
        )
        state: DialogueState = {
            "message": "商品质量有问题，订单号 ORD1001，刚收到的货是破的",
            "session_id": 1,
            "intent": "complaint",
            "filled_slots": {"category": "商品质量"},
        }

        updates = await factory.handle_agent_node(state)

        assert updates["task_executed"] is True
        assert updates["filled_slots"] == {"category": "商品质量", "order_id": "ORD1001"}
        assert updates["pending_slots"] == []

    async def test_read_only_helper_tool_keeps_collection_open(self):
        """Read-only helpers (get_recent_orders before a refund) are
        context gathering, not task completion: the projection keeps
        reporting the missing required slots — live 2026-11-02, a
        get_recent_orders call on 我要退款 emptied pending_slots and
        broke both the probe contract and the bare-answer resume seam."""
        factory, _ = _factory(
            AgentResult(
                response="请问您的订单号是多少？",
                tool_trace=[{"tool": "get_recent_orders", "ok": True}],
            ),
            business_tool="process_refund",
        )
        state: DialogueState = {
            "message": "我要退款",
            "session_id": 1,
            "intent": "refund",
        }

        updates = await factory.handle_agent_node(state)

        assert updates["task_executed"] is True
        assert updates["filled_slots"] == {}
        assert updates["pending_slots"] == ["order_id", "reason"]

    async def test_non_task_intent_leaves_slot_state_untouched(self):
        """Agent turns outside TASK_INTENTS never write slot state —
        the checkpointer's values pass through unchanged."""
        factory, _ = _factory(AgentResult(response="好的，还有什么可以帮您？"))
        state: DialogueState = {
            "message": "今天天气如何",
            "session_id": 1,
            "intent": "unknown",
            "filled_slots": {"order_id": "1111"},
        }

        updates = await factory.handle_agent_node(state)

        assert "filled_slots" not in updates
        assert "pending_slots" not in updates

    async def test_staged_irreversible_action_owns_the_next_turn(self):
        """A staged run (confirmation pending, nothing executed) must
        leave pending_slots EMPTY: the slot-prompt skip branch treats a
        bare short reply as a slot answer while pending_slots is
        non-empty, so a projected list would let "confirm" be eaten as
        a slot value and the staged action could never resolve."""
        factory, _ = _factory(
            AgentResult(
                response="确认要为订单 ORD1001 办理退款吗？",
                pending_confirmation={"intent": "refund", "args": {"order_id": "ORD1001"}},
            )
        )
        state: DialogueState = {
            "message": "订单 ORD1001 退款",
            "session_id": 1,
            "intent": "refund",
        }

        updates = await factory.handle_agent_node(state)

        assert updates["filled_slots"] == {"order_id": "ORD1001"}
        assert updates["pending_slots"] == []
