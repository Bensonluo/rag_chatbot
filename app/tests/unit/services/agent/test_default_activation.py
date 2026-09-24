"""Deep-funnel activation: the agent loop is the default path.

The north star weights the tool-loop funnel as the real need, but
AGENT_TOOLS_ENABLED shipped False with no deployment override anywhere
(compose / env.example / deploy) — the default WAS the product: the
deep funnel never ran, and every agent capability built on top of it
(knowledge grounding, human escalation, loop telemetry) was inert.
This pins the product decision that the agent loop is on by default.
Every safety net is built and tested — step bound, request LLM budget,
provider-fallback into the slot pipeline, staged confirmation for
irreversible tools, claim gate on the final answer.

Also pinned: the system prompt must teach the tool-era doctrines —
ground policy claims via the KB tool, and escalate by CALLING the
handoff tool instead of advising the user to ask for a human (the
pre-tool doctrine made escalate_to_human unreachable in practice).
"""

from app.config.settings import Settings
from app.services.agent.service import SYSTEM_PROMPT


class TestDeepFunnelDefaultActivation:
    def test_agent_loop_is_enabled_by_default(self) -> None:
        assert Settings().AGENT_TOOLS_ENABLED is True

    def test_system_prompt_teaches_kb_grounding(self) -> None:
        # Tool descriptions alone are weak steering; the rule set must
        # name the KB tool so policy questions are grounded by default.
        assert "search_knowledge_base" in SYSTEM_PROMPT

    def test_system_prompt_teaches_tool_escalation(self) -> None:
        # Pre-tool doctrine was "建议转人工" (advise) — the model would
        # talk about escalating instead of calling escalate_to_human.
        assert "escalate_to_human" in SYSTEM_PROMPT
