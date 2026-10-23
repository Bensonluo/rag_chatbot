"""Regressions from the 2026-10-22 live stress test against the deployed bot.

Testers hammered the live deployment and surfaced six failure families
(turn numbers cite the live transcript):

- Turn 21/22: gibberish ("Yeah19 NormaYeah") accepted as complaint
  category, "Okay okay" accepted as description — a garbage ticket with
  an invented id (CP122245) and staff member (客服专员-小李).
- Turn 8: 「那我要人工」 matched no handoff keyword — the LLM leg
  hallucinated「正在为您转接人工客服」with no ticket.
- Turn 15: an English "who are you?" served a cached Chinese reply
  (cross-language semantic-cache replay; FAQ fast path has the same
  hole via the multilingual embedder).
- Turn 15/22: English turns dragged to Chinese by zh-only prompt
  scaffolding (tool/RAG context builders).
- Turn 1: a knowledge-graph tech question broke persona — the bot
  answered as a generic LLM and ignored the embedded 退货 intent.
- Turn 9-family: after a completed task + handoff, an unrelated turn
  re-surfaced the dead refund's staging banner (agent context_note
  replaying executed args; deterministic re-stage invariant pinned).
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from app.services.agent.service import SYSTEM_PROMPT as AGENT_SYSTEM_PROMPT
from app.services.agent.service import AgentResult
from app.services.chat.semantic_cache import SemanticCacheService
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.i18n import detect_language
from app.services.dialogue.nodes import NodeFactory
from app.services.dialogue.slot_extraction import parse_slot_json
from app.services.dialogue.state import DialogueState
from app.services.dialogue.tools import create_default_tool_registry
from app.services.facts.tool_id_gate import ToolIdStreamGate, enforce_tool_id_honesty
from app.services.faq.store import FAQEntry, FAQService, load_faq_entries
from app.services.guardrails.base import GuardrailService
from app.services.guardrails.input_guard import DefaultInputGuardrail
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from app.services.llm.prompt_templates import PromptTemplates
from app.services.slot_filling.slot_types import slot_accepts_value


def _make_factory(llm: LLMServiceBase | None = None) -> NodeFactory:
    intent_detector = Mock()
    intent_detector.detect_with_confidence = AsyncMock()
    return NodeFactory(
        intent_detector=intent_detector,
        slot_filler=Mock(),
        tool_registry=Mock(),
        retrieval_pipeline={},
        llm_service=llm,
        guardrail_service=None,
        graph_retrieval_service=None,
    )


# ── Turn 21/22: garbage slots ────────────────────────────────────────────


class TestGarbageSlotAdmission:
    """The whole-message fallback must not file garbage tickets."""

    async def test_gibberish_never_becomes_complaint_category(self):
        factory = _make_factory()
        state: DialogueState = {
            "message": "Yeah19 NormaYeah",
            "intent": "complaint",
            "filled_slots": {},
            "pending_slots": ["category", "description"],
        }
        updates = await factory.collect_slots_node(state)

        assert "category" not in updates["filled_slots"]
        assert "category" in updates["pending_slots"]
        # The flow re-prompts instead of advancing on junk.
        assert updates["slot_prompt"]

    async def test_acknowledgment_never_becomes_complaint_description(self):
        factory = _make_factory()
        state: DialogueState = {
            "message": "Okay okay",
            "intent": "complaint",
            "filled_slots": {"category": "商品质量"},
            "pending_slots": ["description"],
        }
        updates = await factory.collect_slots_node(state)

        assert "description" not in updates["filled_slots"]

    async def test_refund_reason_still_fills_wholesale(self):
        """The fallback's reason for existing: the bare refund reason
        「不好吃」carries no pattern cue and must keep landing in the
        reason slot (live turn 7 worked; this pins it)."""
        factory = _make_factory()
        state: DialogueState = {
            "message": "不好吃",
            "intent": "refund",
            "filled_slots": {"order_id": "1111"},
            "pending_slots": ["reason"],
        }
        updates = await factory.collect_slots_node(state)

        assert updates["filled_slots"]["reason"] == "不好吃"

    def test_canonical_category_accepts_taxonomy_value(self):
        assert slot_accepts_value("complaint", "category", "商品质量")
        assert slot_accepts_value("complaint", "category", "product quality")
        assert not slot_accepts_value("complaint", "category", "Yeah19 NormaYeah")

    def test_llm_extractor_cannot_echo_gibberish_as_category(self):
        """Same garbage, one layer later: the extractor echoing the
        gibberish turn as the category is rejected by schema (the
        free-form description beside it passes normally)."""
        raw = json.dumps({"category": "Yeah19 NormaYeah", "description": "x" * 12})
        parsed = parse_slot_json(raw, "complaint")
        assert "category" not in parsed
        assert parsed["description"] == "x" * 12

    def test_llm_extractor_canonization_lands_on_taxonomy(self):
        raw = json.dumps({"category": "商品质量"})
        assert parse_slot_json(raw, "complaint") == {"category": "商品质量"}


# ── Turn 8: handoff keyword coverage ─────────────────────────────────────


class TestHandoffKeywordCoverage:
    @pytest.mark.parametrize(
        "message",
        [
            "那我要人工",
            "给我人工",
            "叫人工",
            "I want to talk to a real person please",
            "real human, now",
        ],
    )
    async def test_colloquial_handoff_phrases_detected(self, message):
        """「那我要人工」 must route to the deterministic handoff node
        (template reply, real ticket) — not fall to the LLM leg that
        hallucinated a transfer with no ticket."""
        probe = await RuleBasedIntentDetector().detect_with_confidence(message)
        assert probe.intent.value == "handoff"
        assert probe.confidence >= 0.7


# ── Turn 15: cross-language cache/FAQ replay ─────────────────────────────


class _NearDuplicateEmbeddings:
    """Embeds zh and EN paraphrases to near-identical vectors.

    Only the language filter can separate them — exactly the live
    failure mode (multilingual BGE-M3 scoring「你是谁」and "who are
    you?" as near-duplicates).
    """

    def __init__(self) -> None:
        self.embeddings: list[list[float]] = []

    async def embed(self, texts: list[str]) -> Any:
        self.embeddings = [
            [1.0, 0.0] if detect_language(t) == "zh" else [0.95, 0.05] for t in texts
        ]
        return self


class _FakeRedis:
    """Minimal RedisSeam: one hash namespace in memory."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}

    async def hset(self, name: str, key: str, value: str) -> None:
        self._store.setdefault(name, {})[key] = value

    async def hget(self, name: str, key: str) -> str | None:
        return self._store.get(name, {}).get(key)

    async def hgetall(self, name: str) -> dict[str, str]:
        return dict(self._store.get(name, {}))

    async def hlen(self, name: str) -> int:
        return len(self._store.get(name, {}))

    async def expire(self, name: str, ttl: int) -> None:
        return None


class TestSemanticCacheLanguageScope:
    async def test_english_query_not_served_chinese_entry(self):
        from app.services.chat.answer_cache import CachedAnswer

        settings = Mock(
            SEMANTIC_CACHE_ENABLED=True,
            SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.5,
            SEMANTIC_CACHE_MAX_ENTRIES=100,
            SEMANTIC_CACHE_TTL_SECONDS=60,
            SEMANTIC_CACHE_SERVED_INTENTS="chitchat",
        )
        cache = SemanticCacheService(
            embedding_service=_NearDuplicateEmbeddings(),
            redis_client=_FakeRedis(),
            epoch_provider=_static_epoch,
            settings=settings,
        )
        await cache.put(
            "你是谁",
            CachedAnswer(response="我是电商平台的智能客服助手。", sources=[], intent="chitchat"),
        )

        # zh query still hits its language's entry…
        hit = await cache.get("你是谁吗")
        assert hit is not None and "智能客服助手" in hit.response
        # …but the English visitor is never served the Chinese answer.
        assert await cache.get("who are you?") is None

    async def test_english_entry_serves_english_query(self):
        from app.services.chat.answer_cache import CachedAnswer

        settings = Mock(
            SEMANTIC_CACHE_ENABLED=True,
            SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.5,
            SEMANTIC_CACHE_MAX_ENTRIES=100,
            SEMANTIC_CACHE_TTL_SECONDS=60,
            SEMANTIC_CACHE_SERVED_INTENTS="chitchat",
        )
        cache = SemanticCacheService(
            embedding_service=_NearDuplicateEmbeddings(),
            redis_client=_FakeRedis(),
            epoch_provider=_static_epoch,
            settings=settings,
        )
        await cache.put(
            "who are you?",
            CachedAnswer(
                response="I'm the store's AI support assistant.", sources=[], intent="chitchat"
            ),
        )
        hit = await cache.get("who are you?")
        assert hit is not None and "support assistant" in hit.response


async def _static_epoch() -> str:
    return "test-epoch"


class TestFaqLanguageScope:
    async def test_english_query_skips_chinese_entry(self):
        """The multilingual embedder scores zh/EN paraphrases nearly
        identically — the language filter, not the threshold, is what
        keeps the zh answer away from an English visitor."""
        embeddings = Mock()
        # Same near-identical vector for every text: only the language
        # filter can produce a miss.
        embeddings.embed = AsyncMock(return_value=Mock(embeddings=[[1.0, 0.0]]))
        faq = FAQService(
            embedding_service=embeddings,
            entries=[
                FAQEntry(
                    faq_id="who-zh",
                    question="你是谁",
                    answer="我是电商平台的智能客服助手。",
                )
            ],
            threshold=0.75,
        )

        assert await faq.match("who are you?") is None
        assert await faq.match("你是谁") is not None


# ── Turn 15/22: tool-turn scaffold language + honesty ────────────────────


class _RecordingLLM(LLMServiceBase):
    """Captures prompts; replies with a configurable lie."""

    def __init__(self, reply: str) -> None:
        super().__init__(api_key="test", model="test")
        self.reply = reply
        self.prompts: list[str] = []

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.prompts.append(messages[-1].content)
        return LLMResponse(content=self.reply, model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        self.prompts.append(messages[-1].content)
        for char in self.reply:
            yield char

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages)


class TestToolTurnScaffoldLanguage:
    async def test_english_turn_gets_english_scaffold(self):
        llm = _RecordingLLM("Done.")
        factory = _make_factory(llm)
        state: DialogueState = {
            "message": "okay okay, where is my package now?",
            "session_id": 1,
            "intent": "track_shipping",
            "tool_result": {"status": "in transit"},
        }

        await factory.generate_response_node(state)

        prompt = llm.prompts[-1]
        assert "Tool result:" in prompt
        assert "工具执行结果:" not in prompt
        assert "never invent IDs" in prompt

    async def test_chinese_turn_keeps_chinese_scaffold(self):
        llm = _RecordingLLM("好的。")
        factory = _make_factory(llm)
        state: DialogueState = {
            "message": "我的快递到哪了",
            "session_id": 1,
            "intent": "track_shipping",
            "tool_result": {"status": "运输中"},
        }

        await factory.generate_response_node(state)

        prompt = llm.prompts[-1]
        assert "工具执行结果:" in prompt
        assert "不得编造" in prompt


class TestToolIdHonestyGate:
    """Ticket ids may only be quoted as the tool result states them."""

    async def test_fabricated_ticket_replaced_by_system_reference(self):
        """The live turn-22 shape: zh complaint turn, invented
        CP122245 + invented staff name — replaced by the deterministic
        line quoting the real id."""
        llm = _RecordingLLM("您的投诉已提交，工单号 CP122245，客服专员-小李将在24小时内联系您。")
        factory = _make_factory(llm)
        state: DialogueState = {
            "message": "okay okay",
            "session_id": 1,
            "intent": "complaint",
            "tool_result": {"ticket_id": 5, "status": "recorded"},
        }

        updates = await factory.generate_response_node(state)

        response = updates["response"]
        assert "CP122245" not in response
        assert "客服专员" not in response
        assert "5" in response  # the real reference is quoted

    async def test_streamed_fabrication_never_reaches_the_queue(self):
        """Streaming contract: gated stream and persisted text stay
        byte-identical — the fabricated id is held at the sentence
        boundary, never emitted."""
        llm = _RecordingLLM("Ticket CP122245 filed for you. Thanks!")
        factory = _make_factory(llm)
        queue: asyncio.Queue[Any] = asyncio.Queue()
        config: RunnableConfig = {"configurable": {"stream_queue": queue}}
        state: DialogueState = {
            "message": "okay okay",
            "session_id": 1,
            "intent": "complaint",
            "tool_result": {"ticket_id": 5},
        }

        updates = await factory.generate_response_node(state, config)

        streamed = ""
        while not queue.empty():
            streamed += queue.get_nowait()
        assert "CP122245" not in streamed
        assert updates["response"] == streamed  # byte-identical contract

    async def test_real_identifiers_pass_untouched(self):
        llm = _RecordingLLM("订单 ORD1001 已为您提交退货申请。")
        factory = _make_factory(llm)
        state: DialogueState = {
            "message": "退货",
            "session_id": 1,
            "intent": "return",
            "tool_result": {"order_id": "ORD1001", "return_id": 9},
        }

        updates = await factory.generate_response_node(state)

        assert "ORD1001" in updates["response"]

    def test_gate_is_never_allowed_to_break_chat(self):
        """Unserializable tool payload: the checker degrades to
        pass-through rather than eating the reply."""
        weird: dict[str, Any] = {"x": object()}  # default=str handles it

        out = enforce_tool_id_honesty("Ticket CP122245 filed.", weird, "en")
        assert isinstance(out, str)

    def test_hash_id_honest_when_number_exists(self):
        """「#5」 is fine when the tool result carries 5 — the digits,
        not the formatting, decide."""
        text = "您的工单已创建（#5）。"
        assert enforce_tool_id_honesty(text, {"ticket_id": 5}, "zh") == text

    def test_stream_gate_flush_releases_held_remainder(self):
        gate = ToolIdStreamGate({"ticket_id": 5}, "en")
        assert gate.feed("Ticket CP122245") == ""  # mid-sentence: held
        tail = gate.flush()
        assert "CP122245" not in tail
        assert "5" in tail


# ── Turn 1: persona scope ────────────────────────────────────────────────


class TestPersonaIdentityScope:
    def test_generation_persona_covers_tech_questions(self):
        prompt = PromptTemplates.get_cs_system_prompt()
        # A「你用了知识图谱吗」turn must stay in the CS role…
        assert "技术实现类提问" in prompt
        # …and an embedded after-sales intent gets processed, not ignored.
        assert "主动为其办理" in prompt

    def test_agent_persona_covers_tech_questions(self):
        assert "技术实现类提问" in AGENT_SYSTEM_PROMPT
        assert "主动为其办理" in AGENT_SYSTEM_PROMPT

    def test_named_capabilities_confirmed_at_product_level(self):
        """The live bot once DENIED the knowledge graph the deployment
        actually runs. Both personas must confirm a user-named capability
        at product level (never deny) while keeping stack details out."""
        for prompt in (PromptTemplates.get_cs_system_prompt(), AGENT_SYSTEM_PROMPT):
            assert "如实确认" in prompt
            assert "不否认" in prompt
            # Stack/model selection stays out of the conversation.
            assert "不展开技术栈" in prompt


# ── Capability disclosure via the FAQ fast path (rag-routed probes) ──────


class TestKnowledgeGraphFaqDisclosure:
    """Curated capability answers for「你用了知识图谱吗」-family probes.

    These cover the route where the intent leg classifies the probe as a
    knowledge question (rag → faq_lookup): the curated entry answers
    deterministically — no LLM leg to drift into denial."""

    @staticmethod
    def _faq() -> FAQService:
        def _embed(texts: list[str]) -> Any:
            # Two-bucket geometry: KG-family texts near the probe, the
            # other 28 entries far away — ties among non-KG entries are
            # irrelevant once the KG entry outscores them.
            vectors = [
                [1.0, 0.0] if ("图谱" in t or "knowledge graph" in t) else [0.1, 0.9] for t in texts
            ]
            return Mock(embeddings=vectors)

        embeddings = Mock()
        embeddings.embed = AsyncMock(side_effect=_embed)
        return FAQService(
            embedding_service=embeddings,
            entries=load_faq_entries(Path("app/services/faq/faqs.json")),
            threshold=0.75,
        )

    async def test_zh_probe_hits_curated_capability_answer(self):
        entry = await self._faq().match("你用了知识图谱吗")
        assert entry is not None
        assert entry.faq_id == "knowledge_graph_capability"
        assert "结构化" in entry.answer  # says WHERE it is used
        assert "如实" not in entry.answer  # product voice, not meta voice

    async def test_en_probe_hits_english_entry(self):
        entry = await self._faq().match("do you use a knowledge graph")
        assert entry is not None
        assert entry.faq_id == "knowledge_graph_capability_en"
        assert "cross-check" in entry.answer

    async def test_colloquial_variant_matches(self):
        entry = await self._faq().match("图谱开启了吗")
        assert entry is not None
        assert entry.faq_id == "knowledge_graph_capability"


# ── Turn 9-family: post-execution state hygiene ─────────────────────────


class TestPostExecutionHygiene:
    async def test_agent_context_note_drops_executed_args(self):
        """task_executed ⇒ the agent must not be told the finished
        refund's args are「对话中已知信息」— that note is what let a
        dead action get re-offered."""
        agent = Mock()
        agent.run = AsyncMock(return_value=AgentResult(response="好的，还有什么可以帮您？"))
        factory = NodeFactory(
            intent_detector=Mock(),
            slot_filler=Mock(),
            tool_registry=Mock(),
            retrieval_pipeline={},
            llm_service=None,
            guardrail_service=None,
            graph_retrieval_service=None,
            agent_service=agent,
        )
        state: DialogueState = {
            "message": "今天天气如何",
            "session_id": 1,
            "intent": "unknown",
            "filled_slots": {"order_id": "1111", "reason": "不好吃"},
            "task_executed": True,
        }

        await factory.handle_agent_node(state)

        note = agent.run.call_args.kwargs.get("context_note", "")
        assert "order_id=1111" not in note
        assert "reason=不好吃" not in note

    async def test_unrelated_turn_after_task_and_handoff_never_re_stages(self):
        """The live turn-9 sequence through the compiled graph: refund
        staged → confirmed (executed) → handoff → weather question.
        The weather turn must never resurrect the dead refund's banner
        nor leave a pending confirmation behind."""
        llm = _RecordingLLM("您好，我可以帮您处理订单、退款、退货等售后问题。")
        handoff = Mock()
        handoff.create_ticket_for_session = AsyncMock(
            return_value={"ticket_id": 7, "queue_position": 2, "reused": False}
        )
        graph = build_dialogue_graph(
            intent_detector=RuleBasedIntentDetector(),
            slot_filler=None,
            tool_registry=create_default_tool_registry(),
            retrieval_pipeline={},
            llm_service=llm,
            guardrail_service=GuardrailService(input_guard=DefaultInputGuardrail()),
            checkpointer=MemorySaver(),
            handoff_service=handoff,
        )
        config: RunnableConfig = {"configurable": {"thread_id": "stress-turn9"}}
        base = {"session_id": 1, "user_id": 1}

        ask = await graph.ainvoke({"message": "我要给订单1111退款", **base}, config)
        assert ask["pending_slots"] == ["reason"]  # sanity: reason asked
        staged = await graph.ainvoke({"message": "不好吃", **base}, config)
        assert staged["pending_confirmation"] is not None  # sanity: staged

        confirmed = await graph.ainvoke({"message": "确认", **base}, config)
        assert confirmed["task_executed"] is True

        handoff_turn = await graph.ainvoke({"message": "那我要人工", **base}, config)
        assert "工单号 #7" in handoff_turn["response"]

        weather = await graph.ainvoke({"message": "今天天气如何", **base}, config)

        assert "即将为您执行" not in weather["response"]
        snapshot = await graph.aget_state(config)
        assert snapshot.values.get("pending_confirmation") is None

    async def test_weather_language_detected_english_fallback(self):
        """Sanity for the i18n seam the cache fix leans on."""
        assert detect_language("今天天气如何") == "zh"
        assert detect_language("Yeah19 NormaYeah") == "en"
