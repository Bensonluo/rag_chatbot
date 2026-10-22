"""Isolated evaluation of the production ChatService and compiled dialogue graph.

Real: rule routing, KeywordSearch BM25, graph state, SQLite messages/tickets,
HandoffService, and citation construction. No vector embeddings, Redis, FAQ,
external business tools, or production database factories are used. A supplied
LLM is the only permitted network seam. The default recorded model is a local
contract fixture, NOT a measurement of model answer quality. Semantic judging
is deliberately separate: observations contain its inputs but never its verdict.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, cast

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.database.user import User
from app.services.chat.chat_service import ChatService
from app.services.chat.citations import citations_from_documents
from app.services.chat.persistence import ChatMessagePersister
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.i18n import HANDOFF_NO_TICKET, NO_EVIDENCE, detect_language
from app.services.dialogue.tools import ToolRegistry
from app.services.guardrails.base import GuardrailService
from app.services.guardrails.input_guard import DefaultInputGuardrail
from app.services.handoff.service import HandoffService
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from app.services.retrieval.hybrid_search import KeywordSearch

PIPELINE_GOLDEN_SET_FILE = Path(__file__).with_name("pipeline_golden_set.json")


@dataclass
class PipelineTurn:
    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    utterance: str
    expected_intent: str
    expected_version_ids: list[str] = field(default_factory=list)
    expected_handoff_outcome: str | None = None
    abstention: bool = False
    required_points: list[str] = field(default_factory=list)
    forbidden_promises: list[str] = field(default_factory=list)


@dataclass
class PipelineCase:
    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    id: str
    category: str
    language: str
    documents: list[dict[str, Any]]
    turns: list[PipelineTurn]
    handoff_failure: bool = False


@dataclass
class PipelineObservation:
    case_id: str
    turn_index: int
    category: str
    language: str
    utterance: str
    answer: str
    intent: str
    evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    metadata: dict[str, Any]
    persisted_messages: list[dict[str, Any]]
    history_messages: list[dict[str, Any]]
    expected_history: list[dict[str, Any]]
    handoff_outcome: str | None
    ticket_ids: list[int]
    ticket_records: list[dict[str, Any]]
    session_id: int
    user_id: int
    executed_tools: list[dict[str, Any]]
    tool_result: dict[str, Any]
    retrieval_ran: bool
    required_points: list[str]
    forbidden_promises: list[str]
    abstention: bool


@dataclass
class PipelineTurnResult:
    observation: PipelineObservation
    routing_failures: list[str]
    artifact_failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.routing_failures and not self.artifact_failures


@dataclass
class PipelineEvalReport:
    results: list[PipelineTurnResult]
    model_mode: str
    semantic_quality: str = "not_evaluated"
    retrieval_mode: str = "local_production_bm25_no_embeddings"
    limitations: list[str] = field(
        default_factory=lambda: [
            "Temporary SQLite durability; not production PostgreSQL concurrency.",
            "MemorySaver checkpoints; no distributed checkpoint or cache evaluation.",
            "No FAQ, vector search, reranking, agent loop, or external business tools.",
            "Response transport only; HTTP authentication and SSE framing are not evaluated.",
            "Contract pass does not establish semantic correctness or resolution quality.",
        ]
    )

    @property
    def total_turns(self) -> int:
        return len(self.results)

    @property
    def passed_turns(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def passed(self) -> bool:
        return bool(self.results) and self.passed_turns == self.total_turns


class RecordedPipelineLLM(LLMServiceBase):
    """Prompt-only extractive fixture; never receives golden answers/expectations."""

    def __init__(self) -> None:
        super().__init__(api_key="isolated-fixture", model="recorded-pipeline-fixture")
        self.prompts: list[list[LLMMessage]] = []

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del max_tokens, temperature, kwargs  # The recorded fixture has no sampling parameters.
        self.prompts.append(list(messages))
        prompt = messages[-1].content
        if prompt.startswith("最新消息："):
            answer = prompt.removeprefix("最新消息：")
        elif "参考资料:\n" in prompt:
            answer = prompt.split("参考资料:\n", 1)[1].split("\n\n用户问题:", 1)[0]
            answer = answer.replace("[文档1] ", "", 1)
        else:
            answer = "Hello. How can I help?"
        return LLMResponse(content=answer, model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        yield (await self.generate(messages, max_tokens, temperature, **kwargs)).content

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(self.estimate_tokens(message.content) for message in messages)


class _SeedMetadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    document_id: str = Field(min_length=1, max_length=512)
    version_id: str = Field(min_length=1, max_length=512)
    title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    start_pos: int | None = Field(default=None, ge=0)
    end_pos: int | None = Field(default=None, ge=0)


class _SeedDocument(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    id: str = Field(min_length=1, max_length=512)
    content: str = Field(min_length=1)
    metadata: _SeedMetadata


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate pipeline JSON key: {key}")
        result[key] = value
    return result


def load_pipeline_cases(path: str | Path = PIPELINE_GOLDEN_SET_FILE) -> list[PipelineCase]:
    """Strict typed goldens; malformed fixtures must never silently change a test."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "scope", "cases"}:
        raise ValueError("Pipeline root requires exactly schema_version, scope, cases")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Unsupported pipeline schema_version")
    if not isinstance(raw["scope"], str) or not raw["scope"].strip():
        raise ValueError("Pipeline scope must be a nonempty string")
    cases = TypeAdapter(list[PipelineCase]).validate_json(json.dumps(raw["cases"]), strict=True)
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Pipeline golden set must contain unique, nonempty cases")
    for case in cases:
        if (
            not case.id.strip()
            or not case.category.strip()
            or not case.turns
            or case.language not in {"zh", "en"}
        ):
            raise ValueError("Invalid pipeline case identity, language, or turns")
        chunk_ids = set()
        for document in case.documents:
            seed = _SeedDocument.model_validate(document)
            if seed.id in chunk_ids:
                raise ValueError("Duplicate pipeline chunk ID")
            chunk_ids.add(seed.id)
            if not all(
                value.strip()
                for value in (
                    seed.id,
                    seed.content,
                    seed.metadata.document_id,
                    seed.metadata.version_id,
                )
            ):
                raise ValueError("Pipeline source identity and content must not be blank")
            for start, end in (
                (seed.metadata.page_start, seed.metadata.page_end),
                (seed.metadata.start_pos, seed.metadata.end_pos),
            ):
                if start is not None and end is not None and start > end:
                    raise ValueError("Pipeline source location is reversed")
        for turn in case.turns:
            if not turn.utterance.strip() or not turn.expected_intent.strip():
                raise ValueError("Pipeline turn requires utterance and expected intent")
            if turn.expected_handoff_outcome not in {None, "queued", "failed"}:
                raise ValueError("Invalid expected handoff outcome")
            for values in (
                turn.expected_version_ids,
                turn.required_points,
                turn.forbidden_promises,
            ):
                if any(not value.strip() for value in values):
                    raise ValueError("Pipeline expectation strings must not be blank")
    return cases


def score_pipeline_observation(
    observation: PipelineObservation, expected: PipelineTurn
) -> PipelineTurnResult:
    """Deterministic artifact checks, not an answer-quality proxy.

    Failure acknowledgements are pinned to the fixed production branch template:
    adding a success promise after a disclaimer must fail even if metadata says
    failed. Free-form answer meaning remains the independent judge's job.
    """
    routing: list[str] = []
    artifacts: list[str] = []
    if observation.intent != expected.expected_intent:
        routing.append("intent_mismatch")
    if observation.handoff_outcome != expected.expected_handoff_outcome:
        routing.append("handoff_outcome_mismatch")
    if (observation.metadata.get("handoff_outcome") or None) != observation.handoff_outcome:
        artifacts.append("handoff_metadata_mismatch")
    evidence_citations = citations_from_documents(observation.evidence)
    if observation.citations != evidence_citations:
        artifacts.append("citation_evidence_mismatch")
    # Independent oracle for this document-only BM25 harness: sharing only the
    # production converter would make converter regressions produce false greens.
    if len(observation.citations) != len(observation.evidence):
        artifacts.append("citation_source_count_mismatch")
    for citation, document in zip(observation.citations, observation.evidence, strict=False):
        source = document.get("metadata") or {}
        identity = {
            "source_type": "document",
            "document_id": source.get("document_id"),
            "version_id": source.get("version_id"),
            "chunk_id": source.get("chunk_id"),
            "excerpt": document.get("content", "")[:800],
            "section_path": source.get("section_path", []),
        }
        identity.update(
            {key: source.get(key) for key in ("page_start", "page_end", "start_pos", "end_pos")}
        )
        if any(citation.get(key) != value for key, value in identity.items()):
            artifacts.append("citation_source_identity_mismatch")
            break
    if any(not citation.get("excerpt", "").strip() for citation in observation.citations):
        artifacts.append("empty_citation_evidence")
    versions = {citation.get("version_id") for citation in observation.citations}
    if versions != set(expected.expected_version_ids):
        artifacts.append("citation_version_mismatch")
    if expected.expected_version_ids and not observation.retrieval_ran:
        routing.append("retrieval_not_run")
    if observation.executed_tools or observation.tool_result:
        artifacts.append("unexpected_business_tool_execution")
    if expected.abstention:
        if observation.evidence or observation.citations:
            artifacts.append("abstention_has_evidence")
        if observation.answer != NO_EVIDENCE[detect_language(observation.utterance)]:
            artifacts.append("no_evidence_answer_contract")
    if expected.expected_handoff_outcome == "failed":
        if observation.ticket_ids or observation.metadata.get("handoff_ticket_id"):
            artifacts.append("failed_handoff_has_ticket")
        if observation.answer != HANDOFF_NO_TICKET[detect_language(observation.utterance)]:
            artifacts.append("failed_handoff_false_promise")
    if expected.expected_handoff_outcome == "queued":
        ticket = observation.metadata.get("handoff_ticket_id")
        if not ticket or ticket not in observation.ticket_ids:
            artifacts.append("queued_handoff_missing_ticket")
        records = [row for row in observation.ticket_records if row["id"] == ticket]
        if (
            len(records) != 1
            or records[0]["status"] not in {"open", "claimed"}
            or records[0]["session_id"] != observation.session_id
            or records[0]["user_id"] != observation.user_id
        ):
            artifacts.append("queued_handoff_invalid_ticket")
        if ticket and f"#{ticket}" not in observation.answer:
            artifacts.append("queued_handoff_missing_acknowledgement")
    if any(
        value.casefold() in observation.answer.casefold() for value in expected.forbidden_promises
    ):
        artifacts.append("forbidden_promise")
    if observation.history_messages != observation.expected_history:
        artifacts.append("history_roundtrip_mismatch")
    persisted = observation.persisted_messages
    persisted_history = [
        {
            "role": row["role"],
            "content": row["content"],
            "citations": row["metadata"].get("citations", []),
        }
        for row in persisted
    ]
    if persisted_history != observation.expected_history:
        artifacts.append("persisted_history_mismatch")
    if len(persisted) != observation.turn_index * 2:
        artifacts.append("persisted_turn_count_mismatch")
    if not persisted or persisted[-1]["content"] != observation.answer:
        artifacts.append("persisted_answer_mismatch")
    else:
        metadata = persisted[-1]["metadata"]
        if metadata.get("citations", []) != observation.citations:
            artifacts.append("persisted_citation_mismatch")
        for key in ("handoff_outcome", "handoff_ticket_id", "executed_tools"):
            if metadata.get(key) != observation.metadata.get(key):
                artifacts.append(f"persisted_{key}_mismatch")
    return PipelineTurnResult(observation, routing, artifacts)


async def _run_case(case: PipelineCase, llm: LLMServiceBase) -> list[PipelineTurnResult]:
    # The database URL is constructed here, never read from application settings.
    with TemporaryDirectory(prefix="rag-pipeline-eval-") as directory:
        engine = create_async_engine(f"sqlite+aiosqlite:///{directory}/evaluation.sqlite3")

        @event.listens_for(engine.sync_engine, "connect")
        def enable_foreign_keys(connection: Any, _record: Any) -> None:
            connection.execute("PRAGMA foreign_keys=ON")

        maker = async_sessionmaker(engine, expire_on_commit=False)
        persister = ChatMessagePersister(maker)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(
                    lambda conn: Base.metadata.create_all(
                        conn,
                        tables=[
                            Base.metadata.tables["users"],
                            Base.metadata.tables["chat_sessions"],
                            Base.metadata.tables["messages"],
                            Base.metadata.tables["handoff_tickets"],
                        ],
                    )
                )
                if case.handoff_failure:
                    await connection.execute(
                        text(
                            "CREATE TRIGGER evaluation_ticket_failure BEFORE INSERT ON handoff_tickets "
                            "BEGIN SELECT RAISE(ABORT, 'evaluation ticket write failure'); END"
                        )
                    )
            async with maker() as session:
                session.add(User(id=1, email="pipeline@evaluation.invalid", hashed_password="!"))
                await session.flush()
                session.add(ChatSession(id=1, user_id=1, title="Isolated pipeline evaluation"))
                await session.commit()
            search = KeywordSearch()
            await search.add_documents(case.documents)

            async def history_provider(session_id: int) -> list[LLMMessage]:
                history = await persister.get_history(session_id, limit=100)
                return [LLMMessage(role=item.role, content=item.content) for item in history]

            chat = ChatService(
                graph=build_dialogue_graph(
                    intent_detector=RuleBasedIntentDetector(),
                    slot_filler=None,
                    tool_registry=ToolRegistry(),
                    retrieval_pipeline={"hybrid_search": search},
                    llm_service=llm,
                    guardrail_service=GuardrailService(input_guard=DefaultInputGuardrail()),
                    checkpointer=MemorySaver(),
                    handoff_service=HandoffService(maker),
                    history_provider=history_provider,
                ),
                persister=persister,
            )
            results: list[PipelineTurnResult] = []
            expected_history: list[dict[str, Any]] = []
            for index, turn in enumerate(case.turns, start=1):
                response = await chat.process_message(1, turn.utterance, 1)
                snapshot = await chat.graph.aget_state({"configurable": {"thread_id": "1"}})
                state = cast(dict[str, Any], snapshot.values)
                history = await chat.get_chat_history(1, limit=100)
                expected_history.extend(
                    [
                        {"role": "user", "content": turn.utterance, "citations": []},
                        {
                            "role": "assistant",
                            "content": response.content,
                            "citations": response.citations,
                        },
                    ]
                )
                async with maker() as session:
                    rows = (await session.scalars(select(Message).order_by(Message.id))).all()
                    tickets = (
                        await session.scalars(select(HandoffTicket).order_by(HandoffTicket.id))
                    ).all()
                    persisted = [
                        {
                            "role": row.role.value,
                            "content": row.content,
                            "metadata": row.message_metadata or {},
                        }
                        for row in rows
                    ]
                observation = PipelineObservation(
                    case_id=case.id,
                    turn_index=index,
                    category=case.category,
                    language=case.language,
                    utterance=turn.utterance,
                    answer=response.content,
                    intent=response.intent,
                    evidence=state.get("retrieved_docs") or [],
                    citations=response.citations,
                    metadata=response.metadata or {},
                    persisted_messages=persisted,
                    history_messages=[
                        {"role": item.role, "content": item.content, "citations": item.citations}
                        for item in history
                    ],
                    expected_history=list(expected_history),
                    handoff_outcome=state.get("handoff_outcome") or None,
                    ticket_ids=[ticket.id for ticket in tickets],
                    ticket_records=[
                        {
                            "id": ticket.id,
                            "session_id": ticket.session_id,
                            "user_id": ticket.user_id,
                            "status": ticket.status,
                            "reason": ticket.reason,
                            "priority": ticket.priority,
                            "summary": json.loads(ticket.summary),
                        }
                        for ticket in tickets
                    ],
                    session_id=1,
                    user_id=1,
                    executed_tools=state.get("executed_tools") or [],
                    tool_result=state.get("tool_result") or {},
                    retrieval_ran=bool(state.get("retrieval_ran")),
                    required_points=turn.required_points,
                    forbidden_promises=turn.forbidden_promises,
                    abstention=turn.abstention,
                )
                results.append(score_pipeline_observation(observation, turn))
            return results
        finally:
            await persister.drain()
            await engine.dispose()


async def run_pipeline_eval(
    llm_service: LLMServiceBase | None = None,
    cases: list[PipelineCase] | None = None,
) -> PipelineEvalReport:
    """Run local production components; callers explicitly opt into a live LLM."""
    selected = load_pipeline_cases() if cases is None else cases
    if not selected:
        raise ValueError("Pipeline evaluation requires at least one case")
    llm = llm_service if llm_service is not None else RecordedPipelineLLM()
    results = []
    for case in selected:
        results.extend(await _run_case(case, llm))
    return PipelineEvalReport(
        results=results,
        model_mode="recorded_fixture" if isinstance(llm, RecordedPipelineLLM) else "injected_llm",
    )
