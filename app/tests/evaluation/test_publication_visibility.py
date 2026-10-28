"""Only committed document publications become customer evidence.

Jobs, ingestion, Qdrant's local SDK, BM25, the compiled dialogue graph,
ChatService and SQL persistence are real. External embeddings, Redis epoch
transport and model generation are deterministic local boundaries. SQLite
checks sequential publication visibility, not PostgreSQL worker locking.
"""

import asyncio
import socket
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import event, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import settings
from app.core.exceptions import ConflictError
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.enums.message import MessageRole, MessageStatus
from app.services.chat.chat_service import HEARTBEAT, STREAM_ERROR, ChatService, CitationEvent
from app.services.chat.persistence import ChatMessagePersister
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import ToolRegistry
from app.services.documents.ingestion import DocumentIngestionService
from app.services.documents.jobs import DocumentJobService
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from app.services.retrieval import kb_epoch, keyword_refresh
from app.services.retrieval.factory import RetrievalFactory
from app.services.retrieval.hybrid_search import HybridSearchService, KeywordSearch
from app.services.retrieval.publication import PublicationGuard
from app.services.retrieval.qdrant_client import QdrantClient
from app.services.retrieval.vector_base import VectorSearchRequest
from app.tests.integration.test_document_pipeline import (
    BorrowedQdrant,
    DeterministicEmbeddings,
    LocalEpochStore,
)

_OLD = "Return policy: unopened goods may be returned within 7 days. LEGACY_POLICY."
_NEW = "Return policy: unopened goods may be returned within 30 days. CANDIDATE_POLICY."
_ANSWER = "Please review the cited return policy."


class _FaultClient(BorrowedQdrant):
    def __init__(self, client: AsyncQdrantClient) -> None:
        super().__init__(client)
        self.fail_retirement = False
        self.pause_retirement = False
        self.after_upsert = asyncio.Event()
        self.resume = asyncio.Event()

    async def delete(self, **kwargs: Any) -> Any:
        # replace_document calls this only after the real upsert succeeds.
        if self.pause_retirement:
            self.after_upsert.set()
            await asyncio.wait_for(self.resume.wait(), timeout=5)
        if self.fail_retirement:
            self.fail_retirement = False
            raise TimeoutError("isolated retirement failure after candidate upsert")
        return await self.client.delete(**kwargs)


class _ContextRecorder(LLMServiceBase):
    def __init__(self) -> None:
        super().__init__(api_key="offline", model="local-publication-context-recorder")
        self.prompts: list[list[str]] = []

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.prompts.append([message.content for message in messages])
        return LLMResponse(content=_ANSWER, model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        yield (await self.generate(messages, max_tokens, temperature, **kwargs)).content

    def estimate_tokens(self, value: str) -> int:
        return len(value)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(message.content) for message in messages)


@dataclass
class _Case:
    maker: async_sessionmaker[AsyncSession]
    jobs: DocumentJobService
    sdk: _FaultClient
    vector: QdrantClient
    hybrid: HybridSearchService
    chat: ChatService
    recorder: _ContextRecorder
    embeddings: DeterministicEmbeddings


@pytest.fixture
async def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[_Case, None]:
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "GRAPH_RAG_ENABLED", False)
    monkeypatch.setattr(settings, "CHAT_TRACE_STREAM_ENABLED", False)
    monkeypatch.setattr(kb_epoch, "_client", LocalEpochStore())
    monkeypatch.setattr(keyword_refresh, "_keyword_hybrid", None)
    attempts: list[str] = []

    def no_network(*args: Any, **kwargs: Any) -> None:
        attempts.append(repr(args[1:]))
        raise AssertionError("publication visibility evaluation must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/publication.sqlite3")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection: Any, _record: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with maker() as db:
        db.add(User(id=9, email="publication@evaluation.example.com", hashed_password="!"))
        await db.flush()
        db.add_all([ChatSession(id=index, user_id=9) for index in range(1, 5)])
        await db.commit()
    raw = AsyncQdrantClient(location=":memory:")
    await raw.create_collection(
        "policies", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
    )
    sdk = _FaultClient(raw)
    embeddings = DeterministicEmbeddings()
    vector = QdrantClient(
        url="unused", collection_name="policies", client=sdk, embedding_service=embeddings
    )
    guard = PublicationGuard(maker)
    hybrid = RetrievalFactory.create_hybrid_search(
        vector_client=vector, keyword_search=KeywordSearch(), publication_guard=guard
    )
    keyword_refresh.register_keyword_index(hybrid)
    recorder = _ContextRecorder()
    persister = ChatMessagePersister(maker)
    chat = ChatService(
        graph=build_dialogue_graph(
            intent_detector=RuleBasedIntentDetector(),
            slot_filler=None,
            tool_registry=ToolRegistry(),
            retrieval_pipeline={"hybrid_search": hybrid, "publication_guard": guard},
            llm_service=recorder,
            checkpointer=MemorySaver(),
        ),
        persister=persister,
    )
    try:
        yield _Case(
            maker, DocumentJobService(maker), sdk, vector, hybrid, chat, recorder, embeddings
        )
    finally:
        sdk.resume.set()
        await persister.drain()
        await raw.close()
        await engine.dispose()
        assert attempts == []


async def _factory(case: _Case, _config: dict[str, Any]) -> DocumentIngestionService:
    return DocumentIngestionService(
        qdrant_client=case.vector,
        embedding_service=case.embeddings,
        chunking_strategy="fixed",
        max_chunk_size=256,
        chunk_overlap=0,
        graph_enabled=False,
    )


async def _process(case: _Case) -> bool:
    return await case.jobs.process_next(lambda config: _factory(case, config))


async def _submit(case: _Case, content: str) -> str:
    result = await case.jobs.submit(
        data=content.encode(),
        title="Return policy",
        filename="returns.txt",
        file_type="txt",
        metadata={"lang": "en"},
        pipeline_config={},
        document_id="return-policy",
    )
    version_id = result["version_id"]
    assert isinstance(version_id, str)
    return version_id


async def _raw_versions(case: _Case) -> set[str]:
    # Management/raw-storage evidence intentionally bypasses the customer gate.
    return {chunk["metadata"]["version_id"] for chunk in await case.vector.list_chunks()}


async def _visible_versions(case: _Case) -> set[str]:
    hits = await case.hybrid.search(VectorSearchRequest("return policy", top_k=20))
    return {(hit.metadata or {})["version_id"] for hit in hits}


@dataclass
class _Observed:
    content: str
    versions: set[str]
    prompt: str
    saved: Message


async def _ask(case: _Case, transport: str, session_id: int) -> _Observed:
    start = len(case.recorder.prompts)
    if transport == "normal":
        response = await case.chat.process_message(session_id, "what is the return policy", 9)
        content, citations = response.content, response.citations
    else:
        events = [
            chunk
            async for chunk in case.chat.process_message_stream(
                session_id, "what is the return policy", 9, include_citations=True
            )
        ]
        assert STREAM_ERROR not in events
        content = "".join(
            chunk for chunk in events if isinstance(chunk, str) and chunk != HEARTBEAT
        )
        citation_events = [chunk for chunk in events if isinstance(chunk, CitationEvent)]
        assert len(citation_events) == 1
        citations = citation_events[0].citations
    async with case.maker() as db:
        saved = await db.scalar(
            select(Message)
            .where(Message.session_id == session_id, Message.role == MessageRole.ASSISTANT)
            .order_by(Message.id.desc())
            .limit(1)
        )
    assert saved is not None
    assert saved.status == MessageStatus.COMPLETED
    assert saved.content == content
    assert (saved.message_metadata or {}).get("citations", []) == citations
    versions = {citation["version_id"] for citation in citations}
    prompt = "\n".join(value for block in case.recorder.prompts[start:] for value in block)
    return _Observed(content, versions, prompt, saved)


async def _recover(
    case: _Case, version_id: str, transport: str, *, processing: bool = False
) -> None:
    if not processing:
        await case.jobs.retry(version_id)
    assert await _process(case)
    assert (await case.jobs.get_version(version_id))["status"] == "indexed"
    registry = (await case.jobs.list_documents())[0]
    assert registry["active_version_id"] == version_id
    assert registry["is_active"] is True
    assert await _raw_versions(case) == {version_id}
    assert await _visible_versions(case) == {version_id}
    reply = await _ask(case, transport, 2)
    assert reply.versions == {version_id}
    assert "CANDIDATE_POLICY" in reply.prompt
    assert "LEGACY_POLICY" not in reply.prompt
    normalized, _ = await case.jobs.normalized(version_id)
    assert normalized == _NEW


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_failed_replacement_candidate_is_hidden_until_successful_retry(
    case: _Case, transport: str
) -> None:
    old = await _submit(case, _OLD)
    assert await _process(case)
    assert await _visible_versions(case) == {old}
    candidate = await _submit(case, _NEW)
    case.sdk.fail_retirement = True
    assert await _process(case)
    assert (await case.jobs.get_version(candidate))["status"] == "failed"
    assert (await case.jobs.list_documents())[0]["active_version_id"] == old
    assert await _raw_versions(case) == {old, candidate}
    assert await _visible_versions(case) == {old}
    reply = await _ask(case, transport, 1)
    assert reply.versions == {old}
    assert "LEGACY_POLICY" in reply.prompt
    assert "CANDIDATE_POLICY" not in reply.prompt
    with pytest.raises(ConflictError):
        await case.jobs.normalized(candidate)
    await _recover(case, candidate, transport)


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_failed_first_publication_is_not_customer_evidence(
    case: _Case, transport: str
) -> None:
    candidate = await _submit(case, _NEW)
    case.sdk.fail_retirement = True
    assert await _process(case)
    assert (await case.jobs.get_version(candidate))["status"] == "failed"
    registry = (await case.jobs.list_documents())[0]
    assert registry["active_version_id"] is None
    assert registry["is_active"] is False
    assert await _raw_versions(case) == {candidate}
    assert await _visible_versions(case) == set()
    reply = await _ask(case, transport, 1)
    assert reply.versions == set()
    assert "CANDIDATE_POLICY" not in reply.prompt
    assert "CANDIDATE_POLICY" not in reply.content
    await _recover(case, candidate, transport)


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_rejected_registry_completion_cannot_publish_indexed_candidate(
    case: _Case, transport: str
) -> None:
    old = await _submit(case, _OLD)
    assert await _process(case)
    candidate = await _submit(case, _NEW)
    async with case.maker() as db:
        await db.execute(
            text(
                "CREATE TRIGGER reject_publication BEFORE UPDATE OF status ON document_versions "
                "WHEN NEW.status IN ('indexed', 'partial') BEGIN "
                "SELECT RAISE(ABORT, 'isolated registry completion rejected'); END"
            )
        )
        await db.commit()
    with pytest.raises(IntegrityError, match="isolated registry completion rejected"):
        await _process(case)
    assert (await case.jobs.get_version(candidate))["status"] == "processing"
    assert (await case.jobs.list_documents())[0]["active_version_id"] == old
    assert await _raw_versions(case) == {candidate}
    # Old points were already retired: unavailable evidence is acceptable;
    # inventing publication of the uncommitted candidate is not.
    assert await _visible_versions(case) == set()
    reply = await _ask(case, transport, 1)
    assert reply.versions == set()
    assert "CANDIDATE_POLICY" not in reply.prompt
    assert "CANDIDATE_POLICY" not in reply.content
    async with case.maker() as db:
        await db.execute(text("DROP TRIGGER reject_publication"))
        await db.commit()
    await _recover(case, candidate, transport, processing=True)


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_upserted_candidate_is_hidden_while_worker_has_not_published(
    case: _Case, transport: str
) -> None:
    old = await _submit(case, _OLD)
    assert await _process(case)
    candidate = await _submit(case, _NEW)
    case.sdk.pause_retirement = True
    worker = asyncio.create_task(_process(case))
    try:
        await asyncio.wait_for(case.sdk.after_upsert.wait(), timeout=2)
        assert (await case.jobs.get_version(candidate))["status"] == "processing"
        assert (await case.jobs.list_documents())[0]["active_version_id"] == old
        assert await _raw_versions(case) == {old, candidate}
        assert await _visible_versions(case) == {old}
        reply = await _ask(case, transport, 1)
        assert reply.versions == {old}
        assert "LEGACY_POLICY" in reply.prompt
        assert "CANDIDATE_POLICY" not in reply.prompt
    finally:
        case.sdk.resume.set()
        await asyncio.wait_for(worker, timeout=5)
    assert (await case.jobs.get_version(candidate))["status"] == "indexed"
    assert await _visible_versions(case) == {candidate}
    assert (await _ask(case, transport, 2)).versions == {candidate}


async def test_healthy_replacement_preserves_historical_citation_snapshot(case: _Case) -> None:
    old = await _submit(case, _OLD)
    assert await _process(case)
    original = await _ask(case, "normal", 1)
    assert original.versions == {old}
    candidate = await _submit(case, _NEW)
    assert await _process(case)
    assert await _visible_versions(case) == {candidate}
    current = await _ask(case, "normal", 2)
    assert current.versions == {candidate}
    assert "CANDIDATE_POLICY" in current.prompt
    history = await case.chat.get_chat_history(session_id=1)
    assistant = next(message for message in history if message.role == "assistant")
    assert assistant.message_id == original.saved.id
    assert assistant.content == original.content
    assert {citation["version_id"] for citation in assistant.citations} == {old}
    assert assistant.citations[0]["excerpt"] == _OLD
    assert (await case.jobs.normalized(old))[0] == _OLD
