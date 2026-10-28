"""Committed active revisions, not physical index presence, authorize evidence."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.document import Document, DocumentVersion
from app.services.graph.base import GraphEntity, GraphSearchResult
from app.services.graph.provenance import collect_source_provenance
from app.services.retrieval.publication import PublicationGuard, PublicationUnavailableError
from app.services.retrieval.vector_base import SearchResult


def _result(document_id: str = "a", version_id: str = "a1", **metadata: Any) -> SearchResult:
    return SearchResult(
        document_id=document_id,
        content="policy evidence",
        score=0.8,
        metadata={"document_id": document_id, "version_id": version_id, **metadata},
    )


def _graph(
    documents: list[str] | None = None,
    versions: list[str] | None = None,
    **metadata: Any,
) -> GraphSearchResult:
    return GraphSearchResult(
        content="mixed graph evidence",
        entities=[],
        relations=[],
        score=0.8,
        source_type="graph_embedding",
        metadata={
            "source_document_ids": documents if documents is not None else ["a", "b"],
            "source_version_ids": versions if versions is not None else ["b2", "a1"],
            **metadata,
        },
    )


@pytest.fixture
async def published():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as db:
        for name, active, version_id in [
            ("a", True, "a1"),
            ("b", True, "b2"),
            ("missing", True, "absent"),
            ("inactive", False, "inactive1"),
            ("deleted", True, "deleted1"),
        ]:
            db.add(
                Document(
                    external_doc_id=name,
                    title=name,
                    source="database://test",
                    doc_type="txt",
                    is_active=active,
                    active_version_id=version_id,
                )
            )
        await db.flush()
        for version_id, owner, status in [
            ("a0", "a", "indexed"),
            ("a1", "a", "indexed"),
            ("a2", "a", "failed"),
            ("a3", "a", "processing"),
            ("b2", "b", "partial"),
            ("inactive1", "inactive", "indexed"),
            ("deleted1", "deleted", "deleted"),
        ]:
            db.add(
                DocumentVersion(
                    id=version_id,
                    document_id=owner,
                    title=owner,
                    filename="policy.txt",
                    file_type="txt",
                    content_hash="hash",
                    revision=1,
                    source_metadata={},
                    pipeline_config={},
                    status=status,
                )
            )
        await db.commit()
    try:
        yield maker, PublicationGuard(maker), engine
    finally:
        await engine.dispose()


async def test_only_exact_published_pairs_are_returned_with_original_identity(published):
    _, guard, _ = published
    candidates = [
        _result(),
        _result("a", "a0"),
        _result("a", "a2"),
        _result("a", "a3"),
        _result("b", "b2"),
        _result("legacy", "legacy1"),
        _result("missing", "absent"),
        _result("inactive", "inactive1"),
        _result("deleted", "deleted1"),
        _result("b", "a1"),
    ]
    accepted = await guard.filter_results(candidates)
    assert accepted == [candidates[0], candidates[4]]
    assert accepted[0] is candidates[0]
    assert accepted[1] is candidates[4]


@pytest.mark.parametrize("attempt_status", ["queued", "processing", "failed", "partial"])
async def test_retry_status_does_not_revoke_a_previously_published_revision(
    published, attempt_status
):
    maker, guard, _ = published
    async with maker() as db:
        await db.execute(
            text("UPDATE document_versions SET status = :status WHERE id = 'a1'"),
            {"status": attempt_status},
        )
        await db.commit()
    candidate = _result()
    assert await guard.filter_results([candidate]) == [candidate]


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"version_id": None},
        {"version_id": ""},
        {"version_id": True},
        {"version_id": "a1", "document_id": "b"},
    ],
)
async def test_missing_or_conflicting_vector_provenance_is_rejected(published, metadata):
    _, guard, _ = published
    candidate = SearchResult(document_id="a", content="unsafe", score=1, metadata=metadata)
    assert await guard.filter_results([candidate]) == []


async def test_warm_chunks_use_metadata_identity_without_modifying_their_shape(published):
    _, guard, _ = published
    chunk = {
        "id": "chunk1",
        "content": "policy",
        "metadata": {"document_id": "a", "version_id": "a1"},
    }
    conflict = {**chunk, "document_id": "b"}
    wrong_version = {**chunk, "version_id": "a2"}
    accepted = await guard.filter_documents([chunk, conflict, wrong_version])
    assert accepted == [chunk]
    assert accepted[0] is chunk
    assert "document_id" not in chunk


@pytest.mark.parametrize("source_type", [[], {}, True, 1])
async def test_malformed_source_type_cannot_crash_publication_filter(published, source_type):
    _, guard, _ = published
    candidate = _result(source_type=source_type)
    assert await guard.filter_results([candidate]) == [candidate]
    document = {"document_id": "a", "content": "policy", "metadata": candidate.metadata}
    assert await guard.filter_documents([document]) == [document]


async def test_every_batch_uses_one_registry_statement_without_per_hit_reads(published):
    _, guard, engine = published
    statements = []

    def record(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    candidates = [_result() for _ in range(200)]
    assert await guard.filter_results(candidates) == candidates
    assert len(statements) == 1


async def test_graph_independent_sets_are_mapped_by_version_owner_not_zipped(published):
    _, guard, _ = published
    graph = _graph(["a", "b"], ["b2", "a1"])
    assert await guard.filter_graph_results([graph]) == [graph]


@pytest.mark.parametrize(
    "documents,versions",
    [
        (["a", "b"], ["a1"]),
        (["a"], ["a1", "a2"]),
        (["a"], ["b2"]),
        (["a", "raw"], ["a1", "raw1"]),
        ([], ["a1"]),
        (["a"], []),
    ],
)
async def test_one_unproved_graph_contribution_rejects_the_whole_result(
    published, documents, versions
):
    _, guard, _ = published
    assert await guard.filter_graph_results([_graph(documents, versions)]) == []


async def test_graph_entity_provenance_cannot_hide_behind_valid_top_level_metadata(published):
    _, guard, _ = published
    graph = _graph(["a"], ["a1"])
    graph.entities = [
        GraphEntity(
            "entity",
            "Policy",
            "Policy",
            {
                "source_document_id": "b",
                "source_version_id": "a2",
            },
        )
    ]
    assert await guard.filter_graph_results([graph]) == []


async def test_graph_documents_and_cached_citations_use_the_same_gate(published):
    _, guard, _ = published
    document = {
        "document_id": "graph_synthetic",
        "content": "policy",
        "metadata": {
            "source_type": "graph_embedding",
            "source_document_ids": ["a", "b"],
            "source_version_ids": ["b2", "a1"],
        },
    }
    assert await guard.filter_documents([document]) == [document]
    citation = {
        "source_type": "graph",
        "excerpt": "policy",
        "source_document_ids": ["a", "b"],
        "source_version_ids": ["b2", "a1"],
    }
    assert await guard.answer_allowed(["a", "b"], [citation])
    citation["source_version_ids"] = ["a1", "a2"]
    assert not await guard.answer_allowed(["a", "b"], [citation])


@pytest.mark.parametrize("bad", [True, "unknown"])
async def test_incomplete_graph_marker_is_sticky_and_rejected(published, bad):
    _, guard, _ = published
    assert await guard.filter_graph_results([_graph(provenance_incomplete=bad)]) == []
    provenance = collect_source_provenance(
        {"provenance_incomplete": bad, "source_document_ids": ["a"]}
    )
    assert provenance["provenance_incomplete"] is True


@pytest.mark.parametrize("size", [50, 51, 100])
async def test_saturated_graph_source_sets_are_not_claimed_complete(published, size):
    _, guard, _ = published
    graph = _graph([f"source-{i}" for i in range(size)], [f"version-{i}" for i in range(size)])
    assert await guard.filter_graph_results([graph]) == []


def test_truncation_and_depth_limits_publish_an_incomplete_marker():
    provenance = collect_source_provenance({"source_version_ids": [f"v-{i}" for i in range(51)]})
    assert provenance["provenance_incomplete"] is True
    nested: dict[str, Any] = {"source_document_ids": ["hidden"]}
    for _ in range(10):
        nested = {"nested": nested}
    assert collect_source_provenance(nested)["provenance_incomplete"] is True


async def test_cache_answer_is_rejected_as_a_whole_without_sufficient_citations(published):
    _, guard, _ = published
    good = {"source_type": "document", "document_id": "a", "version_id": "a1", "excerpt": "policy"}
    assert await guard.answer_allowed(["a"], [good])
    assert not await guard.answer_allowed(["a"], [])
    assert not await guard.answer_allowed(["a", "b"], [good])
    assert not await guard.answer_allowed(["a"], [good, {**good, "version_id": "a2"}])
    assert not await guard.answer_allowed(["legacy"], [{**good, "document_id": "legacy"}])


async def test_explicit_faq_citation_is_independent_of_document_registry():
    def unavailable():
        raise AssertionError("FAQ must not query the document registry")

    guard = PublicationGuard(unavailable)
    citation = {
        "source_type": "faq",
        "faq_id": "returns",
        "version_id": "sha256:" + "a" * 64,
        "excerpt": "curated answer",
    }
    assert await guard.answer_allowed(["faq:returns"], [citation])
    assert not await guard.answer_allowed(["a"], [citation])
    assert not await guard.answer_allowed(["faq:returns"], [{**citation, "document_id": "a"}])


async def test_database_failure_raises_typed_error_instead_of_legacy_permission(published):
    maker, guard, _ = published
    async with maker() as db:
        await db.execute(text("DROP TABLE document_versions"))
        await db.commit()
    with pytest.raises(PublicationUnavailableError):
        await guard.filter_results([_result()])
    with pytest.raises(PublicationUnavailableError):
        await guard.answer_allowed(
            ["a"],
            [
                {
                    "source_type": "document",
                    "document_id": "a",
                    "version_id": "a1",
                    "excerpt": "policy",
                }
            ],
        )


async def test_database_wait_is_bounded_and_caller_cancellation_is_not_hidden():
    @asynccontextmanager
    async def stalled():
        await asyncio.Event().wait()
        yield

    guard = PublicationGuard(stalled, timeout_seconds=0.01)
    with pytest.raises(PublicationUnavailableError):
        await asyncio.wait_for(guard.filter_results([_result()]), timeout=0.5)
    task = asyncio.create_task(PublicationGuard(stalled).filter_results([_result()]))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_registry_timeout_must_be_finite_and_positive(timeout):
    with pytest.raises(ValueError):
        PublicationGuard(lambda: None, timeout_seconds=timeout)
