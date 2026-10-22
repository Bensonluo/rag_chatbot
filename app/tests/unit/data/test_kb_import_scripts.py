"""Offline regression coverage for standalone durable document importers."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import database
from app.api.v1 import documents as document_api
from app.api.v1 import graph as graph_api
from app.models.database.base import Base
from app.services.documents.jobs import DocumentJobService
from app.services.graph.factory import GraphFactory
from scripts import document_import, init_kb, seed_demo_kb
from scripts.document_import import ImportDocument

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def data_dir(tmp_path):
    # data/ is ignored in Git; reproduce its existing record format for CI.
    for name in ("faq", "policy", "product"):
        (tmp_path / f"{name}.jsonl").write_text(
            json.dumps({"doc_id": f"{name}-001", "title": "中文标题", "content": "中文正文"}),
            encoding="utf-8",
        )
    return tmp_path


def test_existing_data_format_is_supported(data_dir):
    docs = init_kb.load_documents(data_dir)
    assert {d.document_id for d in docs} == {"faq-001", "policy-001", "product-001"}
    assert all(d.title and d.content for d in docs)


def test_jsonl_and_legacy_pretty_json_preserve_content_and_scope(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"content": "first\nparagraph", "metadata": {"region": "CN"}})
        + '\n\n{"doc_id": "second", "doc_content": "legacy"}\n',
        encoding="utf-8",
    )
    (tmp_path / "b.jsonl").write_text(
        json.dumps({"doc_content": "中文"}, ensure_ascii=False, indent=2), encoding="utf-8-sig"
    )
    docs = init_kb.load_documents(tmp_path)
    assert [d.document_id for d in docs] == ["a-1", "second", "b.jsonl"]
    assert docs[0].content == "first\nparagraph"
    assert docs[0].metadata == {"region": "CN"}
    assert docs[1].content == "legacy"
    assert docs[2].title == "b.jsonl"
    assert docs[2].content == "中文"


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ('{"doc_id":"same","content":"a"}\n{"doc_id":"same","content":"b"}', "Duplicate"),
        ('{"content":"a"}\ninvalid', "a.jsonl:2: Invalid JSON"),
        ("[]", "Expected a JSON object"),
        ('{"content":" "}', "non-empty"),
        ('{"content":"a","title":2}', "Title"),
        ('{"content":"a","metadata":[]}', "Metadata"),
        ('{"doc_id":"../bad","content":"a"}', "doc_id"),
    ],
)
def test_bad_records_fail_before_import(tmp_path, records, message):
    (tmp_path / "a.jsonl").write_text(records, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        init_kb.load_documents(tmp_path)


@pytest.mark.parametrize("script", ["init_kb.py", "seed_demo_kb.py"])
def test_dry_run_needs_no_application_dependencies(script, data_dir):
    # -S removes site-packages. Importing app, database drivers, or providers fails.
    command = [sys.executable, "-S", str(REPO_ROOT / "scripts" / script), "--dry-run"]
    if script == "init_kb.py":
        command.extend(["--data-dir", str(data_dir)])
    result = subprocess.run(
        command,
        cwd=data_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "would ingest" in result.stdout
    assert "indexed" not in result.stdout


@pytest.fixture
def graph_client(monkeypatch):
    client = AsyncMock()
    monkeypatch.setattr(graph_api, "_graph_client", None)
    monkeypatch.setattr(GraphFactory, "create_from_settings", Mock(return_value=client))
    return client


@pytest.mark.parametrize("failure", [None, "connect", "processing"])
async def test_graph_shared_client_lifecycle(graph_client: AsyncMock, failure):
    if failure == "connect":
        graph_client.connect.side_effect = RuntimeError("connection failed")

    async def run() -> None:
        async with document_import.connected_graph():
            assert graph_api.get_graph_client() is graph_client
            graph_client.connect.assert_awaited_once()
            if failure == "processing":
                raise RuntimeError("processing failed")

    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            await run()
    else:
        await run()
    assert graph_api.get_graph_client() is None
    graph_client.close.assert_awaited_once()


async def test_existing_graph_connection_remains_caller_owned(graph_client: AsyncMock, monkeypatch):
    factory = Mock(return_value=graph_client)
    monkeypatch.setattr(GraphFactory, "create_from_settings", factory)
    monkeypatch.setattr(graph_api, "_graph_client", graph_client)
    async with document_import.connected_graph():
        assert graph_api.get_graph_client() is graph_client
    factory.assert_not_called()
    graph_client.connect.assert_not_awaited()
    graph_client.close.assert_not_awaited()
    assert graph_api.get_graph_client() is graph_client


async def test_graph_disabled_allocates_no_client(monkeypatch):
    monkeypatch.setattr(graph_api, "_graph_client", None)
    monkeypatch.setattr(GraphFactory, "create_from_settings", Mock(return_value=None))
    async with document_import.connected_graph():
        assert graph_api.get_graph_client() is None


@pytest.fixture
async def runner(monkeypatch, graph_client: AsyncMock):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "async_session_maker", sessions)
    close_db = AsyncMock()
    monkeypatch.setattr(database, "close_db", close_db)
    services = []
    configurations = []

    async def factory(config):
        assert graph_api.get_graph_client() is graph_client
        graph_client.close.assert_not_awaited()
        configurations.append(config)
        service = AsyncMock()
        service.ingest_bytes.return_value = {
            "chunks_count": 1,
            "graph_status": "disabled",
            "_normalized_text": "normalized",
        }
        services.append(service)
        return service

    monkeypatch.setattr(document_api, "build_ingestion_service", factory)
    try:
        yield DocumentJobService(sessions), services, configurations, close_db
    finally:
        await engine.dispose()


async def test_runner_retains_deduplicates_and_closes_shared_graph(runner, graph_client):
    jobs, services, configurations, close_db = runner
    docs = [
        ImportDocument("a", "Title", "original 中文", {"region": "CN"}),
        ImportDocument("b", "Another", "another source"),
    ]
    revisions = await document_import.ingest_documents(docs, graph_enabled=False)
    version = revisions[0]
    assert version["status"] == "indexed"
    assert await jobs.source(version["version_id"]) == ("original 中文".encode(), "a.txt", "txt")
    assert await jobs.normalized(version["version_id"]) == ("normalized", "a.txt")
    assert configurations[0]["graph_enabled"] is False
    assert services[0].ingest_bytes.call_args.kwargs["metadata"]["region"] == "CN"
    assert len(services) == 2
    for service in services:
        service.close.assert_awaited_once()
    graph_client.close.assert_awaited_once()
    close_db.assert_awaited_once()

    # Repeating the same source does not create another job or revision.
    again = await document_import.ingest_documents(docs, graph_enabled=False)
    assert again[0]["version_id"] == version["version_id"]
    assert len(services) == 2


@pytest.mark.parametrize("status", ["failed", "partial", "deleted"])
async def test_incomplete_revision_is_reported_and_connections_close(
    runner, graph_client, monkeypatch, status
):
    _, _, _, close_db = runner
    monkeypatch.setattr(
        DocumentJobService,
        "get_version",
        AsyncMock(return_value={"document_id": "a", "version_id": "v1", "status": status}),
    )
    with pytest.raises(RuntimeError, match=f"v1, {status}"):
        await document_import.ingest_documents([ImportDocument("a", "Title", "text")])
    graph_client.close.assert_awaited_once()
    close_db.assert_awaited_once()


async def test_invalid_metadata_rejected_before_graph_connection_or_writes(runner, graph_client):
    jobs, services, _, close_db = runner
    with pytest.raises(Exception, match="Reserved metadata"):
        await document_import.ingest_documents(
            [ImportDocument("a", "Title", "text", {"chunk_id": "forged"})]
        )
    assert await jobs.list_documents() == []
    assert services == []
    graph_client.connect.assert_not_awaited()
    close_db.assert_awaited_once()


async def test_both_importers_use_shared_runner(monkeypatch, data_dir):
    importer = AsyncMock(return_value=[{}])
    monkeypatch.setattr(seed_demo_kb, "ingest_documents", importer)
    await seed_demo_kb.seed(seed_demo_kb.DEFAULT_CORPUS, False)
    assert importer.call_args.kwargs == {"graph_enabled": False}
    assert all(d.metadata["synthetic"] for d in importer.call_args.args[0])
    importer.reset_mock()
    monkeypatch.setattr(init_kb, "ingest_documents", importer)
    await init_kb.initialize(data_dir, False)
    assert len(importer.call_args.args[0]) == 3
    assert importer.call_args.kwargs == {}
