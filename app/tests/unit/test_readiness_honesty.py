"""Degraded startup must not report ready (review 2026-09-26, #12).

Silent-degradation paths all ended in a green /ready or a green backup
log: a graph-build failure in the factory produced a graphless
ChatService that the lifespan happily marked ready — every chat call
then died on ``NoneType.ainvoke`` with a 500; a Postgres checkpointer
outage silently swapped in a MemorySaver while readiness stayed green;
and the backup script printed "Backup completed successfully" and
exited 0 after pg_dump failed. Readiness is a claim about capability:
every one of these paths must withdraw it (or the exit code) visibly.
"""

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.v1.chat as chat_api
import app.middleware.rate_limiter_redis as rate_limiter_module
import app.services.dialogue.checkpointer as checkpointer_module
import app.services.handoff.containment_metrics as containment_module
import app.services.retrieval.keyword_refresh as keyword_refresh_module
from app.api.v1.chat import get_chat_service, initialize_chat_service
from app.config.settings import settings
from app.main import create_app, lifespan
from app.services.chat.chat_service import STREAM_ERROR, ChatService

REPO_ROOT = Path(__file__).resolve().parents[3]


# ── Part A: a graphless service must be refused, not served ─────────────────


class TestGraphlessServiceRefusal:
    @pytest.mark.asyncio
    async def test_initialize_refuses_graphless_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The factory's graceful fallback returns a ChatService with
        graph=None; the API wiring must treat that as initialization
        failure so lifespan keeps /ready at not_ready."""
        monkeypatch.setattr(settings, "GRAPH_RAG_ENABLED", False)
        monkeypatch.setattr(settings, "SLOT_FILLING_ENABLED", False)
        monkeypatch.setattr(settings, "RERANKER_ENABLED", False)

        with (
            patch("app.services.llm.LLMFactory.create_from_settings", return_value=Mock()),
            patch(
                "app.services.embeddings.EmbeddingFactory.create_from_settings",
                return_value=Mock(),
            ),
            patch(
                "app.services.retrieval.RetrievalFactory.create_vector_client",
                return_value=Mock(),
            ),
            patch(
                "app.services.retrieval.RetrievalFactory.create_hybrid_search",
                return_value=Mock(),
            ),
            patch(
                "app.services.chat.factory.ChatServiceFactory.create_with_defaults",
                return_value=ChatService(graph=None),
            ),
            pytest.raises(RuntimeError, match="graph"),
        ):
            await initialize_chat_service(AsyncMock())

    async def test_process_message_fails_fast_typed(self) -> None:
        """A graphless service must raise the typed unavailable error —
        not AttributeError from deep inside the pipeline."""
        from app.services.chat.chat_service import GraphUnavailableError

        with pytest.raises(GraphUnavailableError):
            await ChatService(graph=None).process_message(1, "你好", 0)

    async def test_stream_yields_error_sentinel(self) -> None:
        """The stream transport uses the established STREAM_ERROR frame
        convention instead of an async-generator AttributeError."""
        chunks = [c async for c in ChatService(graph=None).process_message_stream(1, "你好", 0)]
        assert chunks == [STREAM_ERROR]

    def test_chat_endpoint_maps_unavailable_graph_to_503(self) -> None:
        """Direct hits during the degraded window get retryable 503
        semantics, not a generic 500."""
        app = create_app()
        app.dependency_overrides[get_chat_service] = lambda: ChatService(graph=None)
        response = TestClient(app).post("/api/v1/chat", json={"message": "你好", "session_id": 1})
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"]


# ── Part B: lifespan gates readiness on real capability ─────────────────────


class _FakeCheckpointerManager:
    def __init__(self) -> None:
        self.degraded = False

    async def start(self) -> object:
        return object()

    async def stop(self) -> None:
        return None


@pytest.fixture
def seams(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch every startup/shutdown seam at its source module (see
    test_lifespan.py — function-local imports re-resolve at call time)."""
    manager = _FakeCheckpointerManager()
    initialize = AsyncMock()
    monkeypatch.setattr(
        checkpointer_module, "create_checkpointer_manager_from_settings", lambda: manager
    )
    monkeypatch.setattr(chat_api, "initialize_chat_service", initialize)
    monkeypatch.setattr(keyword_refresh_module, "stop_keyword_index_refresher", Mock())
    monkeypatch.setattr(rate_limiter_module, "close_rate_limit_redis", AsyncMock())
    monkeypatch.setattr(containment_module, "stop_containment_refresher", Mock())
    return SimpleNamespace(manager=manager, initialize=initialize)


class TestReadinessGating:
    async def test_ready_when_healthy(self, seams) -> None:
        app = FastAPI()
        async with lifespan(app):
            assert app.state.chat_ready is True

    async def test_init_failure_keeps_not_ready(self, seams) -> None:
        seams.initialize.side_effect = RuntimeError("graph build failed")
        app = FastAPI()
        async with lifespan(app):
            assert app.state.chat_ready is False

    async def test_degraded_checkpointer_withholds_readiness(self, seams) -> None:
        """Postgres checkpointer asked for, MemorySaver substituted:
        chat still serves, but /ready must say not_ready — dialogue
        durability is a required dependency, not an optional cache."""
        seams.manager.degraded = True
        app = FastAPI()
        async with lifespan(app):
            assert app.state.chat_ready is False


# ── Part C: the backup script's exit code must tell the truth ───────────────


class TestBackupExitCode:
    def test_backup_script_exits_nonzero_on_failed_dump(self, tmp_path: Path) -> None:
        """pg_dump failure must exit nonzero — cron/monitoring cannot
        see red text, only return codes (review #12)."""
        root = tmp_path / "proj"
        scripts = root / "deployment" / "production" / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(REPO_ROOT / "deployment" / "production" / "scripts" / "backup.sh", scripts)

        # Shadow docker/docker-compose with stubs that fail: the compose
        # probe and the pg_dump exec both take the failure path while
        # date/mkdir/find keep working from the real PATH.
        stub_bin = tmp_path / "bin"
        stub_bin.mkdir()
        for name in ("docker", "docker-compose"):
            stub = stub_bin / name
            stub.write_text("#!/bin/sh\nexit 1\n")
            stub.chmod(0o755)
        env = {**os.environ, "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}"}

        proc = subprocess.run(
            ["bash", str(scripts / "backup.sh")],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

        assert "Database backup failed" in proc.stdout, "must reach the failure branch"
        assert proc.returncode != 0, "a failed backup must not exit 0"
        assert "Backup completed successfully" not in proc.stdout
