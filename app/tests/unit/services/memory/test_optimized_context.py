"""OptimizedContextBuilder: ORM→dict conversion, relevance filtering, budgets."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

from app.models.database.message import Message
from app.repositories.message_repository import MessageRepository
from app.services.embeddings.base import EmbeddingServiceBase
from app.services.memory.optimized_context import OptimizedContextBuilder


def _repo(messages: list[Message], summary: Message | None = None) -> MessageRepository:
    """MessageRepository stand-in returning ORM rows, newest first (prod order)."""
    repo = Mock(spec=MessageRepository)
    repo.get_recent_messages = AsyncMock(return_value=messages)
    repo.get_latest_summary = AsyncMock(return_value=summary)
    return repo


def _embedding(*, fail: bool = False) -> EmbeddingServiceBase:
    """Deterministic 2-D embeddings: 'match' texts align with the query axis."""

    async def embed(text: str) -> list[float]:
        if fail:
            raise RuntimeError("embedding outage")
        return [1.0, 0.0] if "match" in text else [0.0, 1.0]

    embedding = Mock(spec=EmbeddingServiceBase)
    embedding.embed_single = AsyncMock(side_effect=embed)
    return embedding


def _messages(contents: list[str]) -> list[Message]:
    """Build ORM Message rows newest-first (contents[0] is newest), like prod."""
    return [
        Message(id=i, role="user", content=c, created_at=datetime.now())
        for i, c in enumerate(contents, start=1)
    ]


class TestGetContextConversion:
    async def test_returns_subscriptable_dicts_not_orm_rows(self):
        """Regression: the token helpers subscript m["content"]; ORM rows crash.

        Pre-fix, get_context fed ORM Message objects straight into
        truncate_by_tokens -> estimate_tokens and raised TypeError on the
        very first call (the default token budget is always truthy).
        """
        repo = _repo(_messages(["a", "b", "c"]))
        memory = OptimizedContextBuilder(message_repo=repo, embedding_service=_embedding())

        context = await memory.get_context(session_id=1)

        assert context, "expected non-empty context"
        assert all(isinstance(m, dict) for m in context)
        assert all("role" in m and "content" in m for m in context)
        # Repo order is newest-first; context is chronological (oldest first).
        assert context[0]["content"] == "c"

    async def test_without_query_returns_recent_window_only(self):
        repo = _repo(_messages(["r1", "r2", "match old", "other old"]))
        memory = OptimizedContextBuilder(
            message_repo=repo, embedding_service=_embedding(), max_recent_messages=2
        )

        context = await memory.get_context(session_id=1)

        # Chronological: the newest two, oldest of them first.
        assert [m["content"] for m in context] == ["r2", "r1"]


class TestRelevanceFiltering:
    async def test_older_messages_filtered_by_similarity(self):
        repo = _repo(_messages(["r1", "r2", "please match this", "irrelevant old"]))
        memory = OptimizedContextBuilder(
            message_repo=repo,
            embedding_service=_embedding(),
            max_recent_messages=2,
            max_relevant_messages=2,
        )

        context = await memory.get_context(session_id=1, current_query="match please")

        contents = [m["content"] for m in context]
        assert "please match this" in contents
        assert "irrelevant old" not in contents
        # Relevant (older) block first, then the recent window — both
        # chronological, so combined reads like a conversation log.
        assert contents == ["please match this", "r2", "r1"]

    async def test_embedding_outage_falls_back_to_recent_older_messages(self):
        """embed failures degrade to taking the newest older messages, not an error."""
        repo = _repo(_messages(["r1", "r2", "old a", "old b"]))
        memory = OptimizedContextBuilder(
            message_repo=repo,
            embedding_service=_embedding(fail=True),
            max_recent_messages=2,
            max_relevant_messages=1,
        )

        context = await memory.get_context(session_id=1, current_query="anything")

        contents = [m["content"] for m in context]
        assert contents == ["old a", "r2", "r1"]

    async def test_token_budget_truncates_output(self):
        # Newest-first order: the short message is newest, the long one is
        # dropped once the 3-token budget can't fit it (~25 tokens).
        repo = _repo(_messages(["y" * 8, "x" * 100]))
        memory = OptimizedContextBuilder(
            message_repo=repo,
            embedding_service=_embedding(),
            max_recent_messages=2,
            token_budget=3,
        )

        context = await memory.get_context(session_id=1)

        assert [m["content"] for m in context] == ["y" * 8]


class TestEstimateTokenSavings:
    async def test_smoke_savings_report(self):
        repo = _repo(_messages(["x" * 100, "y" * 100, "z" * 100]))
        memory = OptimizedContextBuilder(
            message_repo=repo, embedding_service=_embedding(), max_recent_messages=1
        )

        report = await memory.estimate_token_savings(1, current_query="match")

        assert report["total_messages"] == 3
        assert report["selected_messages"] <= 3
        assert report["original_tokens"] == 75  # 3 × (100 // 4)
        assert "savings_percent" in report


class TestSummaryAwareness:
    """Long-session compression output (system summaries) joins the context.

    SummarizationMemory writes a system-role summary message and archives
    the covered prefix; the optimized strategy must consume that summary
    and not repeat the covered turns.
    """

    @staticmethod
    def _summary(content: str, created_at: datetime) -> Message:
        return Message(id=99, role="system", content=content, created_at=created_at)

    async def test_prepends_latest_summary_as_system_message(self):
        cutoff = datetime(2026, 9, 24, 12, 0, 0)
        summary = self._summary("用户咨询发票问题", cutoff)
        rows = [  # newest first, all after the summary
            Message(id=2, role="user", content="新的问题", created_at=cutoff.replace(minute=5)),
            Message(
                id=1, role="assistant", content="新的回答", created_at=cutoff.replace(minute=1)
            ),
        ]
        memory = OptimizedContextBuilder(message_repo=_repo(rows, summary=summary))

        context = await memory.get_context(session_id=1)

        assert context[0]["role"] == "system"
        assert context[0]["content"] == "Previous conversation: 用户咨询发票问题"
        assert [m["content"] for m in context[1:]] == ["新的回答", "新的问题"]

    async def test_turns_covered_by_the_summary_are_not_repeated(self):
        cutoff = datetime(2026, 9, 24, 12, 0, 0)
        summary = self._summary("前情摘要", cutoff)
        rows = [  # newest first: one post-summary turn, two pre-summary turns
            Message(id=4, role="user", content="之后的问题", created_at=cutoff.replace(minute=3)),
            Message(
                id=2,
                role="assistant",
                content="摘要前的回答",
                created_at=cutoff - timedelta(minutes=1),
            ),
            Message(
                id=1, role="user", content="摘要前的问题", created_at=cutoff - timedelta(minutes=2)
            ),
        ]
        memory = OptimizedContextBuilder(message_repo=_repo(rows, summary=summary))

        context = await memory.get_context(session_id=1)

        contents = [m["content"] for m in context]
        assert contents == ["Previous conversation: 前情摘要", "之后的问题"]

    async def test_summary_only_when_window_fully_covered(self):
        cutoff = datetime(2026, 9, 24, 12, 0, 0)
        summary = self._summary("全部已摘要", cutoff)
        rows = [
            Message(id=1, role="user", content="更早的", created_at=cutoff - timedelta(minutes=5))
        ]
        memory = OptimizedContextBuilder(message_repo=_repo(rows, summary=summary))

        context = await memory.get_context(session_id=1)

        assert [m["content"] for m in context] == ["Previous conversation: 全部已摘要"]

    async def test_summary_newer_than_every_row_still_kept(self):
        cutoff = datetime(2026, 9, 24, 12, 0, 0)
        summary = self._summary("摘要", cutoff)
        memory = OptimizedContextBuilder(message_repo=_repo([], summary=summary))

        context = await memory.get_context(session_id=1)

        assert [m["content"] for m in context] == ["Previous conversation: 摘要"]
