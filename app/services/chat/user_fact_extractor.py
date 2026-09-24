"""Background user-fact extraction: Phase B write side (mem0-style).

Layered-memory pattern distilled from 2026 sources (mem0 v3 docs,
arXiv Mem0 paper, 百度千帆 memory 后台更新, AWS Bedrock event-sourced
memory):

1. **ADD-only single pass** — insert new facts, never UPDATE/DELETE
   existing ones on the hot path; contradictions resolve by recency
   at read time plus retention pruning.
2. **Post-durability background trigger** — the persister schedules
   extraction after the turn is committed, off the request path.
3. **The LLM proposes, code verifies** — light-LLM extraction output
   passes deterministic post-filters (PII drop, exact dedup, length
   cap, per-run cap, retention prune) before touching the table.

``maybe_extract`` never raises: memory is an enhancement, never a
dependency of the chat path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from app.repositories.message_repository import MessageRepository
from app.repositories.session_repository import SessionRepository
from app.repositories.user_fact_repository import UserFactRepository

logger = logging.getLogger(__name__)

# PII never enters the cross-session store. A phone/ID/email captured
# "for convenience" becomes a compliance liability once it outlives the
# session; drop the fact rather than redact it — redacted fragments
# make useless memories anyway.
_PII_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"1[3-9]\d{9}",  # CN mobile
        r"\d{17}[\dXx]",  # CN national ID
        r"[\w.+-]+@[\w-]+\.[\w.]+",  # email
    )
)

_MAX_FACT_LEN = 100
_TRANSCRIPT_MESSAGES = 24

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_EXTRACT_PROMPT = (
    "你是电商客服对话的记忆抽取器。从下面的客服对话中抽取跨会话仍然成立的"
    "用户事实（会员等级、稳定偏好、明确约束），不要抽取一次性信息"
    "（订单号、物流单号、本会话临时话题）。"
    '只输出 JSON 数组，元素为字符串或 {{"fact": ..., "category": ...}}，'
    '例如 ["用户是 PLUS 会员", "偏好上午配送"]。没有可抽取内容时输出 []。\n\n'
    "对话记录:\n{transcript}"
)


class UserFactExtractor:
    """Extracts durable user facts from a session's transcript.

    Trigger cadence mirrors ``SessionCompressor``: fire at ``threshold``
    turns, then every ``interval`` — extraction and summarization are
    independent background consumers of the same persisted turns.
    """

    def __init__(
        self,
        session_maker: Callable[[], Any],
        llm_service: Any | None = None,
        *,
        threshold: int = 12,
        interval: int = 6,
        timeout_seconds: float = 3.0,
        max_facts_per_run: int = 5,
        retention: int = 50,
    ) -> None:
        """
        Args:
            session_maker: Factory producing context-managed async sessions.
            llm_service: Light-tier LLM used for extraction; None disables
                extraction entirely (wiring-level off switch).
            threshold: Turn count that fires the first extraction.
            interval: Re-extraction cadence after the first fire.
            timeout_seconds: Hard ceiling on the extraction LLM call.
            max_facts_per_run: Per-run insert cap (runaway-LLM valve).
            retention: Per-user table ceiling enforced after each run.
        """
        self._session_maker = session_maker
        self._llm_service = llm_service
        self._threshold = threshold
        self._interval = interval
        self._timeout_seconds = timeout_seconds
        self._max_facts_per_run = max_facts_per_run
        self._retention = retention

    async def maybe_extract(self, session_id: int) -> bool:
        """Run one extraction pass if due; returns whether facts were
        written. Never raises — failures log and return False."""
        try:
            return await self._extract(session_id)
        except Exception:  # noqa: BLE001 - availability over memory
            logger.warning("User fact extraction failed (session=%s)", session_id, exc_info=True)
            return False

    def _should_extract(self, count: int) -> bool:
        if count == self._threshold:
            return True
        return count > self._threshold and (count - self._threshold) % self._interval == 0

    async def _extract(self, session_id: int) -> bool:
        if self._llm_service is None:
            return False

        # Phase 1 (DB session open): resolve the owner and the trigger.
        # user_id comes from the session row, never from the caller —
        # extraction must not trust a client-supplied identity.
        async with self._session_maker() as db:
            chat_session = await SessionRepository(db).get_by_id(session_id)
            if chat_session is None or not chat_session.user_id or chat_session.user_id <= 0:
                return False
            user_id = chat_session.user_id
            message_repo = MessageRepository(db)
            if not self._should_extract(await message_repo.count_turns(session_id)):
                return False
            messages = await message_repo.get_recent_messages(
                session_id, limit=_TRANSCRIPT_MESSAGES
            )

        # Phase 2 (DB session closed): transcript → light LLM. The LLM
        # call runs outside any DB session, same contract as the
        # compressor — no DB connection is held across a model call.
        transcript = "\n".join(f"{m.role.value}: {m.content}" for m in reversed(messages))
        prompt = _EXTRACT_PROMPT.format(transcript=transcript)
        from app.services.llm.base import LLMMessage

        payload = await asyncio.wait_for(
            self._llm_service.generate(messages=[LLMMessage(role="user", content=prompt)]),
            timeout=self._timeout_seconds,
        )
        candidates = _parse_facts(payload.content)

        # Phase 3: deterministic post-filters, then ADD-only writes +
        # retention prune inside one DB session.
        async with self._session_maker() as db:
            repo = UserFactRepository(db)
            existing = await repo.existing_texts(user_id=user_id)
            accepted: list[tuple[str, str]] = []
            for text, category in candidates:
                text = text.strip()
                if not text or len(text) > _MAX_FACT_LEN:
                    continue
                if any(pattern.search(text) for pattern in _PII_PATTERNS):
                    continue
                if text in existing:
                    continue
                existing.add(text)
                accepted.append((text, category))
                if len(accepted) >= self._max_facts_per_run:
                    break
            if not accepted:
                return False
            for text, category in accepted:
                await repo.add(
                    user_id=user_id,
                    fact=text,
                    category=category,
                    source_session_id=session_id,
                )
            await repo.prune_for_user(user_id=user_id, keep=self._retention)
        return True


def _parse_facts(content: str) -> list[tuple[str, str]]:
    """Parse the extraction payload into (fact, category) candidates.

    Tolerates ```json fences and mixed str/dict elements; anything that
    is not a JSON list of fact-shaped items yields [] — malformed
    output must never reach the table.
    """
    fenced = _FENCE_RE.search(content)
    raw = fenced.group(1) if fenced else content
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []

    candidates: list[tuple[str, str]] = []
    for item in parsed:
        if isinstance(item, str):
            candidates.append((item, "general"))
        elif isinstance(item, dict):
            fact = item.get("fact")
            category = item.get("category", "general")
            if isinstance(fact, str) and isinstance(category, str):
                candidates.append((fact, category))
    return candidates
