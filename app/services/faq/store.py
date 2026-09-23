"""Curated FAQ fast path (semantic match over pre-approved answers).

Industry baseline (Zendesk Answer Bot canned responses, 阿里小蜜 FAQ
知识库): a small set of top questions covers a large share of customer
service traffic, and serving reviewed, deterministic answers for them
cuts both latency (no retrieval, no LLM) and cost, with no risk of
model hallucination on policy questions.

Design:

- The FAQ table is shipped configuration (``faqs.json``), embedded once
  per process on first use with the pipeline's embedding service and
  matched by cosine similarity against the user message. A read-only
  in-process table is horizontally scalable by construction — every
  replica holds the identical data, so no shared store is needed.
- The fast path is an optimization, never a dependency: any failure
  (missing file, malformed entry, embedding outage) degrades to the
  full RAG pipeline.
- Only pre-approved answer text is served; the output guardrail still
  runs on hits, and hits carry a ``faq:<id>`` source for provenance.
"""

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_FAQ_FILE = Path(__file__).parent / "faqs.json"


@dataclass
class FAQEntry:
    """One curated question/answer pair with paraphrase variants."""

    faq_id: str
    question: str
    answer: str
    variants: list[str] = field(default_factory=list)
    enabled: bool = True


def load_faq_entries(path: str | Path) -> list[FAQEntry]:
    """Load FAQ entries from a JSON file, degrading to empty on failure.

    Missing files and malformed entries never raise: the fast path is
    optional by contract, and a bad config must not take down chat.
    """
    file_path = Path(path)
    if not file_path.exists():
        logger.info("FAQ data file not found (%s); fast path inactive", file_path)
        return []

    try:
        raw: list[dict[str, Any]] = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("FAQ data file unreadable (%s): %s; fast path inactive", file_path, exc)
        return []

    entries: list[FAQEntry] = []
    for index, item in enumerate(raw):
        try:
            entry = FAQEntry(
                faq_id=str(item["id"]),
                question=str(item["question"]),
                answer=str(item["answer"]),
                variants=[str(v) for v in item.get("variants", [])],
                enabled=bool(item.get("enabled", True)),
            )
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping malformed FAQ entry #%d: %s", index, exc)
            continue
        if entry.question and entry.answer:
            entries.append(entry)

    enabled = [e for e in entries if e.enabled]
    logger.info("Loaded %d FAQ entries (%d disabled)", len(enabled), len(entries) - len(enabled))
    return enabled


class FAQService:
    """Semantic matcher over the curated FAQ table.

    The table is embedded lazily on first match (one bulk call) and
    kept for the process lifetime. Match cost is a dot product per
    entry — microseconds at curated-table scale (~hundreds of vectors);
    move to a vector index only if the table grows past that.
    """

    def __init__(
        self,
        embedding_service: Any,
        entries: list[FAQEntry],
        threshold: float = 0.75,
    ) -> None:
        self._embeddings = embedding_service
        self._entries = list(entries)
        self._threshold = threshold
        self._vectors: list[list[float]] | None = None
        self._vector_entries: list[FAQEntry] = []
        self._lock = asyncio.Lock()
        # An empty table can never match; skip all downstream work.
        self._disabled = not self._entries

    async def match(self, message: str) -> FAQEntry | None:
        """Return the best FAQ entry at or above the threshold, else None."""
        if self._disabled or not message.strip():
            return None

        try:
            query_vec = (await self._embeddings.embed([message])).embeddings[0]
        except Exception:  # noqa: BLE001 - availability over fast path
            logger.debug("FAQ query embedding failed; treating as miss", exc_info=True)
            return None

        await self._ensure_table()
        if not self._vectors:
            return None

        best_score = -1.0
        best_entry: FAQEntry | None = None
        for vector, entry in zip(self._vectors, self._vector_entries, strict=True):
            score = _cosine(query_vec, vector)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self._threshold:
            return best_entry
        return None

    async def _ensure_table(self) -> None:
        """Build the embedding table once; a failed build disables matching.

        The lock prevents concurrent first requests from issuing
        duplicate bulk embeds.
        """
        if self._vectors is not None or self._disabled:
            return
        async with self._lock:
            if self._vectors is not None:
                return
            texts: list[str] = []
            owners: list[FAQEntry] = []
            for entry in self._entries:
                texts.append(entry.question)
                owners.append(entry)
                for variant in entry.variants:
                    texts.append(variant)
                    owners.append(entry)
            try:
                result = await self._embeddings.embed(texts)
                self._vector_entries = owners
                self._vectors = list(result.embeddings)
                logger.info("FAQ table embedded: %d vectors", len(self._vectors))
            except Exception:  # noqa: BLE001 - availability over fast path
                # Permanent for the process: retrying a failing provider
                # on every request would add latency to the hot path.
                logger.warning(
                    "FAQ table embedding failed; fast path disabled for this process",
                    exc_info=True,
                )
                self._vectors = []


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b) or not a:
        return -1.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):  # unequal length returned -1.0 above
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return -1.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def create_faq_service(
    embedding_service: Any,
    data_file: str | None = None,
    threshold: float = 0.75,
) -> FAQService | None:
    """Build a FAQService from config; None when there is nothing to serve.

    Returns None (rather than an empty service) when the file yields no
    enabled entries so callers can skip the node entirely.
    """
    path = Path(data_file) if data_file else DEFAULT_FAQ_FILE
    entries = load_faq_entries(path)
    if not entries:
        return None
    return FAQService(embedding_service=embedding_service, entries=entries, threshold=threshold)
