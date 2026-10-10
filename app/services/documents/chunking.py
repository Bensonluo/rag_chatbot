"""Bounded, overlapping document chunks with exact source offsets."""

import json
import re
import uuid
from bisect import bisect_right

from app.core.exceptions import ValidationError
from app.services.documents.base import ChunkingStrategy, Document, DocumentChunk

# Each match ends at a safe boundary. Headings instead start the next section.
_HEADINGS = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+")
_PARAGRAPHS = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")
_LINES = re.compile(r"\n")
_SENTENCES = re.compile(r"""(?:[。！？]+[”’」』）)]*|[.!?]+["'”’)]*(?=\s|$))[ \t]*""")
_WORDS = re.compile(r"[ \t]+")


class _OffsetChunking(ChunkingStrategy):
    """Choose natural boundaries without rewriting or dropping source characters."""

    strategy: str
    boundaries: tuple[re.Pattern[str], ...]

    async def chunk(
        self, document: Document, max_chunk_size: int = 512, chunk_overlap: int = 50
    ) -> list[DocumentChunk]:
        if type(max_chunk_size) is not int or max_chunk_size <= 0:
            raise ValidationError("max_chunk_size must be a positive integer")
        if type(chunk_overlap) is not int or not 0 <= chunk_overlap < max_chunk_size:
            raise ValidationError("chunk_overlap must be an integer from 0 to max_chunk_size - 1")

        text = document.content
        if not text.strip():
            return []

        # Scan each boundary type once, then binary-search within each chunk window.
        # Source offsets keep separators and indentation intact.
        positions = [
            [
                match.start() if pattern is _HEADINGS else match.end()
                for match in pattern.finditer(text)
            ]
            for pattern in self.boundaries
        ]
        chunks: list[DocumentChunk] = []
        start = 0
        while start < len(text):
            end = min(start + max_chunk_size, len(text))
            if end < len(text):
                # Add new characters beyond the overlapping prefix on every step.
                # Prefer reasonably full chunks over tiny sections near the start.
                minimum_end = start + max(chunk_overlap + 1, max_chunk_size // 2)
                for candidates in positions:
                    index = bisect_right(candidates, end) - 1
                    if index >= 0 and candidates[index] >= minimum_end:
                        end = candidates[index]
                        break

            content = text[start:end]
            index = len(chunks)
            identity = json.dumps(
                [
                    document.document_id,
                    document.metadata.get("version_id", ""),
                    self.strategy,
                    index,
                    start,
                    content,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
                    document_id=document.document_id,
                    index=index,
                    content=content,
                    metadata={
                        **document.metadata,
                        "title": document.title,
                        "file_type": document.file_type,
                        "start_pos": start,
                        "end_pos": end,
                        "chunk_size": len(content),
                        "chunking_strategy": self.strategy,
                    },
                )
            )
            if end == len(text):
                break
            start = end - chunk_overlap

        return chunks


class FixedSizeChunking(_OffsetChunking):
    """Prefer word boundaries, falling back to a strict character limit."""

    strategy = "fixed"
    boundaries = (_WORDS,)


class SemanticChunking(_OffsetChunking):
    """Prefer headings, paragraphs, and multilingual sentence boundaries."""

    strategy = "semantic"
    boundaries = (_HEADINGS, _PARAGRAPHS, _SENTENCES, _LINES, _WORDS)


class RecursiveCharacterChunking(_OffsetChunking):
    """Try progressively finer structural boundaries, then split by character."""

    strategy = "recursive"
    boundaries = (_HEADINGS, _PARAGRAPHS, _LINES, _SENTENCES, _WORDS)
