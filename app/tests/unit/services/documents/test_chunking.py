"""Chunking must preserve evidence, enforce limits, and identify source spans."""

from dataclasses import replace

import pytest

from app.core.exceptions import ValidationError
from app.services.documents.base import Document
from app.services.documents.chunking import (
    FixedSizeChunking,
    RecursiveCharacterChunking,
    SemanticChunking,
)

STRATEGIES = [FixedSizeChunking, SemanticChunking, RecursiveCharacterChunking]


def document(content: str) -> Document:
    return Document("doc-policy", "Refund policy", content, "pdf", {"version_id": "v1"})


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("overlap", [0, 5, 31])
@pytest.mark.parametrize(
    "text",
    [
        "退款政策支持七天内申请。超期申请请联系客服！" * 20,
        "# Policy\n\n## Timing\nSeven days.\n\n## Exceptions\nOpened items excluded.\n",
        "A" * 100 + " " + "B" * 1000,
        "X" * 600,
        "    if allowed:\n\treturn 'yes'\n\n  final value  ",
        "Short.",
    ],
)
async def test_chunks_preserve_every_character_with_exact_offsets_and_overlap(
    strategy: type, overlap: int, text: str
) -> None:
    chunks = await strategy().chunk(document(text), max_chunk_size=32, chunk_overlap=overlap)

    assert chunks[0].metadata["start_pos"] == 0
    assert chunks[-1].metadata["end_pos"] == len(text)
    for index, chunk in enumerate(chunks):
        start, end = chunk.metadata["start_pos"], chunk.metadata["end_pos"]
        assert chunk.index == index
        assert chunk.content == text[start:end]
        assert 0 < len(chunk.content) <= 32
        assert chunk.metadata["chunk_size"] == len(chunk.content)
        assert chunk.metadata["file_type"] == "pdf"
        if index:
            previous = chunks[index - 1]
            assert start > previous.metadata["start_pos"]
            assert start == previous.metadata["end_pos"] - overlap

    rebuilt = chunks[0].content + "".join(chunk.content[overlap:] for chunk in chunks[1:])
    assert rebuilt == text


@pytest.mark.asyncio
async def test_semantic_chunking_prefers_section_boundaries() -> None:
    first_section = "# One\nalpha alpha alpha.\n\n"
    second_section = "# Two\nbeta beta beta.\n"

    chunks = await SemanticChunking().chunk(
        document(first_section + second_section), max_chunk_size=35, chunk_overlap=0
    )

    assert [chunk.content for chunk in chunks] == [first_section, second_section]


@pytest.mark.asyncio
async def test_semantic_chunking_recognizes_chinese_sentence_boundaries_without_spaces() -> None:
    chunks = await SemanticChunking().chunk(
        document("甲乙丙。丁戊己。庚辛壬。"), max_chunk_size=9, chunk_overlap=0
    )

    assert [chunk.content for chunk in chunks] == ["甲乙丙。丁戊己。", "庚辛壬。"]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", STRATEGIES)
async def test_chunk_identifiers_are_stable_and_change_with_document_version(
    strategy: type,
) -> None:
    source = document("Refunds are available within seven days. " * 8)
    first = await strategy().chunk(source, 64, 8)
    repeated = await strategy().chunk(source, 64, 8)
    newer = await strategy().chunk(replace(source, metadata={"version_id": "v2"}), 64, 8)
    other_document = await strategy().chunk(replace(source, document_id="other"), 64, 8)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in repeated]
    assert {chunk.chunk_id for chunk in first}.isdisjoint(chunk.chunk_id for chunk in newer)
    assert {chunk.chunk_id for chunk in first}.isdisjoint(
        chunk.chunk_id for chunk in other_document
    )
    assert len({chunk.chunk_id for chunk in first}) == len(first)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (8, -1), (8, 8), (8, 9), (True, 0)])
async def test_invalid_chunk_parameters_are_rejected(
    strategy: type, size: int, overlap: int
) -> None:
    with pytest.raises(ValidationError):
        await strategy().chunk(document("Evidence"), size, overlap)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("text", ["", " \n\t "])
async def test_blank_documents_produce_no_chunks(strategy: type, text: str) -> None:
    assert await strategy().chunk(document(text)) == []
