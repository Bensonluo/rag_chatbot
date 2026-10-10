"""Content-preservation contracts for document preprocessing."""

import pytest

from app.core.exceptions import ValidationError
from app.services.documents.preprocessing import DocumentPreprocessor


@pytest.mark.asyncio
async def test_preprocessing_preserves_document_structure_and_code_indentation() -> None:
    text = (
        "# Refund policy\n\n## Timing\nApply within seven days.\n\n"
        "| Item | Price |\n| Book | 10 |\n\n"
        "```python\n    if eligible:\n\treturn 'refund'\n```\n"
    )

    processed, _ = await DocumentPreprocessor().process(text)

    assert processed == text


@pytest.mark.asyncio
async def test_preprocessing_retains_evidence_after_100000_characters() -> None:
    text = "A" * 100000 + "\n\nTAIL_EVIDENCE"

    processed, _ = await DocumentPreprocessor().process(text)

    assert processed == text


@pytest.mark.asyncio
async def test_explicit_document_length_limit_rejects_instead_of_truncating() -> None:
    processor = DocumentPreprocessor(max_length=10)

    with pytest.raises(ValidationError, match="10"):
        await processor.process("0123456789 TAIL_EVIDENCE")

    assert (await processor.process("0123456789"))[0] == "0123456789"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_document_length_limits_are_rejected(limit: int) -> None:
    with pytest.raises(ValidationError):
        DocumentPreprocessor(max_length=limit)


@pytest.mark.asyncio
async def test_cleanup_normalizes_line_endings_without_flattening_text() -> None:
    text = "\ufeffTitle\r\n\r\n    code\tvalue\rNext\x00line\fNew page"

    processed, _ = await DocumentPreprocessor().process(text)

    assert processed == "Title\n\n    code\tvalue\nNextline\n\nNew page"


@pytest.mark.asyncio
async def test_metadata_is_read_from_preserved_lines_without_mutating_input() -> None:
    metadata = {"category": "policy"}

    _, result = await DocumentPreprocessor().process(
        "Refund policy\nAuthor: Alice\n2026-10-10\nOther content.", metadata
    )

    assert result == {
        "category": "policy",
        "potential_title": "Refund policy",
        "potential_author": "Alice",
        "potential_date": "2026-10-10",
    }
    assert metadata == {"category": "policy"}
