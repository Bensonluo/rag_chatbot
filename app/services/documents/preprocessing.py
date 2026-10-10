"""
Document preprocessing utilities.

Handles text cleaning, normalization, and preparation before chunking.
"""

import re
from typing import Any

from app.core.exceptions import ValidationError


class TextPreprocessor:
    """
    Text preprocessing utilities for document content.

    Provides methods for cleaning and normalizing text before
    chunking and embedding.
    """

    @staticmethod
    def clean(text: str) -> str:
        """
        Remove non-text controls while retaining document layout.

        Args:
            text: Text to clean

        Returns:
            str: Cleaned text
        """
        # Form feeds carry page boundaries in extracted document text.
        text = text.removeprefix("\ufeff").replace("\f", "\n\n")
        # Keep tabs and line endings: indentation and paragraph breaks are evidence.
        return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize platform line endings, preserving spacing and indentation.

        Args:
            text: Text to normalize

        Returns:
            str: Normalized text
        """
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def remove_headers_footers(
        text: str, header_pattern: str | None = None, footer_pattern: str | None = None
    ) -> str:
        """
        Remove repeated headers and footers from text.

        Args:
            text: Text to process
            header_pattern: Regex pattern for headers
            footer_pattern: Regex pattern for footers

        Returns:
            str: Text with headers/footers removed
        """
        # Default patterns: very short lines that appear frequently
        if not header_pattern and not footer_pattern:
            # Simple heuristic: remove lines that are < 20 chars
            # and appear at start/end of many pages
            # This is a simplified approach
            return text

        # Custom patterns
        if header_pattern:
            text = re.sub(header_pattern, "", text)

        if footer_pattern:
            text = re.sub(footer_pattern, "", text)

        return text.strip()

    @staticmethod
    def extract_metadata(text: str) -> dict[str, Any]:
        """
        Extract metadata from text content.

        Attempts to find:
        - Title (first line if it looks like a title)
        - Author (if mentioned)
        - Date (if mentioned)

        Args:
            text: Text to analyze

        Returns:
            dict: Extracted metadata
        """
        metadata = {}
        lines = text.strip().split("\n")

        # First line might be a title
        if lines:
            first_line = lines[0].strip()
            # Title: short, no period at end, might be bold/caps in original
            if len(first_line) < 100 and not first_line.endswith("."):
                metadata["potential_title"] = first_line

        # Look for patterns like "Author: ..." or "By ..."
        author_patterns = [
            r"Author:\s*(.+)",
            r"By\s+(.+?)(?:\n|$)",
            r"Written by\s+(.+?)(?:\n|$)",
        ]

        for pattern in author_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["potential_author"] = match.group(1).strip()
                break

        # Look for date patterns
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # ISO format
            r"\d{1,2}/\d{1,2}/\d{4}",  # US format
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["potential_date"] = match.group(0)
                break

        return metadata

    @staticmethod
    def validate_length(text: str, max_length: int | None = None) -> None:
        """Reject oversized input explicitly instead of indexing a partial document."""
        if max_length is not None and len(text) > max_length:
            raise ValidationError(
                f"Document contains {len(text)} characters; the configured limit is {max_length}. "
                "Split the document or increase the limit; no content was truncated."
            )


class DocumentPreprocessor:
    """
    Complete document preprocessing pipeline.

    Applies multiple preprocessing steps in sequence.
    """

    def __init__(
        self,
        clean: bool = True,
        normalize_whitespace: bool = True,
        remove_headers_footers: bool = False,
        extract_metadata: bool = True,
        max_length: int | None = None,
    ) -> None:
        """
        Initialize preprocessor.

        Args:
            clean: Apply text cleaning
            normalize_whitespace: Normalize whitespace
            remove_headers_footers: Remove headers/footers
            extract_metadata: Extract metadata
            max_length: Optional character limit; oversized documents are rejected.
        """
        if max_length is not None and (type(max_length) is not int or max_length <= 0):
            raise ValidationError("Document max_length must be a positive integer or None")
        self.clean = clean
        self.normalize_whitespace = normalize_whitespace
        self.remove_headers_footers = remove_headers_footers
        self.extract_metadata = extract_metadata
        self.max_length = max_length

    async def process(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Process text through the preprocessing pipeline.

        Args:
            text: Text to process
            metadata: Existing metadata to merge

        Returns:
            tuple: (processed_text, enhanced_metadata)
        """
        result_text = text
        result_metadata = metadata.copy() if metadata else {}

        # Clean text
        if self.clean:
            result_text = TextPreprocessor.clean(result_text)

        # Normalize whitespace
        if self.normalize_whitespace:
            result_text = TextPreprocessor.normalize_whitespace(result_text)

        # Remove headers/footers
        if self.remove_headers_footers:
            result_text = TextPreprocessor.remove_headers_footers(result_text)

        TextPreprocessor.validate_length(result_text, max_length=self.max_length)

        # Extract metadata
        if self.extract_metadata:
            extracted = TextPreprocessor.extract_metadata(result_text)
            result_metadata.update(extracted)

        return result_text, result_metadata
