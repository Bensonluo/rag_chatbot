"""Reference snapshots supplied to answer generation, not claim verification."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

CitationText = Annotated[str, Field(max_length=512)]


class SourceCitation(BaseModel):
    """Allowlisted source identity and a bounded excerpt from this answer's evidence.

    Offsets address normalized source text, with an exclusive end. Graph source
    arrays are independent provenance sets; their positions do not imply pairs.
    FAQ versions are content digests, not downloadable document revision IDs.
    """

    model_config = ConfigDict(extra="ignore", strict=True)

    source_type: Literal["document", "graph", "faq"]
    document_id: str | None = Field(default=None, max_length=512)
    version_id: str | None = Field(default=None, max_length=512)
    chunk_id: str | None = Field(default=None, max_length=512)
    faq_id: str | None = Field(default=None, max_length=512)
    title: str | None = Field(default=None, max_length=500)
    excerpt: str = Field(default="", max_length=800)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    start_pos: int | None = Field(default=None, ge=0)
    end_pos: int | None = Field(default=None, ge=0)
    section_path: list[CitationText] = Field(default_factory=list, max_length=20)
    source_document_ids: list[CitationText] = Field(default_factory=list, max_length=50)
    source_version_ids: list[CitationText] = Field(default_factory=list, max_length=50)
    source_chunk_ids: list[CitationText] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def ordered_locations(self) -> Self:
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must not precede page_start")
        if (
            self.start_pos is not None
            and self.end_pos is not None
            and self.end_pos < self.start_pos
        ):
            raise ValueError("end_pos must not precede start_pos")
        return self
