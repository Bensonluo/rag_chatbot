"""Decode supported uploads and extract PDF pages without silently omitting evidence."""

import asyncio
import io
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.exceptions import ValidationError


@dataclass(frozen=True)
class ParsedPage:
    number: int | None
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    pages: list[ParsedPage]
    warnings: list[str]


async def parse_document(
    data: bytes,
    file_type: str,
    *,
    ocr_languages: str = "eng+chi_sim",
    max_pages: int = 500,
    ocr_timeout: float = 60,
) -> ParsedDocument:
    """Parse a complete upload; missing PDF evidence fails before anything is indexed."""
    if not data:
        raise ValidationError("Document file is empty")
    if type(max_pages) is not int or max_pages <= 0:
        raise ValidationError("max_pages must be a positive integer")
    if (
        not isinstance(ocr_timeout, int | float)
        or not math.isfinite(ocr_timeout)
        or ocr_timeout <= 0
    ):
        raise ValidationError("ocr_timeout must be a positive finite number")
    if not re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", ocr_languages):
        raise ValidationError("OCR languages must be language codes such as eng+chi_sim")

    file_type = file_type.lower().lstrip(".")
    if file_type in {"txt", "md"}:
        return _decode_text(data)
    if file_type != "pdf":
        raise ValidationError(f"Unsupported file type: {file_type}. Supported: txt, md, pdf")
    return await asyncio.to_thread(_parse_pdf, data, ocr_languages, max_pages, ocr_timeout)


def _decode_text(data: bytes) -> ParsedDocument:
    if data.startswith((b"%PDF-", b"PK\x03\x04")):
        raise ValidationError("The uploaded file contains binary document data, not plain text")

    for encoding in ("utf-8-sig", "gb18030", "cp1252"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if any(
            (ord(char) < 32 and char not in "\t\r\n\f") or 0x7F <= ord(char) < 0xA0 for char in text
        ):
            raise ValidationError("The uploaded text contains binary control characters")
        if not text.strip():
            raise ValidationError("Document contains no text")
        warnings = []
        if encoding != "utf-8-sig":
            warnings.append(
                f"Decoded text using {encoding}; save the source as UTF-8 to make its encoding explicit."
            )
        return ParsedDocument(pages=[ParsedPage(number=None, text=text)], warnings=warnings)

    raise ValidationError(
        "Unable to decode document text. Save the file as UTF-8 and upload it again."
    )


def _parse_pdf(
    data: bytes, ocr_languages: str, max_pages: int, ocr_timeout: float
) -> ParsedDocument:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValidationError("The PDF is password protected; unlock it before uploading")
        if len(reader.pages) > max_pages:
            raise ValidationError(
                f"PDF contains {len(reader.pages)} pages; the configured limit is {max_pages}. "
                "Split the PDF or increase the limit."
            )

        pages = []
        warnings = []
        with tempfile.TemporaryDirectory(prefix="rag-ingest-ocr-") as directory:
            source = Path(directory) / "source.pdf"
            for number, page in enumerate(reader.pages, start=1):
                # A truly blank page has no drawing/text instructions or annotations.
                if page.get_contents() is None and not page.get("/Annots"):
                    pages.append(ParsedPage(number=number, text=""))
                    warnings.append(f"PDF page {number} is blank.")
                    continue

                try:
                    text = (
                        page.extract_text(extraction_mode="layout", layout_mode_strip_rotated=False)
                        or ""
                    )
                except Exception:
                    # Damaged/unsupported text layers can still have readable rendered pages.
                    text = ""
                sparse_raster_page = len(re.sub(r"\s", "", text)) < 200 and _has_large_raster(page)
                if not text.strip() or sparse_raster_page:
                    extracted_text = text
                    if not source.exists():
                        source.write_bytes(data)
                    text = _ocr_pdf_page(
                        source,
                        page_number=number,
                        ocr_languages=ocr_languages,
                        ocr_timeout=ocr_timeout,
                    )
                    if not text.strip():
                        raise ValidationError(
                            f"PDF page {number} has no readable text even after OCR. "
                            "Upload a clearer scan or a searchable PDF; no pages were indexed."
                        )
                    if extracted_text.strip() and extracted_text.strip() not in text:
                        text = extracted_text + "\n\n" + text
                    warnings.append(
                        f"PDF page {number} was read using OCR; verify its extracted text."
                    )
                pages.append(ParsedPage(number=number, text=text))

        if not any(page.text.strip() for page in pages):
            raise ValidationError("PDF contains no readable text")
        return ParsedDocument(pages=pages, warnings=warnings)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("Unable to read PDF. Upload a valid, unencrypted PDF file.") from exc


def _has_large_raster(page: Any) -> bool:
    """Detect a likely scanned body behind a small searchable header or page number.

    Read image dictionaries, without decoding pixels or requiring Pillow. Forms
    may wrap images, so inspect their resources too, with a recursion bound.
    """

    def contains_image(resources: Any, depth: int = 0) -> bool:
        if resources is None or depth > 8:
            return False
        objects = resources.get_object().get("/XObject")
        if objects is None:
            return False
        for reference in objects.get_object().values():
            obj = reference.get_object()
            if obj.get("/Subtype") == "/Image":
                width, height = int(obj.get("/Width", 0)), int(obj.get("/Height", 0))
                if min(width, height) >= 256 and width * height >= 90_000:
                    return True
            if obj.get("/Subtype") == "/Form" and contains_image(obj.get("/Resources"), depth + 1):
                return True
        return False

    return contains_image(page.get("/Resources"))


def _ocr_pdf_page(source: Path, *, page_number: int, ocr_languages: str, ocr_timeout: float) -> str:
    """Render one bounded image and run local OCR without a shell or persistent files."""
    renderer = shutil.which("pdftoppm")
    recognizer = shutil.which("tesseract")
    if not renderer or not recognizer:
        raise ValidationError(
            f"PDF page {page_number} needs OCR. Install pdftoppm (Poppler) and Tesseract "
            f"with the {ocr_languages} language data, or upload a searchable PDF. "
            "No pages were indexed."
        )

    with tempfile.TemporaryDirectory(prefix="rag-ingest-page-") as directory:
        output = Path(directory) / "page"
        try:
            subprocess.run(
                [
                    renderer,
                    "-f",
                    str(page_number),
                    "-l",
                    str(page_number),
                    "-singlefile",
                    "-r",
                    "200",
                    "-scale-to",
                    "3000",
                    "-png",
                    str(source),
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=ocr_timeout,
            )
            result = subprocess.run(
                [recognizer, str(output.with_suffix(".png")), "stdout", "-l", ocr_languages],
                check=True,
                capture_output=True,
                text=True,
                timeout=ocr_timeout,
            )
            return result.stdout
        except subprocess.TimeoutExpired as exc:
            raise ValidationError(
                f"OCR for PDF page {page_number} timed out after {ocr_timeout} seconds. "
                "Upload a smaller scan or increase the OCR timeout; no pages were indexed."
            ) from exc
        except (subprocess.CalledProcessError, OSError) as exc:
            raise ValidationError(
                f"OCR failed for PDF page {page_number}. Verify Poppler, Tesseract, and "
                f"the {ocr_languages} language data are installed, or upload a searchable PDF. "
                "No pages were indexed."
            ) from exc
