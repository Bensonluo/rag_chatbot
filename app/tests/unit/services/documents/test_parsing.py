"""File extraction preserves page evidence and fails explicitly when extraction cannot."""

import io
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

from app.core.exceptions import ValidationError
from app.services.documents import parsing


def make_pdf(page_contents: list[str | None], *, with_raster: bool = False) -> bytes:
    writer = PdfWriter()
    for content in page_contents:
        page = writer.add_blank_page(width=300, height=300)
        if content is not None:
            stream = DecodedStreamObject()
            stream.set_data(content.encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {
                            NameObject("/F1"): DictionaryObject(
                                {
                                    NameObject("/Type"): NameObject("/Font"),
                                    NameObject("/Subtype"): NameObject("/Type1"),
                                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                                }
                            )
                        }
                    )
                }
            )
            if with_raster:
                raster = DecodedStreamObject()
                raster.set_data(b"\x00")
                raster.update(
                    {
                        NameObject("/Subtype"): NameObject("/Image"),
                        NameObject("/Width"): NumberObject(1200),
                        NameObject("/Height"): NumberObject(1600),
                    }
                )
                page["/Resources"][NameObject("/XObject")] = DictionaryObject(
                    {NameObject("/Scan"): writer._add_object(raster)}
                )
                stream.set_data(content.encode("ascii") + b"\nq 250 0 0 250 0 0 cm /Scan Do Q")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def text_page(text: str) -> str:
    return f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET"


@pytest.mark.asyncio
@pytest.mark.parametrize("file_type", ["txt", "md"])
async def test_utf8_documents_preserve_structure(file_type: str) -> None:
    text = "# 退款政策\n\n    code\tvalue\n"

    parsed = await parsing.parse_document(text.encode("utf-8-sig"), file_type)

    assert parsed.pages == [parsing.ParsedPage(number=None, text=text)]
    assert parsed.warnings == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoding,text", [("gb18030", "退款政策支持七天内申请。"), ("cp1252", "café")]
)
async def test_legacy_encoding_is_decoded_and_reported(encoding: str, text: str) -> None:
    parsed = await parsing.parse_document(text.encode(encoding), "txt")

    assert parsed.pages[0].text == text
    assert encoding.lower() in parsed.warnings[0].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data", [b"", b"\x00\x01binary", b"PK\x03\x04zipped", b"%PDF-1.7", b"\x81"]
)
async def test_empty_binary_and_undecodable_text_is_rejected(data: bytes) -> None:
    with pytest.raises(ValidationError):
        await parsing.parse_document(data, "txt")


@pytest.mark.asyncio
async def test_pdf_retains_page_numbers_text_and_intentional_blank_pages() -> None:
    data = make_pdf([text_page("First page policy"), None, text_page("Third page exception")])

    parsed = await parsing.parse_document(data, "pdf")

    assert [page.number for page in parsed.pages] == [1, 2, 3]
    assert "First page policy" in parsed.pages[0].text
    assert parsed.pages[1].text == ""
    assert "Third page exception" in parsed.pages[2].text
    assert "blank" in parsed.warnings[0].lower()


@pytest.mark.asyncio
async def test_pdf_layout_preserves_column_spacing() -> None:
    data = make_pdf(["BT /F1 12 Tf 20 200 Td (Item) Tj 150 0 Td (Price) Tj ET"])

    parsed = await parsing.parse_document(data, "pdf")

    text = parsed.pages[0].text
    assert "Item" in text and "Price" in text
    assert len(text[text.index("Item") + 4 : text.index("Price")]) > 1


@pytest.mark.asyncio
async def test_pdf_requires_ocr_for_pages_with_graphics_but_no_text(monkeypatch) -> None:
    data = make_pdf([text_page("First page policy"), "0 0 100 100 re S"])
    monkeypatch.setattr(parsing.shutil, "which", lambda name: None)

    with pytest.raises(ValidationError, match=r"page 2.*OCR"):
        await parsing.parse_document(data, "pdf")


@pytest.mark.asyncio
async def test_pdf_ocr_fills_missing_page_without_replacing_extractable_text(monkeypatch) -> None:
    data = make_pdf([text_page("First page policy"), "0 0 100 100 re S"])
    ocr = Mock(return_value="扫描退款条款")
    monkeypatch.setattr(parsing, "_ocr_pdf_page", ocr)

    parsed = await parsing.parse_document(data, "pdf")

    assert "First page policy" in parsed.pages[0].text
    assert parsed.pages[1].text == "扫描退款条款"
    assert ocr.call_count == 1
    assert ocr.call_args.kwargs["page_number"] == 2
    assert ocr.call_args.kwargs["ocr_languages"] == "eng+chi_sim"
    assert "OCR" in parsed.warnings[0]


@pytest.mark.asyncio
async def test_sparse_searchable_header_does_not_hide_scanned_page_body(monkeypatch) -> None:
    data = make_pdf([text_page("Page 1")], with_raster=True)
    monkeypatch.setattr(parsing, "_ocr_pdf_page", Mock(return_value="退款期限为七天。"))

    parsed = await parsing.parse_document(data, "pdf")

    assert "Page 1" in parsed.pages[0].text
    assert "退款期限为七天。" in parsed.pages[0].text
    assert "OCR" in parsed.warnings[0]


@pytest.mark.asyncio
async def test_sparse_text_over_scanned_body_requires_ocr_tools(monkeypatch) -> None:
    data = make_pdf([text_page("Page 1")], with_raster=True)
    monkeypatch.setattr(parsing.shutil, "which", lambda name: None)

    with pytest.raises(ValidationError, match="needs OCR"):
        await parsing.parse_document(data, "pdf")


@pytest.mark.asyncio
async def test_pdf_fails_if_ocr_cannot_read_page(monkeypatch) -> None:
    monkeypatch.setattr(parsing, "_ocr_pdf_page", Mock(return_value=" \n"))

    with pytest.raises(ValidationError, match="page 1"):
        await parsing.parse_document(make_pdf(["0 0 100 100 re S"]), "pdf")


@pytest.mark.asyncio
async def test_pdf_page_limit_is_checked_before_extraction() -> None:
    with pytest.raises(ValidationError, match="limit"):
        await parsing.parse_document(make_pdf([None, None]), "pdf", max_pages=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [b"not a PDF", make_pdf([None])])
async def test_invalid_and_entirely_blank_pdf_are_rejected(data: bytes) -> None:
    with pytest.raises(ValidationError):
        await parsing.parse_document(data, "pdf")


@pytest.mark.asyncio
async def test_password_protected_pdf_is_rejected() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret")
    output = io.BytesIO()
    writer.write(output)

    with pytest.raises(ValidationError, match="password"):
        await parsing.parse_document(output.getvalue(), "pdf")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"max_pages": 0},
        {"ocr_timeout": 0},
        {"ocr_timeout": float("inf")},
        {"ocr_languages": "--help"},
    ],
)
async def test_invalid_resource_limits_and_ocr_options_are_rejected(options) -> None:
    with pytest.raises(ValidationError):
        await parsing.parse_document(b"Policy", "txt", **options)


@pytest.mark.asyncio
async def test_unsupported_formats_and_whitespace_only_files_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Unsupported"):
        await parsing.parse_document(b"Policy", "docx")
    with pytest.raises(ValidationError, match="no text"):
        await parsing.parse_document(b" \n\t", "txt")


def test_ocr_uses_bounded_subprocess_argument_lists(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(make_pdf([None]))
    monkeypatch.setattr(parsing.shutil, "which", lambda name: f"/usr/bin/{name}")
    run = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "Refund policy", ""),
        ]
    )
    monkeypatch.setattr(parsing.subprocess, "run", run)

    result = parsing._ocr_pdf_page(
        source, page_number=1, ocr_languages="eng+chi_sim", ocr_timeout=3
    )

    assert result == "Refund policy"
    render, recognize = run.call_args_list
    assert render.args[0][0] == "/usr/bin/pdftoppm"
    assert "3000" in render.args[0]
    assert recognize.args[0][0] == "/usr/bin/tesseract"
    assert recognize.args[0][-2:] == ["-l", "eng+chi_sim"]
    assert all(call.kwargs["timeout"] == 3 for call in run.call_args_list)
    assert all(call.kwargs.get("shell", False) is False for call in run.call_args_list)


def test_ocr_timeout_is_actionable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(parsing.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        parsing.subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("pdftoppm", 1))
    )

    with pytest.raises(ValidationError, match="page 3.*timed out"):
        parsing._ocr_pdf_page(
            tmp_path / "source.pdf", page_number=3, ocr_languages="eng", ocr_timeout=1
        )


def test_ocr_failure_is_actionable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(parsing.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        parsing.subprocess,
        "run",
        Mock(side_effect=subprocess.CalledProcessError(1, "tesseract")),
    )

    with pytest.raises(ValidationError, match="OCR failed for PDF page 1"):
        parsing._ocr_pdf_page(
            tmp_path / "source.pdf", page_number=1, ocr_languages="eng", ocr_timeout=1
        )
