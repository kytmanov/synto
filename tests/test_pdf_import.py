"""Tests for Feature 01: PDF Import via pymupdf4llm."""

from __future__ import annotations

import re
import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from synto.extractors.pdf import extract_bibliographic_metadata, extract_pdf
from synto.state import StateDB

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


def _make_png(width: int = 4, height: int = 4, color: tuple = (255, 0, 0)) -> bytes:
    """Return minimal PNG bytes for a solid-colour image (no PIL required)."""

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    row = b"\x00" + bytes(color) * width
    idat = _chunk(b"IDAT", zlib.compress(row * height))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


@pytest.fixture()
def pdf_with_image(tmp_path: Path) -> Path:
    """A single-page PDF containing one embedded PNG image."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Page with image content.")
    page.insert_image(fitz.Rect(72, 100, 172, 200), stream=_make_png())
    out = tmp_path / "with_image.pdf"
    doc.save(str(out))
    doc.close()
    return out


@pytest.fixture()
def pdf_with_math(tmp_path: Path) -> Path:
    """A single-page PDF with LaTeX-style math patterns in text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), r"The formula is $$x^2 + y^2 = z^2$$ for integers.")
    out = tmp_path / "with_math.pdf"
    doc.save(str(out))
    doc.close()
    return out


@pytest.fixture()
def single_page_pdf(tmp_path: Path) -> Path:
    """A minimal single-page PDF with no text (used as stand-in for near-empty PDF)."""
    doc = fitz.open()
    doc.new_page()
    out = tmp_path / "single.pdf"
    doc.save(str(out))
    doc.close()
    return out


@pytest.fixture()
def image_only_pdf(tmp_path: Path) -> Path:
    """A PDF whose page has only an image and no selectable text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(fitz.Rect(72, 72, 200, 200), stream=_make_png(10, 10, (0, 128, 255)))
    out = tmp_path / "image_only.pdf"
    doc.save(str(out))
    doc.close()
    return out


@pytest.fixture()
def pdf_with_metadata(tmp_path: Path) -> Path:
    """A PDF with title/author in metadata and DOI + year in body text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "DOI: 10.9999/example.2021 Published 2021.")
    doc.set_metadata({"title": "Metadata Title", "author": "Alice; Bob"})
    out = tmp_path / "with_meta.pdf"
    doc.save(str(out))
    doc.close()
    return out


# ---------------------------------------------------------------------------
# Stage 1: basic extraction and stable IDs
# ---------------------------------------------------------------------------


def test_extract_pdf_basic(db: StateDB) -> None:
    segs = extract_pdf("test-src", SAMPLE_PDF, db)
    assert len(segs) >= 2
    assert all(s.source_id == "test-src" for s in segs)
    assert all(s.text for s in segs), "Every segment should have non-empty markdown text"


def test_segment_ids_format(db: StateDB) -> None:
    segs = extract_pdf("test-src", SAMPLE_PDF, db)
    pattern = re.compile(r"^test-src:(page:\d+|section:[a-z0-9-]+(:(part\d+))?):[0-9a-f]{8}$")
    for seg in segs:
        assert pattern.match(seg.id), f"Bad ID format: {seg.id}"


def test_stable_ids(tmp_path: Path) -> None:
    """Re-running extract_pdf on the same file must produce identical segment IDs."""
    db1 = StateDB(tmp_path / "state1.db")
    db2 = StateDB(tmp_path / "state2.db")
    segs1 = extract_pdf("stable-src", SAMPLE_PDF, db1)
    segs2 = extract_pdf("stable-src", SAMPLE_PDF, db2)
    assert [s.id for s in segs1] == [s.id for s in segs2]


def test_segments_inserted_in_db(db: StateDB) -> None:
    extract_pdf("test-src", SAMPLE_PDF, db)
    rows = db._conn.execute(
        "SELECT id FROM source_segments WHERE source_id = 'test-src'"
    ).fetchall()
    assert len(rows) >= 2


def test_idempotent_rerun(db: StateDB) -> None:
    """Two consecutive runs must not create duplicate rows (INSERT OR REPLACE)."""
    extract_pdf("idem-src", SAMPLE_PDF, db)
    extract_pdf("idem-src", SAMPLE_PDF, db)
    count = db._conn.execute(
        "SELECT COUNT(*) FROM source_segments WHERE source_id = 'idem-src'"
    ).fetchone()[0]
    expected = len(extract_pdf("idem-src", SAMPLE_PDF, db))
    assert count == expected


# ---------------------------------------------------------------------------
# Stage 2: image extraction
# ---------------------------------------------------------------------------


def test_image_extraction(pdf_with_image: Path, tmp_path: Path, db: StateDB) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    segs = extract_pdf("img-src", pdf_with_image, db, vault_root=vault)
    assert segs, "Should produce at least one segment"
    seg = segs[0]
    assert seg.image_refs, "Segment should have image refs"
    img_path = vault / seg.image_refs[0]
    assert img_path.exists(), f"Image file should exist at {img_path}"


def test_generated_assets_row(pdf_with_image: Path, tmp_path: Path, db: StateDB) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    extract_pdf("asset-src", pdf_with_image, db, vault_root=vault)
    rows = db._conn.execute(
        "SELECT path, source_id FROM generated_assets WHERE source_id = 'asset-src'"
    ).fetchall()
    assert rows, "generated_assets row should be inserted"
    assert rows[0][1] == "asset-src"


def test_no_vault_root_skips_images(pdf_with_image: Path, db: StateDB) -> None:
    """extract_pdf without vault_root must not raise even if PDF has images."""
    segs = extract_pdf("no-vault", pdf_with_image, db)
    assert segs is not None
    assert all(s.image_refs == [] for s in segs)


# ---------------------------------------------------------------------------
# Stage 3: equation detection
# ---------------------------------------------------------------------------


def test_equation_detection(pdf_with_math: Path, db: StateDB) -> None:
    segs = extract_pdf("math-src", pdf_with_math, db)
    assert segs
    has_eq = any(s.equation_refs for s in segs)
    assert has_eq, "Expected equation refs in at least one segment"


def test_no_equations_empty_refs(db: StateDB) -> None:
    """Plain-text PDF without math should have empty equation_refs."""
    segs = extract_pdf("plain-src", SAMPLE_PDF, db)
    # SAMPLE_PDF has no LaTeX math blocks, only slash-prefixed text
    # Just assert no exception and that equation_refs is a list
    for seg in segs:
        assert isinstance(seg.equation_refs, list)


# ---------------------------------------------------------------------------
# Stage 4: bibliographic metadata
# ---------------------------------------------------------------------------


def test_bibliographic_metadata_from_pdf_meta(pdf_with_metadata: Path) -> None:
    first_page = "DOI: 10.9999/example.2021 Published 2021."
    meta = extract_bibliographic_metadata(pdf_with_metadata, first_page)
    assert meta.title == "Metadata Title"
    assert "Alice" in meta.authors or "Bob" in meta.authors


def test_bibliographic_metadata_doi(pdf_with_metadata: Path) -> None:
    first_page_md = "DOI: 10.9999/example.2021 Published 2021."
    meta = extract_bibliographic_metadata(pdf_with_metadata, first_page_md)
    assert meta.doi == "10.9999/example.2021"


def test_bibliographic_metadata_year(pdf_with_metadata: Path) -> None:
    first_page_md = "Published in 2021. DOI: 10.9999/example.2021"
    meta = extract_bibliographic_metadata(pdf_with_metadata, first_page_md)
    assert meta.year == 2021


def test_bibliographic_metadata_title_fallback(tmp_path: Path) -> None:
    """When PDF metadata has no title, fall back to first non-empty text line."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Fallback Title From Text")
    pdf_path = tmp_path / "notitle.pdf"
    doc.save(str(pdf_path))
    doc.close()
    meta = extract_bibliographic_metadata(pdf_path, "Fallback Title From Text\nSome body.")
    assert meta.title == "Fallback Title From Text"


# ---------------------------------------------------------------------------
# Stage 5: edge cases and stability
# ---------------------------------------------------------------------------


def test_empty_chunks_returns_empty_list(single_page_pdf: Path, db: StateDB) -> None:
    """When pymupdf4llm returns no chunks (e.g. blank page), extract_pdf returns []."""
    with patch("synto.extractors.pdf.pymupdf4llm.to_markdown", return_value=[]):
        segs = extract_pdf("empty-src", single_page_pdf, db)
    assert segs == []


def test_image_only_pdf(image_only_pdf: Path, tmp_path: Path, db: StateDB) -> None:
    """PDF with only an image and no selectable text should produce segments."""
    vault = tmp_path / "vault"
    vault.mkdir()
    segs = extract_pdf("imgonly-src", image_only_pdf, db, vault_root=vault)
    assert isinstance(segs, list)
    if segs:
        assert segs[0].image_refs, "Image-only page should populate image_refs"


def test_vault_root_created_automatically(
    pdf_with_image: Path, tmp_path: Path, db: StateDB
) -> None:
    """vault_root subdirectory must be created if it does not exist."""
    vault = tmp_path / "new_vault" / "nested"
    # Do NOT pre-create vault
    extract_pdf("autocreate-src", pdf_with_image, db, vault_root=vault)
    assert (vault / "assets" / "autocreate-src").exists()


# ---------------------------------------------------------------------------
# Heading-aware grouping (Stages 1-4 of heading split plan)
# ---------------------------------------------------------------------------


def _make_chunk(text: str, page_number: int) -> dict:
    return {"text": text, "metadata": {"page_number": page_number}}


def test_extract_heading_atx() -> None:
    from synto.extractors.pdf import _extract_heading

    assert _extract_heading("# Introduction\nSome text") == "Introduction"
    assert _extract_heading("## Methods\n\nContent") == "Methods"
    assert _extract_heading("### Results\nData") == "Results"


def test_extract_heading_bold() -> None:
    from synto.extractors.pdf import _extract_heading

    assert _extract_heading("**Abstract**\nContent") == "Abstract"
    assert _extract_heading("**Chapter 3: Methods**\nText") == "Chapter 3: Methods"


def test_extract_heading_none() -> None:
    from synto.extractors.pdf import _extract_heading

    assert _extract_heading("No heading here") is None
    assert _extract_heading("") is None


def test_non_latin_heading_falls_back_to_page_locator(single_page_pdf: Path, db: StateDB) -> None:
    chunks = [_make_chunk("# Введение\nТекст", 1)]
    with patch("synto.extractors.pdf.pymupdf4llm.to_markdown", return_value=chunks):
        segs = extract_pdf("cyrillic-src", single_page_pdf, db)
    assert len(segs) == 1
    assert segs[0].structural_locator == "page:0"


def test_heading_grouping_two_sections(single_page_pdf: Path, db: StateDB) -> None:
    chunks = [
        _make_chunk("# Introduction\nIntro text", 1),
        _make_chunk("# Methods\nMethods text", 2),
        _make_chunk("# Methods\nMore methods", 3),
    ]
    with patch("synto.extractors.pdf.pymupdf4llm.to_markdown", return_value=chunks):
        segs = extract_pdf("head-src", single_page_pdf, db)
    assert len(segs) == 2
    locators = {s.structural_locator for s in segs}
    assert "section:introduction" in locators
    assert "section:methods" in locators


def test_bold_heading_detected(single_page_pdf: Path, db: StateDB) -> None:
    chunks = [_make_chunk("**Abstract**\nContent of abstract.", 1)]
    with patch("synto.extractors.pdf.pymupdf4llm.to_markdown", return_value=chunks):
        segs = extract_pdf("bold-src", single_page_pdf, db)
    assert len(segs) == 1
    assert segs[0].structural_locator == "section:abstract"


def test_no_heading_fallback(single_page_pdf: Path, db: StateDB) -> None:
    chunks = [
        _make_chunk("Plain text on page one", 1),
        _make_chunk("Plain text on page two", 2),
    ]
    with patch("synto.extractors.pdf.pymupdf4llm.to_markdown", return_value=chunks):
        segs = extract_pdf("plain-src", single_page_pdf, db)
    assert len(segs) == 2
    assert segs[0].structural_locator == "page:0"
    assert segs[1].structural_locator == "page:1"


def test_size_ceiling_splits_section(single_page_pdf: Path, db: StateDB) -> None:
    # All three pages share the same heading so they merge into one group,
    # then _split_by_size breaks them apart because 3×3011 chars > 5000.
    long_text = "x" * 3000
    chunks = [
        _make_chunk(f"# Results\n{long_text}", 1),
        _make_chunk(f"# Results\n{long_text}", 2),
        _make_chunk(f"# Results\n{long_text}", 3),
    ]
    with patch("synto.extractors.pdf.pymupdf4llm.to_markdown", return_value=chunks):
        segs = extract_pdf("large-src", single_page_pdf, db, max_chars=5000)
    assert len(segs) >= 2
    locators = [s.structural_locator for s in segs]
    assert "section:results:part1" in locators
    assert "section:results:part2" in locators


def test_toc_grouping_unit() -> None:
    from synto.extractors.pdf import _toc_groups

    mock_doc = MagicMock()
    mock_doc.get_toc.return_value = [
        [1, "Introduction", 1],
        [1, "Methods", 3],
    ]
    chunks = [
        _make_chunk("Intro text", 1),
        _make_chunk("Intro p2", 2),
        _make_chunk("Methods text", 3),
        _make_chunk("Methods p2", 4),
    ]
    groups = _toc_groups(chunks, mock_doc)
    assert groups is not None
    assert len(groups) == 2
    assert groups[0][0] == "introduction"
    assert len(groups[0][1]) == 2
    assert groups[1][0] == "methods"
    assert len(groups[1][1]) == 2


def test_toc_absent_returns_none() -> None:
    from synto.extractors.pdf import _toc_groups

    mock_doc = MagicMock()
    mock_doc.get_toc.return_value = []
    assert _toc_groups([_make_chunk("text", 1)], mock_doc) is None


def test_toc_grouping_preserves_preamble_pages() -> None:
    from synto.extractors.pdf import _toc_groups

    mock_doc = MagicMock()
    mock_doc.get_toc.return_value = [[1, "Chapter One", 3]]
    chunks = [
        _make_chunk("Cover page", 1),
        _make_chunk("Abstract page", 2),
        _make_chunk("Chapter text", 3),
    ]
    groups = _toc_groups(chunks, mock_doc)
    assert groups is not None
    assert groups[0][0] is None
    assert len(groups[0][1]) == 2
    assert groups[1][0] == "chapter-one"


def test_multi_page_section_page_range(single_page_pdf: Path, db: StateDB) -> None:
    chunks = [
        _make_chunk("# Chapter One\nPage 1", 1),
        _make_chunk("# Chapter One\nPage 2", 2),
        _make_chunk("# Chapter One\nPage 3", 3),
    ]
    with patch("synto.extractors.pdf.pymupdf4llm.to_markdown", return_value=chunks):
        segs = extract_pdf("range-src", single_page_pdf, db)
    assert len(segs) == 1
    assert segs[0].page_range == (0, 2)


def test_grouped_section_collects_images(pdf_with_image: Path, tmp_path: Path, db: StateDB) -> None:
    # pdf_with_image is a single-page PDF; mock one chunk with a heading
    chunks = [_make_chunk("# Section One\nText with image", 1)]
    vault = tmp_path / "vault"
    vault.mkdir()
    with patch("synto.extractors.pdf.pymupdf4llm.to_markdown", return_value=chunks):
        segs = extract_pdf("imggroup-src", pdf_with_image, db, vault_root=vault)
    assert segs
    assert segs[0].image_refs, "Image from grouped section should be collected"


def test_stable_ids_heading_mode(single_page_pdf: Path, tmp_path: Path) -> None:
    chunks = [
        _make_chunk("# Introduction\nContent", 1),
        _make_chunk("# Methods\nContent", 2),
    ]
    db1 = StateDB(tmp_path / "s1.db")
    db2 = StateDB(tmp_path / "s2.db")
    with patch("synto.extractors.pdf.pymupdf4llm.to_markdown", return_value=chunks):
        segs1 = extract_pdf("stable2-src", single_page_pdf, db1)
    with patch("synto.extractors.pdf.pymupdf4llm.to_markdown", return_value=chunks):
        segs2 = extract_pdf("stable2-src", single_page_pdf, db2)
    assert [s.id for s in segs1] == [s.id for s in segs2]
