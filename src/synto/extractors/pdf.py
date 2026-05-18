"""PDF extraction to markdown-based SourceSegment objects.

Uses pymupdf4llm for PDF→markdown conversion and pymupdf for image extraction.
Each page becomes one SourceSegment with a stable content-hash-based ID.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import fitz  # pymupdf — transitive dep of pymupdf4llm
import pymupdf4llm

from ..models import BibliographicMetadata, SourceSegment
from ..state import StateDB

# Patterns used to detect mathematical notation in markdown text
_EQ_PATTERNS = [
    re.compile(r"\$\$.+?\$\$", re.DOTALL),
    re.compile(r"\\\[.+?\\\]", re.DOTALL),
    re.compile(r"\\\(.+?\\\)", re.DOTALL),
    re.compile(r"\\(?:frac|sum|int|prod|sqrt|begin)\b"),
]


def _detect_equations(text: str, page: int) -> list[str]:
    """Return a list of equation ref keys for any math patterns found in text."""
    refs: list[str] = []
    n = 0
    for pat in _EQ_PATTERNS:
        for _ in pat.finditer(text):
            refs.append(f"eq-{page}-{n}")
            n += 1
    return refs


def extract_pdf(
    source_id: str,
    path: Path,
    db: StateDB,
    *,
    vault_root: Path | None = None,
) -> list[SourceSegment]:
    """Extract a PDF into a list of SourceSegment objects.

    Each page becomes one segment whose ``text`` field is markdown produced by
    pymupdf4llm.  Segment IDs are stable across re-runs for the same file
    content: ``<source_id>:<page>-<page>:<content_hash[:8]>``.

    Images are written to ``<vault_root>/assets/<source_id>/`` when
    ``vault_root`` is provided; otherwise image extraction is skipped.
    """
    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    if not chunks:
        return []

    fitz_doc = fitz.open(str(path))
    segments: list[SourceSegment] = []
    now = datetime.now(UTC).isoformat()

    for ordinal, chunk in enumerate(chunks):
        # page_number in pymupdf4llm metadata is 1-indexed
        page = chunk["metadata"].get("page_number", ordinal + 1) - 1
        text = chunk.get("text", "")
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        seg_id = f"{source_id}:{page}-{page}:{content_hash[:8]}"
        identity = f"{source_id}:{page}-{page}"

        image_refs: list[str] = []
        if vault_root is not None and page < len(fitz_doc):
            fitz_page = fitz_doc[page]
            for img_idx, img_tuple in enumerate(fitz_page.get_images(full=True)):
                xref = img_tuple[0]
                base_image = fitz_doc.extract_image(xref)
                ext = base_image.get("ext", "png")
                img_bytes = base_image.get("image", b"")
                if not img_bytes:
                    continue
                img_dir = vault_root / "assets" / source_id
                img_dir.mkdir(parents=True, exist_ok=True)
                rel_path = f"assets/{source_id}/img-{page}-{img_idx}.{ext}"
                (vault_root / rel_path).write_bytes(img_bytes)
                image_refs.append(rel_path)
                db._conn.execute(
                    """INSERT OR REPLACE INTO generated_assets
                       (path, source_id, asset_type, master_path, created_at, referenced_by_json)
                       VALUES (?, ?, 'image', ?, ?, '[]')""",
                    (rel_path, source_id, str(path), now),
                )

        eq_refs = _detect_equations(text, page)

        segment = SourceSegment(
            id=seg_id,
            identity=identity,
            ordinal=ordinal,
            source_id=source_id,
            structural_locator=f"page:{page}",
            content_hash=content_hash,
            text=text,
            page_range=(page, page),
            image_refs=image_refs,
            equation_refs=eq_refs,
        )
        segments.append(segment)

        extra: dict = {}
        if image_refs:
            extra["image_refs"] = image_refs
        if eq_refs:
            extra["equation_refs"] = eq_refs

        db._conn.execute(
            """INSERT OR REPLACE INTO source_segments
               (id, identity, ordinal, source_id, structural_locator, content_hash,
                text, section_path_json, page_start, page_end, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, '[]', ?, ?, ?)""",
            (
                seg_id,
                identity,
                ordinal,
                source_id,
                f"page:{page}",
                content_hash,
                text,
                page,
                page,
                json.dumps(extra) if extra else None,
            ),
        )

    fitz_doc.close()
    db._conn.commit()
    return segments


def extract_bibliographic_metadata(path: Path, first_page_md: str) -> BibliographicMetadata:
    """Heuristic extraction of bibliographic metadata from a PDF file.

    Tries PDF metadata dict first, then falls back to first-page text heuristics
    for title.  DOI and year are extracted via regex from the first page markdown.
    """
    doc = fitz.open(str(path))
    meta = doc.metadata or {}
    doc.close()

    title = meta.get("title", "").strip()
    if not title:
        for line in first_page_md.splitlines():
            line = line.strip().lstrip("#").strip()
            if line:
                title = line
                break

    authors: list[str] = []
    raw_authors = meta.get("author", "").strip()
    if raw_authors:
        authors = [a.strip() for a in re.split(r"[,;&]", raw_authors) if a.strip()]

    doi: str | None = None
    doi_match = re.search(r"10\.\d{4,}/[^\s]+", first_page_md)
    if doi_match:
        doi = doi_match.group(0).rstrip(".,;)")

    year: int | None = None
    year_match = re.search(r"\b(19|20)\d{2}\b", first_page_md)
    if year_match:
        year = int(year_match.group(0))
    if year is None:
        creation_date = meta.get("creationDate", "")
        date_match = re.search(r"D:(\d{4})", creation_date)
        if date_match:
            year = int(date_match.group(1))

    return BibliographicMetadata(
        title=title or "Unknown",
        authors=authors,
        doi=doi,
        year=year,
    )
