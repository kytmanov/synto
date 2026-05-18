"""PDF extraction to markdown-based SourceSegment objects.

Uses pymupdf4llm for PDF→markdown conversion and pymupdf for image extraction.
Pages are grouped by section heading (PDF ToC preferred, heuristic fallback)
with a max_chars ceiling, producing semantically coherent segments rather than
one segment per page.
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

# Heading patterns in pymupdf4llm markdown output
_ATX_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
_BOLD_RE = re.compile(r"^\*{2}(.+?)\*{2}$", re.MULTILINE)

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


def _extract_heading(text: str) -> str | None:
    """Return first H1-H3 or standalone bold line from the first 400 chars, or None."""
    excerpt = text[:400]
    for pat in (_ATX_RE, _BOLD_RE):
        m = pat.search(excerpt)
        if m:
            return m.group(1).strip()
    return None


def _heading_slug(heading: str) -> str:
    """Lowercase alphanumeric slug, max 40 chars."""
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")[:40]


def _group_by_heading(chunks: list[dict]) -> list[tuple[str | None, list[dict]]]:
    """Group consecutive page chunks under the same heading slug.

    Pages with no heading are never merged — each becomes its own singleton group.
    """
    groups: list[tuple[str | None, list[dict]]] = []
    current_slug: str | None = None
    current_chunks: list[dict] = []

    for chunk in chunks:
        heading = _extract_heading(chunk.get("text", ""))
        slug = _heading_slug(heading) if heading else None

        if slug is None:
            if current_chunks:
                groups.append((current_slug, current_chunks))
                current_chunks = []
                current_slug = None
            groups.append((None, [chunk]))
        elif slug == current_slug:
            current_chunks.append(chunk)
        else:
            if current_chunks:
                groups.append((current_slug, current_chunks))
            current_slug = slug
            current_chunks = [chunk]

    if current_chunks:
        groups.append((current_slug, current_chunks))

    return groups


def _toc_groups(chunks: list[dict], fitz_doc: fitz.Document) -> list[tuple[str, list[dict]]] | None:
    """Return [(slug, chunk_list), ...] from the PDF's table of contents.

    Returns None when the ToC is absent or has no level-1 entries.
    Only top-level (level 1) entries are used as chapter/section boundaries.
    """
    toc = fitz_doc.get_toc(simple=True)  # [[level, title, page_1indexed], ...]
    if not toc:
        return None

    level1 = [(title, page) for level, title, page in toc if level == 1]
    if not level1:
        return None

    # Map 1-indexed page numbers → chunks
    page_to_chunks: dict[int, list[dict]] = {}
    for chunk in chunks:
        pn = chunk.get("metadata", {}).get("page_number", 0)
        page_to_chunks.setdefault(pn, []).append(chunk)

    max_page = max(page_to_chunks.keys(), default=0)
    slug_counter: dict[str, int] = {}
    groups: list[tuple[str, list[dict]]] = []

    for i, (title, start_page) in enumerate(level1):
        end_page = level1[i + 1][1] - 1 if i + 1 < len(level1) else max_page

        base_slug = _heading_slug(title)
        slug_counter[base_slug] = slug_counter.get(base_slug, 0) + 1
        count = slug_counter[base_slug]
        slug = base_slug if count == 1 else f"{base_slug}-{count}"

        group_chunks: list[dict] = []
        for pn in range(start_page, end_page + 1):
            group_chunks.extend(page_to_chunks.get(pn, []))

        if group_chunks:
            groups.append((slug, group_chunks))

    return groups if groups else None


def _split_by_size(
    groups: list[tuple[str | None, list[dict]]], max_chars: int
) -> list[tuple[str | None, list[dict], int | None]]:
    """Apply a max_chars ceiling to each group.

    Returns (slug, chunks, part_idx).
    part_idx is None when the group fits; a 1-based int when the group was split.
    """
    result: list[tuple[str | None, list[dict], int | None]] = []

    for slug, chunks in groups:
        combined_len = sum(len(c.get("text", "")) for c in chunks) + max(0, len(chunks) - 1)
        if combined_len <= max_chars:
            result.append((slug, chunks, None))
            continue

        part_idx = 1
        current_part: list[dict] = []
        current_len = 0

        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_len = len(text)
            if not current_part:
                current_part.append(chunk)
                current_len = chunk_len
            elif current_len + 1 + chunk_len <= max_chars:
                current_part.append(chunk)
                current_len += 1 + chunk_len
            else:
                result.append((slug, current_part, part_idx))
                part_idx += 1
                current_part = [chunk]
                current_len = chunk_len

        if current_part:
            result.append((slug, current_part, part_idx))

    return result


def extract_pdf(
    source_id: str,
    path: Path,
    db: StateDB,
    *,
    vault_root: Path | None = None,
    max_chars: int = 8000,
) -> list[SourceSegment]:
    """Extract a PDF into a list of SourceSegment objects.

    Pages are grouped by section (ToC preferred, heuristic heading fallback)
    with a ``max_chars`` ceiling.  Segment IDs are stable across re-runs:
    ``<source_id>:page:<n>:<hash>`` or ``<source_id>:section:<slug>[:<part>]:<hash>``.

    Images are written to ``<vault_root>/assets/<source_id>/`` when
    ``vault_root`` is provided; otherwise image extraction is skipped.
    """
    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    if not chunks:
        return []

    fitz_doc = fitz.open(str(path))
    heading_groups = _toc_groups(chunks, fitz_doc) or _group_by_heading(chunks)
    final_groups = _split_by_size(heading_groups, max_chars)

    segments: list[SourceSegment] = []
    now = datetime.now(UTC).isoformat()

    for ordinal, (slug, group_chunks, part_idx) in enumerate(final_groups):
        text = "\n".join(c.get("text", "") for c in group_chunks)
        content_hash = hashlib.sha256(text.encode()).hexdigest()

        # Build structural locator
        if slug is None:
            pn = group_chunks[0].get("metadata", {}).get("page_number", ordinal + 1) - 1
            locator = f"page:{pn}"
        elif part_idx is None:
            locator = f"section:{slug}"
        else:
            locator = f"section:{slug}:part{part_idx}"

        seg_id = f"{source_id}:{locator}:{content_hash[:8]}"
        identity = f"{source_id}:{locator}"

        # page_range: 0-indexed min/max across all chunks in this group
        pages = [c.get("metadata", {}).get("page_number", 1) - 1 for c in group_chunks]
        page_range = (min(pages), max(pages))

        # Images: collect from every page in the range using fitz
        image_refs: list[str] = []
        if vault_root is not None:
            for page in range(page_range[0], page_range[1] + 1):
                if page >= len(fitz_doc):
                    continue
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
                           (path, source_id, asset_type, master_path,
                            created_at, referenced_by_json)
                           VALUES (?, ?, 'image', ?, ?, '[]')""",
                        (rel_path, source_id, str(path), now),
                    )

        # Equations: collect per chunk, preserving original page number for ref keys
        eq_refs: list[str] = []
        for c in group_chunks:
            pg = c.get("metadata", {}).get("page_number", 1) - 1
            eq_refs.extend(_detect_equations(c.get("text", ""), pg))

        segment = SourceSegment(
            id=seg_id,
            identity=identity,
            ordinal=ordinal,
            source_id=source_id,
            structural_locator=locator,
            content_hash=content_hash,
            text=text,
            page_range=page_range,
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
                locator,
                content_hash,
                text,
                page_range[0],
                page_range[1],
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
