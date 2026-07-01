"""Tests for article quality signals in frontmatter.

See ImpPlan.md §Stage 6 for the original test matrix and the P0–P5 follow-up
plan for the additional cases added later.

Note on MCP coverage: the `list_articles` MCP tool defined inline in
`serve.py:run_server` is a thin projection over `ArticleRef`. We test that
projection at the reader level (`VaultReader.list_articles` carries the new
fields) plus the `read_article` path. An in-process MCP-server invocation
test is intentionally not added — it would exercise the same code through a
heavier transport.
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock

import frontmatter as fm_lib
import pytest
from conftest import as_router

from synto.config import Config
from synto.models import RawNoteRecord, WikiArticleRecord
from synto.pipeline.compile import (
    _source_quality_summary,
    approve_drafts,
    compile_concepts,
)
from synto.pipeline.query import _build_synthesis_file_text, _save_synthesis
from synto.readers import VaultReader
from synto.state import StateDB
from synto.vault import build_wiki_frontmatter, parse_note, write_note


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / "wiki" / "synthesis").mkdir()
    (tmp_path / ".synto").mkdir()
    return tmp_path


@pytest.fixture
def config(vault):
    return Config(vault=vault)


@pytest.fixture
def db(config):
    return StateDB(config.state_db_path)


def _mock_client(article_json: str):
    client = MagicMock()
    client.generate.return_value = article_json
    return as_router(client)


# ── vault.py: build_wiki_frontmatter ──────────────────────────────────────


class TestBuildWikiFrontmatter:
    def test_includes_new_fields(self):
        meta = build_wiki_frontmatter(
            title="Test",
            tags=["tag1"],
            sources=["raw/a.md"],
            confidence=0.75,
            source_count=3,
            single_source=False,
            source_quality="high",
        )
        assert meta["source_count"] == 3
        assert meta["single_source"] is False
        assert meta["source_quality"] == "high"

    def test_omits_when_none(self):
        meta = build_wiki_frontmatter(
            title="Test",
            tags=["tag1"],
            sources=["raw/a.md"],
            confidence=0.5,
        )
        assert "source_count" not in meta
        assert "single_source" not in meta
        assert "source_quality" not in meta

    def test_yaml_serialization_of_quality_fields(self):
        meta = build_wiki_frontmatter(
            title="Test",
            tags=["tag1"],
            sources=["raw/a.md", "raw/b.md", "raw/c.md"],
            confidence=0.75,
            source_count=3,
            single_source=True,
            source_quality="high",
        )
        post = fm_lib.Post("Body.", **meta)
        text = "\n" + fm_lib.dumps(post) + "\n"

        # Newline-anchored asserts pin both the key/value and the YAML scalar
        # form: bool not boxed as a string, int not quoted, enum value bare.
        assert "\nsource_count: 3\n" in text
        assert "\nsingle_source: true\n" in text
        assert "\nsource_quality: high\n" in text


# ── compile.py: _source_quality_summary ────────────────────────────────────


class TestSourceQualitySummary:
    def test_best_of_high_from_mixed(self, db):
        db.upsert_raw(
            RawNoteRecord(path="raw/low.md", content_hash="h1", quality="low", status="ingested")
        )
        db.upsert_raw(
            RawNoteRecord(path="raw/high.md", content_hash="h2", quality="high", status="ingested")
        )
        db.upsert_raw(
            RawNoteRecord(
                path="raw/medium.md",
                content_hash="h3",
                quality="medium",
                status="ingested",
            )
        )

        best = _source_quality_summary(["raw/low.md", "raw/high.md", "raw/medium.md"], db)
        assert best == "high"

    def test_medium_over_low(self, db):
        db.upsert_raw(
            RawNoteRecord(path="raw/low.md", content_hash="h1", quality="low", status="ingested")
        )
        db.upsert_raw(
            RawNoteRecord(
                path="raw/medium.md",
                content_hash="h2",
                quality="medium",
                status="ingested",
            )
        )

        best = _source_quality_summary(["raw/low.md", "raw/medium.md"], db)
        assert best == "medium"

    def test_low_when_all_low(self, db):
        db.upsert_raw(
            RawNoteRecord(path="raw/a.md", content_hash="h1", quality="low", status="ingested")
        )
        db.upsert_raw(
            RawNoteRecord(path="raw/b.md", content_hash="h2", quality="low", status="ingested")
        )

        best = _source_quality_summary(["raw/a.md", "raw/b.md"], db)
        assert best == "low"


# ── compile.py: concept draft frontmatter ──────────────────────────────────


ARTIFACT_JSON = json.dumps(
    {
        "title": "Test Concept",
        "content": "## Overview\n\nTest content.",
        "tags": ["test"],
    }
)


def test_compile_draft_single_source(vault, config, db):
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("---\ntitle: Note\n---\n\nContent.")
    db.upsert_raw(
        RawNoteRecord(path="raw/note.md", content_hash="h1", quality="high", status="ingested")
    )
    db.upsert_concepts("raw/note.md", ["Test Concept"])

    client = _mock_client(ARTIFACT_JSON)
    drafts, _, _ = compile_concepts(config=config, router=client, db=db)
    assert len(drafts) == 1

    meta, _ = parse_note(drafts[0])
    assert meta["source_count"] == 1
    assert meta["single_source"] is True
    assert meta["source_quality"] == "high"


def test_compile_draft_multi_source(vault, config, db):
    for i in range(3):
        p = vault / "raw" / f"note{i}.md"
        p.write_text(f"---\ntitle: Note {i}\n---\n\nContent {i}.")
        db.upsert_raw(
            RawNoteRecord(
                path=f"raw/note{i}.md",
                content_hash=f"h{i}",
                quality="low",
                status="ingested",
            )
        )
    db.upsert_concepts("raw/note0.md", ["Test Concept"])
    db.upsert_concepts("raw/note1.md", ["Test Concept"])
    db.upsert_concepts("raw/note2.md", ["Test Concept"])

    client = _mock_client(ARTIFACT_JSON)
    drafts, _, _ = compile_concepts(config=config, router=client, db=db)
    assert len(drafts) == 1

    meta, _ = parse_note(drafts[0])
    assert meta["source_count"] == 3
    assert meta["single_source"] is False


def test_compile_draft_source_quality_best_of(vault, config, db):
    for name, quality in [("low", "low"), ("high", "high"), ("medium", "medium")]:
        p = vault / "raw" / f"{name}.md"
        p.write_text(f"---\ntitle: {name}\n---\n\nContent.")
        db.upsert_raw(
            RawNoteRecord(
                path=f"raw/{name}.md",
                content_hash=f"h{name}",
                quality=quality,
                status="ingested",
            )
        )
    for p in ["raw/low.md", "raw/high.md", "raw/medium.md"]:
        db.upsert_concepts(p, ["Test Concept"])

    client = _mock_client(ARTIFACT_JSON)
    drafts, _, _ = compile_concepts(config=config, router=client, db=db)
    assert len(drafts) == 1

    meta, _ = parse_note(drafts[0])
    assert meta["source_quality"] == "high"


def test_compile_draft_zero_sources(vault, config, db):
    db.add_stub("Stub Without Sources")

    stub_json = json.dumps(
        {
            "title": "Stub Without Sources",
            "content": "## Overview\n\nBrief stub content.",
            "tags": ["stub"],
        }
    )
    client = _mock_client(stub_json)
    drafts, _, _ = compile_concepts(config=config, router=client, db=db)
    assert len(drafts) == 1

    meta, _ = parse_note(drafts[0])
    assert meta["source_count"] == 0
    assert meta["single_source"] is False
    assert "source_quality" not in meta


# ── compile.py: approve preserves fields ───────────────────────────────────


def test_approve_preserves_new_fields(vault, config, db):
    draft_path = config.drafts_dir / "Article.md"
    meta = build_wiki_frontmatter(
        title="Article",
        tags=["tag1"],
        sources=["raw/a.md"],
        confidence=0.6,
        is_draft=True,
        source_count=2,
        single_source=False,
        source_quality="medium",
    )
    write_note(draft_path, meta, "Body content.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Article",
            sources=["raw/a.md"],
            content_hash="h",
            status="draft",
        )
    )

    published = approve_drafts(config, db, [draft_path])
    assert len(published) == 1

    meta_pub, _ = parse_note(published[0])
    assert meta_pub["status"] == "published"
    assert meta_pub["source_count"] == 2
    assert meta_pub["single_source"] is False
    assert meta_pub["source_quality"] == "medium"


# ── query.py: synthesis frontmatter ────────────────────────────────────────


def _make_index(config, entries: list[str]) -> None:
    lines = ["# Wiki Index", "", "## Articles", ""]
    for entry in entries:
        lines.append(f"- [[{entry}]]")
    (config.wiki_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _make_concept(config, title: str) -> None:
    write_note(
        config.wiki_dir / f"{title}.md",
        {"title": title, "tags": [], "status": "published"},
        f"Content about {title}.",
    )


def test_synthesis_has_source_count_and_single_source(vault, config, db):
    _make_concept(config, "Alpha")
    _make_concept(config, "Beta")
    _make_concept(config, "Gamma")
    _make_index(config, ["Alpha", "Beta", "Gamma"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Alpha.md",
            title="Alpha",
            sources=[],
            content_hash="h1",
            status="published",
        )
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Beta.md",
            title="Beta",
            sources=[],
            content_hash="h2",
            status="published",
        )
    )

    _save_synthesis(
        config,
        db,
        question="Test question?",
        answer="Synthesis answer.",
        source_pages=["Alpha", "Beta"],
        title="Test Synthesis",
        duplicate_strategy="save_with_suffix",
    )

    synthesis_dir = config.synthesis_dir
    found = list(synthesis_dir.glob("*.md"))
    assert len(found) >= 1

    meta, _ = parse_note(found[0])
    assert meta.get("source_count") == 2
    assert meta.get("single_source") is False
    assert "source_quality" not in meta
    assert "confidence" not in meta


def test_synthesis_update_in_place_preserves_fields(vault, config, db):
    from synto.pipeline.query import _body_hash, _question_hash, _render_synthesis_body

    _make_concept(config, "Alpha")
    _make_index(config, ["Alpha"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Alpha.md",
            title="Alpha",
            sources=[],
            content_hash="h1",
            status="published",
        )
    )

    qhash = _question_hash("Same question")

    body = "First synthesis."
    rendered = _render_synthesis_body(body, ["Alpha"])
    content_hash = _body_hash(rendered)
    file_text = _build_synthesis_file_text(
        rendered,
        title="Synth",
        question="Same question",
        source_pages=["Alpha"],
        source_page_hashes=[{"path": "wiki/Alpha.md", "hash": "aaa"}],
        question_hash=qhash,
        content_hash=content_hash,
        created="2025-01-01",
    )
    path = config.synthesis_dir / "Synth.md"
    path.write_text(file_text, encoding="utf-8")
    db.upsert_article(
        WikiArticleRecord(
            path=str(path.relative_to(vault)),
            title="Synth",
            sources=[],
            content_hash=content_hash,
            status="published",
            kind="synthesis",
            question_hash=qhash,
            synthesis_sources=["wiki/Alpha.md"],
            synthesis_source_hashes=[["wiki/Alpha.md", "aaa"]],
        )
    )

    updated_body = "Updated synthesis."
    _save_synthesis(
        config,
        db,
        question="Same question",
        answer=updated_body,
        source_pages=["Alpha"],
        title="Synth",
        duplicate_strategy="update_in_place",
    )

    meta, body_text = parse_note(path)
    assert meta.get("source_count") == 1
    assert meta.get("single_source") is True
    assert "source_quality" not in meta
    assert "confidence" not in meta


def test_synthesis_update_in_place_regenerates_when_hash_blank(vault, config, db):
    """A blank DB content_hash must be regenerable, not misread as a manual edit.

    Regression for review Issue 3: update_in_place compared "" against the real on-disk body
    hash and always raised SynthesisManualEditConflictError, refusing regeneration — while
    compile and lint treat a blank hash as a not-yet-hashed placeholder (#83). The manual-edit
    guard must be disabled by a blank hash consistently across all three stages.
    """
    from synto.pipeline.query import _body_hash, _question_hash, _render_synthesis_body

    _make_concept(config, "Alpha")
    _make_index(config, ["Alpha"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Alpha.md",
            title="Alpha",
            sources=[],
            content_hash="h1",
            status="published",
        )
    )

    qhash = _question_hash("Same question")
    rendered = _render_synthesis_body("First synthesis.", ["Alpha"])
    file_text = _build_synthesis_file_text(
        rendered,
        title="Synth",
        question="Same question",
        source_pages=["Alpha"],
        source_page_hashes=[{"path": "wiki/Alpha.md", "hash": "aaa"}],
        question_hash=qhash,
        content_hash="",
        created="2025-01-01",
    )
    path = config.synthesis_dir / "Synth.md"
    path.write_text(file_text, encoding="utf-8")
    # DB row carries the blank legacy placeholder while the on-disk body hashes to a real digest
    # — the exact mismatch the old guard misread as a manual edit and refused to overwrite.
    db.upsert_article(
        WikiArticleRecord(
            path=str(path.relative_to(vault)),
            title="Synth",
            sources=[],
            content_hash="",
            status="published",
            kind="synthesis",
            question_hash=qhash,
            synthesis_sources=["wiki/Alpha.md"],
            synthesis_source_hashes=[["wiki/Alpha.md", "aaa"]],
        )
    )

    result = _save_synthesis(
        config,
        db,
        question="Same question",
        answer="Updated synthesis.",
        source_pages=["Alpha"],
        title="Synth",
        duplicate_strategy="update_in_place",
    )

    assert result.resolution == "updated_in_place"
    _, body_text = parse_note(path)
    assert "Updated synthesis." in body_text
    art = db.get_article(str(path.relative_to(vault)))
    assert art is not None
    assert art.content_hash == _body_hash(_render_synthesis_body("Updated synthesis.", ["Alpha"]))


# ── MCP / reader path ─────────────────────────────────────────────────────


def test_read_article_includes_quality_fields(vault, config, db):
    meta = build_wiki_frontmatter(
        title="Test Article",
        tags=["tag1"],
        sources=["raw/a.md"],
        confidence=0.75,
        source_count=3,
        single_source=False,
        source_quality="high",
    )
    path = config.wiki_dir / "Test Article.md"
    write_note(path, meta, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(path.relative_to(vault)),
            title="Test Article",
            sources=["raw/a.md"],
            content_hash="h",
            status="published",
        )
    )

    reader = VaultReader(vault)
    article = reader.read_article("Test Article")
    assert article.frontmatter.get("source_count") == 3
    assert article.frontmatter.get("single_source") is False
    assert article.frontmatter.get("source_quality") == "high"


def test_read_article_legacy_returns_none_for_missing_fields(vault, config, db):
    meta = {
        "title": "Legacy",
        "tags": ["old"],
        "status": "published",
        "confidence": 0.5,
    }
    path = config.wiki_dir / "Legacy.md"
    write_note(path, meta, "Old body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(path.relative_to(vault)),
            title="Legacy",
            sources=[],
            content_hash="h",
            status="published",
        )
    )

    reader = VaultReader(vault)
    article = reader.read_article("Legacy")
    assert article.frontmatter.get("source_count") is None
    assert article.frontmatter.get("single_source") is None
    assert article.frontmatter.get("source_quality") is None


# ── lint ───────────────────────────────────────────────────────────────────


def test_lint_does_not_flag_missing_quality_fields(vault, config, db):
    from synto.pipeline.lint import run_lint

    meta = {"title": "Legacy", "tags": ["test"], "status": "published"}
    path = config.wiki_dir / "Legacy.md"
    write_note(path, meta, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(path.relative_to(vault)),
            title="Legacy",
            sources=[],
            content_hash="h",
            status="published",
        )
    )

    result = run_lint(config, db)
    quality_issues = [iss for iss in result.issues if iss.issue_type in ("missing_frontmatter",)]
    for iss in quality_issues:
        assert "source_count" not in iss.description
        assert "single_source" not in iss.description
        assert "source_quality" not in iss.description


# ── VaultReader: ArticleRef carries quality fields ────────────────────────


def test_vault_reader_list_articles_carries_quality_fields(vault, config, db):
    """list_articles populates the three new fields from frontmatter directly,
    so MCP doesn't need a per-ref read_article call."""
    full_meta = build_wiki_frontmatter(
        title="Full",
        tags=["t"],
        sources=["raw/a.md"],
        confidence=0.7,
        is_draft=False,
        source_count=3,
        single_source=False,
        source_quality="high",
    )
    write_note(config.wiki_dir / "Full.md", full_meta, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Full.md",
            title="Full",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )

    # Legacy article: only the pre-feature fields.
    legacy_meta = {"title": "Legacy", "tags": ["t"], "status": "published", "confidence": 0.5}
    write_note(config.wiki_dir / "Legacy.md", legacy_meta, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Legacy.md",
            title="Legacy",
            sources=[],
            content_hash="h2",
            status="published",
        )
    )

    reader = VaultReader(vault)
    refs = {ref.name: ref for ref in reader.list_articles()}

    assert refs["Full"].source_count == 3
    assert refs["Full"].single_source is False
    assert refs["Full"].source_quality == "high"

    assert refs["Legacy"].source_count is None
    assert refs["Legacy"].single_source is None
    assert refs["Legacy"].source_quality is None


def test_vault_reader_rejects_invalid_source_quality(vault, config, db):
    """source_quality is constrained to the {high, medium, low} enum; junk
    values in frontmatter degrade to None rather than poisoning the ref."""
    write_note(
        config.wiki_dir / "Junk.md",
        {
            "title": "Junk",
            "tags": ["t"],
            "status": "published",
            "confidence": 0.5,
            "source_quality": "premium",  # not in the enum
            "source_count": "three",  # non-numeric
            "single_source": "yes",  # not a bool
        },
        "Body.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Junk.md",
            title="Junk",
            sources=[],
            content_hash="h",
            status="published",
        )
    )

    reader = VaultReader(vault)
    ref = reader.list_articles()[0]
    assert ref.source_count is None
    assert ref.single_source is None
    assert ref.source_quality is None


# ── compile.py: single_source via source_documents ────────────────────────


def _seed_source_document(db, source_id: str, origin_uri: str) -> None:
    """Seed a source_documents row. Tests use this together with a raw note
    at `raw/<source_id>.md` to simulate a `synto add` import — the live
    code path that compile.py reads when computing single_source identity.
    """
    db._conn.execute(
        """INSERT OR REPLACE INTO source_documents
           (id, source_type, origin_uri, raw_hash, redistribution)
           VALUES (?, 'notes', ?, ?, 'unknown')""",
        (source_id, origin_uri, hashlib.sha256(source_id.encode()).hexdigest()),
    )
    db._conn.commit()


def test_compile_single_source_from_one_document(vault, config, db):
    """Two raw notes from the same imported document (same source_documents
    origin_uri) → single_source: true, even though source_count == 2. This
    is the dobryakov corroboration check: chunks of one book are one source.
    """
    for i in range(2):
        source_id = f"chunk{i}"
        p = vault / "raw" / f"{source_id}.md"
        p.write_text(f"---\ntitle: Chunk {i}\n---\n\nContent {i}.")
        db.upsert_raw(
            RawNoteRecord(
                path=f"raw/{source_id}.md",
                content_hash=f"h{i}",
                quality="high",
                status="ingested",
            )
        )
        # Same origin_uri across both chunks — they're from the same document.
        _seed_source_document(db, source_id, "file:///books/tanenbaum.pdf")

    for p in ["raw/chunk0.md", "raw/chunk1.md"]:
        db.upsert_concepts(p, ["Test Concept"])

    drafts, _, _ = compile_concepts(config=config, router=_mock_client(ARTIFACT_JSON), db=db)
    meta, _ = parse_note(drafts[0])
    assert meta["source_count"] == 2
    assert meta["single_source"] is True


def test_compile_multi_source_from_two_documents(vault, config, db):
    """Two raw notes with different source_documents origin_uri →
    single_source: false."""
    for i, origin in enumerate(["file:///books/tanenbaum.pdf", "file:///books/kleppmann.pdf"]):
        source_id = f"chunk{i}"
        p = vault / "raw" / f"{source_id}.md"
        p.write_text(f"---\ntitle: Chunk {i}\n---\n\nContent {i}.")
        db.upsert_raw(
            RawNoteRecord(
                path=f"raw/{source_id}.md",
                content_hash=f"h{i}",
                quality="high",
                status="ingested",
            )
        )
        _seed_source_document(db, source_id, origin)

    for p in ["raw/chunk0.md", "raw/chunk1.md"]:
        db.upsert_concepts(p, ["Test Concept"])

    drafts, _, _ = compile_concepts(config=config, router=_mock_client(ARTIFACT_JSON), db=db)
    meta, _ = parse_note(drafts[0])
    assert meta["source_count"] == 2
    assert meta["single_source"] is False


def test_compile_falls_back_to_path_uniqueness_when_origin_uri_missing(vault, config, db):
    """If any source lacks a source_documents row (free-form user notes,
    current production state for non-`synto add` ingest), fall back to
    path-uniqueness. Two distinct paths with no doc-identity → single_source:
    false.
    """
    for i in range(2):
        p = vault / "raw" / f"note{i}.md"
        p.write_text(f"---\ntitle: Note {i}\n---\n\nContent {i}.")
        db.upsert_raw(
            RawNoteRecord(
                path=f"raw/note{i}.md",
                content_hash=f"h{i}",
                quality="medium",
                status="ingested",
            )
        )
        # No source_documents row — manually-dropped raw note.

    for p in ["raw/note0.md", "raw/note1.md"]:
        db.upsert_concepts(p, ["Test Concept"])

    drafts, _, _ = compile_concepts(config=config, router=_mock_client(ARTIFACT_JSON), db=db)
    meta, _ = parse_note(drafts[0])
    assert meta["source_count"] == 2
    assert meta["single_source"] is False


def test_compile_fallback_when_some_sources_lack_source_documents_row(vault, config, db):
    """If some sources have a source_documents row and others don't, we
    cannot trust document identity (mixed signal), so fall back to path
    uniqueness."""
    for i in range(2):
        source_id = f"note{i}"
        p = vault / "raw" / f"{source_id}.md"
        p.write_text(f"---\ntitle: Note {i}\n---\n\nContent {i}.")
        db.upsert_raw(
            RawNoteRecord(
                path=f"raw/{source_id}.md",
                content_hash=f"h{i}",
                quality="medium",
                status="ingested",
            )
        )
    # Only one source has a source_documents row — mixed signal.
    _seed_source_document(db, "note0", "file:///books/tanenbaum.pdf")

    for p in ["raw/note0.md", "raw/note1.md"]:
        db.upsert_concepts(p, ["Test Concept"])

    drafts, _, _ = compile_concepts(config=config, router=_mock_client(ARTIFACT_JSON), db=db)
    meta, _ = parse_note(drafts[0])
    # Mixed source_documents coverage → path-uniqueness fallback says false.
    assert meta["single_source"] is False
