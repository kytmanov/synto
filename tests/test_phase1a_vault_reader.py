from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

from synto.models import WikiArticleRecord
from synto.readers import ArticleFilter, ArticleNotFound, VaultReader, _extract_first_paragraph
from synto.vault import write_note


def _build_v8_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_version (
            id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL
        );
        CREATE TABLE raw_notes (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'new',
            summary TEXT,
            quality TEXT,
            language TEXT,
            ingested_at TEXT,
            compiled_at TEXT,
            error TEXT,
            source_type TEXT NOT NULL DEFAULT 'notes',
            origin_uri TEXT,
            imported_at TEXT,
            normalized_hash TEXT,
            extractor_version TEXT,
            prompt_version TEXT
        );
        CREATE TABLE concepts (
            name TEXT NOT NULL, source_path TEXT NOT NULL,
            PRIMARY KEY (name, source_path)
        );
        CREATE TABLE wiki_articles (
            path TEXT PRIMARY KEY, title TEXT NOT NULL, sources TEXT NOT NULL,
            content_hash TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, is_draft INTEGER NOT NULL DEFAULT 1,
            approved_at TEXT, approval_notes TEXT,
            kind TEXT NOT NULL DEFAULT 'concept', question_hash TEXT,
            synthesis_sources TEXT, synthesis_source_hashes TEXT,
            article_id TEXT
        );
        INSERT INTO schema_version (id, version) VALUES (1, 8);
        INSERT INTO wiki_articles
            (path, title, sources, content_hash, created_at, updated_at, is_draft, kind, article_id)
            VALUES (
                'wiki/Test.md', 'Test Concept', '["raw/old.md"]', 'wh1',
                '2024-01-01T00:00:00', '2024-01-01T00:00:00', 0, 'concept',
                '01TESTULID0000000000000001'
            );
        """
    )
    conn.commit()
    conn.close()


def test_vault_reader_does_not_create_db_when_missing(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()

    reader = VaultReader(tmp_path)

    assert reader.list_articles() == []
    assert not (tmp_path / ".synto" / "state.db").exists()


def test_vault_reader_lists_published_concept_articles_only(vault, config, db) -> None:
    concept_path = config.wiki_dir / "Topic.md"
    synthesis_path = config.synthesis_dir / "Synth.md"
    queries_path = config.queries_dir / "Q.md"
    sources_path = config.sources_dir / "Source.md"
    draft_path = config.drafts_dir / "Draft.md"

    synthesis_path.parent.mkdir(parents=True, exist_ok=True)
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    sources_path.parent.mkdir(parents=True, exist_ok=True)

    write_note(concept_path, {"title": "Topic"}, "Concept body")
    write_note(synthesis_path, {"title": "Synth"}, "Synth body")
    write_note(queries_path, {"title": "Q"}, "Query body")
    write_note(sources_path, {"title": "Source"}, "Source body")
    write_note(draft_path, {"title": "Draft"}, "Draft body")

    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            is_draft=False,
        )
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/synthesis/Synth.md",
            title="Synth",
            sources=[],
            content_hash="h2",
            is_draft=False,
            kind="synthesis",
            question_hash="qh1",
        )
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/.drafts/Draft.md",
            title="Draft",
            sources=["raw/a.md"],
            content_hash="h3",
            is_draft=True,
        )
    )

    reader = VaultReader(vault)
    refs = reader.list_articles()

    assert [ref.name for ref in refs] == ["Topic"]


def test_vault_reader_reads_article_by_title_path_stem_and_article_id(vault, config, db) -> None:
    path = config.wiki_dir / "Nested-Topic.md"
    write_note(path, {"title": "Nested Topic", "tags": ["topic"]}, "Body text")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Nested-Topic.md",
            title="Nested Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            is_draft=False,
        )
    )
    article = db.get_article("wiki/Nested-Topic.md")
    assert article is not None and article.article_id is not None

    reader = VaultReader(vault)
    assert reader.read_article("Nested Topic").body == "Body text"
    assert reader.read_article("wiki/Nested-Topic.md").name == "Nested Topic"
    assert reader.read_article("Nested-Topic").name == "Nested Topic"
    assert reader.read_article(article.article_id).name == "Nested Topic"


def test_vault_reader_filters_by_tag(vault, config, db) -> None:
    write_note(config.wiki_dir / "Topic.md", {"title": "Topic", "tags": ["systems"]}, "Body")
    write_note(config.wiki_dir / "Other.md", {"title": "Other", "tags": ["math"]}, "Body")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            is_draft=False,
        )
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Other.md",
            title="Other",
            sources=["raw/a.md"],
            content_hash="h2",
            is_draft=False,
        )
    )

    reader = VaultReader(vault)

    refs = reader.list_articles(filter=ArticleFilter(tag="systems"))
    assert [ref.name for ref in refs] == ["Topic"]


def test_vault_reader_normalizes_confidence_from_frontmatter(vault, config, db) -> None:
    write_note(config.wiki_dir / "Low.md", {"title": "Low", "confidence": 0.3}, "Body")
    write_note(config.wiki_dir / "Medium.md", {"title": "Medium", "confidence": 0.6}, "Body")
    write_note(config.wiki_dir / "High.md", {"title": "High", "confidence": 0.9}, "Body")
    write_note(config.wiki_dir / "String.md", {"title": "String", "confidence": "low"}, "Body")
    write_note(config.wiki_dir / "Default.md", {"title": "Default"}, "Body")
    for i, title in enumerate(["Low", "Medium", "High", "String", "Default"], start=1):
        db.upsert_article(
            WikiArticleRecord(
                path=f"wiki/{title}.md",
                title=title,
                sources=[f"raw/{i}.md"],
                content_hash=f"h{i}",
                is_draft=False,
            )
        )

    reader = VaultReader(vault)
    refs = {ref.name: ref for ref in reader.list_articles()}

    assert refs["Low"].confidence == "low"
    assert refs["Medium"].confidence == "medium"
    assert refs["High"].confidence == "high"
    assert refs["String"].confidence == "low"
    assert refs["Default"].confidence == "high"


def test_vault_reader_find_concept_resolution_priority(vault, config, db) -> None:
    write_note(config.wiki_dir / "Canonical.md", {"title": "Canonical"}, "Body")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Canonical.md",
            title="Canonical",
            sources=["raw/a.md"],
            content_hash="h1",
            is_draft=False,
        )
    )
    db.upsert_concepts("raw/a.md", ["Canonical", "Substring Target"])
    db.upsert_aliases("Canonical", ["AliasName"])

    reader = VaultReader(vault)

    exact = reader.find_concept("Canonical")
    alias = reader.find_concept("AliasName")
    substring = reader.find_concept("Target")

    assert exact is not None and exact.name == "Canonical"
    assert alias is not None and alias.name == "Canonical"
    assert substring is not None and substring.name == "Substring Target"


def test_vault_reader_article_cache_hits_db_once(vault, config, db) -> None:
    write_note(config.wiki_dir / "Topic.md", {"title": "Topic"}, "Body")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            is_draft=False,
        )
    )

    reader = VaultReader(vault)
    state = reader._state()
    assert state is not None
    with patch.object(state, "list_articles", wraps=state.list_articles) as mocked:
        for _ in range(10):
            refs = reader.list_articles()
            assert [ref.name for ref in refs] == ["Topic"]
        for _ in range(10):
            assert reader.read_article("Topic").name == "Topic"

    assert mocked.call_count == 1


def test_vault_reader_duplicate_title_uses_first_match(vault, config, db) -> None:
    write_note(config.wiki_dir / "First.md", {"title": "Shared"}, "First body")
    write_note(config.wiki_dir / "Second.md", {"title": "Shared"}, "Second body")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/First.md",
            title="Shared",
            sources=["raw/a.md"],
            content_hash="h1",
            is_draft=False,
        )
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Second.md",
            title="Shared",
            sources=["raw/b.md"],
            content_hash="h2",
            is_draft=False,
        )
    )

    reader = VaultReader(vault)

    assert reader.read_article("Shared").path == "wiki/First.md"


def test_vault_reader_capabilities_expand_with_segments(vault, config, db) -> None:
    reader = VaultReader(vault)
    assert reader.capabilities == frozenset({"articles", "concepts"})

    conn = sqlite3.connect(config.state_db_path)
    conn.execute(
        "INSERT INTO source_documents (id, title, source_type) VALUES (?, ?, ?)",
        ("src-1", "Source", "unknown_text"),
    )
    conn.execute(
        (
            "INSERT INTO source_segments "
            "(id, identity, ordinal, source_id, structural_locator, content_hash, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        ),
        ("seg-1", "src-1:loc", 0, "src-1", "loc", "hash", "text"),
    )
    conn.commit()
    conn.close()

    reader2 = VaultReader(vault)
    assert reader2.capabilities == frozenset({"articles", "concepts", "segments", "lifecycle"})
    assert len(reader2.list_segments()) == 1


def test_vault_reader_missing_article_raises(vault) -> None:
    reader = VaultReader(vault)
    try:
        reader.read_article("missing")
    except ArticleNotFound:
        pass
    else:
        raise AssertionError("expected ArticleNotFound")


def test_vault_reader_handles_readonly_v8_db(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / ".synto").mkdir()
    write_note(tmp_path / "wiki" / "Test.md", {"title": "Test Concept"}, "Body")
    db_path = tmp_path / ".synto" / "state.db"
    _build_v8_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE wiki_articles SET article_id = ? WHERE path = ?",
        ("01TESTULID0000000000000001", "wiki/Test.md"),
    )
    conn.commit()
    conn.close()

    reader = VaultReader(tmp_path)

    assert reader.capabilities == frozenset({"articles", "concepts"})
    assert [ref.name for ref in reader.list_articles()] == ["Test Concept"]


def test_vault_reader_ignores_stale_db_rows_with_missing_files(vault, config, db) -> None:
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Missing.md",
            title="Missing",
            sources=["raw/a.md"],
            content_hash="h1",
            is_draft=False,
        )
    )
    write_note(config.wiki_dir / "Present.md", {"title": "Present"}, "Body")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Present.md",
            title="Present",
            sources=["raw/a.md"],
            content_hash="h2",
            is_draft=False,
        )
    )

    reader = VaultReader(vault)

    assert [ref.name for ref in reader.list_articles()] == ["Present"]


# ── _extract_first_paragraph ──────────────────────────────────────────────────


def test_extract_first_paragraph_basic():
    body = "## Heading\n\nThis is the first paragraph. It is informative."
    assert _extract_first_paragraph(body) == "This is the first paragraph. It is informative."


def test_extract_first_paragraph_strips_wikilinks():
    body = "[[Concept|Display Name]] and [[Plain]] are discussed."
    assert _extract_first_paragraph(body) == "Display Name and Plain are discussed."


def test_extract_first_paragraph_strips_citations():
    body = "Kanban is a method [S1](#Sources) for managing work [S2](#Sources)."
    assert _extract_first_paragraph(body) == "Kanban is a method for managing work."


def test_extract_first_paragraph_strips_bold_italic():
    body = "This is **bold** and *italic* text."
    assert _extract_first_paragraph(body) == "This is bold and italic text."


def test_extract_first_paragraph_truncates_at_200():
    long_para = "word " * 60  # 300 chars, no sentence boundary
    result = _extract_first_paragraph(long_para.strip())
    assert result is not None
    assert len(result) <= 200
    assert result.endswith("…")


def test_extract_first_paragraph_sentence_boundary():
    # First sentence ends at ~67 chars; total line is >200 chars so truncation triggers.
    body = "This is the first sentence, which contains enough text to be clear. " + "x" * 200
    result = _extract_first_paragraph(body)
    assert result == "This is the first sentence, which contains enough text to be clear."


def test_extract_first_paragraph_skips_headings():
    body = "## Section\n\n### Subsection\n\nActual content here."
    assert _extract_first_paragraph(body) == "Actual content here."


def test_extract_first_paragraph_empty_body():
    assert _extract_first_paragraph("") is None
    assert _extract_first_paragraph("## Only Heading") is None


def test_vault_reader_list_articles_populates_summary(vault, config, db) -> None:
    write_note(
        config.wiki_dir / "Concept.md",
        {"title": "Concept", "confidence": 0.6},
        "## Heading\n\nThis is the article summary sentence.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Concept.md",
            title="Concept",
            sources=["raw/a.md"],
            content_hash="h1",
            is_draft=False,
        )
    )
    reader = VaultReader(vault)
    refs = reader.list_articles()
    assert len(refs) == 1
    assert refs[0].summary == "This is the article summary sentence."
    assert refs[0].confidence_score == 0.6
    assert refs[0].confidence == "medium"
