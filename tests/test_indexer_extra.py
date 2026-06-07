"""Additional tests for indexer.py uncovered paths."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import frontmatter as fm_lib

from synto.config import Config
from synto.indexer import append_log, generate_index
from synto.models import WikiArticleRecord
from synto.state import StateDB
from synto.vault import atomic_write, write_note


def _setup_vault(tmp_path: Path) -> tuple[Path, Config, StateDB]:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / "wiki" / "synthesis").mkdir()
    (tmp_path / ".synto").mkdir()
    config = Config(vault=tmp_path)
    db = StateDB(config.state_db_path)
    return tmp_path, config, db


def test_generate_index_empty_wiki(tmp_path):
    """Index generated with no articles — just the header."""
    _, config, db = _setup_vault(tmp_path)
    index_path = generate_index(config, db)
    text = index_path.read_text()
    assert "# Wiki Index" in text
    assert "## Concepts" not in text
    assert "## Sources" not in text
    assert "## Synthesis" not in text


def test_generate_index_with_concept_articles(tmp_path):
    """Index includes concept articles section."""
    _, config, db = _setup_vault(tmp_path)
    art_path = config.wiki_dir / "Machine Learning.md"
    write_note(
        art_path,
        {"title": "Machine Learning", "tags": ["ml"], "status": "published"},
        "Body.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Machine Learning.md",
            title="Machine Learning",
            sources=[],
            content_hash="h1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="published",
        )
    )
    text = generate_index(config, db).read_text()
    assert "## Concepts" in text
    assert "[[Machine Learning]]" in text


def test_generate_index_with_source_pages_from_filesystem(tmp_path):
    """Index picks up source pages from wiki/sources/ filesystem."""
    _, config, db = _setup_vault(tmp_path)
    src_path = config.sources_dir / "quantum-notes.md"
    post = fm_lib.Post(
        "Summary of quantum computing.",
        title="Quantum Computing Notes",
        tags=["source"],
        quality="high",
        source_file="raw/quantum.md",
    )
    atomic_write(src_path, fm_lib.dumps(post))

    text = generate_index(config, db).read_text()
    assert "## Sources" in text
    assert "Quantum Computing Notes" in text


def test_generate_index_normalizes_legacy_windows_source_file_hint(tmp_path):
    """Legacy source_file hints are rendered with forward slashes."""
    _, config, db = _setup_vault(tmp_path)
    src_path = config.sources_dir / "legacy-source.md"
    post = fm_lib.Post(
        "Legacy summary.",
        title="Legacy Source",
        tags=["source"],
        source_file=r"raw\quantum.md",
    )
    atomic_write(src_path, fm_lib.dumps(post))

    text = generate_index(config, db).read_text()

    assert "raw/quantum.md" in text
    assert "raw\\quantum.md" not in text


def test_generate_index_source_page_with_parse_error(tmp_path):
    """Source page that can't be parsed — uses stem as title."""
    _, config, db = _setup_vault(tmp_path)
    bad_path = config.sources_dir / "broken.md"
    bad_path.write_bytes(b"\x80\x81\x82")

    text = generate_index(config, db).read_text()
    assert "## Sources" in text
    assert "broken" in text


def test_generate_index_concept_articles_from_filesystem_fallback(tmp_path):
    """Articles in wiki/ root not in DB are picked up as fallback."""
    _, config, db = _setup_vault(tmp_path)
    art_path = config.wiki_dir / "Orphan Article.md"
    post = fm_lib.Post(
        "This article exists on disk but not in DB.",
        title="Orphan Article",
        tags=[],
        status="published",
    )
    atomic_write(art_path, fm_lib.dumps(post))

    text = generate_index(config, db).read_text()
    assert "Orphan Article" in text


def test_generate_index_skips_index_and_log_in_fallback(tmp_path):
    """Fallback scan skips index.md and log.md."""
    _, config, db = _setup_vault(tmp_path)
    # Create index.md and log.md in wiki/ root
    (config.wiki_dir / "index.md").write_text("# Index\n")
    (config.wiki_dir / "log.md").write_text("# Log\n")

    text = generate_index(config, db).read_text()
    # Should not list index.md or log.md as concepts
    assert "[[Index]]" not in text or text.count("[[Index]]") <= 1


def test_generate_index_concept_fallback_parse_error(tmp_path):
    """Fallback scan handles parse errors gracefully."""
    _, config, db = _setup_vault(tmp_path)
    bad_path = config.wiki_dir / "Broken Article.md"
    bad_path.write_bytes(b"\x80\x81\x82")

    # Should not crash
    text = generate_index(config, db).read_text()
    assert "# Wiki Index" in text


def test_generate_index_sources_dir_not_exists(tmp_path):
    """Index generation works when sources_dir doesn't exist."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".synto").mkdir()
    config = Config(vault=tmp_path)
    db = StateDB(config.state_db_path)

    text = generate_index(config, db).read_text()
    assert "# Wiki Index" in text
    assert "## Sources" not in text


# ── append_log ────────────────────────────────────────────────────────────────


def test_append_log_creates_log_file(tmp_path):
    """append_log creates wiki/log.md if it doesn't exist."""
    _, config, db = _setup_vault(tmp_path)
    log_path = config.wiki_dir / "log.md"
    assert not log_path.exists()

    append_log(config, "Test operation")
    assert log_path.exists()
    text = log_path.read_text()
    assert "Test operation" in text


def test_append_log_appends_to_existing_log(tmp_path):
    """append_log appends to existing log.md."""
    _, config, db = _setup_vault(tmp_path)
    log_path = config.wiki_dir / "log.md"
    log_path.write_text("# Operation Log\n\n- First entry\n")

    append_log(config, "Second operation")
    text = log_path.read_text()
    assert "First entry" in text
    assert "Second operation" in text
