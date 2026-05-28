"""Tests for Feature 42 — verbatim source MCP tools.

Covers: FTS5 migration (Stage 1), four MCP handler closures (Stages 2-5),
and the license-driven privacy gate (Stage 6).
All tests use in-memory or tmp_path SQLite; no Ollama required.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from synto.config import Config, McpConfig
from synto.readers import VaultReader
from synto.serve import build_tool_handlers
from synto.state import StateDB


# ── Helpers ──────────────────────────────────────────────────────────────────


def _insert_source(db: StateDB, source_id: str, license: str | None = None) -> None:
    db._conn.execute(
        """INSERT OR REPLACE INTO source_documents
           (id, source_type, origin_uri, title, imported_at, redistribution)
           VALUES (?, 'pdf', ?, ?, '2024-01-01T00:00:00', 'unknown')""",
        (source_id, f"/raw/{source_id}.pdf", source_id),
    )
    if license is not None:
        db._conn.execute(
            "UPDATE source_documents SET license = ? WHERE id = ?",
            (license, source_id),
        )
    db._conn.commit()


def _insert_segment(
    db: StateDB,
    seg_id: str,
    source_id: str,
    text: str,
    ordinal: int = 0,
    content_hash: str = "abc123",
) -> None:
    db._conn.execute(
        """INSERT OR REPLACE INTO source_segments
           (id, identity, ordinal, source_id, structural_locator, content_hash, text)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (seg_id, f"{source_id}:para:{ordinal}", ordinal, source_id, f"p:{ordinal}", content_hash, text),
    )
    db._conn.commit()


# ── Stage 1: FTS5 migration ───────────────────────────────────────────────────


def test_fts5_migration_fresh_db(tmp_path: Path) -> None:
    """Fresh DB at v16 boots with FTS5 table and triggers present, zero FTS rows."""
    db = StateDB(tmp_path / "state.db")
    # FTS table exists
    row = db._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_segments_fts'"
    ).fetchone()
    assert row is not None, "source_segments_fts table not created"
    # Triggers exist
    triggers = {
        r[0]
        for r in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'source_segments_fts_%'"
        ).fetchall()
    }
    assert triggers == {"source_segments_fts_ai", "source_segments_fts_ad", "source_segments_fts_au"}
    # Zero FTS rows on empty DB
    count = db._conn.execute("SELECT count(*) FROM source_segments_fts").fetchone()[0]
    assert count == 0


def test_fts5_migration_backfill(tmp_path: Path) -> None:
    """Pre-existing source_segments rows are backfilled into FTS5 on upgrade.

    Simulate a pre-v16 DB by manipulating schema_version after inserting rows,
    then opening a fresh StateDB instance to trigger the migration.
    """
    db_path = tmp_path / "state.db"
    # Boot to v16 to create schema, then drop FTS+triggers and rewind version to 15
    db = StateDB(db_path)
    _insert_source(db, "src1")
    _insert_segment(db, "src1:p:0:aa", "src1", "alpha beta gamma", ordinal=0)
    _insert_segment(db, "src1:p:1:bb", "src1", "delta epsilon zeta", ordinal=1)

    # Rewind: drop FTS artifacts and set version back to 15 to simulate pre-v16
    db._conn.execute("DROP TABLE IF EXISTS source_segments_fts")
    for trigger in ("source_segments_fts_ai", "source_segments_fts_ad", "source_segments_fts_au"):
        db._conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    db._conn.execute("UPDATE schema_version SET version = 15 WHERE id = 1")
    db._conn.commit()
    db._conn.close()

    # Re-open to trigger v16 migration
    db2 = StateDB(db_path)
    fts_count = db2._conn.execute("SELECT count(*) FROM source_segments_fts").fetchone()[0]
    seg_count = db2._conn.execute("SELECT count(*) FROM source_segments").fetchone()[0]
    assert fts_count == seg_count == 2, f"backfill mismatch: fts={fts_count}, seg={seg_count}"


def test_fts5_trigger_insert(tmp_path: Path) -> None:
    """INSERT into source_segments fires the ai trigger → FTS row appears."""
    db = StateDB(tmp_path / "state.db")
    _insert_source(db, "src1")
    before = db._conn.execute("SELECT count(*) FROM source_segments_fts").fetchone()[0]
    _insert_segment(db, "src1:p:0:aa", "src1", "quantum entanglement")
    after = db._conn.execute("SELECT count(*) FROM source_segments_fts").fetchone()[0]
    assert after == before + 1


def test_fts5_trigger_delete(tmp_path: Path) -> None:
    """DELETE from source_segments fires the ad trigger → FTS row removed."""
    db = StateDB(tmp_path / "state.db")
    _insert_source(db, "src1")
    _insert_segment(db, "src1:p:0:aa", "src1", "wave function collapse")
    before = db._conn.execute("SELECT count(*) FROM source_segments_fts").fetchone()[0]
    db._conn.execute("DELETE FROM source_segments WHERE id = 'src1:p:0:aa'")
    db._conn.commit()
    after = db._conn.execute("SELECT count(*) FROM source_segments_fts").fetchone()[0]
    assert after == before - 1


def test_fts5_trigger_update(tmp_path: Path) -> None:
    """UPDATE on source_segments.text fires au trigger → FTS reflects new text."""
    db = StateDB(tmp_path / "state.db")
    _insert_source(db, "src1")
    _insert_segment(db, "src1:p:0:aa", "src1", "old content here")
    db._conn.execute(
        "UPDATE source_segments SET text = 'new content here' WHERE id = 'src1:p:0:aa'"
    )
    db._conn.commit()
    # Old term should not match; new term should
    old_hits = db._conn.execute(
        "SELECT count(*) FROM source_segments_fts WHERE source_segments_fts MATCH '\"old content here\"'"
    ).fetchone()[0]
    new_hits = db._conn.execute(
        "SELECT count(*) FROM source_segments_fts WHERE source_segments_fts MATCH '\"new content here\"'"
    ).fetchone()[0]
    assert old_hits == 0
    assert new_hits == 1


def test_fts5_migration_idempotent(tmp_path: Path) -> None:
    """Re-running _create_source_segments_fts_v16 when FTS already exists is a no-op."""
    db = StateDB(tmp_path / "state.db")
    _insert_source(db, "src1")
    _insert_segment(db, "src1:p:0:aa", "src1", "idempotency test")
    # Call migration method again — should not raise or duplicate rows
    db._create_source_segments_fts_v16()
    count = db._conn.execute("SELECT count(*) FROM source_segments_fts").fetchone()[0]
    assert count == 1


# ── Shared fixture for handler tests ─────────────────────────────────────────


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / ".synto").mkdir()
    return tmp_path


def _make_handlers(vault: Path, db: StateDB):
    """Build MCP handlers with a real StateDB (no audit)."""
    config = Config(vault=vault, mcp=McpConfig(audit=False))
    reader = VaultReader(vault)
    return build_tool_handlers(reader, config, db, vault_key="test-vault")


# ── Stage 2: read_source_segment ─────────────────────────────────────────────


def test_read_source_segment_returns_body(vault: Path, tmp_path: Path) -> None:
    """Valid segment_id returns the verbatim body."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    _insert_segment(db, "book1:p:0:aa", "book1", "The quick brown fox.", ordinal=0)
    handlers = _make_handlers(vault, db)
    result = handlers["read_source_segment"]("book1:p:0:aa")
    assert result["body"] == "The quick brown fox."
    assert result["segment_id"] == "book1:p:0:aa"
    assert result["source_id"] == "book1"
    assert result["truncated"] is False


def test_read_source_segment_unknown_id_raises(vault: Path) -> None:
    """Unknown segment_id raises a tool error."""
    db = StateDB(vault / ".synto" / "state.db")
    handlers = _make_handlers(vault, db)
    with pytest.raises(Exception, match="unknown segment_id"):
        handlers["read_source_segment"]("nonexistent:id")


def test_read_source_segment_truncation(vault: Path) -> None:
    """max_chars truncates body and sets truncated=True."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    _insert_segment(db, "book1:p:0:aa", "book1", "A" * 200, ordinal=0)
    handlers = _make_handlers(vault, db)
    result = handlers["read_source_segment"]("book1:p:0:aa", max_chars=50)
    assert len(result["body"]) <= 51  # 50 chars + ellipsis
    assert result["truncated"] is True


def test_read_source_segment_max_chars_cap(vault: Path) -> None:
    """max_chars is capped at 16000 even if caller passes higher."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    _insert_segment(db, "book1:p:0:aa", "book1", "B" * 20000, ordinal=0)
    handlers = _make_handlers(vault, db)
    result = handlers["read_source_segment"]("book1:p:0:aa", max_chars=99999)
    # Cap is 16000; body has 20000 chars → must be truncated
    assert result["truncated"] is True
    assert len(result["body"]) <= 16001


def test_read_source_segment_source_path_from_origin_uri(vault: Path) -> None:
    """source_path is populated from origin_uri; null origin_uri → null source_path."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")  # origin_uri = /raw/book1.pdf
    _insert_segment(db, "book1:p:0:aa", "book1", "text", ordinal=0)
    handlers = _make_handlers(vault, db)
    result = handlers["read_source_segment"]("book1:p:0:aa")
    assert result["source_path"] == "/raw/book1.pdf"

    # Source with null origin_uri
    db._conn.execute("INSERT INTO source_documents (id, source_type, redistribution) VALUES ('nouri', 'pdf', 'unknown')")
    db._conn.commit()
    _insert_segment(db, "nouri:p:0:aa", "nouri", "text2", ordinal=0)
    result2 = handlers["read_source_segment"]("nouri:p:0:aa")
    assert result2["source_path"] is None


def test_read_source_segment_not_registered_without_db(vault: Path) -> None:
    """read_source_segment is not registered when db=None."""
    config = Config(vault=vault, mcp=McpConfig(audit=False))
    reader = VaultReader(vault)
    handlers = build_tool_handlers(reader, config, None, vault_key="test-vault")
    assert "read_source_segment" not in handlers
