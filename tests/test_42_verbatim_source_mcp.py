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


# ── Stage 3: search_source_segments ──────────────────────────────────────────


def test_search_source_segments_bm25_ordering(vault: Path) -> None:
    """Results are ordered by BM25 relevance (score = -rank, higher = better)."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    # Segment with many occurrences of "quantum" should rank higher than one with one
    _insert_segment(db, "book1:p:0:aa", "book1", "quantum quantum quantum mechanics quantum field", ordinal=0)
    _insert_segment(db, "book1:p:1:bb", "book1", "classical mechanics has no quantum at all", ordinal=1)
    handlers = _make_handlers(vault, db)
    result = handlers["search_source_segments"]("quantum")
    assert result["hidden_by_policy"] == 0
    scores = [r["score"] for r in result["results"]]
    assert scores == sorted(scores, reverse=True), "results not in descending score order"
    assert result["results"][0]["segment_id"] == "book1:p:0:aa"


def test_search_source_segments_empty_query_raises(vault: Path) -> None:
    """Empty or whitespace-only query raises a tool error."""
    db = StateDB(vault / ".synto" / "state.db")
    handlers = _make_handlers(vault, db)
    with pytest.raises(Exception, match="non-empty"):
        handlers["search_source_segments"]("")
    with pytest.raises(Exception, match="non-empty"):
        handlers["search_source_segments"]("   ")


def test_search_source_segments_limit_clamped(vault: Path) -> None:
    """limit > 50 is silently clamped to 50."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    for i in range(5):
        _insert_segment(db, f"book1:p:{i}:aa", "book1", f"topic alpha segment {i}", ordinal=i)
    handlers = _make_handlers(vault, db)
    # limit=200 is silently clamped; should return all 5 matching segments
    result = handlers["search_source_segments"]("alpha", limit=200)
    assert len(result["results"]) == 5


def test_search_source_segments_limit_zero_raises(vault: Path) -> None:
    """limit=0 raises a tool error."""
    db = StateDB(vault / ".synto" / "state.db")
    handlers = _make_handlers(vault, db)
    with pytest.raises(Exception):
        handlers["search_source_segments"]("anything", limit=0)


def test_search_source_segments_special_chars_in_query(vault: Path) -> None:
    """Query containing FTS5 special characters (quotes, colons) does not raise."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    _insert_segment(db, "book1:p:0:aa", "book1", "some regular content here", ordinal=0)
    handlers = _make_handlers(vault, db)
    # These would blow up FTS5 MATCH parsing if not sanitised
    result = handlers["search_source_segments"]('"quoted"')
    assert "results" in result
    result2 = handlers["search_source_segments"]("field:value AND NOT OR")
    assert "results" in result2


def test_search_source_segments_returns_source_path(vault: Path) -> None:
    """Each result row includes source_path from origin_uri."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    _insert_segment(db, "book1:p:0:aa", "book1", "origin test content", ordinal=0)
    handlers = _make_handlers(vault, db)
    result = handlers["search_source_segments"]("origin")
    assert result["results"][0]["source_path"] == "/raw/book1.pdf"


# ── Stage 4: get_source_passages ─────────────────────────────────────────────


def _insert_concept(db: StateDB, name: str, source_path: str = "raw/book1.md") -> None:
    db._conn.execute(
        "INSERT OR IGNORE INTO concepts (name, source_path) VALUES (?, ?)",
        (name, source_path),
    )
    db._conn.commit()


def _insert_occurrence(
    db: StateDB, concept_name: str, segment_id: str, confidence: float = 1.0, ordinal: int = 0
) -> None:
    db._conn.execute(
        """INSERT OR REPLACE INTO concept_occurrences
           (concept_name, source_segment_id, ordinal, confidence)
           VALUES (?, ?, ?, ?)""",
        (concept_name, segment_id, ordinal, confidence),
    )
    db._conn.commit()


def test_get_source_passages_known_concept(vault: Path) -> None:
    """Known concept returns segments ordered by confidence DESC, ordinal ASC."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    _insert_segment(db, "book1:p:0:aa", "book1", "high confidence passage", ordinal=0)
    _insert_segment(db, "book1:p:1:bb", "book1", "low confidence passage", ordinal=1)
    _insert_concept(db, "Quantum")
    _insert_occurrence(db, "Quantum", "book1:p:0:aa", confidence=0.9)
    _insert_occurrence(db, "Quantum", "book1:p:1:bb", confidence=0.3)
    handlers = _make_handlers(vault, db)
    result = handlers["get_source_passages"]("Quantum")
    assert result["hidden_by_policy"] == 0
    assert len(result["results"]) == 2
    assert result["results"][0]["segment_id"] == "book1:p:0:aa"
    assert result["results"][0]["confidence"] == pytest.approx(0.9)
    assert result["results"][1]["segment_id"] == "book1:p:1:bb"


def test_get_source_passages_alias_resolution(vault: Path) -> None:
    """Alias of a canonical concept returns the same results as the canonical name."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    _insert_segment(db, "book1:p:0:aa", "book1", "the passage body", ordinal=0)
    _insert_concept(db, "Quantum Mechanics")
    _insert_occurrence(db, "Quantum Mechanics", "book1:p:0:aa")
    # Insert alias
    db._conn.execute(
        "INSERT OR IGNORE INTO concept_aliases (concept_name, alias) VALUES (?, ?)",
        ("Quantum Mechanics", "QM"),
    )
    db._conn.commit()
    handlers = _make_handlers(vault, db)
    by_canonical = handlers["get_source_passages"]("Quantum Mechanics")
    by_alias = handlers["get_source_passages"]("QM")
    assert len(by_canonical["results"]) == 1
    assert len(by_alias["results"]) == 1
    assert by_alias["results"][0]["segment_id"] == by_canonical["results"][0]["segment_id"]


def test_get_source_passages_unknown_concept_returns_empty(vault: Path) -> None:
    """Unknown concept returns empty results without raising an error."""
    db = StateDB(vault / ".synto" / "state.db")
    handlers = _make_handlers(vault, db)
    result = handlers["get_source_passages"]("NonexistentConcept")
    assert result == {"results": [], "hidden_by_policy": 0}


def test_get_source_passages_max_passages_exceeded_raises(vault: Path) -> None:
    """max_passages > 20 raises a tool error."""
    db = StateDB(vault / ".synto" / "state.db")
    handlers = _make_handlers(vault, db)
    with pytest.raises(Exception, match="20"):
        handlers["get_source_passages"]("anything", max_passages=21)


def test_get_source_passages_truncation(vault: Path) -> None:
    """max_chars truncates body and sets truncated=True; default cap is 8000."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    _insert_segment(db, "book1:p:0:aa", "book1", "X" * 10000, ordinal=0)
    _insert_concept(db, "Big")
    _insert_occurrence(db, "Big", "book1:p:0:aa")
    handlers = _make_handlers(vault, db)
    # Default cap is 8000
    result = handlers["get_source_passages"]("Big")
    assert result["results"][0]["truncated"] is True
    assert len(result["results"][0]["body"]) <= 8001
    # Explicit max_chars
    result2 = handlers["get_source_passages"]("Big", max_chars=100)
    assert result2["results"][0]["truncated"] is True
    assert len(result2["results"][0]["body"]) <= 101
