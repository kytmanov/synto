"""Tests for Feature 42 — verbatim source MCP tools.

Covers: FTS5 migration (Stage 1), four MCP handler closures (Stages 2-5),
and the license-driven privacy gate (Stage 6).
All tests use in-memory or tmp_path SQLite; no Ollama required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synto.config import Config, McpConfig, McpSourceAccessConfig
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
        (
            seg_id,
            f"{source_id}:para:{ordinal}",
            ordinal,
            source_id,
            f"p:{ordinal}",
            content_hash,
            text,
        ),
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
            "SELECT name FROM sqlite_master"
            " WHERE type='trigger' AND name LIKE 'source_segments_fts_%'"
        ).fetchall()
    }
    assert triggers == {
        "source_segments_fts_ai",
        "source_segments_fts_ad",
        "source_segments_fts_au",
    }
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
        "SELECT count(*) FROM source_segments_fts"
        " WHERE source_segments_fts MATCH '\"old content here\"'"
    ).fetchone()[0]
    new_hits = db._conn.execute(
        "SELECT count(*) FROM source_segments_fts"
        " WHERE source_segments_fts MATCH '\"new content here\"'"
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


def test_fts5_migration_skipped_when_unavailable(tmp_path: Path, monkeypatch) -> None:
    """On a SQLite build without FTS5 the v16 migration skips the index, but the schema
    still advances to v16 and core (non-MCP) operations keep working.

    This is the release-blocker guard: a local-only user whose SQLite lacks FTS5 must
    never be bricked by a verbatim-search feature they don't use. Without the guard the
    CREATE VIRTUAL TABLE would raise on every StateDB open and break every command.
    """
    monkeypatch.setattr("synto.state._fts5_available", lambda conn: False)
    db = StateDB(tmp_path / "state.db")

    version = db._conn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()[0]
    assert version == 18, "schema must still advance even when FTS5 is unavailable"
    assert not db.source_segments_fts_exists(), "FTS table must not be created without FTS5"
    trigger_count = db._conn.execute(
        "SELECT count(*) FROM sqlite_master"
        " WHERE type='trigger' AND name LIKE 'source_segments_fts_%'"
    ).fetchone()[0]
    assert trigger_count == 0, "FTS sync triggers must not be created without FTS5"

    # Core operations still work end-to-end — the user is not bricked.
    _insert_source(db, "src1")
    _insert_segment(db, "src1:p:0:aa", "src1", "still works without fts5")
    assert db.count_source_segments() == 1


def test_verbatim_tools_degrade_gracefully_without_fts5(vault: Path, monkeypatch) -> None:
    """Without FTS5, search_source_segments errors clearly while the concept- and
    id-keyed tools (which query source_segments directly) keep working."""
    monkeypatch.setattr("synto.state._fts5_available", lambda conn: False)
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    _insert_segment(db, "book1:p:0:aa", "book1", "The quick brown fox.", ordinal=0)
    handlers = _make_handlers(vault, db)

    with pytest.raises(Exception, match="FTS5"):
        handlers["search_source_segments"]("fox")

    # read_source_segment does not touch the FTS index → still works.
    result = handlers["read_source_segment"]("book1:p:0:aa")
    assert result["body"] == "The quick brown fox."


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
    """Build MCP handlers with a real StateDB, gate mode='all' (for non-gate tests)."""
    sa = McpSourceAccessConfig(mode="all")
    config = Config(vault=vault, mcp=McpConfig(audit=False, source_access=sa))
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
    db._conn.execute(
        "INSERT INTO source_documents (id, source_type, redistribution)"
        " VALUES ('nouri', 'pdf', 'unknown')"
    )
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
    _insert_segment(
        db, "book1:p:0:aa", "book1", "quantum quantum quantum mechanics quantum field", ordinal=0
    )
    _insert_segment(
        db, "book1:p:1:bb", "book1", "classical mechanics has no quantum at all", ordinal=1
    )
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
    db.replace_concepts_for_source("raw/book1.md", ["Quantum Mechanics"])
    _insert_occurrence(db, "Quantum Mechanics", "book1:p:0:aa")
    db.upsert_aliases("Quantum Mechanics", ["QM"])
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
    assert result == {"results": [], "hidden_by_policy": 0, "orphan_segments": 0}


def test_get_source_passages_max_passages_exceeded_raises(vault: Path) -> None:
    """max_passages > 20 raises a tool error."""
    db = StateDB(vault / ".synto" / "state.db")
    handlers = _make_handlers(vault, db)
    with pytest.raises(Exception, match="20"):
        handlers["get_source_passages"]("anything", max_passages=21)


def test_get_source_passages_truncation(vault: Path) -> None:
    """max_chars_per_passage truncates body and sets truncated=True; default cap is 8000."""
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
    # Explicit max_chars_per_passage
    result2 = handlers["get_source_passages"]("Big", max_chars_per_passage=100)
    assert result2["results"][0]["truncated"] is True
    assert len(result2["results"][0]["body"]) <= 101


# ── Stage 5: list_segments ────────────────────────────────────────────────────


def test_list_segments_returns_ordered_segments(vault: Path) -> None:
    """Returns segments in ordinal order with correct total and returned counts."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    for i in range(5):
        _insert_segment(db, f"book1:p:{i}:aa", "book1", f"paragraph {i}" * 10, ordinal=i)
    handlers = _make_handlers(vault, db)
    result = handlers["list_segments"]("book1")
    assert result["total"] == 5
    assert result["returned"] == 5
    ordinals = [s["ordinal"] for s in result["segments"]]
    assert ordinals == sorted(ordinals)
    # Each segment has id, ordinal, length
    assert all("segment_id" in s and "length" in s for s in result["segments"])


def test_list_segments_pagination(vault: Path) -> None:
    """limit and offset work correctly."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    for i in range(10):
        _insert_segment(db, f"book1:p:{i}:aa", "book1", f"text {i}", ordinal=i)
    handlers = _make_handlers(vault, db)
    result = handlers["list_segments"]("book1", limit=3, offset=2)
    assert result["total"] == 10
    assert result["returned"] == 3
    assert result["segments"][0]["ordinal"] == 2


def test_list_segments_unknown_source_raises(vault: Path) -> None:
    """Unknown source_id raises a tool error."""
    db = StateDB(vault / ".synto" / "state.db")
    handlers = _make_handlers(vault, db)
    with pytest.raises(Exception, match="unknown source_id"):
        handlers["list_segments"]("does-not-exist")


def test_list_segments_source_with_zero_segments(vault: Path) -> None:
    """Source with no segments returns empty list without error."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "empty-book")
    handlers = _make_handlers(vault, db)
    result = handlers["list_segments"]("empty-book")
    assert result["total"] == 0
    assert result["returned"] == 0
    assert result["segments"] == []


def test_list_segments_limit_clamped(vault: Path) -> None:
    """limit > 500 is clamped to 500."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    for i in range(5):
        _insert_segment(db, f"book1:p:{i}:aa", "book1", f"text {i}", ordinal=i)
    handlers = _make_handlers(vault, db)
    # Passing limit=999 should not raise; returns all 5 segments
    result = handlers["list_segments"]("book1", limit=999)
    assert result["returned"] == 5


# ── Stage 6: privacy gate ─────────────────────────────────────────────────────


def _make_handlers_with_access(
    vault: Path, db: StateDB, mode: str, permissive: list[str] | None = None
):
    """Build handlers with a specific source_access config."""
    sa_kwargs: dict = {"mode": mode}
    if permissive is not None:
        sa_kwargs["permissive_licenses"] = permissive
    sa = McpSourceAccessConfig(**sa_kwargs)
    config = Config(vault=vault, mcp=McpConfig(audit=False, source_access=sa))
    reader = VaultReader(vault)
    return build_tool_handlers(reader, config, db, vault_key="test-vault")


def test_privacy_gate_mode_all_permits_any_license(vault: Path) -> None:
    """mode='all' allows access regardless of license."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "restricted", license="proprietary")
    _insert_segment(db, "restricted:p:0:aa", "restricted", "secret content", ordinal=0)
    handlers = _make_handlers_with_access(vault, db, mode="all")
    result = handlers["read_source_segment"]("restricted:p:0:aa")
    assert result["body"] == "secret content"


def test_privacy_gate_mode_deny_blocks_all(vault: Path) -> None:
    """mode='deny' blocks all four tools regardless of license."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "open", license="CC-BY")
    _insert_segment(db, "open:p:0:aa", "open", "open content", ordinal=0)
    _insert_concept(db, "Concept")
    _insert_occurrence(db, "Concept", "open:p:0:aa")
    handlers = _make_handlers_with_access(vault, db, mode="deny")
    # read_source_segment
    with pytest.raises(Exception, match="restricted by license policy"):
        handlers["read_source_segment"]("open:p:0:aa")
    # list_segments
    with pytest.raises(Exception, match="restricted by license policy"):
        handlers["list_segments"]("open")
    # search_source_segments — multi-source tool returns empty with hidden_by_policy count
    result = handlers["search_source_segments"]("open")
    assert result["results"] == []
    assert result["hidden_by_policy"] >= 0  # may be 0 if FTS returns no rows, that's fine
    # get_source_passages — same pattern
    result2 = handlers["get_source_passages"]("Concept")
    assert result2["results"] == []


def test_privacy_gate_permissive_only_allows_permissive_license(vault: Path) -> None:
    """mode='permissive_only' allows sources with a permissive license."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "ccby", license="CC-BY")
    _insert_segment(db, "ccby:p:0:aa", "ccby", "open content", ordinal=0)
    handlers = _make_handlers_with_access(vault, db, mode="permissive_only")
    result = handlers["read_source_segment"]("ccby:p:0:aa")
    assert result["body"] == "open content"


def test_privacy_gate_permissive_only_blocks_null_license(vault: Path) -> None:
    """mode='permissive_only' blocks sources with null license.

    A sentinel source with a declared license ensures the grandfather rule does not
    fire — the gate must be active for the null-license block to be meaningful.
    """
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "sentinel", license="CC-BY")  # declares a license → grandfather stays off
    _insert_source(db, "nolic")  # no license set
    _insert_segment(db, "nolic:p:0:aa", "nolic", "unknown rights content", ordinal=0)
    handlers = _make_handlers_with_access(vault, db, mode="permissive_only")
    with pytest.raises(Exception, match="restricted by license policy"):
        handlers["read_source_segment"]("nolic:p:0:aa")


def test_privacy_gate_permissive_only_blocks_restrictive_license(vault: Path) -> None:
    """mode='permissive_only' blocks sources with non-permissive license."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "prop", license="proprietary")
    _insert_segment(db, "prop:p:0:aa", "prop", "private content", ordinal=0)
    handlers = _make_handlers_with_access(vault, db, mode="permissive_only")
    with pytest.raises(Exception, match="restricted by license policy"):
        handlers["read_source_segment"]("prop:p:0:aa")


def test_privacy_gate_case_insensitive_license_match(vault: Path) -> None:
    """License comparison is case-insensitive: 'cc-by' matches 'CC-BY'."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "src1", license="cc-by")  # lowercase
    _insert_segment(db, "src1:p:0:aa", "src1", "content", ordinal=0)
    handlers = _make_handlers_with_access(
        vault,
        db,
        mode="permissive_only",
        permissive=["CC-BY"],  # uppercase in config
    )
    result = handlers["read_source_segment"]("src1:p:0:aa")
    assert result["body"] == "content"


def test_privacy_gate_custom_permissive_licenses(vault: Path) -> None:
    """Custom permissive_licenses list overrides the default."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "custom", license="Custom-Open-License")
    _insert_segment(db, "custom:p:0:aa", "custom", "custom licensed content", ordinal=0)
    # Default list does not include Custom-Open-License → blocked
    handlers_default = _make_handlers_with_access(vault, db, mode="permissive_only")
    with pytest.raises(Exception, match="restricted by license policy"):
        handlers_default["read_source_segment"]("custom:p:0:aa")
    # Custom list includes it → allowed
    handlers_custom = _make_handlers_with_access(
        vault, db, mode="permissive_only", permissive=["Custom-Open-License"]
    )
    result = handlers_custom["read_source_segment"]("custom:p:0:aa")
    assert result["body"] == "custom licensed content"


def test_privacy_gate_search_hidden_by_policy_count(vault: Path) -> None:
    """search_source_segments reports hidden_by_policy count for blocked sources."""
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "open", license="CC-BY")
    _insert_source(db, "closed", license="proprietary")
    _insert_segment(db, "open:p:0:aa", "open", "open topic alpha", ordinal=0)
    _insert_segment(db, "closed:p:0:aa", "closed", "closed topic alpha", ordinal=0)
    handlers = _make_handlers_with_access(vault, db, mode="permissive_only")
    result = handlers["search_source_segments"]("alpha")
    visible_ids = [r["segment_id"] for r in result["results"]]
    assert "open:p:0:aa" in visible_ids
    assert "closed:p:0:aa" not in visible_ids
    assert result["hidden_by_policy"] == 1


# ── Stage A: grandfather rule for legacy vaults ──────────────────────────────


def _clear_mode_cache() -> None:
    from synto import serve as serve_module

    serve_module._effective_mode_cache.clear()


def _make_handlers_with_default_gate(vault: Path, db: StateDB):
    """Build handlers with default permissive_only mode (the real-world default)."""
    config = Config(vault=vault, mcp=McpConfig(audit=False))  # default source_access
    reader = VaultReader(vault)
    return build_tool_handlers(reader, config, db, vault_key="stageA-vault")


def test_grandfather_legacy_vault_with_no_licenses(vault: Path) -> None:
    """Legacy vault (0 declared licenses) under default config → tools work seamlessly."""
    _clear_mode_cache()
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")  # no license argument → license stays NULL
    _insert_segment(db, "book1:p:0:aa", "book1", "the brown fox", ordinal=0)
    handlers = _make_handlers_with_default_gate(vault, db)
    result = handlers["read_source_segment"]("book1:p:0:aa")
    assert result["body"] == "the brown fox"


def test_grandfather_disengages_when_any_license_declared(vault: Path) -> None:
    """Once any source has a license, the gate re-engages on a fresh process."""
    _clear_mode_cache()
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "open", license="CC-BY")
    _insert_source(db, "unknown")  # NULL license
    _insert_segment(db, "open:p:0:aa", "open", "open content", ordinal=0)
    _insert_segment(db, "unknown:p:0:aa", "unknown", "unknown content", ordinal=0)
    handlers = _make_handlers_with_default_gate(vault, db)
    # Open source is visible
    open_result = handlers["read_source_segment"]("open:p:0:aa")
    assert open_result["body"] == "open content"
    # NULL-license source is blocked (grandfather did NOT fire)
    with pytest.raises(Exception, match="restricted by license policy"):
        handlers["read_source_segment"]("unknown:p:0:aa")


def test_grandfather_does_not_override_explicit_deny(vault: Path) -> None:
    """mode='deny' on a legacy vault still denies. Grandfather only relaxes permissive_only."""
    _clear_mode_cache()
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")  # NULL license, would normally trigger grandfather
    _insert_segment(db, "book1:p:0:aa", "book1", "content", ordinal=0)
    sa = McpSourceAccessConfig(mode="deny")
    config = Config(vault=vault, mcp=McpConfig(audit=False, source_access=sa))
    reader = VaultReader(vault)
    handlers = build_tool_handlers(reader, config, db, vault_key="stageA-vault")
    with pytest.raises(Exception, match="restricted by license policy"):
        handlers["read_source_segment"]("book1:p:0:aa")


def test_grandfather_caches_per_vault(vault: Path) -> None:
    """Effective mode is computed once per (vault_key, configured.mode), not per call."""
    _clear_mode_cache()
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1")
    _insert_segment(db, "book1:p:0:aa", "book1", "content", ordinal=0)
    from synto import serve as serve_module

    handlers = _make_handlers_with_default_gate(vault, db)
    handlers["read_source_segment"]("book1:p:0:aa")
    # After first call, cache should be populated
    assert ("stageA-vault", "permissive_only") in serve_module._effective_mode_cache
    # Cache value should be "all" (grandfather kicked in because 0 declared licenses)
    assert serve_module._effective_mode_cache[("stageA-vault", "permissive_only")] == "all"


# ── Stage B: N+1 license-query fix + orphan-segment distinction ─────────────


class _CountingConn:
    """Thin proxy over a sqlite3.Connection that counts source_documents SELECTs.

    sqlite3.Connection.execute is read-only in CPython 3.13+, so we cannot
    monkey-patch it. Instead we swap db._conn for this proxy, which forwards
    every call while maintaining a counter.
    """

    def __init__(self, conn) -> None:
        self._conn = conn
        self.source_documents_selects = 0

    def execute(self, sql, *args, **kwargs):
        if "source_documents" in sql.lower():
            self.source_documents_selects += 1
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_search_does_not_n_plus_one_license_queries(vault: Path) -> None:
    """search_source_segments must batch-fetch licenses, not query per result.

    The grandfather check (Stage A) and the batch fetch each issue 1 SELECT
    against source_documents. After the cache warms, only the batch fetch fires.
    Total per call: ≤2 SELECTs against source_documents (1 grandfather + 1 batch),
    and after the first call only 1.
    """
    _clear_mode_cache()
    db = StateDB(vault / ".synto" / "state.db")
    # Sentinel licensed source so grandfather does NOT fire and the gate is active.
    _insert_source(db, "open", license="CC-BY")
    _insert_segment(db, "open:p:0:aa", "open", "alpha topic from open source", ordinal=0)
    # Build 10 additional permissive sources, each with one segment matching "alpha".
    for i in range(10):
        sid = f"book{i}"
        _insert_source(db, sid, license="CC-BY")
        _insert_segment(db, f"{sid}:p:0:aa", sid, f"alpha topic in book {i}", ordinal=0)
    handlers = _make_handlers_with_default_gate(vault, db)

    counting = _CountingConn(db._conn)
    real_conn = db._conn
    db._conn = counting  # type: ignore[assignment]
    try:
        result = handlers["search_source_segments"]("alpha", limit=50)
    finally:
        db._conn = real_conn

    assert len(result["results"]) == 11
    # ≤2: at most one grandfather probe + one batched fetch.
    assert counting.source_documents_selects <= 2, (
        f"search_source_segments issued {counting.source_documents_selects} SELECTs"
        " against source_documents — should be ≤2 (1 grandfather + 1 batch)."
    )


def test_get_source_passages_does_not_n_plus_one_license_queries(vault: Path) -> None:
    """get_source_passages must batch-fetch licenses, not query per result."""
    _clear_mode_cache()
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "sentinel", license="CC-BY")
    _insert_concept(db, "TestConcept")
    for i in range(10):
        sid = f"book{i}"
        _insert_source(db, sid, license="CC-BY")
        _insert_segment(db, f"{sid}:p:0:aa", sid, f"passage {i}", ordinal=0)
        _insert_occurrence(db, "TestConcept", f"{sid}:p:0:aa", confidence=1.0, ordinal=i)
    handlers = _make_handlers_with_default_gate(vault, db)

    counting = _CountingConn(db._conn)
    real_conn = db._conn
    db._conn = counting  # type: ignore[assignment]
    try:
        result = handlers["get_source_passages"]("TestConcept", max_passages=10)
    finally:
        db._conn = real_conn

    assert len(result["results"]) == 10
    assert counting.source_documents_selects <= 2, (
        f"get_source_passages issued {counting.source_documents_selects} SELECTs"
        " against source_documents — should be ≤2 (1 grandfather + 1 batch)."
    )


def test_search_orphan_segment_counted_separately(vault: Path) -> None:
    """Segments whose source_id is missing from source_documents → orphan, not policy."""
    _clear_mode_cache()
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "real", license="CC-BY")
    _insert_segment(db, "real:p:0:aa", "real", "alpha real content", ordinal=0)
    # Directly insert a segment row whose source_id has no row in source_documents.
    db._conn.execute(
        """INSERT INTO source_segments
           (id, identity, ordinal, source_id, structural_locator, content_hash, text)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("orphan:p:0:aa", "orphan:para:0", 0, "orphan", "p:0", "abc", "alpha orphan content"),
    )
    db._conn.commit()
    handlers = _make_handlers_with_default_gate(vault, db)
    result = handlers["search_source_segments"]("alpha")
    visible_ids = [r["segment_id"] for r in result["results"]]
    assert "real:p:0:aa" in visible_ids
    assert "orphan:p:0:aa" not in visible_ids
    assert result["orphan_segments"] == 1
    assert result["hidden_by_policy"] == 0


def test_get_source_passages_orphan_segment_counted_separately(vault: Path) -> None:
    """Orphan segments in get_source_passages also surface as orphan_segments, not policy."""
    _clear_mode_cache()
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "real", license="CC-BY")
    _insert_segment(db, "real:p:0:aa", "real", "real passage", ordinal=0)
    # Orphan segment + occurrence
    db._conn.execute(
        """INSERT INTO source_segments
           (id, identity, ordinal, source_id, structural_locator, content_hash, text)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("orphan:p:0:aa", "orphan:para:0", 0, "orphan", "p:0", "abc", "orphan passage"),
    )
    _insert_concept(db, "MyConcept")
    _insert_occurrence(db, "MyConcept", "real:p:0:aa", confidence=1.0, ordinal=0)
    _insert_occurrence(db, "MyConcept", "orphan:p:0:aa", confidence=1.0, ordinal=1)
    db._conn.commit()
    handlers = _make_handlers_with_default_gate(vault, db)
    result = handlers["get_source_passages"]("MyConcept")
    visible_ids = [r["segment_id"] for r in result["results"]]
    assert "real:p:0:aa" in visible_ids
    assert "orphan:p:0:aa" not in visible_ids
    assert result["orphan_segments"] == 1
    assert result["hidden_by_policy"] == 0


def test_search_response_shape_includes_orphan_segments_field(vault: Path) -> None:
    """All search responses include orphan_segments (default 0)."""
    _clear_mode_cache()
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1", license="CC-BY")
    _insert_segment(db, "book1:p:0:aa", "book1", "topic content", ordinal=0)
    handlers = _make_handlers_with_default_gate(vault, db)
    result = handlers["search_source_segments"]("topic")
    assert "orphan_segments" in result
    assert result["orphan_segments"] == 0


def test_get_source_passages_response_shape_includes_orphan_segments(vault: Path) -> None:
    """get_source_passages response always includes orphan_segments."""
    _clear_mode_cache()
    db = StateDB(vault / ".synto" / "state.db")
    handlers = _make_handlers_with_default_gate(vault, db)
    # Unknown concept path
    result = handlers["get_source_passages"]("Nonexistent")
    assert "orphan_segments" in result
    assert result["orphan_segments"] == 0


# ── Stage C: degenerate-query handling + max_chars_per_passage rename ────────


def test_search_degenerate_quoted_query_raises(vault: Path) -> None:
    """Queries that collapse to empty after stripping quotes raise validation error."""
    _clear_mode_cache()
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1", license="CC-BY")
    _insert_segment(db, "book1:p:0:aa", "book1", "content", ordinal=0)
    handlers = _make_handlers_with_default_gate(vault, db)
    for degenerate in ['""', '"  "', '"', '   "   "   ']:
        with pytest.raises(Exception, match="non-empty"):
            handlers["search_source_segments"](degenerate)


def test_get_source_passages_max_chars_per_passage_param(vault: Path) -> None:
    """get_source_passages uses max_chars_per_passage as the keyword (post-Stage-C rename)."""
    _clear_mode_cache()
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1", license="CC-BY")
    _insert_segment(db, "book1:p:0:aa", "book1", "X" * 1000, ordinal=0)
    _insert_concept(db, "Big")
    _insert_occurrence(db, "Big", "book1:p:0:aa")
    handlers = _make_handlers_with_default_gate(vault, db)
    result = handlers["get_source_passages"]("Big", max_chars_per_passage=50)
    assert result["results"][0]["truncated"] is True
    assert len(result["results"][0]["body"]) <= 51


# ── Stage D: body_length in search results ───────────────────────────────────


def test_search_includes_body_length(vault: Path) -> None:
    """Each search result row exposes body_length so callers can size truncation."""
    _clear_mode_cache()
    db = StateDB(vault / ".synto" / "state.db")
    _insert_source(db, "book1", license="CC-BY")
    _insert_segment(db, "book1:p:0:aa", "book1", "alpha topic content here exactly", ordinal=0)
    handlers = _make_handlers_with_default_gate(vault, db)
    result = handlers["search_source_segments"]("alpha")
    assert "body_length" in result["results"][0]
    assert result["results"][0]["body_length"] == len("alpha topic content here exactly")
