"""Tests for Phase 2 of Feature 45: occurrence resolution + match_key matching.

Covers:
  - resolve_by_match_key: plural/singular folding ("Users" → "User" entity)
  - Ambiguous occurrence recording (two-meaning fixture)
  - Source-path occurrence for unsegmented notes (null source_segment_id)
  - Re-ingest → zero new entities (idempotency)
  - Sticky resolution: prior source edge survives re-ingest (decision 18)
  - Alias collision guard: skips alias whose label_key is another entity's preferred label
  - Issue-#54 prevention: "Users" matches "User" entity (alias added, no new entity)
"""

from __future__ import annotations

from pathlib import Path

from synto.models import Concept
from synto.pipeline.ingest import _normalize_concepts
from synto.state import ResolveResult, StateDB

# ---------------------------------------------------------------------------
# resolve_by_match_key: plural/singular folding
# ---------------------------------------------------------------------------


def test_resolve_by_match_key_singular_matches_plural(tmp_path: Path) -> None:
    """'Users' and 'User' share match_key 'user' → resolve returns the User entity."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["User"])

    rr = db.resolve_by_match_key("Users")
    assert not rr.ambiguous
    assert len(rr.ids) == 1
    assert db.preferred_label_for_entity(rr.ids[0]) == "User"


def test_resolve_by_match_key_exact_hit(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Machine Learning"])

    rr = db.resolve_by_match_key("Machine Learning")
    assert not rr.ambiguous
    assert len(rr.ids) == 1


def test_resolve_by_match_key_miss_returns_empty(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    rr = db.resolve_by_match_key("Nonexistent")
    assert isinstance(rr, ResolveResult)
    assert rr.ids == []
    assert not rr.ambiguous


def test_resolve_by_match_key_ambiguous_shared_preferred(tmp_path: Path) -> None:
    """Two entities with same match_key on their preferred labels → ambiguous."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["User"])
    db.upsert_concepts("raw/b.md", ["Users"])  # match_key same as "User"

    rr = db.resolve_by_match_key("User")
    assert rr.ambiguous
    assert len(rr.ids) == 2


# ---------------------------------------------------------------------------
# Ambiguous occurrence recording (two-meaning fixture)
# ---------------------------------------------------------------------------


def test_record_ambiguous_occurrence_creates_row_and_candidates(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["United States"])
    db.upsert_concepts("raw/b.md", ["Ultrasound"])

    id_us = db.entity_id_for_name("United States")
    id_ult = db.entity_id_for_name("Ultrasound")
    assert id_us and id_ult

    db.record_ambiguous_occurrence(
        "US",
        [id_us, id_ult],
        surface="US",
        source_path="raw/note.md",
    )

    rows = db._conn.execute(
        "SELECT resolution_status, surface FROM concept_occurrences WHERE concept_name = 'US'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "ambiguous"
    assert rows[0][1] == "US"

    occ_id = db._conn.execute(
        "SELECT id FROM concept_occurrences WHERE concept_name = 'US'"
    ).fetchone()[0]
    candidates = db._conn.execute(
        "SELECT entity_id FROM concept_occurrence_candidates WHERE occurrence_id = ?",
        (occ_id,),
    ).fetchall()
    assert len(candidates) == 2
    assert {r[0] for r in candidates} == {id_us, id_ult}


def test_record_ambiguous_occurrence_source_path_only(tmp_path: Path) -> None:
    """Unsegmented note: source_segment_id is NULL, source_path is set."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])

    id_a = db.entity_id_for_name("Alpha")
    id_b = db.entity_id_for_name("Beta")

    db.record_ambiguous_occurrence(
        "AB",
        [id_a, id_b],
        surface="AB",
        source_path="raw/note.md",
        source_segment_id=None,
    )

    row = db._conn.execute(
        "SELECT source_segment_id, source_path FROM concept_occurrences WHERE concept_name = 'AB'"
    ).fetchone()
    assert row[0] is None  # source_segment_id is NULL
    assert row[1] == "raw/note.md"  # source_path is set


def test_count_ambiguous_occurrences(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["A"])
    db.upsert_concepts("raw/b.md", ["B"])

    id_a = db.entity_id_for_name("A")
    id_b = db.entity_id_for_name("B")

    assert db.count_ambiguous_occurrences() == 0

    db.record_ambiguous_occurrence("X", [id_a, id_b], surface="X", source_path="raw/n.md")
    assert db.count_ambiguous_occurrences() == 1


# ---------------------------------------------------------------------------
# Re-ingest → zero new entities (idempotency)
# ---------------------------------------------------------------------------


def test_normalize_reingest_mints_zero_new_entities(tmp_path: Path) -> None:
    """Re-ingesting the same concept twice must not mint duplicate entities."""
    db = StateDB(tmp_path / "state.db")
    # First ingest: mints the entity.
    r1 = _normalize_concepts(
        [Concept(name="Machine Learning", aliases=[])],
        db,
        rel_path="raw/a.md",
    )
    assert len(r1) == 1
    assert r1[0][0] == "Machine Learning"
    db.upsert_concepts("raw/a.md", [r1[0][0]])

    before = db._conn.execute("SELECT COUNT(*) FROM concept_entities").fetchone()[0]

    # Second ingest: must find the existing entity, mint zero new ones.
    r2 = _normalize_concepts(
        [Concept(name="Machine Learning", aliases=[])],
        db,
        rel_path="raw/a.md",
    )
    assert len(r2) == 1
    assert r2[0][0] == "Machine Learning"

    after = db._conn.execute("SELECT COUNT(*) FROM concept_entities").fetchone()[0]
    assert after == before


# ---------------------------------------------------------------------------
# Sticky resolution (decision 18)
# ---------------------------------------------------------------------------


def test_normalize_sticky_resolution_keeps_prior_edge(tmp_path: Path) -> None:
    """When a note already has a source edge to one candidate, re-ingest keeps that choice."""
    db = StateDB(tmp_path / "state.db")
    # Two entities with same match_key (homonym).
    db.upsert_concepts("raw/a.md", ["User"])
    db.upsert_concepts("raw/b.md", ["Users"])

    user_id = db.entity_id_for_name("User")

    # Simulate that raw/a.md was previously assigned to "User" (preferred edge via concepts table).
    # The sticky lookup reads concepts.source_path → entity via concept_labels.preferred join.
    # This is already stored because upsert_concepts("raw/a.md", ["User"]) did it.

    # resolve_by_match_key("User") should return both "User" and "Users" (same match_key "user").
    rr = db.resolve_by_match_key("User")
    assert rr.ambiguous  # both entities have match_key "user"

    # Sticky resolution: raw/a.md already has an edge to "User".
    sticky = db.get_sticky_entity_for_source("raw/a.md", rr.ids)
    assert sticky == user_id


def test_normalize_no_sticky_when_source_has_no_prior_edge(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["User"])
    db.upsert_concepts("raw/b.md", ["Users"])

    rr = db.resolve_by_match_key("User")
    assert rr.ambiguous

    # raw/c.md has no prior edge → no sticky.
    sticky = db.get_sticky_entity_for_source("raw/c.md", rr.ids)
    assert sticky is None


# ---------------------------------------------------------------------------
# Issue-#54 prevention: "Users" matches "User" entity via match_key
# ---------------------------------------------------------------------------


def test_normalize_users_matches_user_entity_no_new_entity(tmp_path: Path) -> None:
    """'Users' must resolve to the existing 'User' entity, not mint a duplicate.

    This is the issue-#54 prevention layer: match_key folding collapses plural
    forms to the same entity.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["User"])

    before = db._conn.execute("SELECT COUNT(*) FROM concept_entities").fetchone()[0]

    result = _normalize_concepts(
        [Concept(name="Users", aliases=[])],
        db,
        rel_path="raw/b.md",
    )

    after = db._conn.execute("SELECT COUNT(*) FROM concept_entities").fetchone()[0]

    # Resolved to the existing "User" entity — zero new entities minted.
    assert len(result) == 1
    assert result[0][0] == "User"
    assert after == before


def test_normalize_users_adds_extracted_alias(tmp_path: Path) -> None:
    """When 'Users' maps to the 'User' entity, 'Users' is stored as an extracted alias."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["User"])

    result = _normalize_concepts(
        [Concept(name="Users", aliases=[])],
        db,
        rel_path="raw/b.md",
    )
    # _normalize_concepts returns ("User", [...aliases...]) including "Users"
    assert len(result) == 1
    canonical, aliases = result[0]
    assert canonical == "User"
    # The surface form "Users" should appear as an alias to be persisted.
    assert "Users" in aliases


# ---------------------------------------------------------------------------
# Alias collision guard: extracted "US" alias for United States blocked at extraction
# ---------------------------------------------------------------------------


def test_normalize_collision_guard_blocks_alias_that_is_another_preferred(
    tmp_path: Path,
) -> None:
    """Extracted alias "US" for United States is blocked when "US" is already a preferred entity.

    upsert_aliases() stores all aliases unconditionally (cross-language aliases must
    survive explicit calls).  The collision guard lives in the extraction path:
    _normalize_concepts filters aliases that would make resolve_label ambiguous.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["US"])  # entity whose preferred label IS "US"
    db.upsert_concepts("raw/b.md", ["United States"])

    # Simulate extraction: LLM extracts "US" as an alias of "United States".
    result = _normalize_concepts(
        [Concept(name="United States", aliases=["US"])],
        db,
        rel_path="raw/b.md",
    )

    assert len(result) == 1
    canonical, aliases = result[0]
    assert canonical == "United States"
    # "US" alias is blocked — it's the preferred label of another active entity.
    assert "US" not in aliases

    # "US" entity's preferred label remains unambiguous.
    rr = db.resolve_label("US")
    assert not rr.ambiguous
    assert len(rr.ids) == 1
    assert db.preferred_label_for_entity(rr.ids[0]) == "US"
