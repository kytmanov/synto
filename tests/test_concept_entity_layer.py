"""Tests for the concept identity layer (Phase 1, Feature 45).

Covers:
  - concept_entities / concept_labels schema and partial unique indexes
  - name→id→name round-trip via entity_id_for_name / preferred_label_for_entity
  - resolve_label: exact 1-hit, miss, ambiguous (homonym fixture)
  - Bridge facade methods backed by concept_labels
  - Backfill idempotency
  - INDEX.json seed carries entity_id (integration)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from synto.concept_text import concept_key as _ck
from synto.concept_text import match_key as _mk
from synto.state import ResolveResult, StateDB

# ---------------------------------------------------------------------------
# Schema guards
# ---------------------------------------------------------------------------


def test_concept_entities_table_exists(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    tables = {
        r[0]
        for r in db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "concept_entities" in tables
    assert "concept_labels" in tables


def test_concept_entities_has_no_preferred_label_column(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    cols = {r[1] for r in db._conn.execute("PRAGMA table_info(concept_entities)").fetchall()}
    assert "preferred_label" not in cols
    assert "id" in cols
    assert "status" in cols


def test_concept_labels_partial_unique_index_preferred_global(tmp_path: Path) -> None:
    """Two active entities cannot share the same preferred label_key."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])

    alpha_id = db.entity_id_for_name("Alpha")
    beta_id = db.entity_id_for_name("Beta")
    assert alpha_id and beta_id

    # Attempt to steal Alpha's preferred label_key for Beta → must fail.
    with pytest.raises(sqlite3.IntegrityError):
        with db._tx():
            db._conn.execute(
                "INSERT INTO concept_labels"
                " (entity_id, label, label_key, match_key, role, source, created_at)"
                " VALUES (?, ?, ?, ?, 'preferred', 'extracted', '2026-01-01')",
                (beta_id, "Alpha", "alpha", "alpha"),
            )


def test_concept_labels_partial_unique_index_preferred_per_entity(tmp_path: Path) -> None:
    """Each entity can have at most one preferred label."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    alpha_id = db.entity_id_for_name("Alpha")
    assert alpha_id

    # Inserting a second preferred row for the same entity must fail.
    with pytest.raises(sqlite3.IntegrityError):
        with db._tx():
            db._conn.execute(
                "INSERT INTO concept_labels"
                " (entity_id, label, label_key, match_key, role, source, created_at)"
                " VALUES (?, ?, ?, ?, 'preferred', 'extracted', '2026-01-01')",
                (alpha_id, "Alpha2", _ck("Alpha2"), _mk("Alpha2")),
            )


# ---------------------------------------------------------------------------
# Round-trip: name → id → name
# ---------------------------------------------------------------------------


def test_name_to_id_to_name_round_trip(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Quantum Computing"])

    entity_id = db.entity_id_for_name("Quantum Computing")
    assert entity_id is not None
    assert len(entity_id) > 0

    back = db.preferred_label_for_entity(entity_id)
    assert back == "Quantum Computing"


def test_entity_id_for_name_case_insensitive(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Machine Learning"])

    id1 = db.entity_id_for_name("Machine Learning")
    id2 = db.entity_id_for_name("machine learning")
    id3 = db.entity_id_for_name("MACHINE LEARNING")
    assert id1 is not None
    assert id1 == id2 == id3


def test_entity_id_for_unknown_name_returns_none(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    assert db.entity_id_for_name("Nonexistent Concept") is None


def test_preferred_label_for_unknown_entity_returns_none(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    assert db.preferred_label_for_entity("01AAAAAAAAAAAAAAAAAAAAAAAAA") is None


# ---------------------------------------------------------------------------
# resolve_label: 1-hit, 0-hit, ambiguous
# ---------------------------------------------------------------------------


def test_resolve_label_exact_hit(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Backpropagation"])

    result = db.resolve_label("Backpropagation")
    assert isinstance(result, ResolveResult)
    assert len(result.ids) == 1
    assert not result.ambiguous
    assert db.preferred_label_for_entity(result.ids[0]) == "Backpropagation"


def test_resolve_label_miss_returns_empty(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    result = db.resolve_label("Nonexistent")
    assert result.ids == []
    assert not result.ambiguous


def test_resolve_label_via_alias(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Gradient Descent"])
    db.upsert_aliases("Gradient Descent", ["GD"])

    result = db.resolve_label("GD")
    assert len(result.ids) == 1
    assert not result.ambiguous
    assert db.preferred_label_for_entity(result.ids[0]) == "Gradient Descent"


def test_resolve_label_ambiguous_shared_alias(tmp_path: Path) -> None:
    """Two entities sharing the same alias label_key → resolve returns both, ambiguous=True."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["United States"])
    db.upsert_concepts("raw/b.md", ["Ultrasound"])
    db.upsert_aliases("United States", ["US"])
    db.upsert_aliases("Ultrasound", ["US"])

    result = db.resolve_label("US")
    assert result.ambiguous
    assert len(result.ids) == 2
    labels = {db.preferred_label_for_entity(eid) for eid in result.ids}
    assert labels == {"United States", "Ultrasound"}


def test_resolve_label_preferred_label_wins_over_alias(tmp_path: Path) -> None:
    """A preferred label stored as an alias of another entity creates an ambiguous surface.

    The collision guard lives in the extraction path (_normalize_concepts), not in
    upsert_aliases.  Explicit programmatic alias writes (cross-language aliases, tests)
    are allowed; resolve_label will return ambiguous when they collide.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Python"])
    db.upsert_concepts("raw/b.md", ["Python Snake"])
    db.upsert_aliases("Python Snake", ["Python"])  # explicit alias; stored unconditionally

    # "Python" is preferred label of entity A and an alias of entity B → ambiguous.
    result = db.resolve_label("Python")
    assert result.ambiguous
    assert len(result.ids) == 2


# ---------------------------------------------------------------------------
# Bridge facade methods backed by concept_labels
# ---------------------------------------------------------------------------


def test_upsert_aliases_creates_entity_if_missing(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    # No prior upsert_concepts call — entity must be minted by upsert_aliases.
    db.upsert_aliases("NewConcept", ["nc", "new-c"])

    entity_id = db.entity_id_for_name("NewConcept")
    assert entity_id is not None

    aliases = db.get_aliases("NewConcept")
    assert set(aliases) == {"nc", "new-c"}


def test_get_aliases_reads_from_concept_labels(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Tensor"])
    db.upsert_aliases("Tensor", ["tensor flow", "T"])

    aliases = db.get_aliases("Tensor")
    assert "T" in aliases
    assert "tensor flow" in aliases


def test_count_aliases(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_aliases("Alpha", ["a1", "a2", "a3"])
    assert db.count_aliases() == 3


def test_list_frequent_aliases_detects_shared_alias(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["EntityA"])
    db.upsert_concepts("raw/b.md", ["EntityB"])
    db.upsert_aliases("EntityA", ["shared", "only-a"])
    db.upsert_aliases("EntityB", ["shared", "only-b"])

    frequent = db.list_frequent_aliases()
    assert "shared" in frequent  # label_key of "shared"
    assert "only-a" not in frequent
    assert "only-b" not in frequent


def test_resolve_alias_returns_canonical_name(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Neural Network"])
    db.upsert_aliases("Neural Network", ["NN"])

    result = db.resolve_alias("NN")
    assert result == "Neural Network"


def test_resolve_alias_returns_none_for_ambiguous(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["A"])
    db.upsert_concepts("raw/b.md", ["B"])
    db.upsert_aliases("A", ["shared"])
    db.upsert_aliases("B", ["shared"])

    assert db.resolve_alias("shared") is None


# ---------------------------------------------------------------------------
# Backfill idempotency
# ---------------------------------------------------------------------------


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    """Opening the same DB multiple times must not duplicate entities."""
    db_path = tmp_path / "state.db"
    db = StateDB(db_path)
    db.upsert_concepts("raw/a.md", ["Qubit"])
    db.upsert_aliases("Qubit", ["qubit", "q"])
    db.close()

    db2 = StateDB(db_path)
    q = "SELECT COUNT(*) FROM concept_entities WHERE status='active'"
    count = db2._conn.execute(q).fetchone()[0]

    db3 = StateDB(db_path)
    count2 = db3._conn.execute(q).fetchone()[0]

    assert count == count2


# ---------------------------------------------------------------------------
# INDEX.json seed carries entity_id
# ---------------------------------------------------------------------------


def test_index_json_seed_carries_entity_id(tmp_path: Path) -> None:
    import json

    from synto.config import Config
    from synto.indexer import generate_index_json
    from synto.models import RawNoteRecord, WikiArticleRecord
    from synto.vault import write_note

    vault = tmp_path
    (vault / "raw").mkdir()
    (vault / "wiki").mkdir()
    (vault / ".synto").mkdir()

    config = Config(vault=vault)
    db = StateDB(config.state_db_path)

    write_note(vault / "wiki" / "Qubit.md", {"title": "Qubit"}, "Qubit body.")
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Qubit"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Qubit.md",
            title="Qubit",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )

    index_path = generate_index_json(config, db)
    payload = json.loads(index_path.read_text(encoding="utf-8"))

    article = payload["articles"][0]
    assert "entity_id" in article
    assert article["entity_id"] == db.entity_id_for_name("Qubit")

    sc = payload["source_concepts"][0]
    concept_entry = sc["concepts"][0]
    assert isinstance(concept_entry, dict)
    assert concept_entry["name"] == "Qubit"
    assert concept_entry["entity_id"] == db.entity_id_for_name("Qubit")

    assert "identity_log" in payload
    assert payload["identity_log"] == []
