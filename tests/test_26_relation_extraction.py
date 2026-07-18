"""Tests for Feature 26: Relation Extraction."""

from __future__ import annotations

from pathlib import Path

from synto.models import RelationCandidate, RelationExtractionResult
from synto.state import StateDB

# ---------------------------------------------------------------------------
# Stage 1: RelationExtractionResult model
# ---------------------------------------------------------------------------


def test_relation_extraction_result_model() -> None:
    relation = RelationCandidate(
        subject="Vector Clocks",
        predicate="implemented_by",
        object="Causal Consistency",
        evidence="Vector clocks are used to implement causal consistency.",
        source_segment_id="doc:0-0:abc123",
        provenance="extracted",
        confidence=0.85,
    )
    result = RelationExtractionResult(
        relations=[relation],
        source_segment_id="doc:0-0:abc123",
        model="gemma4:e4b",
    )
    data = result.model_dump()
    assert data["source_segment_id"] == "doc:0-0:abc123"
    assert data["model"] == "gemma4:e4b"
    assert len(data["relations"]) == 1
    assert data["relations"][0]["subject"] == "Vector Clocks"

    # round-trip: reconstructing from model_dump() must produce an equal model
    reconstructed = RelationExtractionResult(**data)
    assert reconstructed == result


def test_relation_extraction_result_empty_relations() -> None:
    result = RelationExtractionResult(relations=[], source_segment_id="x", model="m")
    assert result.model_dump_json() is not None
    assert result.relations == []


def test_relation_extraction_result_model_is_plain_string() -> None:
    # `model` must accept any provider/model string, not an enum of known models.
    result = RelationExtractionResult(
        relations=[], source_segment_id="x", model="some-arbitrary-model-id"
    )
    assert result.model == "some-arbitrary-model-id"


# ---------------------------------------------------------------------------
# Stage 2: relations, relation_evidence, relation_candidates tables (v28)
# ---------------------------------------------------------------------------


def _tables(db: StateDB) -> set[str]:
    return {
        row[0]
        for row in db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }


def _indexes(db: StateDB) -> set[str]:
    return {
        row[0]
        for row in db._conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    }


def test_fresh_db_has_relation_tables_and_indexes(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    tables = _tables(db)
    assert {"relations", "relation_evidence", "relation_candidates"}.issubset(tables)
    indexes = _indexes(db)
    assert {"idx_relations_subject", "idx_relations_object"}.issubset(indexes)


def test_current_schema_version_v28() -> None:
    from synto.state import _CURRENT_SCHEMA_VERSION

    assert _CURRENT_SCHEMA_VERSION == 28


def test_upsert_relation_dedups_case_insensitively_and_keeps_max_confidence(
    tmp_path: Path,
) -> None:
    db = StateDB(tmp_path / "state.db")

    # Same relation, different subject/object casing — must not create a second row
    # (concept_key folds case so LLM casing drift can't fork identity).
    first_id = db.upsert_relation(
        subject="Vector Clocks",
        predicate="implemented_by",
        object_="Causal Consistency",
        confidence=0.6,
        source_segment_id="doc:0-0:abc123",
        evidence_text="Vector clocks implement causal consistency.",
    )
    second_id = db.upsert_relation(
        subject="vector clocks",
        predicate="implemented_by",
        object_="causal consistency",
        confidence=0.9,
        source_segment_id="doc:1-0:def456",
        evidence_text="Another passage restating the same relation.",
    )

    assert first_id == second_id
    assert db.count_relations() == 1

    relation = db.get_relation(first_id)
    assert relation is not None
    assert relation["confidence"] == 0.9  # max of 0.6 and 0.9, not overwritten by the lower value

    evidence = db.list_relation_evidence(first_id)
    assert len(evidence) == 2


def test_list_relations_filters_by_subject_and_object(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_relation(
        subject="A",
        predicate="depends_on",
        object_="B",
        confidence=0.5,
        source_segment_id="s1",
        evidence_text="A depends on B.",
    )
    db.upsert_relation(
        subject="C",
        predicate="depends_on",
        object_="D",
        confidence=0.5,
        source_segment_id="s2",
        evidence_text="C depends on D.",
    )

    assert len(db.list_relations()) == 2
    by_subject = db.list_relations(subject="A")
    assert len(by_subject) == 1
    assert by_subject[0]["object"] == "B"
    by_object = db.list_relations(object_="D")
    assert len(by_object) == 1
    assert by_object[0]["subject"] == "C"


def test_list_relations_for_concept_orders_by_confidence(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_relation(
        subject="X",
        predicate="related_to",
        object_="Y",
        confidence=0.3,
        source_segment_id="s1",
        evidence_text="weak link",
    )
    db.upsert_relation(
        subject="Z",
        predicate="related_to",
        object_="X",
        confidence=0.9,
        source_segment_id="s2",
        evidence_text="strong link",
    )

    rows = db.list_relations_for_concept("X")
    assert [r["confidence"] for r in rows] == [0.9, 0.3]


def test_list_relation_neighbors_respects_min_confidence(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_relation(
        subject="Gradient Descent",
        predicate="depends_on",
        object_="Calculus",
        confidence=0.8,
        source_segment_id="s1",
        evidence_text="strong evidence",
    )
    db.upsert_relation(
        subject="Adam",
        predicate="related_to",
        object_="Gradient Descent",
        confidence=0.1,
        source_segment_id="s2",
        evidence_text="weak evidence",
    )

    neighbors = db.list_relation_neighbors("Gradient Descent", min_confidence=0.5)
    assert neighbors == ["Calculus"]

    all_neighbors = set(db.list_relation_neighbors("Gradient Descent", min_confidence=0.0))
    assert all_neighbors == {"Calculus", "Adam"}


def test_insert_relation_candidates_round_trip(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    candidates = [
        RelationCandidate(
            subject="Vector Clocks",
            predicate="implemented_by",
            object="Causal Consistency",
            evidence="...",
            source_segment_id="doc:0-0:abc123",
            provenance="extracted",
            confidence=0.85,
        )
    ]

    db.insert_relation_candidates(candidates, source_segment_id="doc:0-0:abc123")

    rows = db._conn.execute("SELECT * FROM relation_candidates").fetchall()
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["subject"] == "Vector Clocks"
    assert row["predicate"] == "implemented_by"
    assert row["object"] == "Causal Consistency"
    assert row["source_segment_id"] == "doc:0-0:abc123"
    assert row["confidence"] == 0.85
    assert row["created_at"]
