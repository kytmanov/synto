"""Tests for Feature 26: Relation Extraction."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from conftest import as_endpoint, as_router, make_mock_client

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


# ---------------------------------------------------------------------------
# Stage 3: extract_relations() function
# ---------------------------------------------------------------------------


def _segment(text: str = "Vector clocks implement causal consistency."):
    from synto.models import SourceSegment

    return SourceSegment(
        id="doc:0-0:abc123",
        identity="doc:0-0",
        ordinal=0,
        source_id="doc",
        structural_locator="0-0",
        content_hash="abc123",
        text=text,
    )


def test_extract_relations_empty_response(config) -> None:
    from synto.pipeline.ingest import extract_relations

    segment = _segment()
    client = make_mock_client('{"relations": []}')
    result = extract_relations(
        segment,
        ["Vector Clocks", "Causal Consistency"],
        as_endpoint(client, model=config.model_name("fast")),
        config,
    )
    assert isinstance(result, RelationExtractionResult)
    assert result.relations == []
    assert result.source_segment_id == segment.id
    assert result.model == config.model_name("fast")


def test_extract_relations_two_valid(config) -> None:
    from synto.pipeline.ingest import extract_relations

    segment = _segment()
    response = json.dumps(
        {
            "relations": [
                {
                    "subject": "Vector Clocks",
                    "predicate": "implemented_by",
                    "object": "Causal Consistency",
                    "evidence": "Vector clocks implement causal consistency.",
                    "confidence": 0.9,
                },
                {
                    "subject": "Causal Consistency",
                    "predicate": "depends_on",
                    "object": "Vector Clocks",
                    "evidence": "Causal consistency depends on vector clocks.",
                    "confidence": 0.7,
                },
            ]
        }
    )
    client = make_mock_client(response)
    result = extract_relations(
        segment,
        ["Vector Clocks", "Causal Consistency"],
        as_endpoint(client, model=config.model_name("fast")),
        config,
    )
    assert len(result.relations) == 2
    first = result.relations[0]
    assert isinstance(first, RelationCandidate)
    assert first.subject == "Vector Clocks"
    assert first.predicate == "implemented_by"
    assert first.object == "Causal Consistency"
    assert first.source_segment_id == segment.id
    assert result.relations[1].predicate == "depends_on"


def test_extract_relations_drops_invalid_predicate(config) -> None:
    from synto.pipeline.ingest import extract_relations

    segment = _segment()
    response = json.dumps(
        {
            "relations": [
                {
                    "subject": "Vector Clocks",
                    "predicate": "implemented_by",
                    "object": "Causal Consistency",
                    "evidence": "valid one",
                    "confidence": 0.8,
                },
                {
                    "subject": "Vector Clocks",
                    "predicate": "related_to_somehow",
                    "object": "Causal Consistency",
                    "evidence": "bad predicate",
                    "confidence": 0.5,
                },
            ]
        }
    )
    client = make_mock_client(response)
    result = extract_relations(
        segment,
        ["Vector Clocks", "Causal Consistency"],
        as_endpoint(client, model=config.model_name("fast")),
        config,
    )
    assert len(result.relations) == 1
    assert result.relations[0].predicate == "implemented_by"


def test_extract_relations_clamps_confidence(config) -> None:
    from synto.pipeline.ingest import extract_relations

    segment = _segment()
    response = json.dumps(
        {
            "relations": [
                {
                    "subject": "Vector Clocks",
                    "predicate": "implemented_by",
                    "object": "Causal Consistency",
                    "evidence": "overconfident",
                    "confidence": 1.7,
                }
            ]
        }
    )
    client = make_mock_client(response)
    result = extract_relations(
        segment,
        ["Vector Clocks", "Causal Consistency"],
        as_endpoint(client, model=config.model_name("fast")),
        config,
    )
    assert result.relations[0].confidence == 1.0


def test_extract_relations_empty_concepts_skips_llm_call(config) -> None:
    from synto.pipeline.ingest import extract_relations

    segment = _segment()
    client = make_mock_client('{"relations": []}')
    result = extract_relations(
        segment,
        [],
        as_endpoint(client, model=config.model_name("fast")),
        config,
    )
    assert result.relations == []
    assert result.source_segment_id == segment.id
    assert result.model == config.model_name("fast")
    client.generate.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 4: config flag + persistence + ingest wiring
# ---------------------------------------------------------------------------


def _analysis_response(concepts: list[str]) -> str:
    return json.dumps(
        {
            "summary": "A summary.",
            "concepts": [{"name": c, "aliases": []} for c in concepts],
            "suggested_topics": [],
            "named_references": [],
            "quality": "high",
        }
    )


def test_extract_and_persist_relations_dedups_across_segments(tmp_path: Path, config) -> None:
    """Two segments restating the same relation must collapse to one relations row (upsert_relation
    dedup), while both raw LLM candidates and both evidence rows are still kept for provenance."""
    from synto.pipeline.ingest import _extract_and_persist_relations

    db = StateDB(tmp_path / "state.db")
    segments = [
        SimpleNamespace(id="seg-1", text="Vector clocks implement causal consistency."),
        SimpleNamespace(id="seg-2", text="Restated: vector clocks implement causal consistency."),
    ]
    response_1 = json.dumps(
        {
            "relations": [
                {
                    "subject": "Vector Clocks",
                    "predicate": "implemented_by",
                    "object": "Causal Consistency",
                    "evidence": "Vector clocks implement causal consistency.",
                    "confidence": 0.6,
                }
            ]
        }
    )
    response_2 = json.dumps(
        {
            "relations": [
                {
                    "subject": "vector clocks",
                    "predicate": "implemented_by",
                    "object": "causal consistency",
                    "evidence": "Restated relation.",
                    "confidence": 0.9,
                }
            ]
        }
    )
    client = make_mock_client()
    client.generate.side_effect = [response_1, response_2]
    fast = as_endpoint(client, model=config.model_name("fast"))

    n = _extract_and_persist_relations(
        db, segments, ["Vector Clocks", "Causal Consistency"], fast, config
    )

    assert n == 2  # two relations upserted (one per segment call), even though they dedup
    relations = db.list_relations()
    assert len(relations) == 1
    assert relations[0]["confidence"] == 0.9  # max of 0.6 and 0.9

    evidence = db.list_relation_evidence(relations[0]["id"])
    assert len(evidence) == 2

    raw_rows = db._conn.execute("SELECT * FROM relation_candidates").fetchall()
    assert len(raw_rows) == 2


def test_extract_and_persist_relations_isolates_segment_failures(tmp_path: Path, config) -> None:
    """A StructuredOutputError on one segment's relation call must not skip the remaining
    segments — best-effort per segment, matching the "log + continue" convention used
    elsewhere in ingest for non-fatal LLM failures."""
    from synto.pipeline.ingest import _extract_and_persist_relations
    from synto.structured_output import StructuredOutputError

    db = StateDB(tmp_path / "state.db")
    segments = [
        SimpleNamespace(id="seg-1", text="Alpha depends on Beta."),
        SimpleNamespace(id="seg-2", text="Gamma depends on Delta."),
    ]
    response_2 = json.dumps(
        {
            "relations": [
                {
                    "subject": "Gamma",
                    "predicate": "depends_on",
                    "object": "Delta",
                    "evidence": "Gamma depends on Delta.",
                    "confidence": 0.7,
                }
            ]
        }
    )
    client = make_mock_client()
    client.generate.side_effect = [StructuredOutputError("boom"), response_2]
    fast = as_endpoint(client, model=config.model_name("fast"))

    n = _extract_and_persist_relations(db, segments, ["Gamma", "Delta"], fast, config)

    assert n == 1  # seg-1 failed and was skipped; seg-2 still persisted
    relations = db.list_relations()
    assert len(relations) == 1
    assert relations[0]["subject"] == "Gamma"
    assert relations[0]["source_segment_id"] == "seg-2"


def test_ingest_note_relation_extraction_off_by_default(vault, config, db) -> None:
    """The flag guard in ingest_note must skip the whole relation-extraction pass when
    pipeline.relation_extraction is False (its default): no relations rows AND no extra
    LLM call beyond the analysis pass. (Asserting count_relations()==0 alone would also
    pass if the guard were deleted, since the broad except around the extraction pass
    would swallow the resulting failure on the analysis-shaped mock response.)"""
    from synto.pipeline.ingest import ingest_note

    assert config.pipeline.relation_extraction is False
    path = vault / "raw" / "note.md"
    path.write_text("# Note\n\nAlpha depends on Beta.", encoding="utf-8")
    client = as_router(make_mock_client(_analysis_response(["Alpha", "Beta"])), config)

    ingest_note(path, config, client, db)

    assert client.generate.call_count == 1  # analysis pass only, no relation-extraction call
    assert db.count_relations() == 0


def test_ingest_note_relation_extraction_pseudo_segment_for_plain_note(vault, config, db) -> None:
    """Plain notes (no tracked source_segments) must still get relation extraction with a
    traceable pseudo-segment id, so `trace relation` can point back at note+chunk."""
    from synto.pipeline.ingest import ingest_note

    config.pipeline.relation_extraction = True
    path = vault / "raw" / "note.md"
    path.write_text("# Note\n\nAlpha depends on Beta.", encoding="utf-8")
    relation_response = json.dumps(
        {
            "relations": [
                {
                    "subject": "Alpha",
                    "predicate": "depends_on",
                    "object": "Beta",
                    "evidence": "Alpha depends on Beta.",
                    "confidence": 0.8,
                }
            ]
        }
    )
    client = as_router(MagicMock())
    client.generate.side_effect = [_analysis_response(["Alpha", "Beta"]), relation_response]

    ingest_note(path, config, client, db)

    assert db.count_relations() == 1
    relation = db.list_relations()[0]
    assert relation["source_segment_id"].startswith("note:")
