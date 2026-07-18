"""Tests for Feature 26: Relation Extraction."""

from __future__ import annotations

from synto.models import RelationCandidate, RelationExtractionResult

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
