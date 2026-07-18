"""Tests for Feature 03: Term Extraction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from conftest import as_endpoint

from synto.models import TermExtractionResult, TermRecord
from synto.state import StateDB


def _mock_client(response: str = "{}") -> MagicMock:
    client = MagicMock()
    client.generate.return_value = response
    return client


# ---------------------------------------------------------------------------
# Stage 1: TermExtractionResult model
# ---------------------------------------------------------------------------


def test_term_extraction_result_model() -> None:
    term = TermRecord(
        name="Gradient Descent",
        definition="An optimization algorithm.",
        aliases=["GD"],
        source_segment_id="doc:0-0:abc123",
        provenance="extracted",
        confidence=0.9,
    )
    result = TermExtractionResult(
        terms=[term],
        source_segment_id="doc:0-0:abc123",
        model="gemma4:e4b",
    )
    data = result.model_dump()
    assert data["source_segment_id"] == "doc:0-0:abc123"
    assert data["model"] == "gemma4:e4b"
    assert len(data["terms"]) == 1
    assert data["terms"][0]["name"] == "Gradient Descent"


def test_term_extraction_result_empty_terms() -> None:
    result = TermExtractionResult(terms=[], source_segment_id="x", model="m")
    assert result.model_dump_json() is not None
    assert result.terms == []


def test_term_record_confidence_validation() -> None:
    import pytest

    with pytest.raises(Exception):
        TermRecord(
            name="X",
            definition="Y",
            source_segment_id="s",
            provenance="extracted",
            confidence=1.5,  # out of range
        )


# ---------------------------------------------------------------------------
# Stage 2: concept_occurrences DB migration v13
# ---------------------------------------------------------------------------


def test_migration_v13(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    tables = {
        row[0]
        for row in db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "concept_occurrences" in tables


def test_concept_occurrences_schema(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    cols = {row[1] for row in db._conn.execute("PRAGMA table_info(concept_occurrences)").fetchall()}
    expected = {
        "id",
        "concept_name",
        "source_segment_id",
        "ordinal",
        "confidence",
        "extraction_run",
    }
    assert expected.issubset(cols)


def test_current_schema_version_v13() -> None:
    from synto.state import _CURRENT_SCHEMA_VERSION

    assert _CURRENT_SCHEMA_VERSION == 28


# ---------------------------------------------------------------------------
# Stage 3: extract_terms() function
# ---------------------------------------------------------------------------


def test_extract_terms_empty_response(tmp_path: Path, config) -> None:
    from synto.models import SourceSegment
    from synto.pipeline.ingest import extract_terms

    segment = SourceSegment(
        id="doc:0-0:abc123",
        identity="doc:0-0",
        ordinal=0,
        source_id="doc",
        structural_locator="0-0",
        content_hash="abc123",
        text="Machine learning uses gradient descent.",
    )
    client = _mock_client('{"terms": []}')
    result = extract_terms(segment, as_endpoint(client, model=config.model_name("fast")), config)
    assert isinstance(result, TermExtractionResult)
    assert result.terms == []
    assert result.source_segment_id == segment.id
    assert result.model == config.models.fast


def test_extract_terms_two_terms(tmp_path: Path, config) -> None:
    from synto.models import SourceSegment
    from synto.pipeline.ingest import extract_terms

    segment = SourceSegment(
        id="doc:1-1:def456",
        identity="doc:1-1",
        ordinal=1,
        source_id="doc",
        structural_locator="1-1",
        content_hash="def456",
        text="Backpropagation and Adam optimizer are common techniques.",
    )
    response = json.dumps(
        {
            "terms": [
                {
                    "name": "Backpropagation",
                    "definition": "Algorithm for training neural networks.",
                    "aliases": ["backprop"],
                    "provenance": "extracted",
                    "confidence": 0.95,
                },
                {
                    "name": "Adam Optimizer",
                    "definition": "Adaptive gradient descent method.",
                    "aliases": [],
                    "provenance": "extracted",
                    "confidence": 0.9,
                },
            ]
        }
    )
    client = _mock_client(response)
    result = extract_terms(segment, as_endpoint(client, model=config.model_name("fast")), config)
    assert len(result.terms) == 2
    assert result.terms[0].name == "Backpropagation"
    assert result.terms[0].source_segment_id == segment.id
    assert result.terms[1].name == "Adam Optimizer"


# ---------------------------------------------------------------------------
# Stage 4: Persist terms to DB
# ---------------------------------------------------------------------------


def test_persist_terms(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    terms = [
        TermRecord(
            name="Gradient Descent",
            definition="Optimization algo.",
            source_segment_id="seg1",
            provenance="extracted",
            confidence=0.9,
        ),
        TermRecord(
            name="Backprop",
            definition="Training algo.",
            source_segment_id="seg1",
            provenance="extracted",
            confidence=0.85,
        ),
    ]
    db.upsert_concept_occurrences(terms, "seg1")
    count = db._conn.execute("SELECT COUNT(*) FROM concept_occurrences").fetchone()[0]
    assert count == 2


def test_persist_terms_idempotent(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    term = TermRecord(
        name="DuplicateTerm",
        definition="Def.",
        source_segment_id="seg2",
        provenance="extracted",
        confidence=0.8,
    )
    db.upsert_concept_occurrences([term], "seg2")
    db.upsert_concept_occurrences([term], "seg2")  # second insert should not duplicate
    count = db._conn.execute(
        "SELECT COUNT(*) FROM concept_occurrences WHERE concept_name = 'DuplicateTerm'"
    ).fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# Stage 5: VaultReader.list_terms()
# ---------------------------------------------------------------------------


def test_vault_reader_list_terms_empty(config) -> None:
    from synto.readers import VaultReader

    reader = VaultReader(config.vault)
    terms = reader.list_terms()
    assert terms == []


def test_vault_reader_list_terms_populated(tmp_path: Path, config, db) -> None:
    from synto.readers import VaultReader

    terms = [
        TermRecord(
            name="AlphaTerm",
            definition="First term.",
            source_segment_id="seg-a",
            provenance="extracted",
            confidence=0.9,
        ),
        TermRecord(
            name="BetaTerm",
            definition="Second term.",
            source_segment_id="seg-a",
            provenance="extracted",
            confidence=0.8,
        ),
    ]
    db.upsert_concept_occurrences(terms, "seg-a")

    reader = VaultReader(config.vault)
    result = reader.list_terms()
    names = [t.name for t in result]
    assert "AlphaTerm" in names
    assert "BetaTerm" in names
