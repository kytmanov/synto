"""Tests for structure-aware segment chunking + concept→segment attribution.

Covers `_build_segment_units` (segment-aligned analysis chunking) and
`_persist_concept_occurrences` (mapping a chunk's extracted concepts to the segments
that fed it, canonicalized), which together make `get_source_passages` functional on
pipeline-built vaults. Offline — tmp_path SQLite, no LLM.
"""

from __future__ import annotations

from pathlib import Path

from synto.pipeline.ingest import _build_segment_units, _persist_concept_occurrences
from synto.state import StateDB


def _seg(seg_id: str, text: str) -> dict:
    return {"id": seg_id, "text": text}


def _insert_segment(db: StateDB, seg_id: str, source_id: str, text: str = "x") -> None:
    db._conn.execute(
        """INSERT OR IGNORE INTO source_segments
           (id, identity, ordinal, source_id, structural_locator, content_hash, text)
           VALUES (?, ?, 0, ?, '', '', ?)""",
        (seg_id, seg_id, source_id, text),
    )
    db._conn.commit()


# ── _build_segment_units ──────────────────────────────────────────────────────


def test_build_segment_units_packs_without_splitting() -> None:
    """Whole segments pack up to chunk_size; a segment is never split, even when its
    own text contains a '##' heading (which a naive body regex would wrongly split on)."""
    segs = [
        _seg("s1", "alpha " * 10),  # ~60 chars
        _seg("s2", "## not a boundary\nbeta " * 5),  # contains '##' inside the segment
        _seg("s3", "gamma " * 10),
    ]
    units = _build_segment_units(segs, chunk_size=120)
    # Every segment id appears exactly once across units, in order, never split.
    ids = [sid for _t, sids in units for sid in sids]
    assert ids == ["s1", "s2", "s3"]
    # Each unit's text is the join of its segments' full texts (no truncation/split).
    for text, sids in units:
        for sid in sids:
            seg_text = next(s["text"] for s in segs if s["id"] == sid)
            assert seg_text in text


def test_build_segment_units_oversized_segment_is_own_unit() -> None:
    """A single segment larger than chunk_size becomes its own unit rather than being cut."""
    segs = [_seg("big", "x" * 500), _seg("small", "y" * 10)]
    units = _build_segment_units(segs, chunk_size=100)
    assert units[0][1] == ["big"]
    assert "x" * 500 in units[0][0]  # full text retained, not truncated


# ── _persist_concept_occurrences ──────────────────────────────────────────────


def _setup_source(db: StateDB) -> None:
    _insert_segment(db, "seg-a", "src1")
    _insert_segment(db, "seg-b", "src1")
    db.replace_concepts_for_source("raw/src1.md", ["Capital Goods", "Knowledge Compounding"])
    db.upsert_aliases("Capital Goods", ["capital goods reclassification"])


def test_persist_attributes_concepts_to_segments_and_canonicalizes(tmp_path: Path) -> None:
    """Each chunk's concepts attach to its segments; raw names resolve to canonical;
    concepts not in the final canonical set are dropped."""
    db = StateDB(tmp_path / "state.db")
    _setup_source(db)
    units = [("textA", ["seg-a"]), ("textB", ["seg-b"])]
    attribution = {
        0: ["capital goods reclassification"],  # alias → canonical "Capital Goods"
        1: ["Knowledge Compounding", "Noise Concept"],  # Noise not in canonical set → dropped
    }
    _persist_concept_occurrences(
        db, "src1", units, attribution, ["Capital Goods", "Knowledge Compounding"]
    )
    rows = {(r["concept_name"], r["source_segment_id"]) for r in db.list_concept_occurrences()}
    assert ("Capital Goods", "seg-a") in rows
    assert ("Knowledge Compounding", "seg-b") in rows
    assert not any(c == "Noise Concept" for c, _ in rows)


def test_persist_replaces_on_reingest(tmp_path: Path) -> None:
    """Re-running attribution for a source clears its prior links (no stale/dup rows)."""
    db = StateDB(tmp_path / "state.db")
    _setup_source(db)
    units = [("t", ["seg-a", "seg-b"])]
    _persist_concept_occurrences(db, "src1", units, {0: ["Capital Goods"]}, ["Capital Goods"])
    first = {(r["concept_name"], r["source_segment_id"]) for r in db.list_concept_occurrences()}
    assert first == {("Capital Goods", "seg-a"), ("Capital Goods", "seg-b")}

    # Re-ingest now attributes only Knowledge Compounding — the old Capital Goods links go.
    _persist_concept_occurrences(
        db, "src1", units, {0: ["Knowledge Compounding"]}, ["Knowledge Compounding"]
    )
    second = {(r["concept_name"], r["source_segment_id"]) for r in db.list_concept_occurrences()}
    assert second == {("Knowledge Compounding", "seg-a"), ("Knowledge Compounding", "seg-b")}
