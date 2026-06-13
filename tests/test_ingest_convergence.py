"""Order-independent concept identity (v26).

The headline guarantee of the v26 classifier: a vault's concept identity is a pure function of
the accumulated claim-set, not of ingest order. These tests drive the REAL ingest pipeline with
deterministic mocked clients over every permutation of a fixed paper set and assert the final
identity structure is byte-identical — comparing by label_key (entity_ids are random), per
CLAUDE.md's non-determinism note (identity only, never LLM prose).
"""

from __future__ import annotations

import json
from itertools import permutations
from unittest.mock import MagicMock

import pytest
from conftest import as_router

from synto.concept_text import concept_key as _ck
from synto.config import Config
from synto.pipeline.ingest import ingest_note
from synto.state import StateDB


@pytest.fixture
def vault(tmp_path):
    for sub in ("raw", "wiki", "wiki/.drafts", "wiki/sources", ".synto"):
        (tmp_path / sub).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def config(vault):
    return Config(vault=vault)


def _paper_json(concepts: list[dict]) -> str:
    """Analysis JSON for one paper. ``concepts`` is a list of {name, aliases}."""
    return json.dumps(
        {
            "summary": "A summary.",
            "concepts": [{"name": c["name"], "aliases": c.get("aliases", [])} for c in concepts],
            "suggested_topics": [],
            "named_references": [],
            "quality": "high",
        }
    )


def _client(analysis_json: str, config: Config):
    client = MagicMock()
    client.generate.return_value = analysis_json
    return as_router(client, config)


def _ingest_papers(config: Config, db: StateDB, papers: dict[str, dict], order: tuple[str, ...]):
    """Ingest the given papers in ``order`` into a fresh DB."""
    for fname in order:
        spec = papers[fname]
        path = config.vault / "raw" / fname
        path.write_text(spec["body"], encoding="utf-8")
        ingest_note(path, config, _client(_paper_json(spec["concepts"]), config), db)


def _identity_snapshot(db: StateDB) -> dict:
    """Capture concept identity by label_key (entity_ids are random and excluded).

    Returns: preferred label_keys, the alias label_key set per preferred concept, and the
    merge-candidate set as ordered (label_key, label_key, surface_key) tuples.
    """
    rows = db._conn.execute(
        """SELECT cl.entity_id, cl.label, cl.label_key, cl.role
           FROM concept_labels cl
           JOIN concept_entities ce ON ce.id = cl.entity_id AND ce.status = 'active'"""
    ).fetchall()
    preferred_by_entity: dict[str, str] = {}
    aliases_by_entity: dict[str, set[str]] = {}
    for entity_id, _label, label_key, role in rows:
        if role == "preferred":
            preferred_by_entity[entity_id] = label_key
        else:
            aliases_by_entity.setdefault(entity_id, set()).add(label_key)

    preferred = set(preferred_by_entity.values())
    alias_map = {
        preferred_by_entity[eid]: tuple(sorted(aliases_by_entity.get(eid, set())))
        for eid in preferred_by_entity
    }
    candidates = sorted(
        (_ck(c["label_a"]), _ck(c["label_b"]), _ck(c["surface"]))
        for c in db.list_merge_candidates()
    )
    return {"preferred": preferred, "aliases": alias_map, "candidates": candidates}


def _converges(config: Config, papers: dict[str, dict]) -> dict:
    """Assert every ingest-order permutation yields the SAME identity; return that snapshot."""
    snapshots = []
    for order in permutations(papers):
        db = StateDB(config.state_db_path)
        try:
            _ingest_papers(config, db, papers, order)
            snapshots.append(_identity_snapshot(db))
        finally:
            db.close()
        # Reset for the next permutation.
        config.state_db_path.unlink()
        for f in (config.vault / "raw").glob("*.md"):
            f.unlink()
    first = snapshots[0]
    for snap, order in zip(snapshots[1:], list(permutations(papers))[1:]):
        assert snap == first, f"identity diverged for order {order}: {snap} != {first}"
    return first


def test_two_host_block_is_order_independent(config):
    # The headline bug: "Agentic ROI Framework" is a weak alias on TWO concepts, then a paper
    # extracts it as a concept. Old behavior: ambiguous → never minted (blocked). New: it mints
    # its own entity in EVERY order, with a merge candidate against each host.
    papers = {
        "h1.md": {
            "body": "Knowledge Compounding underlies the Agentic ROI Framework approach. "
            "Knowledge Compounding is central. Agentic ROI Framework recurs here.",
            "concepts": [{"name": "Knowledge Compounding", "aliases": ["Agentic ROI Framework"]}],
        },
        "h2.md": {
            "body": "Value Accretion is shaped by the Agentic ROI Framework lens. "
            "Value Accretion matters. Agentic ROI Framework appears again.",
            "concepts": [{"name": "Value Accretion", "aliases": ["Agentic ROI Framework"]}],
        },
        "arf.md": {
            "body": "The Agentic ROI Framework is a method in its own right. "
            "Agentic ROI Framework deserves its own treatment. Agentic ROI Framework.",
            "concepts": [{"name": "Agentic ROI Framework", "aliases": []}],
        },
    }
    snap = _converges(config, papers)
    # ARF is minted as its own concept regardless of order.
    assert _ck("Agentic ROI Framework") in snap["preferred"]
    assert _ck("Knowledge Compounding") in snap["preferred"]
    assert _ck("Value Accretion") in snap["preferred"]
    # Two merge candidates surface the promotion (one per former host), each via the ARF surface.
    surfaces = {c[2] for c in snap["candidates"]}
    assert surfaces == {_ck("Agentic ROI Framework")}
    assert len(snap["candidates"]) == 2


def test_abbreviation_alias_promotion_is_order_independent(config):
    # The "GD" case: a paper extracts "Gradient Descent" with alias "GD"; another extracts "GD"
    # as a concept. In every order the result is two concepts plus one merge candidate — never a
    # silent absorb and never order-dependent.
    papers = {
        "grad.md": {
            "body": "Gradient Descent optimizes the loss. Gradient Descent is iterative. "
            "GD is the shorthand used throughout. GD GD.",
            "concepts": [{"name": "Gradient Descent", "aliases": ["GD"]}],
        },
        "gd.md": {
            "body": "GD is presented here as a standalone idea. GD has its own page. GD GD GD.",
            "concepts": [{"name": "GD", "aliases": []}],
        },
    }
    snap = _converges(config, papers)
    assert _ck("Gradient Descent") in snap["preferred"]
    assert _ck("GD") in snap["preferred"]
    assert len(snap["candidates"]) == 1
    assert snap["candidates"][0][2] == _ck("GD")


def test_reingest_is_idempotent(config):
    # Re-ingesting the same paper must not flap identity: ingesting twice == ingesting once
    # (no new entities, no alias churn, no duplicate merge candidates). This is the real-world
    # failure mode the permutation sweep alone does not exercise.
    papers = {
        "grad.md": {
            "body": "Gradient Descent optimizes the loss. Gradient Descent is iterative. "
            "GD is the shorthand. GD GD.",
            "concepts": [{"name": "Gradient Descent", "aliases": ["GD"]}],
        },
        "gd.md": {
            "body": "GD is a standalone idea here. GD has its own page. GD GD GD.",
            "concepts": [{"name": "GD", "aliases": []}],
        },
    }
    db = StateDB(config.state_db_path)
    try:
        _ingest_papers(config, db, papers, ("grad.md", "gd.md"))
        once = _identity_snapshot(db)
        # Re-ingest both with force (same content, content-hash skip would otherwise no-op).
        for fname in ("grad.md", "gd.md"):
            path = config.vault / "raw" / fname
            ingest_note(
                path,
                config,
                _client(_paper_json(papers[fname]["concepts"]), config),
                db,
                force=True,
            )
        twice = _identity_snapshot(db)
    finally:
        db.close()
    assert twice == once
