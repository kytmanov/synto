"""Tests for Phase 4 of Feature 45: merge/split/homonym machinery.

Covers:
  - merge_entities: source edges moved, loser retired, labels absorbed, identity log entry
  - merge_entities: blocked concept propagation
  - merge_entities: raises on unknown or self-merge
  - split_entity: two senses minted, compile state seeded, original retired, log entry
  - split_entity: raises on unclaimed sources, fewer than 2 senses
  - merge_concepts vault choreography: article retired, links rewritten (dry-run safe)
  - split_concept vault choreography: stub articles created, disambiguation stub
  - suggest_concept_merges: match_key collision signal
  - suggest_concept_splits: bimodal source heuristic
  - find_match_key_collisions: shared match_key between active entities
  - list_identity_log: records merge and split ops
  - resolve_ambiguous_occurrences: marks rows resolved and returns count
  - lint identity checks: label_collision + orphan_entity issues
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from synto.models import WikiArticleRecord
from synto.state import StateDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _article(db: StateDB, vault: Path, name: str, body: str = "body") -> WikiArticleRecord:
    """Write an article stub to disk and register it in the DB."""
    wiki = vault / "wiki"
    wiki.mkdir(parents=True, exist_ok=True)
    path = wiki / f"{name}.md"
    path.write_text(f"---\ntitle: {name}\n---\n{body}")
    art = WikiArticleRecord(
        path=f"wiki/{name}.md",
        title=name,
        sources=[],
        content_hash="",
        status="published",
    )
    db.upsert_article(art)
    return art


# ---------------------------------------------------------------------------
# merge_entities — DB-level
# ---------------------------------------------------------------------------


def test_merge_entities_moves_source_edges(tmp_path: Path) -> None:
    """After merge, all loser source rows now reference the winner name."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["ML"])
    db.upsert_concepts("raw/b.md", ["Machine Learning"])

    db.merge_entities("Machine Learning", "ML")

    rows = db._conn.execute("SELECT name FROM concepts WHERE source_path='raw/a.md'").fetchall()
    assert rows, "source edge should still exist"
    assert all(r[0] == "Machine Learning" for r in rows)


def test_merge_entities_collapses_shared_source_edge(tmp_path: Path) -> None:
    """When winner and loser both cite one source, the edge collapses to a single row.

    The entity_id move is UPDATE OR IGNORE + delete-leftover, so the shared (winner, src)
    PK does not duplicate or raise — it merges to one edge and one compile-state row.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/shared.md", ["A"])
    db.upsert_concepts("raw/shared.md", ["B"])
    db.upsert_concepts("raw/b_only.md", ["B"])

    db.merge_entities("B", "A")
    winner_id = db.entity_id_for_name("B")

    assert set(db.get_sources_for_concept("B")) == {"raw/shared.md", "raw/b_only.md"}
    # No duplicate concept rows for the shared source.
    crows = db._conn.execute(
        "SELECT entity_id FROM concepts WHERE source_path='raw/shared.md'"
    ).fetchall()
    assert [r[0] for r in crows] == [winner_id]
    # Compile-state likewise collapsed onto the winner, no orphan loser rows.
    ccs = db._conn.execute(
        "SELECT DISTINCT entity_id FROM concept_compile_state WHERE source_path='raw/shared.md'"
    ).fetchall()
    assert [r[0] for r in ccs] == [winner_id]


def test_rename_concept_moves_no_db_identity_keys(tmp_path: Path) -> None:
    """rename is a relabel: the entity_id PK never moves, only labels + name caches.

    Architecture X end-state check — identity does not flow through the name string.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Old Name"])
    eid = db.entity_id_for_name("Old Name")

    db.rename_concept("Old Name", "New Name")

    # Same entity, no new entity minted.
    assert db.entity_id_for_name("New Name") == eid
    assert db.entity_id_for_name("Old Name") is None or db.entity_id_for_name("Old Name") == eid
    entity_count = db._conn.execute(
        "SELECT COUNT(*) FROM concept_entities WHERE status='active'"
    ).fetchone()[0]
    assert entity_count == 1
    # Identity key unchanged on concepts and compile_state; only the name cache moved.
    crow = db._conn.execute(
        "SELECT entity_id, name FROM concepts WHERE source_path='raw/a.md'"
    ).fetchone()
    assert crow["entity_id"] == eid and crow["name"] == "New Name"
    ccs = db._conn.execute(
        "SELECT entity_id, concept_name FROM concept_compile_state WHERE source_path='raw/a.md'"
    ).fetchone()
    assert ccs["entity_id"] == eid and ccs["concept_name"] == "New Name"


def test_merge_entities_absorbs_loser_preferred_as_alias(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["ML"])
    db.upsert_concepts("raw/b.md", ["Machine Learning"])

    result = db.merge_entities("Machine Learning", "ML")

    assert "ML" in result["labels_absorbed"]
    # "ML" alias now lives on the Machine Learning entity.
    aliases = db.aliases_for_concept("Machine Learning")
    assert "ML" in aliases


def test_merge_entities_retires_loser_entity(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["A"])
    db.upsert_concepts("raw/b.md", ["B"])
    loser_id = db.entity_id_for_name("A")

    db.merge_entities("B", "A")

    row = db._conn.execute(
        "SELECT status, merged_into FROM concept_entities WHERE id=?", (loser_id,)
    ).fetchone()
    assert row[0] == "merged"
    assert row[1] == db.entity_id_for_name("B")


def test_merge_entities_propagates_blocked_flag(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["LoserConcept"])
    db.upsert_concepts("raw/b.md", ["WinnerConcept"])

    now = "2026-01-01T00:00:00"
    db._conn.execute(
        "INSERT OR IGNORE INTO blocked_concepts (concept, blocked_at) VALUES (?,?)",
        ("LoserConcept", now),
    )
    db._conn.connection.commit() if hasattr(db._conn, "connection") else None

    db.merge_entities("WinnerConcept", "LoserConcept")

    blocked = db._conn.execute(
        "SELECT 1 FROM blocked_concepts WHERE lower(concept)='winnerconcept'"
    ).fetchone()
    assert blocked is not None, "winner should inherit loser's blocked flag"


def test_merge_entities_raises_on_unknown_loser(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Exists"])

    with pytest.raises(ValueError, match="not found"):
        db.merge_entities("Exists", "DoesNotExist")


def test_merge_entities_raises_on_self_merge(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])

    with pytest.raises(ValueError, match="same entity"):
        db.merge_entities("Alpha", "Alpha")


def test_merge_entities_logs_to_identity_log(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["A"])
    db.upsert_concepts("raw/b.md", ["B"])

    db.merge_entities("B", "A")

    entries = db.list_identity_log()
    assert len(entries) >= 1
    assert entries[0]["op"] == "merge"


def test_merge_entities_collapses_shared_segment_occurrence(tmp_path: Path) -> None:
    """Occurrences for both concepts in one segment must collapse, not raise.

    concept_occurrences has UNIQUE(concept_name, source_segment_id). A plain rename of the
    loser onto the winner would duplicate (winner, seg) and raise IntegrityError, aborting
    the whole merge. The move must OR IGNORE + drop the leftover loser row instead.
    """
    from collections import namedtuple

    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])

    occ = namedtuple("occ", ["name", "confidence"])
    # Both concepts cited in the same segment — the collision trigger.
    db.upsert_concept_occurrences([occ("Alpha", 1.0), occ("Beta", 1.0)], "seg-1")

    # Pre-fix this raises sqlite3.IntegrityError and rolls the merge back.
    db.merge_entities("Alpha", "Beta")

    rows = db._conn.execute(
        "SELECT concept_name FROM concept_occurrences WHERE source_segment_id='seg-1'"
    ).fetchall()
    assert [r[0] for r in rows] == ["Alpha"], "shared-segment occurrence should collapse to winner"
    # Loser entity retired, so no occurrence row should still carry the loser name.
    assert (
        db._conn.execute(
            "SELECT COUNT(*) FROM concept_occurrences WHERE lower(concept_name)='beta'"
        ).fetchone()[0]
        == 0
    )


# ---------------------------------------------------------------------------
# unmerge_entities — DB-level (reversibility)
# ---------------------------------------------------------------------------


def test_unmerge_restores_loser_identity(tmp_path: Path) -> None:
    """merge A→B then unmerge A round-trips: A is active again, owns its source, loses
    nothing of B's, and B no longer claims A's label."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    loser_id = db.entity_id_for_name("Alpha")
    winner_id = db.entity_id_for_name("Beta")

    db.merge_entities("Beta", "Alpha")
    # Sanity: Alpha is retired and resolves to Beta now.
    assert db.entity_id_for_name("Alpha") == winner_id

    # The loser must be found via get_merged_entity_id, NOT entity_id_for_name.
    assert db.get_merged_entity_id("Alpha") == loser_id

    result = db.unmerge_entities(loser_id)
    assert result["loser"] == "Alpha"
    assert result["winner"] == "Beta"

    # Alpha is active again with no merge pointer.
    status, merged_into = db._conn.execute(
        "SELECT status, merged_into FROM concept_entities WHERE id=?", (loser_id,)
    ).fetchone()
    assert status == "active"
    assert merged_into is None

    # Alpha owns its source edge again; Beta keeps its own.
    assert db.get_sources_for_concept("Alpha") == ["raw/a.md"]
    assert db.get_sources_for_concept("Beta") == ["raw/b.md"]

    # Beta no longer carries "Alpha" as an alias.
    beta_aliases = {a.lower() for a in db.get_aliases("Beta")}
    assert "alpha" not in beta_aliases

    # Alpha is reseeded pending so the next compile regenerates it.
    pending = db._conn.execute(
        "SELECT status FROM concept_compile_state WHERE entity_id=? AND source_path='raw/a.md'",
        (loser_id,),
    ).fetchone()
    assert pending is not None and pending[0] == "pending"


def test_unmerge_logs_identity_op(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    loser_id = db.entity_id_for_name("Alpha")
    db.merge_entities("Beta", "Alpha")

    db.unmerge_entities(loser_id)

    assert db.list_identity_log()[0]["op"] == "unmerge"


def test_unmerge_raises_without_a_merge(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    alpha_id = db.entity_id_for_name("Alpha")

    with pytest.raises(ValueError, match="No merge to reverse"):
        db.unmerge_entities(alpha_id)


def test_unmerge_concept_recreates_stub(tmp_path: Path) -> None:
    """The maintain wrapper resolves the merged loser and recreates its page."""
    from synto.config import Config
    from synto.pipeline.maintain import unmerge_concept

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    _article(db, vault, "Alpha")
    _article(db, vault, "Beta")
    config = Config.model_validate({"vault": str(vault)})

    db.merge_entities("Beta", "Alpha")
    # Merge retires the loser file; simulate that so the stub recreation path runs.
    alpha_file = vault / "wiki" / "Alpha.md"
    if alpha_file.exists():
        alpha_file.unlink()

    report = unmerge_concept(config, db, "Alpha")

    assert report.loser == "Alpha"
    assert report.winner == "Beta"
    assert (vault / "wiki" / "Alpha.md").exists()


# ---------------------------------------------------------------------------
# split_entity — DB-level
# ---------------------------------------------------------------------------


def test_split_entity_mints_two_senses(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/planets.md", ["Mercury"])
    db.upsert_concepts("raw/chemistry.md", ["Mercury"])

    result = db.split_entity(
        "Mercury",
        [
            {"name": "Mercury (planet)", "sources": ["raw/planets.md"]},
            {"name": "Mercury (element)", "sources": ["raw/chemistry.md"]},
        ],
    )

    assert result["original"] == "Mercury"
    assert len(result["senses"]) == 2
    sense_names = {s["name"] for s in result["senses"]}
    assert "Mercury (planet)" in sense_names
    assert "Mercury (element)" in sense_names
    assert result["stub_needed"] is True


def test_split_entity_retires_original(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Mercury"])
    db.upsert_concepts("raw/b.md", ["Mercury"])
    orig_id = db.entity_id_for_name("Mercury")

    db.split_entity(
        "Mercury",
        [
            {"name": "Mercury (planet)", "sources": ["raw/a.md"]},
            {"name": "Mercury (element)", "sources": ["raw/b.md"]},
        ],
    )

    row = db._conn.execute("SELECT status FROM concept_entities WHERE id=?", (orig_id,)).fetchone()
    assert row[0] == "merged"


def test_split_entity_seeds_compile_state(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Mercury"])
    db.upsert_concepts("raw/b.md", ["Mercury"])

    db.split_entity(
        "Mercury",
        [
            {"name": "Mercury (planet)", "sources": ["raw/a.md"]},
            {"name": "Mercury (element)", "sources": ["raw/b.md"]},
        ],
    )

    pending = db._conn.execute(
        "SELECT concept_name, source_path FROM concept_compile_state WHERE status='pending'"
    ).fetchall()
    by_concept: dict[str, list[str]] = {}
    for name, src in pending:
        by_concept.setdefault(name, []).append(src)

    assert "raw/a.md" in by_concept.get("Mercury (planet)", [])
    assert "raw/b.md" in by_concept.get("Mercury (element)", [])


def test_rebuild_restores_entity_ids_and_log_losslessly(tmp_path: Path) -> None:
    """Durability (decision 13): a fresh DB restored from a seed keeps original entity ids.

    Deleting state.db but keeping the INDEX.json seed must rebuild without re-minting ids,
    so the merge/split history in the identity log still references real entities.
    """
    db = StateDB(tmp_path / "s1.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    beta = db.entity_id_for_name("Beta")
    db.merge_entities("Beta", "Alpha")
    log_before = db.list_identity_log()
    assert log_before and log_before[0]["op"] == "merge"

    # Seed as INDEX.json would carry it: active entity Beta (Alpha merged away).
    db2 = StateDB(tmp_path / "s2.db")
    assert db2.restore_entities_from_seed([("Beta", beta)]) == 1
    assert db2.entity_id_for_name("Beta") == beta  # original id, not re-minted
    db2.restore_identity_log(log_before)
    assert [e["op"] for e in db2.list_identity_log()] == [e["op"] for e in log_before]


def test_restore_identity_from_index_end_to_end(tmp_path: Path) -> None:
    """The rebuild hook reads INDEX.json and restores entity ids + log into a fresh DB."""
    import json

    from synto.config import Config
    from synto.pipeline.ingest import _restore_identity_from_index

    vault = tmp_path / "v"
    (vault / ".synto").mkdir(parents=True)
    config = Config.model_validate({"vault": str(vault)})
    index = {
        "source_concepts": [
            {
                "source_path": "raw/a.md",
                "content_hash": "h",
                "concepts": [{"name": "Alpha", "entity_id": "EIDALPHA0000000000000000AA"}],
            }
        ],
        "articles": [
            {"name": "Beta", "entity_id": "EIDBETA00000000000000000BB", "path": "wiki/Beta.md"}
        ],
        "identity_log": [
            {
                "op": "merge",
                "ts": "2026-01-01T00:00:00",
                "entity_ids": ["EIDBETA00000000000000000BB"],
                "labels": {},
            }
        ],
    }
    (config.app_dir / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")

    db = StateDB(config.app_dir / "state.db")
    _restore_identity_from_index(config, db)

    assert db.entity_id_for_name("Alpha") == "EIDALPHA0000000000000000AA"
    assert db.entity_id_for_name("Beta") == "EIDBETA00000000000000000BB"
    assert any(e["op"] == "merge" for e in db.list_identity_log())


def test_restore_is_noop_when_state_db_has_entities(tmp_path: Path) -> None:
    """Precedence: state.db wins — a seed never overwrites a live DB's identities."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    alpha = db.entity_id_for_name("Alpha")

    assert db.restore_entities_from_seed([("Alpha", "fake-divergent-id")]) == 0
    assert db.entity_id_for_name("Alpha") == alpha


def test_doctor_reconcile_restores_from_seed(tmp_path: Path) -> None:
    """doctor --reconcile restores identity into an empty DB from the committed seed."""
    import json

    from synto.cli import _render_identity_section
    from synto.config import Config

    vault = tmp_path / "v"
    (vault / ".synto").mkdir(parents=True)
    config = Config.model_validate({"vault": str(vault)})
    index = {
        "source_concepts": [
            {
                "source_path": "raw/a.md",
                "content_hash": "h",
                "concepts": [{"name": "Alpha", "entity_id": "EIDALPHA0000000000000000AA"}],
            }
        ],
        "articles": [],
        "identity_log": [],
    }
    (config.app_dir / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")
    db = StateDB(config.app_dir / "state.db")

    _render_identity_section(config, db, reconcile=True)

    assert db.entity_id_for_name("Alpha") == "EIDALPHA0000000000000000AA"


def test_article_entity_binding_routes_publish_to_correct_homonym(tmp_path: Path) -> None:
    """v24: verifying one homonym's article marks only that entity's compile state.

    Two homonyms cite one source. With title-keyed recovery the bare label would be
    ambiguous; the article's stored entity_id routes the status to the right entity.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Mercury (planet)"])
    db.upsert_concepts("raw/a.md", ["Mercury (element)"])
    planet = db.entity_id_for_name("Mercury (planet)")
    element = db.entity_id_for_name("Mercury (element)")
    assert planet and element and planet != element

    planet_path = "wiki/.drafts/Mercury (planet).md"
    db.upsert_article(
        WikiArticleRecord(
            path=planet_path,
            title="Mercury (planet)",
            sources=["raw/a.md"],
            content_hash="h",
            status="draft",
            entity_id=planet,
        )
    )
    # entity_id round-trips through persistence.
    assert db.get_article(planet_path).entity_id == planet

    db.verify_article(planet_path)

    planet_state = db.get_compile_state_for_entity(planet, "raw/a.md")
    element_state = db.get_compile_state_for_entity(element, "raw/a.md")
    assert planet_state is not None and planet_state["status"] == "compiled"
    # The other homonym is untouched — its row is still pending, not compiled.
    assert element_state is not None and element_state["status"] != "compiled"


def test_compile_state_is_keyed_on_entity_id(tmp_path: Path) -> None:
    """v23: compile-state rows carry entity_id; two homonyms key on two distinct entities."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Mercury"])
    db.upsert_concepts("raw/b.md", ["Mercury"])
    db.split_entity(
        "Mercury",
        [
            {"name": "Mercury (planet)", "sources": ["raw/a.md"]},
            {"name": "Mercury (element)", "sources": ["raw/b.md"]},
        ],
    )

    planet_id = db.entity_id_for_name("Mercury (planet)")
    element_id = db.entity_id_for_name("Mercury (element)")
    assert planet_id and element_id and planet_id != element_id

    rows = db._conn.execute(
        "SELECT entity_id, source_path FROM concept_compile_state WHERE status='pending'"
    ).fetchall()
    by_entity = {r["entity_id"]: r["source_path"] for r in rows}
    assert by_entity.get(planet_id) == "raw/a.md"
    assert by_entity.get(element_id) == "raw/b.md"

    # The scheduler surfaces both senses as distinct qualified labels (homonym-safe).
    needing = db.concepts_needing_compile()
    assert "Mercury (planet)" in needing
    assert "Mercury (element)" in needing


def test_split_entity_removes_stale_original_compile_state(tmp_path: Path) -> None:
    """Original entity's compile state rows must be deleted after split.

    If they survive, refresh_raw_compile_status overcounts 'compiled' rows for
    sources that now belong to the senses.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Mercury"])
    db.upsert_concepts("raw/b.md", ["Mercury"])
    # Simulate a previously-compiled state for the original entity.
    db.mark_concept_compile_state("Mercury", ["raw/a.md", "raw/b.md"], "compiled")

    db.split_entity(
        "Mercury",
        [
            {"name": "Mercury (planet)", "sources": ["raw/a.md"]},
            {"name": "Mercury (element)", "sources": ["raw/b.md"]},
        ],
    )

    stale = db._conn.execute(
        "SELECT COUNT(*) FROM concept_compile_state WHERE lower(concept_name)='mercury'"
    ).fetchone()[0]
    assert stale == 0, "original entity's compile state rows must be deleted after split"

    # Senses are scheduled for recompile.
    pending_names = {
        r[0]
        for r in db._conn.execute(
            "SELECT concept_name FROM concept_compile_state WHERE status='pending'"
        ).fetchall()
    }
    assert "Mercury (planet)" in pending_names
    assert "Mercury (element)" in pending_names


def test_split_entity_raises_on_unclaimed_source(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Mercury"])
    db.upsert_concepts("raw/b.md", ["Mercury"])
    db.upsert_concepts("raw/c.md", ["Mercury"])

    with pytest.raises(ValueError, match="not assigned"):
        db.split_entity(
            "Mercury",
            [
                {"name": "Mercury (planet)", "sources": ["raw/a.md"]},
                {"name": "Mercury (element)", "sources": ["raw/b.md"]},
                # raw/c.md unclaimed
            ],
        )


def test_split_entity_raises_on_fewer_than_two_senses(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Mercury"])

    with pytest.raises(ValueError, match="at least 2"):
        db.split_entity("Mercury", [{"name": "Mercury (planet)", "sources": ["raw/a.md"]}])


def test_split_entity_raises_when_sense_reuses_original_label(tmp_path: Path) -> None:
    """A sense reusing the bare label would resolve back to the original entity (then be
    retired) and collide with the disambiguation page — reject it at the DB seam."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Mercury"])
    db.upsert_concepts("raw/b.md", ["Mercury"])

    with pytest.raises(ValueError, match="reserved for the disambiguation page"):
        db.split_entity(
            "Mercury",
            [
                {"name": "Mercury", "sources": ["raw/a.md"]},
                {"name": "Mercury (element)", "sources": ["raw/b.md"]},
            ],
        )


def test_split_entity_logs_to_identity_log(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Mercury"])
    db.upsert_concepts("raw/b.md", ["Mercury"])

    db.split_entity(
        "Mercury",
        [
            {"name": "Mercury (planet)", "sources": ["raw/a.md"]},
            {"name": "Mercury (element)", "sources": ["raw/b.md"]},
        ],
    )

    entries = db.list_identity_log()
    assert any(e["op"] == "split" for e in entries)


# ---------------------------------------------------------------------------
# merge_concepts — vault choreography
# ---------------------------------------------------------------------------


def test_merge_concepts_dry_run_returns_report_no_fs_changes(tmp_path: Path) -> None:
    from synto.config import Config
    from synto.pipeline.maintain import merge_concepts

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    (vault / "wiki").mkdir(parents=True)

    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    _article(db, vault, "Alpha")
    _article(db, vault, "Beta")

    config = Config.model_validate({"vault": str(vault)})

    # Snapshot identity state before the dry run.
    concepts_before = db._conn.execute(
        "SELECT name, source_path FROM concepts ORDER BY name, source_path"
    ).fetchall()
    entities_before = db._conn.execute(
        "SELECT id, status, merged_into FROM concept_entities ORDER BY id"
    ).fetchall()

    report = merge_concepts(config, db, "Alpha", "Beta", dry_run=True)

    assert report.dry_run is True
    assert report.loser == "Alpha"
    assert report.winner == "Beta"
    # Loser article must still exist on disk (no writes in dry_run).
    assert (vault / "wiki" / "Alpha.md").exists()

    # dry_run must NOT mutate identity state — merge_entities commits, so calling
    # it here (the original bug) would corrupt the DB even on a dry run.
    concepts_after = db._conn.execute(
        "SELECT name, source_path FROM concepts ORDER BY name, source_path"
    ).fetchall()
    entities_after = db._conn.execute(
        "SELECT id, status, merged_into FROM concept_entities ORDER BY id"
    ).fetchall()
    assert [tuple(r) for r in concepts_after] == [tuple(r) for r in concepts_before]
    assert [tuple(r) for r in entities_after] == [tuple(r) for r in entities_before]


def test_merge_concepts_retires_loser_article(tmp_path: Path) -> None:
    from synto.config import Config
    from synto.pipeline.maintain import merge_concepts

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    (vault / "wiki").mkdir(parents=True)

    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    _article(db, vault, "Alpha")
    _article(db, vault, "Beta")

    config = Config.model_validate({"vault": str(vault)})

    merge_concepts(config, db, "Alpha", "Beta", dry_run=False)

    assert not (vault / "wiki" / "Alpha.md").exists()
    drafts = list((vault / "wiki" / ".drafts").glob("Alpha_retired_*.md"))
    assert drafts, "retired article should be in .drafts/"


def test_merge_concepts_deletes_loser_article_row(tmp_path: Path) -> None:
    """Merge must not leave a published wiki_articles row pointing at a vanished file.

    The loser's file is retired to .drafts/, so a surviving 'published' row would make
    generate_index emit a dangling [[Loser]] and break query/pack/MCP (they resolve the
    row to a missing path). The row must go, not just the file.
    """
    from synto.config import Config
    from synto.indexer import generate_index
    from synto.pipeline.maintain import merge_concepts

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    (vault / "wiki").mkdir(parents=True)

    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    _article(db, vault, "Alpha")
    _article(db, vault, "Beta")

    config = Config.model_validate({"vault": str(vault)})

    merge_concepts(config, db, "Alpha", "Beta", dry_run=False)

    # No tracked row may survive at the retired loser path.
    assert db.get_article("wiki/Alpha.md") is None
    assert all(a.path != "wiki/Alpha.md" for a in db.list_articles())

    # The rebuilt index must carry no dangling link to the retired concept.
    generate_index(config, db)
    index_body = (vault / "wiki" / "index.md").read_text()
    assert "[[Alpha]]" not in index_body
    assert "Alpha.md" not in index_body


def test_merge_concepts_raises_on_manual_edit_mismatch(tmp_path: Path) -> None:

    from synto.config import Config
    from synto.pipeline.maintain import ConceptMergeError, merge_concepts

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    (vault / "wiki").mkdir(parents=True)

    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/a.md", ["LoserC"])
    db.upsert_concepts("raw/b.md", ["WinnerC"])

    # Write article with a specific content hash recorded in DB.
    art_path = vault / "wiki" / "LoserC.md"
    art_path.write_text("---\ntitle: LoserC\n---\noriginal body")
    known_hash = hashlib.sha256(b"original body").hexdigest()
    art = WikiArticleRecord(
        path="wiki/LoserC.md",
        title="LoserC",
        sources=[],
        content_hash=known_hash,
        status="published",
    )
    db.upsert_article(art)
    _article(db, vault, "WinnerC")

    # Modify the article on disk so hash mismatches.
    art_path.write_text("---\ntitle: LoserC\n---\nmanually edited body")

    config = Config.model_validate({"vault": str(vault)})

    with pytest.raises(ConceptMergeError, match="manually edited"):
        merge_concepts(config, db, "LoserC", "WinnerC", dry_run=False)


# ---------------------------------------------------------------------------
# split_concept — vault choreography
# ---------------------------------------------------------------------------


def test_split_concept_creates_sense_stubs(tmp_path: Path) -> None:
    from synto.config import Config
    from synto.pipeline.maintain import split_concept

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    (vault / "wiki").mkdir(parents=True)

    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/planets.md", ["Mercury"])
    db.upsert_concepts("raw/chem.md", ["Mercury"])
    _article(db, vault, "Mercury")

    config = Config.model_validate({"vault": str(vault)})

    report = split_concept(
        config,
        db,
        "Mercury",
        [
            ("Mercury (planet)", ["raw/planets.md"]),
            ("Mercury (element)", ["raw/chem.md"]),
        ],
        dry_run=False,
    )

    assert (vault / "wiki" / "Mercury (planet).md").exists()
    assert (vault / "wiki" / "Mercury (element).md").exists()
    assert len(report.senses) == 2


def test_split_concept_rejects_sense_reusing_original_label(tmp_path: Path) -> None:
    from synto.config import Config
    from synto.pipeline.maintain import ConceptSplitError, split_concept

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    (vault / "wiki").mkdir(parents=True)

    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/planets.md", ["Mercury"])
    db.upsert_concepts("raw/chem.md", ["Mercury"])

    config = Config.model_validate({"vault": str(vault)})

    with pytest.raises(ConceptSplitError, match="reserved for the disambiguation page"):
        split_concept(
            config,
            db,
            "Mercury",
            [
                ("Mercury", ["raw/planets.md"]),
                ("Mercury (element)", ["raw/chem.md"]),
            ],
            dry_run=False,
        )


def test_split_concept_creates_disambiguation_stub(tmp_path: Path) -> None:
    from synto.config import Config
    from synto.pipeline.maintain import split_concept

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    (vault / "wiki").mkdir(parents=True)

    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/planets.md", ["Mercury"])
    db.upsert_concepts("raw/chem.md", ["Mercury"])

    config = Config.model_validate({"vault": str(vault)})

    report = split_concept(
        config,
        db,
        "Mercury",
        [
            ("Mercury (planet)", ["raw/planets.md"]),
            ("Mercury (element)", ["raw/chem.md"]),
        ],
        dry_run=False,
    )

    assert report.stub_path
    dis_path = vault / report.stub_path
    assert dis_path.exists()
    content = dis_path.read_text()
    assert "Mercury (planet)" in content
    assert "Mercury (element)" in content


def test_split_concept_dry_run_no_fs_changes(tmp_path: Path) -> None:
    from synto.config import Config
    from synto.pipeline.maintain import split_concept

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    (vault / "wiki").mkdir(parents=True)

    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/a.md", ["Mercury"])
    db.upsert_concepts("raw/b.md", ["Mercury"])

    config = Config.model_validate({"vault": str(vault)})

    # Snapshot identity state before the dry run.
    concepts_before = db._conn.execute(
        "SELECT name, source_path FROM concepts ORDER BY name, source_path"
    ).fetchall()
    entities_before = db._conn.execute(
        "SELECT id, status, merged_into FROM concept_entities ORDER BY id"
    ).fetchall()

    split_concept(
        config,
        db,
        "Mercury",
        [
            ("Mercury (planet)", ["raw/a.md"]),
            ("Mercury (element)", ["raw/b.md"]),
        ],
        dry_run=True,
    )

    # No stubs written.
    assert not (vault / "wiki" / "Mercury (planet).md").exists()
    assert not (vault / "wiki" / "Mercury (element).md").exists()

    # dry_run must NOT call db.split_entity (which commits) — identity unchanged.
    concepts_after = db._conn.execute(
        "SELECT name, source_path FROM concepts ORDER BY name, source_path"
    ).fetchall()
    entities_after = db._conn.execute(
        "SELECT id, status, merged_into FROM concept_entities ORDER BY id"
    ).fetchall()
    assert [tuple(r) for r in concepts_after] == [tuple(r) for r in concepts_before]
    assert [tuple(r) for r in entities_after] == [tuple(r) for r in entities_before]


def _edited_article(db: StateDB, vault: Path, name: str) -> None:
    """Register an article whose on-disk body diverges from its DB content_hash.

    Simulates a user who hand-edited a published page after compile.
    """
    wiki = vault / "wiki"
    wiki.mkdir(parents=True, exist_ok=True)
    path = wiki / f"{name}.md"
    path.write_text(f"---\ntitle: {name}\n---\nhand-edited body the user wants to keep\n")
    # content_hash records the ORIGINAL compiled body, not what is on disk now.
    db.upsert_article(
        WikiArticleRecord(
            path=f"wiki/{name}.md",
            title=name,
            sources=[],
            content_hash=hashlib.sha256(b"original compiled body").hexdigest(),
            status="published",
        )
    )


def test_split_concept_refuses_manually_edited_article(tmp_path: Path) -> None:
    """Decision 19: split must not silently discard a hand-edited body."""
    from synto.config import Config
    from synto.pipeline.maintain import ConceptSplitError, split_concept

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/planets.md", ["Mercury"])
    db.upsert_concepts("raw/chem.md", ["Mercury"])
    _edited_article(db, vault, "Mercury")
    config = Config.model_validate({"vault": str(vault)})

    with pytest.raises(ConceptSplitError, match="manually edited"):
        split_concept(
            config,
            db,
            "Mercury",
            [
                ("Mercury (planet)", ["raw/planets.md"]),
                ("Mercury (element)", ["raw/chem.md"]),
            ],
        )
    # Refusal happens in preflight — no split occurred, original entity still active.
    assert (
        db._conn.execute(
            "SELECT status FROM concept_entities WHERE id=?", (db.entity_id_for_name("Mercury"),)
        ).fetchone()[0]
        == "active"
    )


def test_split_concept_absorb_edits_carries_body_into_primary_sense(tmp_path: Path) -> None:
    """With --absorb-edits, the hand-edited body lands on the first sense."""
    from synto.config import Config
    from synto.pipeline.maintain import split_concept

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/planets.md", ["Mercury"])
    db.upsert_concepts("raw/chem.md", ["Mercury"])
    _edited_article(db, vault, "Mercury")
    config = Config.model_validate({"vault": str(vault)})

    split_concept(
        config,
        db,
        "Mercury",
        [
            ("Mercury (planet)", ["raw/planets.md"]),
            ("Mercury (element)", ["raw/chem.md"]),
        ],
        absorb_edits=True,
    )

    primary = (vault / "wiki" / "Mercury (planet).md").read_text()
    assert "hand-edited body the user wants to keep" in primary


# ---------------------------------------------------------------------------
# find_match_key_collisions + suggest_concept_merges
# ---------------------------------------------------------------------------


def test_find_match_key_collisions_returns_shared_match_key_pairs(tmp_path: Path) -> None:
    """'User' and 'Users' share match_key 'user' → collision detected."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["User"])
    db.upsert_concepts("raw/b.md", ["Users"])

    # Confirm both are distinct active entities.
    uid = db.entity_id_for_name("User")
    usid = db.entity_id_for_name("Users")
    assert uid != usid

    collisions = db.find_match_key_collisions()
    assert any(mk == "user" for _, _, mk in collisions), (
        "expected 'user' match_key collision between 'User' and 'Users'"
    )


def test_find_match_key_collisions_empty_when_no_overlap(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])

    collisions = db.find_match_key_collisions()
    assert collisions == []


def test_suggest_concept_merges_includes_match_key_pair(tmp_path: Path) -> None:
    from synto.config import Config
    from synto.pipeline.maintain import suggest_concept_merges

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/a.md", ["User"])
    db.upsert_concepts("raw/b.md", ["Users"])

    config = Config.model_validate({"vault": str(vault)})
    suggestions = suggest_concept_merges(config, db)

    names = {(a, b) for a, b, _ in suggestions}
    assert any(
        {"a", "b"} <= {x.lower() for x in pair}
        or "user" in pair[0].lower()
        or "user" in pair[1].lower()
        for pair in names
    ), f"expected user/users merge suggestion, got: {suggestions}"


# ---------------------------------------------------------------------------
# list_identity_log
# ---------------------------------------------------------------------------


def test_list_identity_log_merge_and_split(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["A"])
    db.upsert_concepts("raw/b.md", ["B"])
    db.upsert_concepts("raw/c.md", ["C"])
    db.upsert_concepts("raw/d.md", ["C"])

    db.merge_entities("B", "A")
    db.split_entity(
        "C",
        [
            {"name": "C1", "sources": ["raw/c.md"]},
            {"name": "C2", "sources": ["raw/d.md"]},
        ],
    )

    log = db.list_identity_log()
    ops = {e["op"] for e in log}
    assert "merge" in ops
    assert "split" in ops


# ---------------------------------------------------------------------------
# resolve_ambiguous_occurrences
# ---------------------------------------------------------------------------


def test_resolve_ambiguous_occurrences_updates_count(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["US"])
    db.upsert_concepts("raw/b.md", ["United States"])

    id_us = db.entity_id_for_name("US")
    id_united = db.entity_id_for_name("United States")

    db.record_ambiguous_occurrence(
        "US", [id_us, id_united], surface="US", source_path="raw/note.md"
    )
    assert db.count_ambiguous_occurrences() == 1

    count = db.resolve_ambiguous_occurrences("US", id_us)
    assert count == 1
    assert db.count_ambiguous_occurrences() == 0


# ---------------------------------------------------------------------------
# lint identity checks
# ---------------------------------------------------------------------------


def test_lint_reports_label_collision_for_shared_match_key(tmp_path: Path) -> None:
    from synto.config import Config
    from synto.pipeline.lint import run_lint

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    (vault / "wiki").mkdir(parents=True)

    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/a.md", ["User"])
    db.upsert_concepts("raw/b.md", ["Users"])

    config = Config.model_validate({"vault": str(vault)})

    result = run_lint(config, db)
    collision_issues = [i for i in result.issues if i.issue_type == "label_collision"]
    assert collision_issues, "expected label_collision lint issue for User/Users match_key conflict"


def test_lint_reports_orphan_entity_for_active_with_no_article(tmp_path: Path) -> None:
    from synto.config import Config
    from synto.pipeline.lint import run_lint

    vault = tmp_path / "vault"
    (vault / "raw").mkdir(parents=True)
    (vault / "wiki").mkdir(parents=True)

    db = StateDB(vault / ".synto" / "state.db")
    db.upsert_concepts("raw/a.md", ["OrphanConcept"])
    # No article registered for OrphanConcept.

    config = Config.model_validate({"vault": str(vault)})

    result = run_lint(config, db)
    orphan_issues = [i for i in result.issues if i.issue_type == "orphan_entity"]
    assert orphan_issues, "expected orphan_entity lint issue for active entity with no article"
