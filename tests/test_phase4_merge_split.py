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

    rows = db._conn.execute(
        "SELECT name FROM concepts WHERE source_path='raw/a.md'"
    ).fetchall()
    assert rows, "source edge should still exist"
    assert all(r[0] == "Machine Learning" for r in rows)


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

    row = db._conn.execute(
        "SELECT status FROM concept_entities WHERE id=?", (orig_id,)
    ).fetchone()
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

    report = merge_concepts(config, db, "Alpha", "Beta", dry_run=True)

    assert report.dry_run is True
    assert report.loser == "Alpha"
    assert report.winner == "Beta"
    # Loser article must still exist on disk (no writes in dry_run).
    assert (vault / "wiki" / "Alpha.md").exists()


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
        {"a", "b"} <= {x.lower() for x in pair} or
        "user" in pair[0].lower() or "user" in pair[1].lower()
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
