"""Regression tests for the v0.6.0 concept-identity review fixes.

Each test encodes the invariant a specific fix protects, and fails against the pre-fix code:
  - merge does not crash when both concepts have a stub row (#10)
  - _ensure_entity_for_name resolves a merge-loser's name to the active winner, not the dead
    entity, and never to a retired split-original (Root A / #1)
  - match_key does not over-fold short singular -s nouns (#11)
  - unmerge keeps an absorbed alias another still-merged loser also contributed (#12)
  - the index payload uses the article's bound entity_id, not name re-resolution (#3)
"""

from __future__ import annotations

from pathlib import Path

from synto.concept_text import match_key
from synto.indexer import _build_index_payload
from synto.models import WikiArticleRecord
from synto.state import StateDB

_NOW = "2026-01-01T00:00:00"


def test_merge_does_not_crash_when_both_concepts_have_stub(tmp_path: Path) -> None:
    """stubs.concept is a PRIMARY KEY; merging two stubbed concepts must collapse, not abort.

    The pre-fix plain UPDATE collided on the PK and raised IntegrityError inside the merge
    transaction, aborting the whole merge.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Foo"])
    db.upsert_concepts("raw/b.md", ["Bar"])
    db.add_stub("Foo")
    db.add_stub("Bar")

    db.merge_entities("Bar", "Foo")  # must not raise

    stubs = {r[0] for r in db._conn.execute("SELECT concept FROM stubs").fetchall()}
    assert "Foo" not in stubs, "loser stub should be collapsed away"
    assert "Bar" in stubs, "winner stub survives"


def test_ensure_entity_resolves_merged_loser_name_to_winner(tmp_path: Path) -> None:
    """After merge, the loser keeps its preferred label row (unmerge depends on it). The write
    resolver must follow merged_into to the active winner, not return the retired entity.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["GPU"])
    db.upsert_concepts("raw/b.md", ["Graphics Card"])
    winner_id = db.entity_id_for_name("Graphics Card")
    loser_id = db.entity_id_for_name("GPU")
    assert winner_id is not None and loser_id is not None and winner_id != loser_id

    db.merge_entities("Graphics Card", "GPU")

    with db._tx():
        resolved = db._ensure_entity_for_name("GPU", _NOW)
    assert resolved == winner_id, "loser's name must resolve to the active winner, not the dead id"


def test_ensure_entity_returns_none_for_split_retired_original(tmp_path: Path) -> None:
    """A split retires the original with merged_into NULL (no single successor). Its bare label
    must resolve to None, never to the dead entity (and never re-mint → global-unique collision).
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Mercury"])
    db.upsert_concepts("raw/b.md", ["Mercury"])  # second source on the same entity
    db.split_entity(
        "Mercury",
        [
            {"name": "Mercury (planet)", "sources": ["raw/a.md"]},
            {"name": "Mercury (element)", "sources": ["raw/b.md"]},
        ],
    )

    with db._tx():
        resolved = db._ensure_entity_for_name("Mercury", _NOW)
    assert resolved is None


def test_match_key_does_not_overfold_short_singular_s_nouns() -> None:
    """'Lens' must not fold onto the unrelated concept 'Len'; legit plurals still fold."""
    assert match_key("Lens") != match_key("Len")
    assert match_key("Users") == match_key("User"), "real plural fold must be preserved"


def test_unmerge_keeps_alias_owed_by_other_merge(tmp_path: Path) -> None:
    """merge A→B and C→B both absorbing 'Shared'; unmerging A keeps 'Shared' (C still owns it)."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_aliases("Alpha", ["Shared"])
    db.upsert_concepts("raw/c.md", ["Gamma"])
    db.upsert_aliases("Gamma", ["Shared"])
    db.upsert_concepts("raw/b.md", ["Beta"])

    db.merge_entities("Beta", "Alpha")
    db.merge_entities("Beta", "Gamma")
    assert "Shared" in {a for a in db.get_aliases("Beta")}

    alpha_id = db.get_merged_entity_id("Alpha")
    assert alpha_id is not None
    db.unmerge_entities(alpha_id)

    aliases = {a.casefold() for a in db.get_aliases("Beta")}
    assert "shared" in aliases, "alias still owed by the Gamma merge must survive"
    assert "alpha" not in aliases, "the unmerged loser's own label is dropped"


def test_index_payload_uses_bound_entity_id_not_name(config, db) -> None:
    """The index emits the row's bound entity_id even when the title no longer resolves by name."""
    db.upsert_concepts("raw/a.md", ["Real Concept"])
    eid = db.entity_id_for_name("Real Concept")
    assert eid is not None

    # Article whose title does NOT resolve via entity_id_for_name, but carries a bound entity_id.
    path = config.wiki_dir / "Ghost.md"
    path.write_text("---\ntitle: Ghost\n---\nbody")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Ghost.md",
            title="Ghost",
            sources=["raw/a.md"],
            content_hash="",
            status="published",
            kind="concept",
            entity_id=eid,
        )
    )
    assert db.entity_id_for_name("Ghost") is None  # name resolution would yield ""

    payload = _build_index_payload(config, db)
    ghost = next(a for a in payload["articles"] if a["name"] == "Ghost")
    assert ghost["entity_id"] == eid
