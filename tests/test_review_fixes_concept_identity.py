"""Regression tests for the v0.6.0 concept-identity review fixes.

Each test encodes the invariant a specific fix protects, and fails against the pre-fix code:
  - merge does not crash when both concepts have a stub row (#10)
  - _ensure_entity_for_name resolves a merge-loser's name to the active winner, not the dead
    entity, and never to a retired split-original (Root A / #1)
  - match_key does not over-fold short singular -s nouns (#11), and the deliberate cost of that
    guard (short acronym plurals no longer fold) is pinned so it can't drift silently
  - unmerge keeps an absorbed alias another still-merged loser also contributed (#12), and uses
    the winner's CURRENT preferred label when it was renamed after the merge
  - the index payload AND the pack export use the article's bound entity_id, not name
    re-resolution (#3)
  - upgrading a schema < 6 vault preserves the "already compiled" mark instead of aborting
    the migration with an entity_id INSERT against the name-keyed compile-state table
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from synto.concept_text import match_key
from synto.indexer import _build_index_payload
from synto.models import WikiArticleRecord
from synto.pack_export import _concepts_payload
from synto.state import _CURRENT_SCHEMA_VERSION, StateDB

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


def test_match_key_does_not_fold_short_acronym_plurals() -> None:
    """Pins the DELIBERATE cost of the >= 4-char-stem guard: 4-char acronym plurals do NOT fold.

    'GPUs'/'GPU' and 'APIs'/'API' staying separate is an accepted tradeoff — folding them would
    require folding 'Lens' onto 'Len', which lets ingest dedup silently drop a distinct concept
    (data loss, worse than recoverable fragmentation). This test exists so that tradeoff cannot be
    flipped by accident; if it ever needs revisiting, it must be a conscious change here.
    """
    assert match_key("GPUs") != match_key("GPU")
    assert match_key("APIs") != match_key("API")


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


def test_concepts_payload_uses_bound_entity_id_not_name(config, db) -> None:
    """Pack export mirrors the index fix: emit the published article's bound entity_id, not a
    name re-resolution. Pre-fix used db.entity_id_for_name(name), which diverges for a homonym
    or renamed title — here the article is bound to a different entity than its title resolves to.
    """
    db.upsert_concepts("raw/a.md", ["Real Concept"])
    db.upsert_concepts("raw/b.md", ["Other Concept"])
    eid_a = db.entity_id_for_name("Real Concept")
    eid_b = db.entity_id_for_name("Other Concept")
    assert eid_a and eid_b and eid_a != eid_b

    # Published concept article titled "Real Concept" but BOUND to the other entity. The bound
    # id is authoritative; name re-resolution would (wrongly) yield eid_a.
    path = config.wiki_dir / "Real Concept.md"
    path.write_text("---\ntitle: Real Concept\n---\nbody")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Real Concept.md",
            title="Real Concept",
            sources=["raw/b.md"],
            content_hash="",
            status="published",
            kind="concept",
            entity_id=eid_b,
        )
    )

    payload = _concepts_payload(db)
    rc = next(c for c in payload["concepts"] if c["name"] == "Real Concept")
    assert rc["entity_id"] == eid_b, "must emit the article's bound id, not the name resolution"


def test_unmerge_uses_winner_current_label_after_rename(tmp_path: Path) -> None:
    """If the winner is renamed AFTER a merge, unmerge must restore name-keyed state using the
    winner's CURRENT preferred label, not the stale merge-time snapshot. The name-keyed restore
    (concept_occurrences) matches on the winner label, so the snapshot would match nothing.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    alpha_id = db.entity_id_for_name("Alpha")
    assert alpha_id is not None
    # An occurrence on Alpha's source; the merge will repoint its name to the winner.
    db.record_resolved_occurrence("Alpha", alpha_id, "Alpha", source_path="raw/a.md")

    db.merge_entities("Beta", "Alpha")  # winner Beta absorbs Alpha; occurrence → "Beta"
    db.rename_concept("Beta", "Beta Renamed")  # occurrence → "Beta Renamed"

    merged_id = db.get_merged_entity_id("Alpha")
    assert merged_id is not None
    db.unmerge_entities(merged_id)

    restored = db._conn.execute(
        "SELECT concept_name FROM concept_occurrences WHERE source_path='raw/a.md'"
    ).fetchone()
    assert restored is not None
    assert restored[0] == "Alpha", (
        "occurrence must be restored to the loser; the stale snapshot 'Beta' would have matched "
        "nothing after the winner rename"
    )


def _build_pre_v6_db(path: Path) -> None:
    """A schema_version=5 DB with a published concept article whose source is compiled.

    Built with raw sqlite3 (not live _SCHEMA, which is the current shape) so the v6→v26
    migration chain runs for real. _row_to_article needs the 7 columns below on wiki_articles;
    later migrations ADD any columns they need.
    """
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE schema_version (
            id      INTEGER PRIMARY KEY CHECK(id = 1),
            version INTEGER NOT NULL
        );
        INSERT INTO schema_version (id, version) VALUES (1, 5);
        CREATE TABLE raw_notes (
            path         TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'new',
            summary      TEXT,
            quality      TEXT,
            language     TEXT,
            ingested_at  TEXT,
            compiled_at  TEXT,
            error        TEXT
        );
        INSERT INTO raw_notes (path, content_hash, status)
            VALUES ('raw/a.md', 'h', 'ingested');
        CREATE TABLE concepts (
            name        TEXT NOT NULL,
            source_path TEXT NOT NULL
        );
        INSERT INTO concepts (name, source_path) VALUES ('My Concept', 'raw/a.md');
        CREATE TABLE concept_aliases (
            concept_name TEXT NOT NULL,
            alias        TEXT NOT NULL
        );
        CREATE TABLE wiki_articles (
            path           TEXT PRIMARY KEY,
            title          TEXT NOT NULL,
            sources        TEXT NOT NULL,
            content_hash   TEXT NOT NULL,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'draft',
            approved_at    TEXT,
            approval_notes TEXT
        );
        INSERT INTO wiki_articles
            (path, title, sources, content_hash, created_at, updated_at, status)
            VALUES ('wiki/My Concept.md', 'My Concept', '["raw/a.md"]', 'h',
                    '2026-01-01T00:00:00', '2026-01-01T00:00:00', 'published');
        """
    )
    conn.commit()
    conn.close()


def test_v6_upgrade_preserves_compiled_marks(tmp_path: Path) -> None:
    """Upgrading from schema < 6 must keep the 'already compiled' mark for a published article.

    Pre-fix the v6 backfill routed through mark_concept_compile_state, which mints an entity
    (concept_labels already exists via _SCHEMA) and then INSERTs entity_id into the still
    name-keyed concept_compile_state — raising "no such column: entity_id" and ABORTING the
    upgrade. The fix writes the name-keyed row directly at v6; v23's rebuild carries it onto
    entity_id. This drives the real v6→current migration chain end to end.
    """
    db_path = tmp_path / ".synto" / "state.db"
    db_path.parent.mkdir(parents=True)
    _build_pre_v6_db(db_path)

    db = StateDB(db_path)  # opening triggers _migrate through every version

    assert (
        db._conn.execute("SELECT version FROM schema_version WHERE id=1").fetchone()[0]
        == _CURRENT_SCHEMA_VERSION
    )
    status = db._conn.execute(
        "SELECT status FROM concept_compile_state WHERE source_path='raw/a.md'"
    ).fetchone()
    assert status is not None, "compile-state row must survive the upgrade"
    assert status[0] == "compiled", "the published article's compiled mark must be preserved"


def test_merge_repoints_wiki_article_entity_id_and_source_seed_eids(tmp_path: Path) -> None:
    """Merge repoints wiki_articles.entity_id (so published exports don't emit retired eids)
    and source seeds now carry the (repointed) eid from the concepts row rather than
    re-resolving the (stale) name at export time.

    These pin the review fixes for residual identity durability gaps.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["LoserName"])
    db.upsert_concepts("raw/b.md", ["WinnerName"])
    # Seeds query JOINs raw_notes for the content_hash; ensure the source rows exist.
    db._conn.execute(
        "INSERT OR IGNORE INTO raw_notes (path, content_hash, status) VALUES (?, 'h', 'ingested')",
        ("raw/a.md",),
    )
    db._conn.execute(
        "INSERT OR IGNORE INTO raw_notes (path, content_hash, status) VALUES (?, 'h', 'ingested')",
        ("raw/b.md",),
    )
    loser_eid = db.entity_id_for_name("LoserName")
    winner_eid = db.entity_id_for_name("WinnerName")
    assert loser_eid and winner_eid and loser_eid != winner_eid

    # Simulate a published article bound to the loser (pre-merge).
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/LoserName.md",
            title="LoserName",
            sources=["raw/a.md"],
            content_hash="h",
            status="published",
            kind="concept",
            entity_id=loser_eid,
        )
    )

    # Pre-merge seed should carry the original binding.
    seeds = db.list_source_concept_seeds()
    a_seed = next(s for s in seeds if s[0] == "raw/a.md")
    assert any(n == "LoserName" and e == loser_eid for (n, e) in a_seed[2])

    # Now merge (this is what the review found was missing).
    db.merge_entities("WinnerName", "LoserName")

    # wiki_articles row for the (still-titled) loser concept must now point at winner.
    art_row = db._conn.execute(
        "SELECT entity_id FROM wiki_articles WHERE path='wiki/LoserName.md'"
    ).fetchone()
    assert art_row is not None and art_row[0] == winner_eid

    # Source seed for the loser's old source now carries the winner eid (repointed in concepts).
    seeds = db.list_source_concept_seeds()
    a_seed = next(s for s in seeds if s[0] == "raw/a.md")
    # The concept entry for this source now uses the followed (winner) eid.
    assert any(
        n == "WinnerName" and e == winner_eid for (n, e) in a_seed[2]
    )  # name also updated on move
    # And no stale loser eid remains for that source.
    assert not any(e == loser_eid for (n, e) in a_seed[2])


def test_unmerge_restores_wiki_article_entity_id_to_loser(tmp_path: Path) -> None:
    """merge→unmerge must round-trip wiki_articles.entity_id back to the loser.

    Merge repoints the loser's published article onto the winner so exports never emit a
    retired eid; unmerge must reverse that exact repoint, or the restored loser concept's own
    article keeps reporting the winner's identity in INDEX.json / pack exports. The merge
    records the moved article paths in meta, so the reversal is precise (not a title heuristic).
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["LoserName"])
    db.upsert_concepts("raw/b.md", ["WinnerName"])
    loser_eid = db.entity_id_for_name("LoserName")
    winner_eid = db.entity_id_for_name("WinnerName")
    assert loser_eid and winner_eid and loser_eid != winner_eid

    db.upsert_article(
        WikiArticleRecord(
            path="wiki/LoserName.md",
            title="LoserName",
            sources=["raw/a.md"],
            content_hash="h",
            status="published",
            kind="concept",
            entity_id=loser_eid,
        )
    )

    db.merge_entities("WinnerName", "LoserName")
    merged_id = db.get_merged_entity_id("LoserName")
    assert merged_id is not None
    db.unmerge_entities(merged_id)

    art_row = db._conn.execute(
        "SELECT entity_id FROM wiki_articles WHERE path='wiki/LoserName.md'"
    ).fetchone()
    assert art_row is not None
    assert art_row[0] == loser_eid, (
        "unmerge must repoint the loser's article back; otherwise it keeps the winner's eid"
    )
