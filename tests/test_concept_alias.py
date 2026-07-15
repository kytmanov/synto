"""Tests for `synto concept alias add/remove/move` — discussion #94.

The fast model sometimes attaches a wrong alias to an entity (real case: an npm package
name attached as an alias of a *project* entity). These tests cover the three load-bearing
requirements: the denial tombstone survives re-ingest and merge absorption, denials seed-
roundtrip through a state.db rebuild (ordered before blessed-alias restore), and removing/
moving an alias un-rewrites the piped `[[Canonical|Alias]]` wiki links that
`normalize_published_alias_links` had already produced for it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import frontmatter as fm_lib
import pytest
from click.testing import CliRunner

from synto.cli import cli
from synto.config import Config
from synto.pipeline.maintain import unlink_alias_links
from synto.state import StateDB
from synto.vault import atomic_write, parse_note, sanitize_filename


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    for sub in ("raw", "wiki", "wiki/.drafts", ".synto"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def config(vault: Path) -> Config:
    return Config(vault=vault)


@pytest.fixture
def db(config: Config) -> StateDB:
    return StateDB(config.state_db_path)


def _write_article(config: Config, title: str, body: str = "## Body\n\nContent.") -> Path:
    path = config.wiki_dir / f"{sanitize_filename(title)}.md"
    post = fm_lib.Post(body, title=title, status="published", tags=[], sources=[])
    atomic_write(path, fm_lib.dumps(post))
    return path


# ── denial tombstone blocks re-attachment ───────────────────────────────────────


def test_remove_alias_blocks_reattachment_by_upsert_aliases(config, db):
    db.upsert_aliases("Mocha Project", ["@mocha/engine"])
    assert "@mocha/engine" in db.get_aliases("Mocha Project")

    db.remove_alias("Mocha Project", "@mocha/engine")
    assert "@mocha/engine" not in db.get_aliases("Mocha Project")

    # Simulate the next ingest re-extracting the same wrong alias.
    db.upsert_aliases("Mocha Project", ["@mocha/engine"], source="extracted")
    assert "@mocha/engine" not in db.get_aliases("Mocha Project")


def test_remove_denial_blocks_resurrection_via_merge(config, db):
    """A merge absorbing the loser's preferred label must not resurrect a denied alias."""
    db.upsert_aliases("Mocha Project", ["Widget"])
    db.remove_alias("Mocha Project", "Widget")

    db.upsert_concepts("raw/b.md", ["Widget"])  # separate entity, preferred label "Widget"
    report = db.merge_entities("Mocha Project", "Widget")

    assert "Widget" not in report["labels_absorbed"]
    # The surface is denied, not silently reattached to anyone.
    assert db.entity_id_for_name("Widget") is None


# ── wiki link un-rewrite ─────────────────────────────────────────────────────────


def test_unlink_alias_links_remove_un_rewrites_piped_links(config, db):
    _write_article(config, "Mocha Project", "## About\n\nThe project.")
    linker = _write_article(
        config,
        "Overview",
        "See [[Mocha Project|@mocha/engine]] for details. Plain [[@mocha/engine]] stays.",
    )

    modified = unlink_alias_links(config, "@mocha/engine", "Mocha Project")
    assert modified == 1

    _, body = parse_note(linker)
    assert "See @mocha/engine for details." in body
    # A plain (unpiped) mention of the same text is not a link to Mocha Project at all —
    # it must not be touched.
    assert "Plain [[@mocha/engine]] stays." in body


def test_unlink_alias_links_move_repoints_piped_links(config, db):
    _write_article(config, "Mocha Project", "## About\n\nThe project.")
    _write_article(config, "Mocha Engine", "## About\n\nThe engine.")
    linker = _write_article(config, "Overview", "See [[Mocha Project|@mocha/engine]] for details.")

    modified = unlink_alias_links(
        config, "@mocha/engine", "Mocha Project", retarget_title="Mocha Engine"
    )
    assert modified == 1

    _, body = parse_note(linker)
    assert "[[Mocha Engine|@mocha/engine]]" in body


def test_unlink_alias_links_leaves_unrelated_display_text_alone(config, db):
    _write_article(config, "Mocha Project", "## About\n\nThe project.")
    linker = _write_article(config, "Overview", "See [[Mocha Project|the framework]] for details.")

    modified = unlink_alias_links(config, "@mocha/engine", "Mocha Project")
    assert modified == 0

    _, body = parse_note(linker)
    assert "[[Mocha Project|the framework]]" in body


def test_remove_end_to_end_cyrillic_alias(config, db):
    """The reporter runs a Russian vault — label_key matching must hold for Cyrillic."""
    canonical = "Мой Проект"
    alias = "Проект"
    db.upsert_aliases(canonical, [alias])
    _write_article(config, canonical, "## О проекте\n\nОписание.")
    linker = _write_article(config, "Обзор", f"См. [[{canonical}|{alias}]] для деталей.")

    db.remove_alias(canonical, alias)
    modified = unlink_alias_links(config, alias, canonical)

    assert modified == 1
    _, body = parse_note(linker)
    assert f"См. {alias} для деталей." in body
    assert alias not in db.get_aliases(canonical)
    # Re-ingest must not silently reattach it.
    db.upsert_aliases(canonical, [alias], source="extracted")
    assert alias not in db.get_aliases(canonical)


# ── durability: seed export → fresh-DB restore roundtrip ────────────────────────


def test_add_alias_is_blessed_and_survives_rebuild_denial_order_respected(tmp_path: Path):
    """add uses source='user' (durable); denials restore before blessed aliases so a
    blessed alias later denied stays gone after rebuild — not resurrected by its own seed
    entry.
    """
    import json

    from synto.indexer import generate_index_json
    from synto.models import RawNoteRecord
    from synto.pipeline.ingest import _restore_identity_from_index

    for sub in ("raw", "wiki", ".synto"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    config = Config(vault=tmp_path)
    db = StateDB(config.state_db_path)

    # A raw_notes row is required for list_source_concept_seeds (the entity-restore seed) —
    # mirrors test_blessed_aliases_survive_state_db_rebuild_from_seed.
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Mocha Project"])
    db.add_alias("Mocha Project", "@mocha/lib")  # stays blessed
    db.add_alias("Mocha Project", "@mocha/engine")  # blessed, then denied below
    eid = db.entity_id_for_name("Mocha Project")

    row = db._conn.execute(
        "SELECT source FROM concept_labels WHERE entity_id=? AND label='@mocha/lib'", (eid,)
    ).fetchone()
    assert row[0] == "user"

    db.remove_alias("Mocha Project", "@mocha/engine")

    payload = json.loads(generate_index_json(config, db).read_text(encoding="utf-8"))
    seed_labels = {a["label"] for a in payload["entity_aliases"]}
    seed_denials = {d["label"] for d in payload["alias_denials"]}
    assert seed_labels == {"@mocha/lib"}
    assert seed_denials == {"@mocha/engine"}

    db.close()
    config.state_db_path.unlink()
    db2 = StateDB(config.state_db_path)
    _restore_identity_from_index(config, db2)

    assert db2.entity_id_for_name("Mocha Project") == eid
    assert "@mocha/lib" in db2.get_aliases("Mocha Project")
    assert "@mocha/engine" not in db2.get_aliases("Mocha Project")

    # And the rebuilt guard still holds: a re-ingest can't resurrect the denied surface.
    db2.upsert_aliases("Mocha Project", ["@mocha/engine"], source="extracted")
    assert "@mocha/engine" not in db2.get_aliases("Mocha Project")


def test_add_after_remove_clears_denial(config, db):
    db.upsert_aliases("Mocha Project", ["@mocha/engine"])
    db.remove_alias("Mocha Project", "@mocha/engine")
    eid = db.entity_id_for_name("Mocha Project")
    assert db._conn.execute(
        "SELECT 1 FROM concept_alias_denials WHERE entity_id=? AND label='@mocha/engine'",
        (eid,),
    ).fetchone()

    db.add_alias("Mocha Project", "@mocha/engine")

    assert "@mocha/engine" in db.get_aliases("Mocha Project")
    assert (
        db._conn.execute(
            "SELECT 1 FROM concept_alias_denials WHERE entity_id=? AND label='@mocha/engine'",
            (eid,),
        ).fetchone()
        is None
    )


# ── error handling ────────────────────────────────────────────────────────────────


def test_remove_preferred_label_refused(config, db):
    db.upsert_concepts("raw/a.md", ["Mocha Project"])
    with pytest.raises(ValueError, match="concept rename"):
        db.remove_alias("Mocha Project", "Mocha Project")


def test_remove_unattached_alias_errors(config, db):
    db.upsert_concepts("raw/a.md", ["Mocha Project"])
    with pytest.raises(ValueError, match="not attached"):
        db.remove_alias("Mocha Project", "@mocha/engine")


def test_remove_unknown_entity_does_not_mint_entity(config, db):
    before = db._conn.execute("SELECT COUNT(*) FROM concept_entities").fetchone()[0]
    with pytest.raises(ValueError, match="not found"):
        db.remove_alias("Nonexistent Concept", "whatever")
    after = db._conn.execute("SELECT COUNT(*) FROM concept_entities").fetchone()[0]
    assert after == before


# ── move (state-level: remove-from + add-to in one transaction) ─────────────────


def test_move_alias_denies_source_and_blesses_target(config, db):
    # "Mocha Toolkit" is deliberately NOT "Mocha Engine" — concept_key strips punctuation,
    # so "@mocha/engine" and "Mocha Engine" fold to the identical label_key, which would
    # make the alias a no-op self-match on that target and mask the bug this test checks.
    db.upsert_concepts("raw/a.md", ["Mocha Project"])
    db.upsert_concepts("raw/b.md", ["Mocha Toolkit"])
    db.upsert_aliases("Mocha Project", ["@mocha/engine"])

    db.move_alias("@mocha/engine", "Mocha Project", "Mocha Toolkit")

    assert "@mocha/engine" not in db.get_aliases("Mocha Project")
    assert "@mocha/engine" in db.get_aliases("Mocha Toolkit")

    # Re-ingest attaching it back to the wrong (source) entity must stay blocked.
    db.upsert_aliases("Mocha Project", ["@mocha/engine"], source="extracted")
    assert "@mocha/engine" not in db.get_aliases("Mocha Project")


def test_move_alias_unknown_target_rolls_back_the_remove(config, db):
    db.upsert_concepts("raw/a.md", ["Mocha Project"])
    db.upsert_aliases("Mocha Project", ["@mocha/engine"])

    with pytest.raises(ValueError, match="not found"):
        db.move_alias("@mocha/engine", "Mocha Project", "Nonexistent Target")

    # The failed add must not have left the alias detached from the source.
    assert "@mocha/engine" in db.get_aliases("Mocha Project")


# ── `synto undo` refuses a batch containing an alias op ─────────────────────────


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _commit(vault: Path, subject: str) -> None:
    _git(["add", "-A"], vault)
    _git(["commit", "-m", subject, "--allow-empty"], vault)


def test_undo_refuses_alias_op(vault: Path) -> None:
    _git(["init"], vault)
    _git(["config", "user.email", "t@t"], vault)
    _git(["config", "user.name", "t"], vault)
    _git(["config", "commit.gpgsign", "false"], vault)
    (vault / "wiki" / "seed.md").write_text("seed")
    _commit(vault, "initial")

    (vault / "wiki" / "seed.md").write_text("after alias remove")
    _commit(vault, "[synto] concept alias remove: Mocha Project :: @mocha/engine")

    result = CliRunner().invoke(cli, ["undo", "--vault", str(vault)])

    assert result.exit_code == 1, result.output
    assert "concept alias add" in result.output
