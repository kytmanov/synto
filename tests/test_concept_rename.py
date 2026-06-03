"""Tests for `synto concept rename` — issue #29.

Covers the DB identity migration, inbound-link rewriting with content_hash refresh,
behavioral-state survival (block/rejection), and the re-ingest durability contract.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import frontmatter as fm_lib
import pytest

from synto.config import Config
from synto.models import Concept, WikiArticleRecord
from synto.pipeline.ingest import _normalize_concepts
from synto.pipeline.maintain import ConceptRenameError, rename_concept
from synto.state import StateDB
from synto.vault import atomic_write, parse_note, sanitize_filename


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".synto").mkdir()
    return tmp_path


@pytest.fixture
def config(vault: Path) -> Config:
    return Config(vault=vault)


@pytest.fixture
def db(config: Config) -> StateDB:
    return StateDB(config.state_db_path)


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def _make_concept_article(
    config: Config, db: StateDB, name: str, body: str, *, source: str = "raw/note.md"
) -> Path:
    """Register a published concept (concepts + knowledge_items + article row + file)."""
    db.upsert_concepts(source, [name])
    path = config.wiki_dir / f"{sanitize_filename(name)}.md"
    post = fm_lib.Post(body, title=name, status="published", tags=[], sources=[source])
    atomic_write(path, fm_lib.dumps(post))
    db.upsert_article(
        WikiArticleRecord(
            path=str(path.relative_to(config.vault)),
            title=name,
            sources=[source],
            content_hash=_body_hash(body),
            status="published",
        )
    )
    return path


def _write_linking_article(config: Config, db: StateDB, title: str, body: str) -> Path:
    path = config.wiki_dir / f"{sanitize_filename(title)}.md"
    post = fm_lib.Post(body, title=title, status="published", tags=[], sources=[])
    atomic_write(path, fm_lib.dumps(post))
    db.upsert_article(
        WikiArticleRecord(
            path=str(path.relative_to(config.vault)),
            title=title,
            sources=[],
            content_hash=_body_hash(body),
            status="published",
        )
    )
    return path


# ── End-to-end rename ─────────────────────────────────────────────────────────


def test_rename_moves_file_updates_title_and_rewrites_inbound_links(config, db):
    _make_concept_article(config, db, "Quantm Computing", "## About\n\nA topic.")
    linker = _write_linking_article(config, db, "Overview", "We discuss [[Quantm Computing]] here.")

    report = rename_concept(config, db, "Quantm Computing", "Quantum Computing")

    # File moved + frontmatter title corrected.
    assert not (config.wiki_dir / "Quantm Computing.md").exists()
    new_file = config.wiki_dir / "Quantum Computing.md"
    assert new_file.exists()
    assert parse_note(new_file)[0]["title"] == "Quantum Computing"

    # Inbound link repointed.
    assert "[[Quantum Computing]]" in linker.read_text()
    assert "Quantm Computing" not in linker.read_text()
    assert report.links_rewritten == 1


def test_rename_refreshes_content_hash_of_rewritten_pages(config, db):
    """A rewritten tracked page must have its DB content_hash refreshed, or compile's
    manual-edit protection would falsely treat it as hand-edited and stop updating it."""
    _make_concept_article(config, db, "Quantm Computing", "Body.")
    linker = _write_linking_article(config, db, "Overview", "See [[Quantm Computing]].")

    rename_concept(config, db, "Quantm Computing", "Quantum Computing")

    rel = str(linker.relative_to(config.vault))
    on_disk_body = parse_note(linker)[1]
    assert db.get_article(rel).content_hash == _body_hash(on_disk_body)


def test_rename_migrates_concept_identity_tables(config, db):
    _make_concept_article(config, db, "Old Concept", "Body.")
    db.upsert_aliases("Old Concept", ["OC"])

    rename_concept(config, db, "Old Concept", "New Concept", keep_alias=True)

    assert "New Concept" in db.list_all_concept_names()
    assert "Old Concept" not in db.list_all_concept_names()
    # Existing surface alias re-keyed to the new canonical.
    assert "OC" in db.get_aliases("New Concept")
    # Old name kept as a resolvable alias (durability).
    assert db.resolve_alias("Old Concept") == "New Concept"
    # knowledge_items row migrated.
    assert db.get_item("New Concept") is not None
    assert db.get_item("Old Concept") is None


def test_rename_preserves_blocked_state(config, db):
    """Skipping blocked_concepts would silently unblock the concept on rename."""
    _make_concept_article(config, db, "Spam Topic", "Body.")
    db.mark_concept_blocked("Spam Topic")

    rename_concept(config, db, "Spam Topic", "Clean Topic")

    assert db.is_concept_blocked("Clean Topic") is True
    assert db.is_concept_blocked("Spam Topic") is False


def test_rename_preserves_rejection_guidance(config, db):
    """Rejections are looked up by exact title; they must follow the rename."""
    _make_concept_article(config, db, "Vague Topic", "Body.")
    db.add_rejection("Vague Topic", "Too vague")

    rename_concept(config, db, "Vague Topic", "Sharp Topic")

    assert db.rejection_count("Sharp Topic") == 1
    assert db.get_rejections("Sharp Topic")[0]["feedback"] == "Too vague"


# ── Durability against re-ingest (the core bug) ───────────────────────────────


def test_kept_alias_canonicalizes_reextracted_old_name(config, db):
    """With the old name kept as alias, ingest normalization maps a re-extracted old
    name back to the new concept — the rename survives re-ingest."""
    _make_concept_article(config, db, "Quantm Computing", "Body.")
    rename_concept(config, db, "Quantm Computing", "Quantum Computing", keep_alias=True)

    result = _normalize_concepts([Concept(name="Quantm Computing")], db)
    assert result[0][0] == "Quantum Computing"


def test_dropped_alias_lets_old_name_resurrect(config, db):
    """Documents the tradeoff: without the alias, a re-extracted old name becomes a
    brand-new concept again (would re-create the old article on next compile)."""
    _make_concept_article(config, db, "Quantm Computing", "Body.")
    rename_concept(config, db, "Quantm Computing", "Quantum Computing", keep_alias=False)

    assert db.resolve_alias("Quantm Computing") is None
    result = _normalize_concepts([Concept(name="Quantm Computing")], db)
    assert result[0][0].casefold() == "quantm computing"


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_rename_refuses_existing_target_name(config, db):
    _make_concept_article(config, db, "Alpha", "Body.")
    _make_concept_article(config, db, "Beta", "Body.", source="raw/b.md")

    with pytest.raises(ConceptRenameError, match="already exists"):
        rename_concept(config, db, "Alpha", "Beta")


def test_collision_check_is_exact_not_fuzzy(config, db):
    """Renaming to 'Net' must not be blocked by the substring match against 'Network'."""
    _make_concept_article(config, db, "Network", "Body.")
    assert db.concept_name_exists_exact("Net") is False
    assert db.concept_name_exists_exact("network") is True


def test_rename_unknown_concept_errors(config, db):
    with pytest.raises(ConceptRenameError, match="not found"):
        rename_concept(config, db, "Nonexistent", "Whatever")


def test_dry_run_writes_nothing(config, db):
    _make_concept_article(config, db, "Quantm Computing", "Body.")
    linker = _write_linking_article(config, db, "Overview", "See [[Quantm Computing]].")
    before = linker.read_text()

    report = rename_concept(config, db, "Quantm Computing", "Quantum Computing", dry_run=True)

    assert report.dry_run is True
    assert report.links_rewritten == 1  # counted, not written
    assert linker.read_text() == before
    assert (config.wiki_dir / "Quantm Computing.md").exists()
    assert "Quantm Computing" in db.list_all_concept_names()


def test_rename_db_only_concept_without_article(config, db):
    """A concept that was never compiled has no article — rename still migrates the DB."""
    db.upsert_concepts("raw/a.md", ["Orphan Concept"])

    report = rename_concept(config, db, "Orphan Concept", "Adopted Concept")

    assert report.files_moved == []
    assert "Adopted Concept" in db.list_all_concept_names()


def test_rename_draft_article(config, db):
    db.upsert_concepts("raw/a.md", ["Draft Topic"])
    draft = config.drafts_dir / "Draft Topic.md"
    post = fm_lib.Post("Body.", title="Draft Topic", status="draft", tags=[], sources=[])
    atomic_write(draft, fm_lib.dumps(post))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft.relative_to(config.vault)),
            title="Draft Topic",
            sources=[],
            content_hash=_body_hash("Body."),
            status="draft",
        )
    )

    rename_concept(config, db, "Draft Topic", "Final Topic")

    assert not draft.exists()
    assert (config.drafts_dir / "Final Topic.md").exists()
