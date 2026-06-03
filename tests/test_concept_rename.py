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
from synto.models import Concept, ItemMentionRecord, TermRecord, WikiArticleRecord
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


def test_rename_preserves_manual_edit_protection(config, db):
    """The issue's workflow is a hand-fixed article (review process). The rename must not
    sync the DB content_hash to the on-disk body — that would erase manual-edit
    protection and let the next compile clobber the user's fix."""
    path = _make_concept_article(config, db, "Quantm Computing", "Hand-fixed body.")
    # Simulate a manual on-disk edit after the last compile: DB hash is now stale, so
    # compile.py treats the page as manually edited (on-disk hash != DB hash).
    rel = str(path.relative_to(config.vault))
    art = db.get_article(rel)
    db.upsert_article(art.model_copy(update={"content_hash": "STALE_PRE_EDIT_HASH"}))

    rename_concept(config, db, "Quantm Computing", "Quantum Computing")

    new_rel = "wiki/Quantum Computing.md"
    moved = db.get_article(new_rel)
    # Stale (protective) hash preserved → still differs from the on-disk body → protected.
    assert moved.content_hash == "STALE_PRE_EDIT_HASH"
    assert moved.content_hash != _body_hash(parse_note(config.wiki_dir / "Quantum Computing.md")[1])


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


def test_find_concept_exact_rejects_substring(config, db):
    """find_concept_exact resolves exact name/alias (any case) but never a substring."""
    _make_concept_article(config, db, "Network", "Body.")

    assert db.find_concept_exact("Net") is None
    assert db.find_concept_exact("Network")[0] == "Network"
    assert db.find_concept_exact("network")[0] == "Network"


def test_rename_unknown_concept_errors(config, db):
    with pytest.raises(ConceptRenameError, match="not found"):
        rename_concept(config, db, "Nonexistent", "Whatever")


def test_rename_source_lookup_is_exact_not_fuzzy(config, db):
    """A substring of an existing concept must NOT resolve as the rename source —
    fuzzy matching here would silently rename the wrong concept (the destructive bug).
    The error suggests the near-match without acting on it."""
    article = _make_concept_article(config, db, "Network", "Body.")

    with pytest.raises(ConceptRenameError, match="not found.*Network"):
        rename_concept(config, db, "Net", "Foo")

    # "Network" untouched: still tracked, file unmoved, no "Foo" article created.
    assert "Network" in db.list_all_concept_names()
    assert article.exists()
    assert not (config.wiki_dir / "Foo.md").exists()


def test_rename_by_exact_alias_still_works(config, db):
    """Exact-alias resolution (tier 2) must survive the exact-only source lookup."""
    _make_concept_article(config, db, "Program Counter", "Body.")
    db.upsert_aliases("Program Counter", ["PC"])

    rename_concept(config, db, "PC", "Instruction Pointer")

    assert "Instruction Pointer" in db.list_all_concept_names()
    assert "Program Counter" not in db.list_all_concept_names()


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


def test_rename_does_not_clobber_unrelated_article_sharing_an_alias(config, db):
    """An unrelated article that merely matches one of the concept's aliases must not be
    moved/overwritten — only the concept's own page (by canonical name/stem) moves."""
    _make_concept_article(config, db, "Alpha", "Alpha body.")
    db.upsert_aliases("Alpha", ["A"])
    other = _make_concept_article(config, db, "A", "Unrelated A body.", source="raw/a2.md")

    rename_concept(config, db, "Alpha", "Renamed", keep_alias=False)

    # The aliased-but-unrelated article is untouched.
    assert other.exists()
    assert "Unrelated A body." in other.read_text()
    assert db.get_article("wiki/A.md") is not None
    # The concept's own page moved.
    assert (config.wiki_dir / "Renamed.md").exists()
    assert not (config.wiki_dir / "Alpha.md").exists()


def test_rename_can_promote_own_alias_to_canonical(config, db):
    """Renaming a concept to one of its own aliases is not a collision."""
    _make_concept_article(config, db, "Program Counter", "Body.")
    db.upsert_aliases("Program Counter", ["PC"])

    # Must not raise — "PC" is this concept's own alias, not a foreign name.
    rename_concept(config, db, "Program Counter", "PC")

    assert "PC" in db.list_all_concept_names()
    assert "Program Counter" not in db.list_all_concept_names()
    assert (config.wiki_dir / "PC.md").exists()


def test_rename_errors_when_article_missing_on_disk(config, db):
    """A tracked row whose file was deleted must fail preflight, not crash mid-rename."""
    path = _make_concept_article(config, db, "Ghost", "Body.")
    path.unlink()

    with pytest.raises(ConceptRenameError, match="missing on disk"):
        rename_concept(config, db, "Ghost", "Spectre")


def test_rename_refuses_stale_db_row_at_target_path(config, db):
    """A stale wiki_articles row at the target path must block the rename loudly rather
    than be silently deleted."""
    _make_concept_article(config, db, "Old", "Body.")
    # A DB row at the target path with no file on disk (stale metadata).
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/New.md",
            title="Something Else",
            sources=[],
            content_hash="x",
            status="published",
        )
    )

    with pytest.raises(ConceptRenameError, match="already exists at wiki/New.md"):
        rename_concept(config, db, "Old", "New")


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


# ── State.rename_concept: DB re-key completeness (unit) ─────────────────────────
# The high-level rename is covered above; these pin the lower-level State method's
# obligation to re-key *every* behavioral table it documents. Each asserts a table the
# end-to-end tests don't reach (compile state, occurrence provenance, stubs, self-alias).


def test_state_rename_rekeys_compile_state_occurrences_and_stubs(db):
    """If any UPDATE in State.rename_concept is dropped, the rename silently strips compile
    progress, occurrence provenance, or a pending stub — none of which raise. Re-keying is
    the only thing that keeps those rows reachable under the new canonical name."""
    db.upsert_concepts("raw/n.md", ["Old Name"])
    db.mark_concept_compile_state("Old Name", ["raw/n.md"], "compiled")
    db.upsert_concept_occurrences(
        [
            TermRecord(
                name="Old Name",
                definition="d",
                source_segment_id="seg-1",
                provenance="extracted",
                confidence=0.9,
            )
        ],
        "seg-1",
    )
    db.add_stub("Old Name")

    db.rename_concept("Old Name", "New Name")

    assert db.get_compile_state("Old Name", "raw/n.md") is None
    assert db.get_compile_state("New Name", "raw/n.md") is not None
    assert {r["concept_name"] for r in db.list_concept_occurrences()} == {"New Name"}
    assert db.has_stub("New Name") is True
    assert db.has_stub("Old Name") is False


def test_state_rename_drops_redundant_self_alias(db):
    """Renaming onto one of the concept's own aliases must delete the now self-referential
    row (alias == canonical). A surviving PC→PC alias would resolve a name to itself and
    pollute the alias surface."""
    db.upsert_concepts("raw/n.md", ["Program Counter"])
    db.upsert_aliases("Program Counter", ["PC"])

    db.rename_concept("Program Counter", "PC")

    assert "PC" not in db.aliases_for_concept("PC")


def test_state_rename_leaves_item_mentions_untouched(db):
    """Documented invariant (State.rename_concept docstring): item_mentions are generic
    source-evidence, not canonical concept binding, so the rename deliberately does NOT
    re-key them. Pinning the contract makes any future change to it a conscious one."""
    db.upsert_concepts("raw/n.md", ["Old Name"])
    db.add_item_mention(
        ItemMentionRecord(
            item_name="Old Name",
            source_path="raw/n.md",
            mention_text="Old Name",
            evidence_level="source_supported",
        )
    )

    db.rename_concept("Old Name", "New Name")

    assert len(db.get_item_mentions("Old Name")) == 1
    assert db.get_item_mentions("New Name") == []


# ── Inbound-link rewrite breadth across the wiki tree ───────────────────────────


def test_rename_rewrites_inbound_links_in_every_subdir(config, db):
    """_rewrite_inbound_links rglobs the whole wiki tree (sources/, queries/, synthesis/,
    drafts), not just article root. A regression that narrowed the scan would leave broken
    links in the other folders. Seed one inbound link per folder and assert all repoint."""
    (config.wiki_dir / "sources").mkdir(parents=True, exist_ok=True)
    (config.wiki_dir / "queries").mkdir(parents=True, exist_ok=True)
    (config.wiki_dir / "synthesis").mkdir(parents=True, exist_ok=True)

    _make_concept_article(config, db, "Quantm Computing", "Body.")
    pages = {
        "root": config.wiki_dir / "Overview.md",
        "source": config.wiki_dir / "sources" / "Some Source.md",
        "query": config.wiki_dir / "queries" / "2026-01-01 A question.md",
        "synthesis": config.wiki_dir / "synthesis" / "A synthesis.md",
        "draft": config.drafts_dir / "Draft.md",
    }
    for label, path in pages.items():
        post = fm_lib.Post(
            f"Mentions [[Quantm Computing]] in {label}.", title=label, status="published"
        )
        atomic_write(path, fm_lib.dumps(post))

    report = rename_concept(config, db, "Quantm Computing", "Quantum Computing")

    assert report.links_rewritten == len(pages)
    for path in pages.values():
        text = path.read_text()
        assert "[[Quantum Computing]]" in text
        assert "Quantm Computing" not in text
