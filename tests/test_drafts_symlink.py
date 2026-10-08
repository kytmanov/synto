"""wiki/.drafts as a symlink (#130): approve/verify/reject must find the row compile wrote.

Obsidian hides symlinked folders that point inside the vault, so users keep drafts in a
real folder and point wiki/.drafts at it. Rows are keyed wiki/.drafts/X.md; resolving the
draft path before building the key used to miss them and leave stale rows behind.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from synto.cli import cli
from synto.config import Config
from synto.models import RawNoteRecord, WikiArticleRecord
from synto.pipeline.compile import approve_drafts, reject_draft, verify_drafts
from synto.pipeline.lint import run_lint
from synto.state import StateDB

SOURCE = "raw/a.md"


def _make_vault(root: Path, drafts_target: Path) -> Path:
    vault = root / "vault"
    for d in ("raw", "wiki", ".synto"):
        (vault / d).mkdir(parents=True)
    drafts_target.mkdir(parents=True, exist_ok=True)
    try:
        (vault / "wiki" / ".drafts").symlink_to(drafts_target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    return vault


@pytest.fixture(params=["inside_vault", "outside_vault"])
def config(request, tmp_path: Path) -> Config:
    if request.param == "inside_vault":
        target = tmp_path / "vault" / "Drafts"
    else:
        target = tmp_path / "elsewhere" / "Drafts"
    return Config(vault=_make_vault(tmp_path, target))


@pytest.fixture
def db(config: Config) -> StateDB:
    return StateDB(config.state_db_path)


def _seed_draft(config: Config, db: StateDB, title: str = "Alpha") -> str:
    """Write a draft and its row the way compile does; return the entity id."""
    db.upsert_raw(RawNoteRecord(path=SOURCE, content_hash="h1", status="ingested"))
    db.upsert_concepts(SOURCE, [title])
    entity_id = db.entity_id_for_name(title)
    assert entity_id is not None
    draft = config.drafts_dir / f"{title}.md"
    draft.write_text(f"---\ntitle: {title}\nstatus: draft\nsources: [{SOURCE}]\n---\nBody.\n")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft.relative_to(config.vault)),
            title=title,
            sources=[SOURCE],
            content_hash="h",
            status="draft",
            entity_id=entity_id,
        )
    )
    db.mark_concept_compile_state(title, [SOURCE], "compiled")
    return entity_id


def _rows(db: StateDB) -> dict[str, str]:
    return {a.path: a.status for a in db.list_articles()}


# ── approve / verify / reject ────────────────────────────────────────────────


def test_approve_moves_the_tracked_row(config, db):
    entity_id = _seed_draft(config, db)

    approve_drafts(config, db, [config.drafts_dir / "Alpha.md"])

    assert _rows(db) == {"wiki/Alpha.md": "published"}
    assert db.get_article("wiki/Alpha.md").entity_id == entity_id
    assert not (config.drafts_dir / "Alpha.md").exists()


def test_reject_drops_the_row_and_requeues_the_concept(config, db):
    _seed_draft(config, db)

    reject_draft(config.drafts_dir / "Alpha.md", config, db, feedback="Nope")

    assert _rows(db) == {}
    assert db.get_compile_state("Alpha", SOURCE)["status"] == "pending"
    assert not (config.drafts_dir / "Alpha.md").exists()


def test_verify_updates_the_tracked_row(config, db):
    _seed_draft(config, db)

    verify_drafts(config, db, [config.drafts_dir / "Alpha.md"])

    assert _rows(db) == {"wiki/.drafts/Alpha.md": "verified"}


@pytest.mark.parametrize("arg", ["Alpha.md", "wiki/.drafts/Alpha.md", "real"])
def test_cli_approve_accepts_any_spelling_of_the_draft(config, db, arg):
    _seed_draft(config, db)
    if arg == "real":
        arg = str((config.drafts_dir / "Alpha.md").resolve())

    result = CliRunner().invoke(cli, ["approve", "--vault", str(config.vault), arg])

    assert result.exit_code == 0, result.output
    assert _rows(db) == {"wiki/Alpha.md": "published"}


# ── lint repairs rows left by the pre-fix code ────────────────────────────────


def _stray_key(config: Config, name: str) -> str:
    """The key pre-fix code built from the resolved draft path (e.g. Drafts/Alpha.md)."""
    real = (config.drafts_dir / name).resolve()
    if not real.is_relative_to(config.vault.resolve()):
        pytest.skip("pre-fix code could not key a draft outside the vault")
    return real.relative_to(config.vault.resolve()).as_posix()


def _stale_issues(result) -> list[str]:
    return sorted(i.path for i in result.issues if i.issue_type == "stale_draft_row")


@pytest.mark.parametrize("published_title", ["Alpha", "Alpha (edited before approve)"])
def test_lint_drops_stale_row_and_restores_entity_binding(config, db, published_title):
    """Pre-fix approve: draft row left behind, published row built without entity_id.

    The published row takes its title from the draft's frontmatter, so a title edited
    before approve must not stop the binding from carrying over.
    """
    entity_id = _seed_draft(config, db)
    (config.drafts_dir / "Alpha.md").unlink()
    (config.wiki_dir / "Alpha.md").write_text(
        f"---\ntitle: {published_title}\nstatus: published\n---\nB\n"
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Alpha.md",
            title=published_title,
            sources=[SOURCE],
            content_hash="h",
            status="published",
        )
    )

    assert _stale_issues(run_lint(config, db)) == ["wiki/.drafts/Alpha.md"]
    run_lint(config, db, fix=True)

    assert _rows(db) == {"wiki/Alpha.md": "published"}
    assert db.get_article("wiki/Alpha.md").entity_id == entity_id
    assert _stale_issues(run_lint(config, db)) == []


@pytest.mark.parametrize("file_status", ["verified", "draft"])
def test_lint_merges_stray_verified_row_into_tracked_row(config, db, file_status):
    """Pre-fix verify: a second row keyed by the symlink target, the real one untouched.

    Verify also wrote `status: verified` into the file. If the file says draft, a recompile
    rewrote it after that verify, and the new draft must not inherit the old approval.
    """
    _seed_draft(config, db)
    (config.drafts_dir / "Alpha.md").write_text(
        f"---\ntitle: Alpha\nstatus: {file_status}\nsources: [{SOURCE}]\n---\nBody.\n"
    )
    stray_key = _stray_key(config, "Alpha.md")
    approved_at = datetime(2026, 1, 2, 3, 4, 5)
    db.upsert_article(
        WikiArticleRecord(
            path=stray_key,
            title="Alpha",
            sources=[SOURCE],
            content_hash="h",
            status="verified",
            approved_at=approved_at,
            approval_notes="ok",
        )
    )

    assert _stale_issues(run_lint(config, db)) == [stray_key]
    run_lint(config, db, fix=True)

    row = db.get_article("wiki/.drafts/Alpha.md")
    if file_status == "verified":
        assert _rows(db) == {"wiki/.drafts/Alpha.md": "verified"}
        assert (row.approved_at, row.approval_notes) == (approved_at, "ok")
    else:
        assert _rows(db) == {"wiki/.drafts/Alpha.md": "draft"}
        assert (row.approved_at, row.approval_notes) == (None, None)


def test_lint_rekeys_stray_row_without_tracked_twin(config, db):
    _seed_draft(config, db)
    stray_key = _stray_key(config, "Alpha.md")
    row = db.get_article("wiki/.drafts/Alpha.md")
    db.delete_article(row.path)
    db.upsert_article(row.model_copy(update={"path": stray_key}))

    run_lint(config, db, fix=True)

    assert _rows(db) == {"wiki/.drafts/Alpha.md": "draft"}


def test_lint_leaves_healthy_drafts_and_split_stubs_alone(tmp_path):
    """No symlink: live drafts and draft-status stubs under wiki/ are not stale."""
    vault = tmp_path
    for d in ("raw", "wiki/.drafts", ".synto"):
        (vault / d).mkdir(parents=True)
    config = Config(vault=vault)
    db = StateDB(config.state_db_path)
    _seed_draft(config, db)
    (config.wiki_dir / "Stub.md").write_text("---\ntitle: Stub\nstatus: draft\n---\nS\n")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Stub.md", title="Stub", sources=[], content_hash="s", status="draft"
        )
    )

    run_lint(config, db, fix=True)

    assert _stale_issues(run_lint(config, db)) == []
    assert _rows(db) == {"wiki/.drafts/Alpha.md": "draft", "wiki/Stub.md": "draft"}
