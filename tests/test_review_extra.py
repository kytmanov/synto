"""Additional tests for pipeline/review.py uncovered paths."""

from __future__ import annotations

from pathlib import Path

import frontmatter as fm_lib
import pytest

from synto.config import Config
from synto.pipeline.review import (
    compute_diff,
    compute_rejection_diff,
    list_drafts,
)
from synto.state import StateDB
from synto.vault import atomic_write


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / ".synto").mkdir()
    return tmp_path


@pytest.fixture
def config(vault: Path) -> Config:
    return Config(vault=vault)


@pytest.fixture
def db(config: Config) -> StateDB:
    return StateDB(config.state_db_path)


def test_list_drafts_skips_unreadable_draft(config, db):
    """Unreadable draft files are skipped, not crashed on."""
    config.drafts_dir.mkdir(parents=True, exist_ok=True)
    bad_draft = config.drafts_dir / "Broken.md"
    bad_draft.write_bytes(b"\x80\x81\x82")

    summaries = list_drafts(config, db)
    assert summaries == []


def test_list_drafts_sources_not_list(config, db):
    """Non-list sources field → source_count = 0."""
    config.drafts_dir.mkdir(parents=True, exist_ok=True)
    draft_path = config.drafts_dir / "Topic.md"
    meta = {
        "title": "Topic",
        "status": "draft",
        "tags": [],
        "sources": "not-a-list",
        "confidence": 0.5,
    }
    post = fm_lib.Post("Body.", **meta)
    atomic_write(draft_path, fm_lib.dumps(post))

    summaries = list_drafts(config, db)
    assert len(summaries) == 1
    assert summaries[0].source_count == 0


def test_list_drafts_detects_legacy_annotation(config, db):
    """Legacy olw-auto annotation is detected."""
    config.drafts_dir.mkdir(parents=True, exist_ok=True)
    draft_path = config.drafts_dir / "Legacy.md"
    meta = {
        "title": "Legacy",
        "status": "draft",
        "tags": [],
        "sources": [],
        "confidence": 0.5,
    }
    body = "<!-- olw-auto: low-confidence -->\n\n## Body"
    post = fm_lib.Post(body, **meta)
    atomic_write(draft_path, fm_lib.dumps(post))

    summaries = list_drafts(config, db)
    assert summaries[0].has_annotations is True


def test_compute_rejection_diff_no_differences(config, db):
    """Rejection diff returns '(no differences...)' when bodies match."""
    config.drafts_dir.mkdir(parents=True, exist_ok=True)
    draft_path = config.drafts_dir / "Topic.md"
    body = "## Body\n\nSame content."
    meta = {
        "title": "Topic",
        "status": "draft",
        "tags": [],
        "sources": [],
        "confidence": 0.5,
    }
    post = fm_lib.Post(body, **meta)
    atomic_write(draft_path, fm_lib.dumps(post))

    db.add_rejection("Topic", "Feedback", body=body)
    result = compute_rejection_diff(draft_path, db, "Topic")
    assert result == "(no differences from rejected version)"


def test_compute_rejection_diff_unreadable_draft(config, db):
    """Unreadable draft → returns None."""
    config.drafts_dir.mkdir(parents=True, exist_ok=True)
    draft_path = config.drafts_dir / "Broken.md"
    draft_path.write_bytes(b"\x80\x81\x82")

    db.add_rejection("Broken", "Feedback", body="old body")
    result = compute_rejection_diff(draft_path, db, "Broken")
    assert result is None


def test_compute_diff_unreadable_published(config, db):
    """Unreadable published file → returns None."""
    config.drafts_dir.mkdir(parents=True, exist_ok=True)
    draft_path = config.drafts_dir / "Topic.md"
    meta = {"title": "Topic", "status": "draft", "tags": [], "sources": [], "confidence": 0.5}
    post = fm_lib.Post("Draft body.", **meta)
    atomic_write(draft_path, fm_lib.dumps(post))

    wiki_path = config.wiki_dir / "Topic.md"
    wiki_path.write_bytes(b"\x80\x81\x82")

    result = compute_diff(draft_path, wiki_path)
    assert result is None


# ── review 'e' action: cross-platform editor launch (#92) ─────────────────────


def _write_editable_draft(config: Config, title: str = "Alpha") -> Path:
    config.drafts_dir.mkdir(parents=True, exist_ok=True)
    post = fm_lib.Post(
        "Draft body.",
        title=title,
        status="draft",
        tags=[],
        sources=[],
        confidence=0.8,
        created="2024-01-01",
        updated="2024-01-01",
    )
    path = config.drafts_dir / f"{title}.md"
    atomic_write(path, fm_lib.dumps(post))
    return path


def test_review_edit_action_uses_click_edit(config, db, monkeypatch):
    """The 'e' action must go through click.edit, which resolves VISUAL/EDITOR and
    falls back per platform (notepad on Windows) — not a hand-rolled 'vi' default."""
    import click
    from click.testing import CliRunner

    from synto.cli import cli

    path = _write_editable_draft(config)
    monkeypatch.setenv("VISUAL", "true")  # keep an unfixed build from launching vi
    monkeypatch.setenv("EDITOR", "true")
    calls: list[str | None] = []
    monkeypatch.setattr(click, "edit", lambda *a, **kw: calls.append(kw.get("filename")))

    result = CliRunner().invoke(cli, ["review", "--vault", str(config.vault)], input="\ne\nl\nq\n")

    assert result.exit_code == 0, result.output
    assert calls == [str(path)]


def test_review_edit_action_survives_missing_editor(config, db, monkeypatch):
    """#92: no usable editor must print a hint and keep the review session alive,
    not unwind the whole command with a traceback."""
    import click
    from click.testing import CliRunner

    from synto.cli import cli

    _write_editable_draft(config)
    monkeypatch.setenv("VISUAL", "true")
    monkeypatch.setenv("EDITOR", "true")

    def boom(*args, **kwargs):
        raise click.UsageError("vi: not found")

    monkeypatch.setattr(click, "edit", boom)

    result = CliRunner().invoke(cli, ["review", "--vault", str(config.vault)], input="\ne\nl\nq\n")

    assert result.exit_code == 0, result.output
    assert "Could not launch an editor" in result.output
    # The single-draft loop survived the failure: it prompted for an action again.
    assert result.output.count("Action") >= 2
