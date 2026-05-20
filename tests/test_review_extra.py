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
