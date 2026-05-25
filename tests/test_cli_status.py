from __future__ import annotations

import re

from click.testing import CliRunner

from synto.cli import cli
from synto.config import Config
from synto.models import WikiArticleRecord
from synto.pipeline.lock import pipeline_lock
from synto.state import StateDB
from synto.vault import write_note


def _init_status_vault(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / ".synto").mkdir()
    (tmp_path / "synto.toml").write_text(
        """
[models]
fast = "test-fast"
heavy = "test-heavy"

[provider]
name = "ollama"
url = "http://localhost:11434"
""".strip()
    )
    return Config(vault=tmp_path)


def test_status_hides_released_lock_file(tmp_path):
    _init_status_vault(tmp_path)
    StateDB(tmp_path / ".synto" / "state.db")

    with pipeline_lock(tmp_path) as acquired:
        assert acquired is True

    result = CliRunner().invoke(cli, ["status", "--vault", str(tmp_path)])

    assert result.exit_code == 0
    assert "Lock file present" not in result.output
    assert "Pipeline lock held" not in result.output


def test_status_shows_live_pipeline_lock(tmp_path):
    _init_status_vault(tmp_path)
    StateDB(tmp_path / ".synto" / "state.db")

    # CliRunner invokes the command in-process, so this assertion depends on POSIX flock
    # treating a second open() on the same path as a contending live lock.
    with pipeline_lock(tmp_path) as acquired:
        assert acquired is True
        result = CliRunner().invoke(cli, ["status", "--vault", str(tmp_path)])

    assert result.exit_code == 0
    assert "Pipeline lock held by PID" in result.output


def test_status_counts_uningested_raw_files_as_new(tmp_path):
    _init_status_vault(tmp_path)
    StateDB(tmp_path / ".synto" / "state.db")
    (tmp_path / "raw" / "imported.md").write_text("# Imported\n", encoding="utf-8")

    result = CliRunner().invoke(cli, ["status", "--vault", str(tmp_path)])

    assert result.exit_code == 0
    assert re.search(r"Raw: new\s+│\s+1\s+│", result.output)


def test_status_separates_draft_and_verified_counts(tmp_path):
    config = _init_status_vault(tmp_path)
    db = StateDB(tmp_path / ".synto" / "state.db")

    draft_path = config.drafts_dir / "Draft.md"
    verified_path = config.drafts_dir / "Verified.md"
    write_note(draft_path, {"title": "Draft", "status": "draft", "tags": []}, "Body")
    write_note(verified_path, {"title": "Verified", "status": "verified", "tags": []}, "Body")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/.drafts/Draft.md",
            title="Draft",
            sources=[],
            content_hash="h",
            status="draft",
        )
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/.drafts/Verified.md",
            title="Verified",
            sources=[],
            content_hash="h",
            status="verified",
        )
    )

    result = CliRunner().invoke(cli, ["status", "--vault", str(tmp_path)])

    assert result.exit_code == 0
    assert "Verified pending" in result.output
    assert "verified article(s) pending publish" in result.output
