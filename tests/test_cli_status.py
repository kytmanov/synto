from __future__ import annotations

import re

from click.testing import CliRunner

from synto.cli import cli
from synto.config import Config
from synto.pipeline.lock import pipeline_lock
from synto.state import StateDB


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
