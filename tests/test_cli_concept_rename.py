"""CLI wiring for `synto concept rename` (offline — no LLM).

The rename pipeline itself is covered in test_concept_rename.py. These tests pin the
command-layer behavior the pipeline tests can't reach: the non-TTY alias default, the
drop-alias warning, dry-run short-circuiting index/commit, the error exit code, and the
auto-commit side effect.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import frontmatter as fm_lib
from click.testing import CliRunner

from synto.cli import cli
from synto.config import Config
from synto.models import WikiArticleRecord
from synto.state import StateDB
from synto.vault import atomic_write


def _seed_vault(tmp_path: Path) -> Config:
    for sub in ("raw", "wiki", "wiki/.drafts", ".synto"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    config = Config(vault=tmp_path)
    db = StateDB(config.state_db_path)
    db.upsert_concepts("raw/n.md", ["Old Topic"])
    body = "## About\n\nA topic."
    art = config.wiki_dir / "Old Topic.md"
    atomic_write(art, fm_lib.dumps(fm_lib.Post(body, title="Old Topic", status="published")))
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Old Topic.md",
            title="Old Topic",
            sources=["raw/n.md"],
            content_hash=hashlib.sha256(body.encode()).hexdigest(),
            status="published",
        )
    )
    linker = config.wiki_dir / "Linker.md"
    atomic_write(
        linker, fm_lib.dumps(fm_lib.Post("See [[Old Topic]].", title="Linker", status="published"))
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Linker.md",
            title="Linker",
            sources=[],
            content_hash="x",
            status="published",
        )
    )
    db.close()
    return config


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def _git_init(vault: Path) -> None:
    _git(["init"], vault)
    _git(["config", "user.name", "Test"], vault)
    _git(["config", "user.email", "t@e.com"], vault)
    _git(["add", "."], vault)
    _git(["commit", "-m", "baseline"], vault)


def test_cli_rename_moves_file_rewrites_link_and_reports(tmp_path):
    config = _seed_vault(tmp_path)

    result = CliRunner().invoke(
        cli, ["concept", "rename", "Old Topic", "New Topic", "--vault", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "Renamed concept" in result.output
    assert not (config.wiki_dir / "Old Topic.md").exists()
    assert (config.wiki_dir / "New Topic.md").exists()
    assert "[[New Topic]]" in (config.wiki_dir / "Linker.md").read_text()


def test_cli_rename_non_tty_keeps_old_name_as_alias(tmp_path):
    """With no TTY and no flag, the command must default to keeping the old name as an
    alias — silently dropping it would let re-ingest resurrect the old concept."""
    config = _seed_vault(tmp_path)

    result = CliRunner().invoke(
        cli, ["concept", "rename", "Old Topic", "New Topic", "--vault", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    db = StateDB(config.state_db_path)
    try:
        assert db.resolve_alias("Old Topic") == "New Topic"
    finally:
        db.close()


def test_cli_rename_drop_alias_warns_and_drops(tmp_path):
    config = _seed_vault(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "concept",
            "rename",
            "Old Topic",
            "New Topic",
            "--drop-old-alias",
            "--vault",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dropping the old name as alias" in result.output
    db = StateDB(config.state_db_path)
    try:
        assert db.resolve_alias("Old Topic") is None
    finally:
        db.close()


def test_cli_rename_dry_run_writes_nothing_and_skips_commit(tmp_path):
    config = _seed_vault(tmp_path)
    _git_init(config.vault)
    before = (config.wiki_dir / "Linker.md").read_text()

    result = CliRunner().invoke(
        cli, ["concept", "rename", "Old Topic", "New Topic", "--dry-run", "--vault", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "Would rename" in result.output
    assert (config.wiki_dir / "Old Topic.md").exists()
    assert (config.wiki_dir / "Linker.md").read_text() == before
    # No index regeneration and no auto-commit on a dry run.
    assert not (config.wiki_dir / "index.md").exists()
    last = _git(["log", "--format=%s", "-1"], config.vault).stdout.strip()
    assert last == "baseline"


def test_cli_rename_auto_commits_when_enabled(tmp_path):
    config = _seed_vault(tmp_path)
    _git_init(config.vault)

    result = CliRunner().invoke(
        cli, ["concept", "rename", "Old Topic", "New Topic", "--vault", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    last = _git(["log", "--format=%s", "-1"], config.vault).stdout.strip()
    assert "concept rename: Old Topic → New Topic" in last


def test_cli_rename_unknown_concept_exits_nonzero(tmp_path):
    _seed_vault(tmp_path)

    result = CliRunner().invoke(
        cli, ["concept", "rename", "Nonexistent", "Whatever", "--vault", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "not found" in result.output.lower()
