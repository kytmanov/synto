"""CLI wiring for `synto maintain` (offline — maintain uses no LLM).

The maintain pipeline functions are covered in test_maintain.py / test_maintain_fix_links.py.
These pin the command layer: clear-cache short-circuit, --fix disk side effects, the
read-only dry-run, and the lock-held guard.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import frontmatter as fm_lib
import pytest
from click.testing import CliRunner

from synto.cache import LLMCache
from synto.cli import cli
from synto.config import Config
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


def _write_article(config: Config, title: str, body: str) -> Path:
    path = config.wiki_dir / f"{sanitize_filename(title)}.md"
    atomic_write(path, fm_lib.dumps(fm_lib.Post(body, title=title, status="published")))
    return path


def test_clear_cache_deletes_all_entries_and_reports(config, tmp_path):
    db = StateDB(config.state_db_path)
    cache = LLMCache(db)
    cache.put("m", [{"role": "user", "content": "a"}], "resp-a")
    cache.put("m", [{"role": "user", "content": "b"}], "resp-b")
    assert cache.stats()["total_entries"] == 2
    db.close()

    result = CliRunner().invoke(cli, ["maintain", "--clear-cache", "--vault", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Cleared 2 LLM cache entries." in result.output
    db2 = StateDB(config.state_db_path)
    try:
        assert LLMCache(db2).stats()["total_entries"] == 0
    finally:
        db2.close()


def test_clear_cache_older_than_reports_window(config, tmp_path):
    db = StateDB(config.state_db_path)
    LLMCache(db).put("m", [{"role": "user", "content": "a"}], "resp")
    db.close()

    result = CliRunner().invoke(
        cli, ["maintain", "--clear-cache", "--older-than", "365", "--vault", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    # A fresh entry is younger than 365 days → nothing deleted, but the windowed message fires.
    assert "older than 365 days" in result.output


def test_fix_normalizes_alias_link_on_disk(config, tmp_path):
    """--fix rewrites a raw alias link to its canonical|alias form. Lint resolves known
    aliases, so this travels the alias-normalization path, not broken-link repair."""
    db = StateDB(config.state_db_path)
    _write_article(config, "Machine Learning", "## Body\n\nContent.")
    db.upsert_aliases("Machine Learning", ["ML"])
    ref = _write_article(config, "Ref", "See [[ML]] for details.")
    db.close()

    result = CliRunner().invoke(cli, ["maintain", "--fix", "--vault", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Normalized alias links in 1 article(s)." in result.output
    assert "[[Machine Learning|ML]]" in parse_note(ref)[1]


def test_fix_creates_stub_for_unresolvable_link(config, tmp_path):
    """A link to a name with no article and no alias is genuinely broken; --fix creates a
    stub draft for it so the link resolves."""
    db = StateDB(config.state_db_path)
    _write_article(config, "Ref", "See [[Totally Missing Topic]] here.")
    db.close()

    result = CliRunner().invoke(cli, ["maintain", "--fix", "--vault", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "stub" in result.output.lower()
    assert db_has_stub_file(config, "Totally Missing Topic")


def db_has_stub_file(config: Config, name: str) -> bool:
    return (config.drafts_dir / f"{sanitize_filename(name)}.md").exists()


def test_dry_run_reports_but_writes_nothing(config, tmp_path):
    db = StateDB(config.state_db_path)
    _write_article(config, "Machine Learning", "## Body\n\nContent.")
    db.upsert_aliases("Machine Learning", ["ML"])
    ref = _write_article(config, "Ref", "See [[ML]] for details.")
    db.close()
    before = ref.read_text()

    result = CliRunner().invoke(cli, ["maintain", "--fix", "--dry-run", "--vault", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert ref.read_text() == before  # the alias link is untouched on disk


def test_lock_held_exits_nonzero(config, tmp_path, monkeypatch):
    """A concurrent pipeline run holds the lock; maintain must refuse, not race the DB."""

    @contextlib.contextmanager
    def _held(_vault):
        yield False  # acquired=False

    monkeypatch.setattr("synto.pipeline.lock.pipeline_lock", _held)

    result = CliRunner().invoke(cli, ["maintain", "--vault", str(tmp_path)])

    assert result.exit_code == 1
    assert "lock held" in result.output.lower()
