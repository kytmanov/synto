"""Tests for the `synto compile` CLI command."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from click.testing import CliRunner
from conftest import as_router

from synto.cli import cli
from synto.config import Config
from synto.models import RawNoteRecord
from synto.state import StateDB


def test_compile_cli_concept_alias_resolution(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".synto").mkdir()
    config = Config(vault=tmp_path)
    db = StateDB(config.state_db_path)
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Product Backlog"])
    db.upsert_aliases("Product Backlog", ["Backlog"])
    (tmp_path / "raw" / "a.md").write_text("Body.")

    client = MagicMock()
    client.generate.return_value = json.dumps(
        {"title": "Product Backlog", "content": "Body.", "tags": []}
    )

    monkeypatch.setattr("synto.cli._load_deps", lambda cfg: (as_router(client), db))

    result = CliRunner().invoke(
        cli,
        ["compile", "--vault", str(tmp_path), "--concept", "Backlog"],
    )

    assert result.exit_code == 0
    assert (tmp_path / "wiki" / ".drafts" / "Product Backlog.md").exists()


def test_compile_cli_retry_failed_retries_failed_concepts(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".synto").mkdir()
    config = Config(vault=tmp_path)
    db = StateDB(config.state_db_path)
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.mark_concept_compile_state("Alpha", ["raw/a.md"], "failed", error="bad json")
    (tmp_path / "raw" / "a.md").write_text("Body.")

    client = MagicMock()
    client.generate.return_value = json.dumps({"title": "Alpha", "content": "Body.", "tags": []})

    monkeypatch.setattr("synto.cli._load_deps", lambda cfg: (as_router(client), db))

    result = CliRunner().invoke(
        cli,
        ["compile", "--vault", str(tmp_path), "--retry-failed"],
    )

    assert result.exit_code == 0
    assert (tmp_path / "wiki" / ".drafts" / "Alpha.md").exists()


def test_compile_auto_approve_commits_synto_dir(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".synto").mkdir()
    (tmp_path / "synto.toml").write_text("[pipeline]\nauto_commit = true\n")
    config = Config.from_vault(tmp_path)
    db = StateDB(config.state_db_path)
    client = MagicMock()
    commit_calls = []

    monkeypatch.setattr("synto.cli._load_deps", lambda cfg: (as_router(client), db))
    monkeypatch.setattr(
        "synto.pipeline.compile.compile_concepts",
        lambda **kwargs: ([tmp_path / "wiki" / ".drafts" / "Alpha.md"], [], {}),
    )
    monkeypatch.setattr(
        "synto.pipeline.compile.approve_drafts",
        lambda *args, **kwargs: [tmp_path / "wiki" / "Alpha.md"],
    )
    monkeypatch.setattr("synto.indexer.generate_index", lambda *args, **kwargs: None)
    monkeypatch.setattr("synto.indexer.append_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "synto.git_ops.git_commit",
        lambda vault, message, paths=None: commit_calls.append((vault, message, paths)) or True,
    )

    result = CliRunner().invoke(cli, ["compile", "--vault", str(tmp_path), "--auto-approve"])

    assert result.exit_code == 0
    assert commit_calls
    assert commit_calls[0][2] == ["wiki/", ".synto/"]


def test_compile_noop_does_not_update_index(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".synto").mkdir()
    config = Config(vault=tmp_path)
    db = StateDB(config.state_db_path)
    client = MagicMock()

    gen_idx = MagicMock()
    monkeypatch.setattr("synto.cli._load_deps", lambda cfg: (as_router(client), db))
    monkeypatch.setattr("synto.indexer.generate_index", gen_idx)

    result = CliRunner().invoke(cli, ["compile", "--vault", str(tmp_path)])

    assert result.exit_code == 0
    gen_idx.assert_not_called()
