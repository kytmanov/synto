"""Tests for the `synto set-default-vault` CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from synto.cli import cli
from synto.global_config import load_global_config


def test_set_default_vault_sets_global_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Redirect global config to a temp dir and run the CLI
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))

    vault = tmp_path / "my-wiki"
    vault.mkdir()

    runner = CliRunner()
    result = runner.invoke(cli, ["set-default-vault", str(vault)])
    assert result.exit_code == 0

    cfg = load_global_config()
    assert cfg is not None
    assert cfg.vault == str(vault.resolve())


def test_set_default_vault_errors_on_missing_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))

    missing = Path("/nonexistent/path/which/does/not/exist")
    runner = CliRunner()
    result = runner.invoke(cli, ["set-default-vault", str(missing)])
    assert result.exit_code == 1
    # global config should remain unset
    cfg = load_global_config()
    assert cfg is None
