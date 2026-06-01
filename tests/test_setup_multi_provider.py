"""Tests for the multi-provider `synto setup` branch (#24).

The branch persists per-role providers to the user-private global config (api_key_env
references, never raw keys) so `synto init` reproduces the split, and can optionally apply
the new format to an existing vault. Tests isolate XDG_CONFIG_HOME so they never touch the
real ~/.config/synto/config.toml.
"""

from __future__ import annotations

import re
import tomllib
from unittest.mock import patch

import pytest
from rich.console import Console

from synto.cli import _setup_multi_provider
from synto.config import Config
from synto.global_config import _global_config_path, load_global_config


@pytest.fixture(autouse=True)
def _isolate_global_config(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _run(answers: list[str]):
    with patch("synto.cli.Prompt.ask", side_effect=answers):
        _setup_multi_provider(Console())


def test_persists_per_role_split_to_global_config():
    # fast=ollama (no key prompt), heavy=nvidia (key env), then skip the vault-apply prompt.
    _run(["1", "", "gemma4:e4b", "nvidia", "", "NVIDIA_API_KEY", "qwen2.5:14b", ""])

    g = load_global_config()
    assert g is not None and g.is_multi_provider
    assert g.models["fast"].provider == "default"
    assert g.models["fast"].model == "gemma4:e4b"
    heavy_alias = g.models["heavy"].provider
    assert g.models["heavy"].model == "qwen2.5:14b"
    assert g.providers[heavy_alias].name == "nvidia"
    assert g.providers[heavy_alias].api_key_env == "NVIDIA_API_KEY"
    # Secrets are never written to the global config — only the env-var name.
    assert not re.search(r"(?m)^\s*api_key\s*=", _global_config_path().read_text())


def test_optionally_applies_to_existing_vault_preserving_pipeline(tmp_path):
    vault = tmp_path / "wiki"
    vault.mkdir()
    (vault / "synto.toml").write_text(
        '[providers.default]\nname = "ollama"\nurl = "http://localhost:11434"\n\n'
        '[models.fast]\nprovider = "default"\nmodel = "old"\nctx = 16384\n\n'
        '[models.heavy]\nprovider = "default"\nmodel = "old"\nctx = 32768\n\n'
        "[pipeline]\nmax_concepts_per_source = 25\nauto_commit = true\n"
    )
    _run(["1", "", "gemma4:e4b", "groq", "", "GROQ_API_KEY", "llama-3.3-70b", str(vault)])

    # Global config persisted...
    assert load_global_config().is_multi_provider
    # ...and the existing vault was rewritten with the split, preserving [pipeline].
    data = tomllib.loads((vault / "synto.toml").read_text())
    assert data["models"]["heavy"]["model"] == "llama-3.3-70b"
    assert data["pipeline"]["max_concepts_per_source"] == 25
    heavy = Config.from_vault(vault).resolve_role("heavy")
    assert heavy.provider_kind == "groq" and heavy.model == "llama-3.3-70b"
