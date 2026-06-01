"""Tests for the multi-provider `synto setup` branch (#24).

Why: a user must be able to put each model role on a different provider/account from
the wizard, with the result written to the vault in the new [providers.*] format using
env-var key references (never raw secrets).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from synto.cli import _setup_multi_provider
from synto.config import Config


def _run(answers: list[str]):
    console = Console()
    with patch("synto.cli.Prompt.ask", side_effect=answers):
        _setup_multi_provider(console)


def test_multi_provider_writes_split_vault(tmp_path: Path):
    vault = tmp_path / "wiki"
    vault.mkdir()
    answers = [
        # Fast role: provider, url, model (ollama -> no key prompt)
        "1",
        "",
        "gemma4:e4b",
        # Heavy role: provider, url, api-key env var, model
        "nvidia",
        "",
        "NVIDIA_API_KEY",
        "qwen2.5:14b",
        # Vault path
        str(vault),
    ]
    _run(answers)

    text = (vault / "synto.toml").read_text()
    data = tomllib.loads(text)
    # Two distinct connections; the first is "default" so embed/string roles resolve.
    assert data["providers"]["default"]["name"] == "ollama"
    assert data["providers"]["nvidia"]["name"] == "nvidia"
    assert data["providers"]["nvidia"]["api_key_env"] == "NVIDIA_API_KEY"
    # Secrets are never written to the vault.
    assert "NVIDIA_API_KEY" in text and "api_key =" not in text

    cfg = Config.from_vault(vault)
    fast = cfg.resolve_role("fast")
    heavy = cfg.resolve_role("heavy")
    assert fast.provider_kind == "ollama"
    assert heavy.provider_kind == "nvidia"
    assert heavy.url == "https://integrate.api.nvidia.com/v1"
    assert fast.think is False  # role-aware default still applies


def test_multi_provider_preserves_existing_pipeline_settings(tmp_path: Path):
    vault = tmp_path / "wiki"
    vault.mkdir()
    # Pre-existing vault with a customized pipeline setting that must survive.
    (vault / "synto.toml").write_text(
        '[providers.default]\nname = "ollama"\nurl = "http://localhost:11434"\n\n'
        '[models.fast]\nprovider = "default"\nmodel = "old"\nctx = 16384\n\n'
        '[models.heavy]\nprovider = "default"\nmodel = "old"\nctx = 32768\n\n'
        "[pipeline]\nmax_concepts_per_source = 25\nauto_commit = true\n"
    )
    answers = [
        "1",
        "",
        "gemma4:e4b",
        "groq",
        "",
        "GROQ_API_KEY",
        "llama-3.3-70b",
        str(vault),
    ]
    _run(answers)

    data = tomllib.loads((vault / "synto.toml").read_text())
    assert data["models"]["heavy"]["provider"] == "groq"
    assert data["models"]["heavy"]["model"] == "llama-3.3-70b"
    # The user's pipeline customization is preserved through the rewrite.
    assert data["pipeline"]["max_concepts_per_source"] == 25
