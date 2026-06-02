"""Tests for the per-role provider finalizer behind the `synto setup` advanced branch (#24).

`_finalize_per_role_providers` persists a per-role split to the user-private global config
(api_key_env references, never raw keys) so `synto init` reproduces the split, and can optionally
apply the new format to an existing vault. A raw key typed for the reused primary (fast) provider
is carried into the user-private global config under the fast alias — never into a vault. Tests
isolate XDG_CONFIG_HOME so they never touch the real ~/.config/synto/config.toml.
"""

from __future__ import annotations

import re
import tomllib

import pytest
from rich.console import Console

from synto.cli import _finalize_per_role_providers
from synto.config import Config
from synto.global_config import (
    GlobalConfig,
    _global_config_path,
    load_global_config,
    save_global_config,
)
from synto.providers import PROVIDER_REGISTRY


@pytest.fixture(autouse=True)
def _isolate_global_config(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _spec(name: str, model: str, api_key_env: str | None = None) -> dict:
    """A connection spec in the shape `_collect_role_provider` / the setup branch produce."""
    prov = PROVIDER_REGISTRY[name]
    return {
        "name": name,
        "url": prov.default_url or "",
        "api_key_env": api_key_env,
        "azure_api_version": None,
        "model": model,
        "timeout": int(prov.default_timeout),
    }


def _finalize(fast, heavy, *, vault_input="", citations=False, fast_api_key=None):
    _finalize_per_role_providers(
        Console(),
        fast=fast,
        heavy=heavy,
        vault_input=vault_input,
        citations=citations,
        fast_api_key=fast_api_key,
    )


def test_persists_per_role_split_to_global_config():
    # fast=ollama (no key), heavy=nvidia (key env); the first connection becomes "default".
    _finalize(_spec("ollama", "gemma4:e4b"), _spec("nvidia", "qwen2.5:14b", "NVIDIA_API_KEY"))

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


def test_rerun_preserves_existing_global_provider_keys():
    # The user-private per-alias key fallback ([provider_keys]) must survive re-running setup —
    # rebuilding GlobalConfig from scratch would silently delete it.
    save_global_config(GlobalConfig(provider_keys={"ngc": "secret-key"}))

    _finalize(_spec("ollama", "gemma4:e4b"), _spec("nvidia", "qwen2.5:14b", "NVIDIA_API_KEY"))

    g = load_global_config()
    assert g is not None and g.is_multi_provider
    assert g.provider_keys == {"ngc": "secret-key"}  # not dropped
    assert "ngc" in _global_config_path().read_text()


def test_reused_primary_raw_key_lands_in_provider_keys_under_fast_alias():
    # When the reused primary (fast) provider used a raw key (no env var), it must be carried into
    # the user-private global config under the fast alias — the multi-provider format has no other
    # home for it, and resolve_api_key reads it from there (step 3).
    _finalize(
        _spec("groq", "llama-fast"),  # api_key_env=None → the raw-key path
        _spec("nvidia", "qwen-heavy", "NVIDIA_API_KEY"),
        fast_api_key="raw-fast-secret",
    )

    g = load_global_config()
    fast_alias = g.models["fast"].provider
    assert g.provider_keys[fast_alias] == "raw-fast-secret"
    assert g.providers[fast_alias].api_key_env is None


def test_optionally_applies_to_existing_vault_preserving_pipeline(tmp_path):
    vault = tmp_path / "wiki"
    vault.mkdir()
    (vault / "synto.toml").write_text(
        '[providers.default]\nname = "ollama"\nurl = "http://localhost:11434"\n\n'
        '[models.fast]\nprovider = "default"\nmodel = "old"\nctx = 16384\n\n'
        '[models.heavy]\nprovider = "default"\nmodel = "old"\nctx = 32768\n\n'
        "[pipeline]\nmax_concepts_per_source = 25\nauto_commit = true\n"
    )
    _finalize(
        _spec("ollama", "gemma4:e4b"),
        _spec("groq", "llama-3.3-70b", "GROQ_API_KEY"),
        vault_input=str(vault),
    )

    # Global config persisted...
    assert load_global_config().is_multi_provider
    # ...and the existing vault was rewritten with the split, preserving [pipeline].
    data = tomllib.loads((vault / "synto.toml").read_text())
    assert data["models"]["heavy"]["model"] == "llama-3.3-70b"
    assert data["pipeline"]["max_concepts_per_source"] == 25
    heavy = Config.from_vault(vault).resolve_role("heavy")
    assert heavy.provider_kind == "groq" and heavy.model == "llama-3.3-70b"


def test_applies_to_legacy_wiki_toml_vault_writes_loadable_synto_toml(tmp_path):
    """#3: a legacy-only (wiki.toml) vault must end up loadable — setup writes synto.toml,
    not a rewritten wiki.toml that Config.from_vault would still refuse."""
    vault = tmp_path / "legacy"
    vault.mkdir()
    (vault / "wiki.toml").write_text(
        '[models]\nfast = "old"\nheavy = "old"\n\n'
        '[ollama]\nurl = "http://localhost:11434"\n\n'
        "[pipeline]\nmax_concepts_per_source = 12\n"
    )
    _finalize(
        _spec("ollama", "gemma4:e4b"),
        _spec("groq", "llama-3.3-70b", "GROQ_API_KEY"),
        vault_input=str(vault),
    )

    assert (vault / "synto.toml").exists(), "setup must write synto.toml for a legacy vault"
    c = Config.from_vault(vault)  # must not raise on the legacy-only-vault guard
    assert c.resolve_role("heavy").provider_kind == "groq"
    data = tomllib.loads((vault / "synto.toml").read_text())
    assert data["pipeline"]["max_concepts_per_source"] == 12  # legacy [pipeline] migrated forward
