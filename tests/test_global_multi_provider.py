"""Global config persistence + reproduction of multi-provider setups (#24).

`synto setup` saves a per-role multi-provider layout to the user-private global config
(api_key_env references only), and `synto init` reproduces it for new vaults — symmetric
with the single-provider path. These tests isolate XDG_CONFIG_HOME.
"""

from __future__ import annotations

import re

import pytest
from click.testing import CliRunner

from synto.cli import cli
from synto.config import Config, ModelProfile, ProviderBlock
from synto.global_config import (
    GlobalConfig,
    _global_config_path,
    load_global_config,
    save_global_config,
)
from synto.paths import CONFIG_FILE_NAME


@pytest.fixture(autouse=True)
def _isolate_global_config(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _multi() -> GlobalConfig:
    return GlobalConfig(
        providers={
            "default": ProviderBlock(name="ollama", url="http://localhost:11434"),
            "ngc": ProviderBlock(
                name="nvidia",
                url="https://integrate.api.nvidia.com/v1",
                api_key_env="NGC_API_KEY",
            ),
        },
        models={
            "fast": ModelProfile(provider="default", model="gemma4:e4b", ctx=16384),
            "heavy": ModelProfile(provider="ngc", model="qwen2.5:14b", ctx=32768),
        },
    )


def test_multi_provider_save_load_roundtrip():
    save_global_config(_multi())
    g = load_global_config()
    assert g is not None and g.is_multi_provider
    assert g.providers["ngc"].api_key_env == "NGC_API_KEY"
    assert g.models["heavy"].provider == "ngc"
    assert g.models["heavy"].model == "qwen2.5:14b"
    # No raw secret persisted — only the env-var name.
    assert not re.search(r"(?m)^\s*api_key\s*=", _global_config_path().read_text())


def test_save_load_preserves_every_field():
    """Durable guard against serializer drift: save_global_config hand-rolls TOML while the
    loader reads the full Pydantic model, so any added ProviderBlock/ModelProfile field that the
    serializer forgets is silently dropped on the next save (e.g. the `synto init --default`
    load→mutate→save round-trip rewrites the user's auth/generation config). Populate every
    lossy field and assert the round-trip is identity — this fails the moment a new field drifts.
    """
    original = GlobalConfig(
        vault="/tmp/vault",
        experimental_inline_source_citations=True,
        api_key="raw-secret",
        providers={
            "default": ProviderBlock(name="ollama", url="http://localhost:11434"),
            "cloud": ProviderBlock(
                name="groq",
                url="https://api.groq.com/openai/v1",
                timeout=120,
                api_key_env="GROQ_KEY",
                headers={"X-Org": "acme"},
                options={"top_k": 40},
            ),
        },
        models={
            "fast": ModelProfile(provider="default", model="gemma4:e4b", ctx=16384, think=False),
            "heavy": ModelProfile(
                provider="cloud",
                model="llama-3.3-70b",
                ctx=32768,
                think=True,
                temperature=0.4,
                options={"top_p": 0.9, "thinking": {"budget": 1}},
            ),
            "embed": ModelProfile(provider="default", model="nomic-embed-text", ctx=8192),
        },
        provider_keys={"cloud": "fallback-key"},
    )
    save_global_config(original)
    reloaded = load_global_config()
    assert reloaded == original


def test_legacy_flat_config_still_loads():
    save_global_config(GlobalConfig(provider_name="ollama", fast_model="m", heavy_model="h"))
    g = load_global_config()
    assert g is not None
    assert not g.is_multi_provider
    assert g.fast_model == "m" and g.heavy_model == "h"


def test_single_provider_setup_after_multi_clears_tables():
    """Mode switch: re-saving a single-provider config must drop the [providers.*]/[models.*]
    tables, so `init` reproduces the single provider, not the stale split."""
    save_global_config(_multi())
    assert load_global_config().is_multi_provider
    save_global_config(GlobalConfig(provider_name="ollama", fast_model="m", heavy_model="h"))
    g = load_global_config()
    assert not g.is_multi_provider
    assert not g.providers and not g.models


def test_init_reproduces_multi_provider_vault(tmp_path):
    save_global_config(_multi())
    vault = tmp_path / "new-vault"
    result = CliRunner().invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0, result.output

    cfg = Config.from_vault(vault)
    fast = cfg.resolve_role("fast")
    heavy = cfg.resolve_role("heavy")
    assert fast.provider_kind == "ollama" and fast.model == "gemma4:e4b"
    assert heavy.provider_kind == "nvidia"
    assert heavy.model == "qwen2.5:14b"
    assert heavy.url == "https://integrate.api.nvidia.com/v1"


def test_provider_keys_roundtrip_and_resolution(monkeypatch):
    """#4: per-alias raw key in the global config round-trips and is used by resolution."""
    from synto.api_keys import resolve_api_key

    for var in ("NGC_API_KEY", "NVIDIA_API_KEY", "SYNTO_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    g = _multi()
    g.provider_keys = {"ngc": "nvapi-secret"}
    save_global_config(g)

    loaded = load_global_config()
    assert loaded.provider_keys == {"ngc": "nvapi-secret"}
    assert "[provider_keys]" in _global_config_path().read_text()
    # Per-alias key is used when no env var is set; aliases without a key resolve to None.
    assert resolve_api_key("nvidia", alias="ngc") == "nvapi-secret"
    assert resolve_api_key("ollama", alias="default") is None


def test_init_leaves_multi_provider_vault_untouched(tmp_path):
    """A vault that already splits roles across providers must survive `init`.

    Regression: the respect-the-vault guard only compared the *default* provider name, so a vault
    with fast->default(ollama) + heavy->groq and a global config that also defaults to ollama (but
    a different heavy) fell through to the multi-provider rewrite and lost its heavy split.
    """
    vault = tmp_path / "split-vault"
    vault.mkdir()
    (vault / CONFIG_FILE_NAME).write_text(
        '[providers.default]\nname = "ollama"\nurl = "http://localhost:11434"\n\n'
        '[providers.gq]\nname = "groq"\napi_key_env = "GROQ_API_KEY"\n\n'
        '[models.fast]\nprovider = "default"\nmodel = "gemma4:e4b"\n\n'
        '[models.heavy]\nprovider = "gq"\nmodel = "llama-3.3-70b"\n'
    )
    # Global default also resolves to ollama, but its heavy provider is nvidia, not groq.
    save_global_config(_multi())

    result = CliRunner().invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0, result.output
    assert "uses multiple providers" in result.output

    # The user's deliberate heavy split survives — not rewritten to the global nvidia provider.
    heavy = Config.from_vault(vault).resolve_role("heavy")
    assert heavy.provider_kind == "groq"
    assert heavy.model == "llama-3.3-70b"


def test_per_role_setup_drops_stale_key_when_default_repointed(monkeypatch):
    """Re-running per-role setup must not send a previous provider's key to a new one.

    `provider_keys` is keyed by alias, and the fast role's alias is always "default". If the
    default provider is switched (Groq -> OpenRouter) without entering a new key, the old Groq
    secret under "default" must be dropped, not silently reused for OpenRouter.
    """
    from rich.console import Console

    from synto.cli import _finalize_per_role_providers

    for var in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "NGC_API_KEY", "SYNTO_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    # Prior setup: default == Groq, with a raw key saved under "default".
    save_global_config(
        GlobalConfig(
            providers={"default": ProviderBlock(name="groq", url="https://api.groq.com/openai/v1")},
            models={"fast": ModelProfile(provider="default", model="llama-3.1-8b")},
            provider_keys={"default": "old-groq-key"},
        )
    )

    # Re-run: switch the fast/default provider to OpenRouter, no new raw key typed.
    _finalize_per_role_providers(
        Console(),
        fast={
            "name": "openrouter",
            "url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "azure_api_version": None,
            "model": "x/y",
            "timeout": 120,
        },
        heavy={
            "name": "nvidia",
            "url": "https://integrate.api.nvidia.com/v1",
            "api_key_env": "NGC_API_KEY",
            "azure_api_version": None,
            "model": "qwen2.5:14b",
            "timeout": 120,
        },
        vault_input="",
        citations=False,
        fast_api_key=None,
    )

    from synto.api_keys import resolve_api_key

    # The harm: the stale Groq key must not be returned for the new OpenRouter default.
    assert resolve_api_key("openrouter", alias="default") is None
    assert "default" not in (load_global_config().provider_keys or {})


def test_per_role_setup_keeps_key_when_default_unchanged(monkeypatch):
    """Companion to the stale-key drop: an unchanged default connection keeps its key."""
    from rich.console import Console

    from synto.api_keys import resolve_api_key
    from synto.cli import _finalize_per_role_providers

    for var in ("GROQ_API_KEY", "NGC_API_KEY", "SYNTO_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    groq_url = "https://api.groq.com/openai/v1"
    save_global_config(
        GlobalConfig(
            providers={"default": ProviderBlock(name="groq", url=groq_url)},
            models={"fast": ModelProfile(provider="default", model="llama-3.1-8b")},
            provider_keys={"default": "groq-secret"},
        )
    )

    # Re-run keeping the same default (Groq, same URL); only the heavy provider differs.
    _finalize_per_role_providers(
        Console(),
        fast={
            "name": "groq",
            "url": groq_url,
            "api_key_env": "GROQ_API_KEY",
            "azure_api_version": None,
            "model": "llama-3.1-8b",
            "timeout": 120,
        },
        heavy={
            "name": "nvidia",
            "url": "https://integrate.api.nvidia.com/v1",
            "api_key_env": "NGC_API_KEY",
            "azure_api_version": None,
            "model": "qwen2.5:14b",
            "timeout": 120,
        },
        vault_input="",
        citations=False,
        fast_api_key=None,
    )

    assert (load_global_config().provider_keys or {}).get("default") == "groq-secret"
    assert resolve_api_key("groq", alias="default") == "groq-secret"
