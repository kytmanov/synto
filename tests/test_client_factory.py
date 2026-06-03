"""Tests for client_factory.py API key resolution and provider selection."""

from __future__ import annotations

from unittest.mock import patch

from synto.api_keys import resolve_api_key
from synto.client_factory import build_client
from synto.config import Config


def test_resolve_api_key_explicit_env_override(monkeypatch):
    """Explicit api_key_env parameter takes priority."""
    monkeypatch.setenv("TEST_OVERRIDE_KEY", "explicit-key")
    key = resolve_api_key("groq", api_key_env_override="TEST_OVERRIDE_KEY")
    assert key == "explicit-key"


def test_resolve_api_key_explicit_env_empty_returns_none(monkeypatch):
    """Explicit env var set but empty → falls through."""
    monkeypatch.setenv("TEST_EMPTY_KEY", "")
    monkeypatch.delenv("SYNTO_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with patch("synto.global_config.load_global_config", return_value=None):
        key = resolve_api_key("groq", api_key_env_override="TEST_EMPTY_KEY")
        assert key is None


def test_resolve_api_key_provider_specific_env(monkeypatch):
    """Provider-specific env var (e.g. GROQ_API_KEY) is used."""
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.delenv("SYNTO_API_KEY", raising=False)
    key = resolve_api_key("groq")
    assert key == "groq-secret"


def test_resolve_api_key_generic_env(monkeypatch):
    """Generic SYNTO_API_KEY is used when provider-specific is absent."""
    monkeypatch.setenv("SYNTO_API_KEY", "generic-secret")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    key = resolve_api_key("groq")
    assert key == "generic-secret"


def test_resolve_api_key_global_config(monkeypatch, tmp_path):
    """API key from global config is used as fallback."""
    monkeypatch.delenv("SYNTO_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text('api_key = "global-secret"\n')

    with patch("synto.global_config.load_global_config") as mock_load:
        from synto.global_config import GlobalConfig

        mock_load.return_value = GlobalConfig(api_key="global-secret")
        key = resolve_api_key("groq")
        assert key == "global-secret"


def test_resolve_api_key_returns_none_when_no_key_found(monkeypatch):
    """Returns None when no key source is available."""
    monkeypatch.delenv("SYNTO_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with patch("synto.global_config.load_global_config", return_value=None):
        key = resolve_api_key("groq")
        assert key is None


def test_resolve_api_key_unknown_provider(monkeypatch):
    """Unknown provider with no env var → falls through to generic/global."""
    monkeypatch.setenv("SYNTO_API_KEY", "fallback-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    key = resolve_api_key("unknown")
    assert key == "fallback-key"


# ── resolve_api_key precedence INTERACTIONS ───────────────────────────────────
# The single-source tests above don't constrain ordering. These set several sources at
# once and assert which one wins — the contract documented in api_keys.py's module docstring.


def _gcfg(**kw):
    from synto.global_config import GlobalConfig

    return patch("synto.global_config.load_global_config", return_value=GlobalConfig(**kw))


def test_block_env_beats_provider_registry_env(monkeypatch):
    monkeypatch.setenv("MY_BLOCK_KEY", "block-key")
    monkeypatch.setenv("GROQ_API_KEY", "registry-key")
    with patch("synto.global_config.load_global_config", return_value=None):
        assert resolve_api_key("groq", block_api_key_env="MY_BLOCK_KEY") == "block-key"


def test_provider_registry_env_beats_per_alias_global_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "registry-key")
    monkeypatch.delenv("SYNTO_API_KEY", raising=False)
    with _gcfg(provider_keys={"acct1": "alias-key"}):
        assert resolve_api_key("groq", alias="acct1") == "registry-key"


def test_per_alias_global_key_beats_generic_env(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("SYNTO_API_KEY", "generic-key")
    with _gcfg(provider_keys={"acct1": "alias-key"}):
        assert resolve_api_key("groq", alias="acct1") == "alias-key"


def test_generic_env_beats_legacy_single_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("SYNTO_API_KEY", "generic-key")
    with _gcfg(api_key="legacy-key"):
        assert resolve_api_key("groq") == "generic-key"


def test_alias_none_does_not_pick_up_a_per_alias_key(monkeypatch):
    """A None alias must skip the per-alias step entirely — otherwise a role with no account
    binding could silently borrow some other account's key."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("SYNTO_API_KEY", "generic-key")
    with _gcfg(provider_keys={"acct1": "alias-key"}):
        assert resolve_api_key("groq", alias=None) == "generic-key"


def test_build_client_ollama(tmp_path):
    """build_client returns OllamaClient for ollama provider."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "raw").mkdir()
    (vault / "wiki").mkdir()
    (vault / ".synto").mkdir()
    (vault / "synto.toml").write_text(
        '[ollama]\nurl = "http://localhost:11434"\nfast_ctx = 8192\nheavy_ctx = 16384\n'
    )
    config = Config.from_vault(vault)
    client = build_client(config)
    from synto.ollama_client import OllamaClient

    assert isinstance(client, OllamaClient)
    client.close()


def test_build_client_openai_compat(tmp_path, monkeypatch):
    """build_client returns OpenAICompatClient for non-ollama providers."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "raw").mkdir()
    (vault / "wiki").mkdir()
    (vault / ".synto").mkdir()
    (vault / "synto.toml").write_text(
        '[provider]\nname = "groq"\nurl = "https://api.groq.com/openai/v1"\n'
    )
    config = Config.from_vault(vault)
    client = build_client(config)
    from synto.openai_compat_client import OpenAICompatClient

    assert isinstance(client, OpenAICompatClient)
    client.close()


def test_build_client_custom_provider(tmp_path, monkeypatch):
    """build_client handles custom/unknown provider names."""
    monkeypatch.setenv("SYNTO_API_KEY", "custom-key")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "raw").mkdir()
    (vault / "wiki").mkdir()
    (vault / ".synto").mkdir()
    (vault / "synto.toml").write_text(
        '[provider]\nname = "custom"\nurl = "http://localhost:9999/v1"\n'
    )
    config = Config.from_vault(vault)
    client = build_client(config)
    from synto.openai_compat_client import OpenAICompatClient

    assert isinstance(client, OpenAICompatClient)
    assert client.base_url == "http://localhost:9999/v1"
    client.close()


def test_build_client_kimi_returns_anthropic_client(tmp_path, monkeypatch):
    """build_client returns AnthropicCompatClient for Kimi provider."""
    monkeypatch.setenv("KIMI_API_KEY", "kimi-test-key")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "raw").mkdir()
    (vault / "wiki").mkdir()
    (vault / ".synto").mkdir()
    (vault / "synto.toml").write_text(
        '[provider]\nname = "kimi"\nurl = "https://api.kimi.com/coding"\n'
    )
    config = Config.from_vault(vault)
    client = build_client(config)
    from synto.anthropic_compat_client import AnthropicCompatClient

    assert isinstance(client, AnthropicCompatClient)
    assert client.base_url == "https://api.kimi.com/coding"
    client.close()
