"""Tests for per-role provider/parameter resolution (Config.resolve_role).

These encode *why* the resolution matters: a user must be able to point each role at a
different account (own key or none), get role-aware thinking defaults, and have config
mistakes fail loud at load rather than mid-run.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synto.config import Config, ProviderBlock


def _cfg(**kw) -> Config:
    return Config(vault="/tmp/vault", **kw)


# ── shape: string vs table form ───────────────────────────────────────────────


def test_string_form_uses_default_provider_with_role_ctx():
    c = _cfg()
    fast = c.resolve_role("fast")
    heavy = c.resolve_role("heavy")
    assert fast.provider_kind == "ollama"
    assert fast.model == "gemma4:e4b"
    assert fast.ctx == 16384  # legacy fast_ctx
    assert heavy.ctx == 32768  # legacy heavy_ctx


def test_table_form_per_role_provider_and_params():
    c = _cfg(
        providers={
            "default": ProviderBlock(name="ollama"),
            "ngc": ProviderBlock(name="custom", url="https://x/v1", api_key_env="NGC_API_KEY"),
        },
        models={
            "fast": {"provider": "default", "model": "gemma4:e4b"},
            "heavy": {"provider": "ngc", "model": "big", "ctx": 40000},
        },
    )
    fast = c.resolve_role("fast")
    heavy = c.resolve_role("heavy")
    assert (fast.provider_kind, fast.url) == ("ollama", "http://localhost:11434")
    assert (heavy.provider_kind, heavy.url, heavy.ctx) == ("custom", "https://x/v1", 40000)
    # Different connections => different clients downstream.
    assert fast.connection_key != heavy.connection_key


# ── role-aware think default (the #31 decision) ───────────────────────────────


def test_think_default_fast_off_heavy_on():
    c = _cfg()
    # fast extraction must not waste budget thinking; heavy keeps the model's own default.
    assert c.resolve_role("fast").think is False
    assert c.resolve_role("heavy").think is None


def test_think_explicit_override_wins_for_both_roles():
    c = _cfg(
        models={
            "fast": {"model": "m", "think": True},
            "heavy": {"model": "m", "think": False},
        }
    )
    assert c.resolve_role("fast").think is True
    assert c.resolve_role("heavy").think is False


# ── options passthrough merge (provider then model) ───────────────────────────


def test_options_merge_model_overrides_provider():
    c = _cfg(
        providers={"default": ProviderBlock(name="ollama", options={"top_k": 10, "seed": 1})},
        models={"heavy": {"provider": "default", "model": "m", "options": {"top_k": 99}}},
    )
    opts = c.resolve_role("heavy").options
    assert opts == {"top_k": 99, "seed": 1}


# ── precedence: default block > [provider] > [ollama] ─────────────────────────


def test_default_block_takes_precedence_over_legacy_provider():
    c = _cfg(
        provider={"name": "groq", "url": "https://api.groq.com/openai/v1"},
        providers={"default": ProviderBlock(name="ollama", url="http://local:1234")},
    )
    # String-form role with no explicit provider -> the "default" block, not legacy [provider].
    assert c.resolve_role("fast").provider_kind == "ollama"


def test_legacy_provider_block_still_resolves():
    c = _cfg(provider={"name": "groq", "url": "https://api.groq.com/openai/v1"})
    r = c.resolve_role("heavy")
    assert r.provider_kind == "groq"
    assert r.url == "https://api.groq.com/openai/v1"


def test_legacy_ollama_block_still_resolves():
    c = _cfg(ollama={"url": "http://host:9999", "heavy_ctx": 4096})
    r = c.resolve_role("heavy")
    assert r.provider_kind == "ollama"
    assert (r.url, r.ctx) == ("http://host:9999", 4096)


# ── fail loud at load (Rule 11) ───────────────────────────────────────────────


def test_unknown_alias_raises_at_load():
    with pytest.raises(ValidationError):
        _cfg(models={"heavy": {"provider": "ghost", "model": "m"}})


def test_unknown_provider_name_without_url_raises():
    with pytest.raises(ValidationError):
        _cfg(providers={"x": ProviderBlock(name="not-a-real-provider")})


def test_unknown_provider_name_with_url_is_allowed():
    # Custom/self-hosted endpoint: unknown name is fine as long as a url is given.
    c = _cfg(
        providers={"x": ProviderBlock(name="my-llm", url="http://h/v1")},
        models={"heavy": {"provider": "x", "model": "m"}},
    )
    assert c.resolve_role("heavy").url == "http://h/v1"


def test_misspelled_top_level_key_raises():
    with pytest.raises(ValidationError):
        _cfg(models={"heavy": {"model": "m", "temprature": 0.5}})


# ── per-account API keys: own key, or none ────────────────────────────────────


def test_local_block_resolves_to_no_key(monkeypatch):
    monkeypatch.delenv("SYNTO_API_KEY", raising=False)
    c = _cfg(providers={"default": ProviderBlock(name="ollama")})
    assert c.resolve_role("fast").api_key is None


def test_two_accounts_same_provider_get_distinct_keys(monkeypatch):
    monkeypatch.setenv("KEY_A", "aaa")
    monkeypatch.setenv("KEY_B", "bbb")
    c = _cfg(
        providers={
            "a": ProviderBlock(name="openrouter", api_key_env="KEY_A"),
            "b": ProviderBlock(name="openrouter", api_key_env="KEY_B"),
        },
        models={
            "fast": {"provider": "a", "model": "m"},
            "heavy": {"provider": "b", "model": "m"},
        },
    )
    fast = c.resolve_role("fast")
    heavy = c.resolve_role("heavy")
    assert (fast.api_key, heavy.api_key) == ("aaa", "bbb")
    # Same provider+url but different keys must still be two distinct connections.
    assert fast.connection_key != heavy.connection_key


# ── per-invocation CLI overrides (--provider / --provider-url) ────────────────


def test_provider_url_override_alone_keeps_kind_and_key(monkeypatch):
    # `synto query --provider-url http://other` must hit the SAME provider kind at a different
    # endpoint, keeping the configured api_key_env (it's the same account, just relocated).
    monkeypatch.setenv("GROQ_KEY", "secret")
    c = _cfg(
        providers={
            "default": ProviderBlock(
                name="groq", url="https://api.groq.com/openai/v1", api_key_env="GROQ_KEY"
            )
        },
        models={"heavy": {"provider": "default", "model": "m"}},
        provider_override_url="http://localhost:1234/v1",
    )
    r = c.resolve_role("heavy")
    assert r.provider_kind == "groq"  # kind unchanged
    assert r.url == "http://localhost:1234/v1"  # only the endpoint moved
    assert r.api_key == "secret"  # configured key preserved
    assert r.api_key_env == "GROQ_KEY"


def test_provider_override_replaces_kind_and_url():
    c = _cfg(
        providers={"default": ProviderBlock(name="ollama")},
        models={"heavy": {"provider": "default", "model": "m"}},
        provider_override="groq",
        provider_override_url="https://api.groq.com/openai/v1",
    )
    r = c.resolve_role("heavy")
    assert r.provider_kind == "groq"
    assert r.url == "https://api.groq.com/openai/v1"


def test_override_preserves_per_role_model_and_params():
    # An override changes only the connection; per-role model/ctx/think/temperature survive.
    c = _cfg(
        models={"heavy": {"model": "big", "ctx": 40000, "think": True, "temperature": 0.3}},
        provider_override_url="http://relocated/v1",
    )
    r = c.resolve_role("heavy")
    assert (r.model, r.ctx, r.think, r.temperature) == ("big", 40000, True, 0.3)
    assert r.url == "http://relocated/v1"
