"""Pin the NVIDIA / NGC connection contract (#24).

Issue #24 hosts a large model with NVIDIA NGC. All realistic NGC inference paths are
OpenAI-compatible; synto reaches them by pointing a provider block at the right url
(+ optional api_key_env). These tests lock that down so a future provider-table or
client change can't silently break NVIDIA, and document the no-auth recommendation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synto.client_factory import build_router
from synto.config import Config, ProviderBlock
from synto.openai_compat_client import OpenAICompatClient
from synto.providers import get_provider


@pytest.fixture
def clean_keys(monkeypatch):
    """Isolate key resolution from the dev/CI environment and global config."""
    for var in ("NVIDIA_API_KEY", "NGC_API_KEY", "SYNTO_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("synto.global_config.load_global_config", lambda: None)


def _cfg(block: ProviderBlock, model: str = "qwen/qwen2.5-72b-instruct") -> Config:
    return Config(
        vault="/tmp/v",
        providers={"x": block},
        models={"heavy": {"provider": "x", "model": model}},
    )


def _chat_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {},
    }
    return resp


# ── registry entry ────────────────────────────────────────────────────────────


def test_nvidia_registry_entry():
    p = get_provider("nvidia")
    assert p is not None
    assert p.default_url == "https://integrate.api.nvidia.com/v1"
    assert p.requires_auth is True
    assert p.anthropic_compat is False
    assert p.supports_json_mode is True
    assert p.env_var == "NVIDIA_API_KEY"


# ── API Catalog (cloud, nvapi- key) ─────────────────────────────────────────────


def test_nvidia_api_catalog_resolve(clean_keys, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    r = _cfg(ProviderBlock(name="nvidia")).resolve_role("heavy")
    assert r.provider_kind == "nvidia"
    assert r.url == "https://integrate.api.nvidia.com/v1"
    assert r.api_key == "nvapi-test"
    assert r.anthropic_compat is False
    assert r.supports_json_mode is True


def test_nvidia_router_dispatches_openai_compat(clean_keys, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    router = build_router(_cfg(ProviderBlock(name="nvidia")))
    try:
        assert isinstance(router.endpoint("heavy").client, OpenAICompatClient)
    finally:
        router.close()


def test_nvidia_posts_to_v1_chat_completions_with_bearer():
    client = OpenAICompatClient(
        base_url="https://integrate.api.nvidia.com/v1",
        provider_name="nvidia",
        api_key="nvapi-test",
    )
    # Auth header is set at construction (standard Bearer, not Azure api-key).
    assert client._client.headers.get("authorization") == "Bearer nvapi-test"

    captured: dict = {}

    def fake_post(url, json=None, **kw):
        captured["url"] = url
        return _chat_response()

    with patch.object(client._client, "post", side_effect=fake_post):
        client.generate(prompt="hi", model="qwen/qwen2.5-72b-instruct")
    # No double /v1; exactly the OpenAI chat-completions route.
    assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    client.close()


# ── self-hosted NIM (model hosted with NGC, run on own infra) ───────────────────


def test_self_hosted_nim_custom_no_auth(clean_keys):
    """Recommended no-auth path: name='custom' + own url → no key, no auth header."""
    r = _cfg(
        ProviderBlock(name="custom", url="http://nim.local:8000/v1"), model="meta/llama-3.1-70b"
    ).resolve_role("heavy")
    assert r.provider_kind == "custom"
    assert r.url == "http://nim.local:8000/v1"
    assert r.api_key is None

    client = OpenAICompatClient(base_url=r.url, provider_name="custom", api_key=None)
    assert "authorization" not in {k.lower() for k in client._client.headers.keys()}

    captured: dict = {}

    def fake_post(url, json=None, **kw):
        captured["url"] = url
        return _chat_response()

    with patch.object(client._client, "post", side_effect=fake_post):
        client.generate(prompt="hi", model="meta/llama-3.1-70b")
    assert captured["url"] == "http://nim.local:8000/v1/chat/completions"
    client.close()


def test_self_hosted_nvidia_block_leaks_env_key(clean_keys, monkeypatch):
    """Footgun pinned: name='nvidia' + own url, no api_key_env, but NVIDIA_API_KEY in env
    → the registry env var is auto-sent. Documents why 'custom' is the no-auth recommendation.
    If url-override is ever made to suppress the registry env var, this assertion flips."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-leak")
    r = _cfg(ProviderBlock(name="nvidia", url="http://nim.local:8000/v1")).resolve_role("heavy")
    assert r.url == "http://nim.local:8000/v1"
    assert r.api_key == "nvapi-leak"


# ── NVCF LLM Gateway (NV-managed, NGC key) ──────────────────────────────────────


def test_nvcf_gateway_uses_ngc_key(clean_keys, monkeypatch):
    monkeypatch.setenv("NGC_API_KEY", "ngc-test")
    r = _cfg(
        ProviderBlock(
            name="nvidia",
            url="https://fid.invocation.api.nvcf.nvidia.com/v1",
            api_key_env="NGC_API_KEY",
        )
    ).resolve_role("heavy")
    # Block api_key_env beats the registry NVIDIA_API_KEY default.
    assert r.api_key == "ngc-test"
    assert r.url == "https://fid.invocation.api.nvcf.nvidia.com/v1"

    client = OpenAICompatClient(base_url=r.url, provider_name="nvidia", api_key=r.api_key)
    assert client._client.headers.get("authorization") == "Bearer ngc-test"
    client.close()
