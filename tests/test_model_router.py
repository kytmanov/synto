"""Tests for ModelRouter: per-role clients, connection de-duplication, dispatch.

Why: #24 needs fast and heavy on different providers/accounts, but roles sharing one
connection must share a client (don't open N sockets for one Ollama). require_healthy
must not let an optional embed endpoint block the pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from synto.anthropic_compat_client import AnthropicCompatClient
from synto.client_factory import ModelRouter, build_router
from synto.config import Config, ProviderBlock
from synto.ollama_client import OllamaClient
from synto.openai_compat_client import OpenAICompatClient


def _cfg(**kw) -> Config:
    return Config(vault="/tmp/v", **kw)


def test_roles_on_same_connection_share_one_client():
    router = build_router(_cfg())  # default: fast + heavy both default Ollama
    fast = router.endpoint("fast")
    heavy = router.endpoint("heavy")
    assert fast.client is heavy.client  # de-duplicated
    assert isinstance(fast.client, OllamaClient)


def test_split_providers_get_distinct_clients_and_classes():
    cfg = _cfg(
        providers={
            "local": ProviderBlock(name="ollama"),
            "cloud": ProviderBlock(name="groq", url="https://api.groq.com/openai/v1"),
        },
        models={
            "fast": {"provider": "local", "model": "gemma4:e4b"},
            "heavy": {"provider": "cloud", "model": "llama-3.3-70b"},
        },
    )
    router = build_router(cfg)
    fast = router.endpoint("fast")
    heavy = router.endpoint("heavy")
    assert fast.client is not heavy.client
    assert isinstance(fast.client, OllamaClient)
    assert isinstance(heavy.client, OpenAICompatClient)


def test_anthropic_compat_dispatch_for_kimi():
    cfg = _cfg(
        providers={"k": ProviderBlock(name="kimi", url="https://api.kimi.com/coding")},
        models={"heavy": {"provider": "k", "model": "kimi-model"}},
    )
    router = build_router(cfg)
    assert isinstance(router.endpoint("heavy").client, AnthropicCompatClient)


def test_per_role_params_flow_to_endpoint():
    cfg = _cfg(
        models={
            "fast": {"model": "f"},  # think defaults False
            "heavy": {"model": "h", "ctx": 9000, "think": True, "options": {"top_p": 0.8}},
        }
    )
    router = build_router(cfg)
    heavy = router.endpoint("heavy")
    assert (heavy.model, heavy.ctx, heavy.think, heavy.options) == ("h", 9000, True, {"top_p": 0.8})
    assert router.endpoint("fast").think is False


def test_require_healthy_probes_unique_fast_heavy_only():
    cfg = _cfg(
        providers={
            "local": ProviderBlock(name="ollama"),
            "cloud": ProviderBlock(name="groq", url="https://api.groq.com/openai/v1"),
        },
        models={
            "fast": {"provider": "local", "model": "f"},
            "heavy": {"provider": "cloud", "model": "h"},
            "embed": {"provider": "cloud", "model": "e"},  # shares cloud; not separately probed
        },
    )
    fake_clients: list[MagicMock] = []

    def fake_build(resolved, cache):
        m = MagicMock()
        fake_clients.append(m)
        return m

    with patch("synto.client_factory._build_client_for", side_effect=fake_build):
        router = ModelRouter(cfg)
        router.require_healthy()
        # Two unique connections (local, cloud) => two clients, each healthchecked once.
        assert len(fake_clients) == 2
        for c in fake_clients:
            c.require_healthy.assert_called_once()


def test_close_closes_each_client_once():
    fake_clients: list[MagicMock] = []

    def fake_build(resolved, cache):
        m = MagicMock()
        fake_clients.append(m)
        return m

    with patch("synto.client_factory._build_client_for", side_effect=fake_build):
        router = ModelRouter(_cfg())
        router.endpoint("fast")
        router.endpoint("heavy")  # same connection => same client
        router.close()
        assert len(fake_clients) == 1
        fake_clients[0].close.assert_called_once()
