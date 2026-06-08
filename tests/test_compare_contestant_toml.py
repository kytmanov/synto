"""The `synto compare` contestant vault must faithfully reproduce per-role providers.

Why: #24's headline feature is fast and heavy on *different* providers/accounts with their
own params. If the ephemeral compare vault collapses both roles onto one provider (or drops
think/temperature/options), the compare measures the wrong configuration — a silently invalid
result. These tests pin that the materialized synto.toml round-trips the split and the params.
"""

from __future__ import annotations

from synto.compare.runner import _write_effective_compare_toml
from synto.config import Config, ProviderBlock


def _write_and_reload(tmp_path, config: Config) -> Config:
    (tmp_path / "synto.toml")  # path used by the writer
    _write_effective_compare_toml(tmp_path, config)
    return Config.from_vault(tmp_path)


def test_split_providers_survive_materialization(tmp_path):
    config = Config(
        vault=str(tmp_path / "active"),
        providers={
            "local": ProviderBlock(name="ollama", url="http://localhost:11434"),
            "cloud": ProviderBlock(
                name="groq", url="https://api.groq.com/openai/v1", api_key_env="GROQ_KEY"
            ),
        },
        models={
            "fast": {"provider": "local", "model": "gemma4:e4b"},
            "heavy": {"provider": "cloud", "model": "llama-3.3-70b"},
        },
    )
    reloaded = _write_and_reload(tmp_path, config)
    fast = reloaded.resolve_role("fast")
    heavy = reloaded.resolve_role("heavy")
    # The fast role must NOT have been silently moved onto the heavy (cloud) provider.
    assert (fast.provider_kind, fast.url) == ("ollama", "http://localhost:11434")
    assert (heavy.provider_kind, heavy.url) == ("groq", "https://api.groq.com/openai/v1")
    assert fast.model == "gemma4:e4b" and heavy.model == "llama-3.3-70b"
    # The env-var name is reproduced (so the contestant resolves the right account)...
    assert heavy.api_key_env == "GROQ_KEY"
    # ...but the contestant toml never contains a raw secret.
    assert "api_key =" not in (tmp_path / "synto.toml").read_text()


def test_per_role_params_survive_materialization(tmp_path):
    config = Config(
        vault=str(tmp_path / "active"),
        providers={"default": ProviderBlock(name="ollama")},
        models={
            "fast": {"provider": "default", "model": "f", "think": False},
            "heavy": {
                "provider": "default",
                "model": "h",
                "ctx": 12000,
                "think": True,
                "temperature": 0.4,
                "options": {"top_p": 0.9},
            },
        },
    )
    reloaded = _write_and_reload(tmp_path, config)
    heavy = reloaded.resolve_role("heavy")
    assert (heavy.ctx, heavy.think, heavy.temperature) == (12000, True, 0.4)
    assert heavy.options == {"top_p": 0.9}
    # fast think stays explicitly off (not lost to the role default).
    assert reloaded.resolve_role("fast").think is False


def test_all_fields_survive_one_materialization(tmp_path):
    # Drift guard: a single materialization must carry *every* per-role/per-provider field at once
    # (split providers, headers, options, think, temperature, embed). Because the contestant is now
    # serialized from the Pydantic models, a field added to the schema cannot be silently dropped
    # here — this test fails the moment resolution stops reproducing one of them.
    config = Config(
        vault=str(tmp_path / "active"),
        providers={
            "local": ProviderBlock(name="ollama", url="http://localhost:11434"),
            "cloud": ProviderBlock(
                name="groq",
                url="https://api.groq.com/openai/v1",
                api_key_env="GROQ_KEY",
                headers={"X-Org": "acme"},
            ),
        },
        models={
            "fast": {"provider": "local", "model": "f", "think": False},
            "heavy": {
                "provider": "cloud",
                "model": "h",
                "ctx": 12000,
                "think": True,
                "temperature": 0.4,
                "options": {"top_p": 0.9, "thinking": {"budget": 1}},
            },
            "embed": {"provider": "local", "model": "nomic-embed-text"},
        },
    )
    reloaded = _write_and_reload(tmp_path, config)
    fast, heavy, embed = (reloaded.resolve_role(r) for r in ("fast", "heavy", "embed"))
    assert fast.provider_kind == "ollama" and fast.think is False
    assert heavy.provider_kind == "groq" and heavy.api_key_env == "GROQ_KEY"
    assert heavy.headers == {"X-Org": "acme"}
    assert (heavy.ctx, heavy.think, heavy.temperature) == (12000, True, 0.4)
    assert heavy.options == {"top_p": 0.9, "thinking": {"budget": 1}}
    assert embed.provider_kind == "ollama" and embed.model == "nomic-embed-text"
    assert "api_key =" not in (tmp_path / "synto.toml").read_text()


def test_embed_split_survives_materialization(tmp_path):
    # Guards the not-yet-live RAG path: resolve_role("embed") is unreachable until a RAG store
    # is wired in, but a hand-configured embed split (heavy=cloud, embed=local) must still be
    # reproduced faithfully so the contestant/SWITCH snippet does not misrepresent the config.
    config = Config(
        vault=str(tmp_path / "active"),
        providers={
            "local": ProviderBlock(name="ollama", url="http://localhost:11434"),
            "cloud": ProviderBlock(
                name="groq", url="https://api.groq.com/openai/v1", api_key_env="GROQ_KEY"
            ),
        },
        models={
            "fast": {"provider": "local", "model": "gemma4:e4b"},
            "heavy": {"provider": "cloud", "model": "llama-3.3-70b"},
            "embed": {"provider": "local", "model": "nomic-embed-text"},
        },
    )
    reloaded = _write_and_reload(tmp_path, config)
    embed = reloaded.resolve_role("embed")
    # The embed role must keep its distinct (local) provider, not collapse onto heavy's cloud.
    assert (embed.provider_kind, embed.url) == ("ollama", "http://localhost:11434")
    assert embed.model == "nomic-embed-text"


def test_default_embed_emits_no_embed_block(tmp_path):
    # String-form (default) embed is covered by [providers.default]; no [models.embed] churn.
    config = Config(
        vault=str(tmp_path / "active"),
        providers={"default": ProviderBlock(name="ollama")},
        models={"fast": {"provider": "default", "model": "f"}, "heavy": {"model": "h"}},
    )
    _write_effective_compare_toml(tmp_path, config)
    assert "[models.embed]" not in (tmp_path / "synto.toml").read_text()


def test_legacy_active_config_still_materializes(tmp_path):
    # A legacy [provider] active vault (no [providers.*]) must still produce a loadable contestant.
    config = Config(
        vault=str(tmp_path / "active"),
        provider={"name": "groq", "url": "https://api.groq.com/openai/v1"},
        models={"fast": "f", "heavy": "h"},
    )
    reloaded = _write_and_reload(tmp_path, config)
    assert reloaded.resolve_role("heavy").provider_kind == "groq"
    assert reloaded.resolve_role("fast").model == "f"


def test_local_openai_compare_materialization_uses_full_article_budget(tmp_path):
    config = Config(
        vault=str(tmp_path / "active"),
        providers={"local": ProviderBlock(name="lm_studio", url="http://localhost:1234/v1")},
        models={
            "fast": {"provider": "local", "model": "gemma4:e4b"},
            "heavy": {"provider": "local", "model": "qwen/qwen3.5-9b"},
        },
    )

    reloaded = _write_and_reload(tmp_path, config)

    assert reloaded.pipeline.auto_approve is True
    assert reloaded.pipeline.auto_commit is False
    assert reloaded.pipeline.concept_draft_soft_cap == "article_max_tokens"


def test_ollama_compare_materialization_keeps_existing_soft_cap(tmp_path):
    config = Config(
        vault=str(tmp_path / "active"),
        providers={"default": ProviderBlock(name="ollama", url="http://localhost:11434")},
        models={
            "fast": {"provider": "default", "model": "gemma4:e4b"},
            "heavy": {"provider": "default", "model": "qwen2.5:14b"},
        },
    )

    reloaded = _write_and_reload(tmp_path, config)

    assert reloaded.pipeline.concept_draft_soft_cap == 2400
