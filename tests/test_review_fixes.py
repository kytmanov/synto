"""Regression tests for the external-review fixes on PR #41.

Covers: #2 (CLI --provider override on new-format vaults) and #6 (per-role temperature
actually reaching the compile/query LLM calls).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from conftest import as_router

from synto.config import Config, ModelProfile, default_wiki_toml


def _new_format_vault(tmp_path):
    d = tmp_path / "v"
    for sub in ("raw", "wiki", "wiki/.drafts", ".synto"):
        (d / sub).mkdir(parents=True)
    (d / "synto.toml").write_text(default_wiki_toml())  # [providers.default] + [models.<role>]
    return d


# ── #2: --provider/--provider-url override works on new-format vaults ──────────


def test_provider_override_supersedes_new_format_vault(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    d = _new_format_vault(tmp_path)
    c = Config.from_vault(d, provider_override="groq")
    fast, heavy = c.resolve_role("fast"), c.resolve_role("heavy")
    # Both roles routed to the override, at the override's registry URL...
    assert fast.provider_kind == "groq" and heavy.provider_kind == "groq"
    assert heavy.url == "https://api.groq.com/openai/v1"
    # ...while each role keeps its own configured model.
    assert fast.model == "gemma4:e4b" and heavy.model == "qwen2.5:14b"


def test_provider_override_url_honored(tmp_path):
    d = _new_format_vault(tmp_path)
    c = Config.from_vault(d, provider_override="custom", provider_override_url="http://h:9/v1")
    assert c.resolve_role("heavy").url == "http://h:9/v1"


def test_no_override_uses_configured_provider(tmp_path):
    d = _new_format_vault(tmp_path)
    assert Config.from_vault(d).resolve_role("heavy").provider_kind == "ollama"


def test_cli_model_override_kwargs_maps_to_provider_override():
    from synto.cli import _model_override_kwargs

    kw = _model_override_kwargs(None, None, "groq", "http://x/v1")
    assert kw == {"provider_override": "groq", "provider_override_url": "http://x/v1"}


# ── #6: per-role temperature reaches the compile call ─────────────────────────


def test_heavy_temperature_reaches_compile(config, db):
    from synto.models import RawNoteRecord, SingleArticle
    from synto.pipeline import compile as compile_mod

    (config.vault / "raw").mkdir(exist_ok=True)
    (config.vault / "raw" / "n.md").write_text("# N\n\nContent about Topic and ideas.")
    db.upsert_raw(RawNoteRecord(path="raw/n.md", content_hash="h", status="ingested"))
    db.upsert_concepts("raw/n.md", ["Topic"])
    # Configure a per-role heavy temperature.
    config.models.heavy = ModelProfile(model="heavy-m", temperature=0.33)

    captured: list[float | None] = []

    def fake_request_structured(*args, **kwargs):
        captured.append(kwargs.get("temperature"))
        return SingleArticle(title="Topic", content="Body.", tags=[], summary="s")

    with patch.object(compile_mod, "request_structured", side_effect=fake_request_structured):
        compile_mod.compile_concepts(config=config, router=as_router(MagicMock(), config), db=db)

    assert captured, "compile made no LLM call"
    assert 0.33 in captured, f"heavy temperature 0.33 never reached request_structured: {captured}"


# ── #5: Azure api_version survives the multi-provider vault path ──────────────


def test_azure_api_version_preserved_in_multi_provider(tmp_path):
    from synto.config import multi_provider_vault_toml

    providers = [
        {"alias": "default", "name": "ollama", "url": "http://localhost:11434", "timeout": 600},
        {
            "alias": "az",
            "name": "azure",
            "url": "https://r.openai.azure.com/openai/deployments/gpt4",
            "timeout": 120,
            "api_key_env": "AZURE_OPENAI_API_KEY",
            "azure_api_version": "2025-01-01-preview",
        },
    ]
    models = {
        "fast": {"provider": "default", "model": "m", "ctx": 8192},
        "heavy": {"provider": "az", "model": "gpt-4", "ctx": 32768},
    }
    d = tmp_path / "v"
    for sub in ("raw", "wiki", "wiki/.drafts", ".synto"):
        (d / sub).mkdir(parents=True)
    (d / "synto.toml").write_text(multi_provider_vault_toml(providers, models))
    rh = Config.from_vault(d).resolve_role("heavy")
    assert rh.azure is True
    assert rh.azure_api_version == "2025-01-01-preview"


def test_provider_models_head_omits_optional_keys_when_absent():
    # Setup/init callers pass no think/temperature/options/headers — output must stay minimal
    # (no spurious keys) so their existing round-trips are byte-identical.
    from synto.config import _provider_models_head

    head = _provider_models_head(
        [{"alias": "default", "name": "ollama", "url": "http://x", "timeout": 600}],
        {
            "fast": {"provider": "default", "model": "f", "ctx": 8192},
            "heavy": {"provider": "default", "model": "h", "ctx": 16384},
        },
    )
    for absent in ("think", "temperature", "options", "headers"):
        assert absent not in head


def test_provider_models_head_emits_optional_keys_when_present():
    from synto.config import _provider_models_head

    head = _provider_models_head(
        [
            {
                "alias": "default",
                "name": "ollama",
                "url": "http://x",
                "timeout": 600,
                "headers": {"X-Org": "acme"},
            }
        ],
        {
            "fast": {"provider": "default", "model": "f", "ctx": 8192, "think": False},
            "heavy": {
                "provider": "default",
                "model": "h",
                "ctx": 16384,
                "think": True,
                "temperature": 0.5,
                "options": {"top_p": 0.9},
            },
        },
    )
    assert 'headers = { "X-Org" = "acme" }' in head
    assert "think = false" in head and "think = true" in head
    assert "temperature = 0.5" in head
    assert 'options = { "top_p" = 0.9 }' in head


def test_model_string_override_keeps_role_provider_binding(tmp_path):
    # `synto compare/query --heavy-model X` overrides only the model id. On a new-format vault
    # whose alias isn't "default", a naive replace would drop provider="local" and silently fall
    # back to default/legacy (e.g. Ollama). The role's provider + ctx must survive.
    d = tmp_path / "v"
    for sub in ("raw", "wiki", "wiki/.drafts", ".synto"):
        (d / sub).mkdir(parents=True)
    (d / "synto.toml").write_text(
        '[providers.local]\nname = "lm_studio"\nurl = "http://localhost:1234/v1"\ntimeout = 600\n\n'
        '[models.fast]\nprovider = "local"\nmodel = "f"\nctx = 8192\n\n'
        '[models.heavy]\nprovider = "local"\nmodel = "h"\nctx = 8192\n'
    )
    c = Config.from_vault(d, models={"heavy": "qwen/qwen3.5-9b"})
    heavy = c.resolve_role("heavy")
    assert heavy.provider_kind == "lm_studio"  # NOT dropped to ollama/legacy
    assert heavy.url == "http://localhost:1234/v1"
    assert heavy.model == "qwen/qwen3.5-9b"  # only the model id changed
    assert heavy.ctx == 8192  # role ctx preserved


def test_dedup_role_connections_shares_and_splits():
    from synto.config import dedup_role_connections

    # Same connection -> one "default" alias shared by both roles.
    providers, role_alias = dedup_role_connections(
        {
            "fast": {"name": "ollama", "url": "http://x", "timeout": 600, "api_key_env": None},
            "heavy": {"name": "ollama", "url": "http://x", "timeout": 600, "api_key_env": None},
        }
    )
    assert len(providers) == 1 and providers[0]["alias"] == "default"
    assert role_alias["fast"] == role_alias["heavy"] == "default"

    # Different api_key_env (a different account) -> two distinct providers.
    providers, role_alias = dedup_role_connections(
        {
            "fast": {"name": "openrouter", "url": "http://x", "api_key_env": "KEY_A"},
            "heavy": {"name": "openrouter", "url": "http://x", "api_key_env": "KEY_B"},
        }
    )
    assert len(providers) == 2
    assert role_alias["fast"] != role_alias["heavy"]
    assert providers[0]["alias"] == "default"


def test_nested_options_round_trip_through_toml_writers(tmp_path):
    # Provider-native options can be nested ({"thinking": {"budget": 1}}) and the runtime accepts
    # them; the TOML writers must serialize+reload them, not crash.
    import tomllib

    from synto.compare.runner import _write_effective_compare_toml
    from synto.config import multi_provider_vault_toml

    providers = [{"alias": "default", "name": "ollama", "url": "http://x", "timeout": 600}]
    models = {
        "fast": {"provider": "default", "model": "f", "ctx": 8192},
        "heavy": {
            "provider": "default",
            "model": "h",
            "ctx": 16384,
            "options": {"thinking": {"budget": 1}, "stop": ["a", "b"]},
        },
    }
    text = multi_provider_vault_toml(providers, models)
    parsed = tomllib.loads(text)
    assert parsed["models"]["heavy"]["options"] == {"thinking": {"budget": 1}, "stop": ["a", "b"]}

    # And through the compare materializer (which goes via resolve_role -> role_providers_head).
    cfg = Config(
        vault=str(tmp_path / "active"),
        providers={"default": {"name": "ollama"}},
        models={
            "fast": {"provider": "default", "model": "f"},
            "heavy": {"provider": "default", "model": "h", "options": {"thinking": {"budget": 1}}},
        },
    )
    d = tmp_path / "contestant"
    d.mkdir()
    _write_effective_compare_toml(d, cfg)  # must not raise
    reloaded = Config.from_vault(d).resolve_role("heavy")
    assert reloaded.options == {"thinking": {"budget": 1}}


def test_role_providers_head_renders_split(tmp_path):
    # The switch snippet / contestant head must reproduce a fast≠heavy split, not collapse it.
    from synto.config import role_providers_head

    cfg = Config(
        vault=str(tmp_path / "v"),
        providers={
            "local": {"name": "ollama", "url": "http://localhost:11434"},
            "cloud": {"name": "groq", "url": "https://api.groq.com/openai/v1"},
        },
        models={
            "fast": {"provider": "local", "model": "f"},
            "heavy": {"provider": "cloud", "model": "h"},
        },
    )
    head = role_providers_head(cfg)
    d = tmp_path / "applied"
    for sub in ("raw", "wiki", "wiki/.drafts", ".synto"):
        (d / sub).mkdir(parents=True)
    (d / "synto.toml").write_text(head + "\n[pipeline]\n")
    applied = Config.from_vault(d)
    assert applied.resolve_role("fast").provider_kind == "ollama"
    assert applied.resolve_role("heavy").provider_kind == "groq"
