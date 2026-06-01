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
