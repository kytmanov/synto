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
