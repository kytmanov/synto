"""Global user-level config for Synto."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .config import ModelProfile, ProviderBlock, to_toml
from .paths import APP_NAME


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vault: str | None = None
    ollama_url: str | None = None  # legacy — kept for backward compat
    fast_model: str | None = None
    heavy_model: str | None = None
    # Provider fields (new in v0.3)
    provider_name: str | None = None
    provider_url: str | None = None
    api_key: str | None = None  # never stored in wiki.toml; this file is user-private
    azure_api_version: str | None = None  # Azure OpenAI API version (e.g. "2024-02-15-preview")
    experimental_inline_source_citations: bool | None = None  # new-vault default only
    # Multi-provider (per-role) defaults: when both are set they supersede the flat
    # single-provider fields above, and `synto init` reproduces a multi-provider vault.
    # Provider blocks carry api_key_env references (the recommended path).
    providers: dict[str, ProviderBlock] = Field(default_factory=dict)
    models: dict[str, ModelProfile] | None = None
    # Optional raw API key per provider alias (user-private — same trust as the legacy
    # `api_key` above). Used as a fallback when no env var is set. `api_key_env` is preferred.
    provider_keys: dict[str, str] | None = None

    @property
    def is_multi_provider(self) -> bool:
        return bool(self.providers and self.models)


def _global_config_path() -> Path:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(appdata) / APP_NAME / "config.toml"
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(xdg) / APP_NAME / "config.toml"


def load_global_config() -> GlobalConfig | None:
    """Load global config. Returns None if missing or malformed — never raises."""
    path = _global_config_path()
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return GlobalConfig(**data)
    except Exception:
        return None


def save_global_config(cfg: GlobalConfig) -> None:
    """Write global config to disk. Creates parent directory if needed.

    Serialized through the single `to_toml` seam (model_dump → TOML), so this is the exact inverse
    of the `GlobalConfig(**tomllib.load(...))` read path: every field — including provider
    headers/options and model think/temperature/options — round-trips, and a new field needs no
    change here. Only set fields are written (exclude_unset), so a partial config stays minimal.
    """
    path = _global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_toml(cfg), encoding="utf-8")
