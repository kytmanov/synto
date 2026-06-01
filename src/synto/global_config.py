"""Global user-level config for Synto."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .config import ModelProfile, ProviderBlock
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
    """Write global config to disk. Creates parent directory if needed."""
    path = _global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if cfg.vault is not None:
        lines.append(f"vault = {_toml_str(cfg.vault)}")
    if cfg.ollama_url is not None:
        lines.append(f"ollama_url = {_toml_str(cfg.ollama_url)}")
    if cfg.fast_model is not None:
        lines.append(f"fast_model = {_toml_str(cfg.fast_model)}")
    if cfg.heavy_model is not None:
        lines.append(f"heavy_model = {_toml_str(cfg.heavy_model)}")
    if cfg.provider_name is not None:
        lines.append(f"provider_name = {_toml_str(cfg.provider_name)}")
    if cfg.provider_url is not None:
        lines.append(f"provider_url = {_toml_str(cfg.provider_url)}")
    if cfg.api_key is not None:
        lines.append(f"api_key = {_toml_str(cfg.api_key)}")
    if cfg.azure_api_version is not None:
        lines.append(f"azure_api_version = {_toml_str(cfg.azure_api_version)}")
    if cfg.experimental_inline_source_citations is not None:
        value = "true" if cfg.experimental_inline_source_citations else "false"
        lines.append(f"experimental_inline_source_citations = {value}")
    # Multi-provider tables must come AFTER all flat top-level keys (TOML requirement).
    for alias, block in cfg.providers.items():
        lines.append("")
        lines.append(f"[providers.{alias}]")
        lines.append(f"name = {_toml_str(block.name)}")
        if block.url:
            lines.append(f"url = {_toml_str(block.url)}")
        if block.timeout is not None:
            lines.append(f"timeout = {int(block.timeout)}")
        if block.api_key_env:
            lines.append(f"api_key_env = {_toml_str(block.api_key_env)}")
        if block.name == "azure" and block.azure_api_version:
            lines.append(f"azure_api_version = {_toml_str(block.azure_api_version)}")
    for role, prof in (cfg.models or {}).items():
        lines.append("")
        lines.append(f"[models.{role}]")
        if prof.provider:
            lines.append(f"provider = {_toml_str(prof.provider)}")
        lines.append(f"model = {_toml_str(prof.model)}")
        if prof.ctx is not None:
            lines.append(f"ctx = {int(prof.ctx)}")
    if cfg.provider_keys:
        lines.append("")
        lines.append("[provider_keys]")
        for key_alias, key_value in cfg.provider_keys.items():
            lines.append(f"{_toml_str(key_alias)} = {_toml_str(key_value)}")
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def _toml_str(value: str) -> str:
    """Minimal safe TOML string quoting — escapes backslashes, double quotes, and control chars."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'
