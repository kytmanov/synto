"""Global user-level config for Synto."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .config import ModelProfile, ProviderBlock, to_toml
from .paths import APP_NAME
from .vault import atomic_write


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


class GlobalConfigUnreadableError(Exception):
    """config.toml exists on disk but cannot be parsed as a GlobalConfig."""


def load_global_config_strict() -> GlobalConfig | None:
    """Like load_global_config, but distinguishes missing from unreadable.

    Returns None only when the file does not exist; raises GlobalConfigUnreadableError when it
    exists but cannot be parsed. Read-modify-write paths must use this: falling back to a fresh
    GlobalConfig() over a present-but-unparseable file would silently discard provider settings
    and API keys on save.
    """
    path = _global_config_path()
    if not path.exists():
        return None
    cfg = load_global_config()
    if cfg is None:
        raise GlobalConfigUnreadableError(str(path))
    return cfg


def save_global_config(cfg: GlobalConfig) -> None:
    """Write global config to disk. Creates parent directory if needed.

    Serialized through the single `to_toml` seam (model_dump → TOML), so this is the exact inverse
    of the `GlobalConfig(**tomllib.load(...))` read path: every field — including provider
    headers/options and model think/temperature/options — round-trips, and a new field needs no
    change here. Only set fields are written (exclude_unset), so a partial config stays minimal.
    """
    path = _global_config_path()
    atomic_write(path, to_toml(cfg))


# ── known-vault registry (sidecar: vaults.toml, next to config.toml) ──────────
# Deliberately NOT a GlobalConfig field: GlobalConfig is extra="forbid" and load_global_config
# fails open to None, so an unknown key in config.toml would make every older synto silently
# lose its entire global config (vault, providers, API keys) on downgrade.


def _known_vaults_path() -> Path:
    return _global_config_path().parent / "vaults.toml"


def _vault_key(vault: Path | str) -> str:
    # normcase: paths differing only by case are the same vault on Windows.
    return os.path.normcase(str(Path(vault).expanduser().resolve()))


def load_known_vaults() -> list[str]:
    """Registered vault paths (resolved strings, insertion order).

    [] on missing or malformed file — unlike config.toml, the registry holds no secrets and is
    regenerable by re-running `vault use`, so reads fail open and the next write repairs it.
    """
    path = _known_vaults_path()
    if not path.exists():
        return []
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        vaults = data.get("vaults")
        if not isinstance(vaults, list):
            return []
        return [v for v in vaults if isinstance(v, str)]
    except Exception:
        return []


def save_known_vaults(paths: list[str]) -> None:
    """Write the registry to disk. Creates the parent directory if needed."""
    import tomli_w

    path = _known_vaults_path()
    atomic_write(path, tomli_w.dumps({"vaults": paths}))


def register_known_vault(vault: Path | str) -> None:
    """Best-effort add to the registry (dedup by resolved, case-normalized path).

    Never raises: registry bookkeeping must not crash init/setup/vault-use.
    """
    try:
        resolved = str(Path(vault).expanduser().resolve())
        vaults = load_known_vaults()
        if _vault_key(resolved) in {_vault_key(v) for v in vaults}:
            return
        save_known_vaults([*vaults, resolved])
    except Exception:
        pass


def forget_known_vault(vault: Path | str) -> bool:
    """Remove the registry entry matching the resolved path. True if one was removed.

    Never raises.
    """
    try:
        key = _vault_key(vault)
        vaults = load_known_vaults()
        kept = [v for v in vaults if _vault_key(v) != key]
        if len(kept) == len(vaults):
            return False
        save_known_vaults(kept)
        return True
    except Exception:
        return False
