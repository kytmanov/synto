"""Per-account API key resolution.

The key belongs to a provider block (= one account). Resolution order, per §4 of the
per-role-provider design:

1. block `api_key_env` (or an explicit override, e.g. a compare contestant) -> env var
2. provider registry `env_var` (e.g. GROQ_API_KEY)
3. per-alias key in the user-private global config (providers.<alias>.api_key)
4. generic SYNTO_API_KEY
5. legacy single global `api_key`
6. None  (correct for local/keyless blocks like Ollama)

Secrets are never read from the vault synto.toml — only env-var *names* live there.
"""

from __future__ import annotations

import os

from .paths import API_KEY_ENV_VAR
from .providers import get_provider

_LOCAL_URL_PREFIXES = ("http://localhost", "http://127.0.0.1")


def resolve_api_key(
    provider_kind: str,
    *,
    alias: str | None = None,
    block_api_key_env: str | None = None,
    api_key_env_override: str | None = None,
) -> str | None:
    # 1. Explicit override / block-named env var
    for env_name in (api_key_env_override, block_api_key_env):
        if env_name:
            val = os.environ.get(env_name)
            if val:
                return val

    # 2. Provider-conventional env var
    prov = get_provider(provider_kind)
    if prov and prov.env_var:
        val = os.environ.get(prov.env_var)
        if val:
            return val

    # 3 + 5. User-private global config (per-alias key, then legacy single key)
    from .global_config import load_global_config

    gcfg = load_global_config()
    per_alias = getattr(gcfg, "provider_keys", None) if gcfg is not None else None
    if alias and per_alias and per_alias.get(alias):
        return per_alias[alias]

    # 4. Generic env var
    val = os.environ.get(API_KEY_ENV_VAR)
    if val:
        return val

    # 5. Legacy single global key
    if gcfg is not None and getattr(gcfg, "api_key", None):
        return gcfg.api_key

    return None


def credential_gap(
    provider_kind: str,
    *,
    api_key: str | None,
    api_key_env: str | None,
    url: str,
    has_custom_headers: bool,
) -> tuple[str, str | None] | None:
    """(reason, suggested_env_var) when a role has no working credential, else None.

    Takes primitives rather than a ResolvedModel: config.py imports this module, so
    importing ResolvedModel back would cycle. Reason is "declared" (the block named an
    env var that resolved to nothing) or "missing" (nothing was configured at all); the
    suggested var may be None for "missing" when the provider has no conventional env var.
    """
    if api_key:
        return None
    if has_custom_headers:
        return None
    prov = get_provider(provider_kind)
    if prov is None:
        return None
    if not prov.requires_auth:
        return None
    if url.startswith(_LOCAL_URL_PREFIXES):
        return None
    if api_key_env:
        return ("declared", api_key_env)
    return ("missing", prov.env_var)
