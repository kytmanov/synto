"""Factory for building the appropriate LLM client from config."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .config import Config
from .openai_compat_client import LLMError, OpenAICompatClient
from .paths import API_KEY_ENV_VAR
from .protocols import LLMClientProtocol
from .providers import ProviderInfo, get_provider

if TYPE_CHECKING:
    from .cache import LLMCache


def build_client(
    config: Config,
    api_key_env: str | None = None,
    cache: LLMCache | None = None,
) -> LLMClientProtocol:
    """Return the appropriate LLM client for the vault's provider config."""
    prov = config.effective_provider

    if prov.name == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient(base_url=prov.url, timeout=prov.timeout, cache=cache)

    prov_info = get_provider(prov.name)
    api_key = _resolve_api_key(prov.name, prov_info, api_key_env=api_key_env)

    if prov_info and prov_info.anthropic_compat:
        from .anthropic_compat_client import AnthropicCompatClient

        return AnthropicCompatClient(
            base_url=prov.url,
            provider_name=prov.name,
            api_key=api_key,
            timeout=prov.timeout,
            cache=cache,
        )

    return OpenAICompatClient(
        base_url=prov.url,
        provider_name=prov.name,
        api_key=api_key,
        timeout=prov.timeout,
        supports_json_mode=prov_info.supports_json_mode if prov_info else True,
        supports_embeddings=prov_info.supports_embeddings if prov_info else False,
        azure=prov_info.azure if prov_info else False,
        azure_api_version=prov.azure_api_version,
        cache=cache,
    )


def _resolve_api_key(
    provider_name: str,
    prov_info: ProviderInfo | None,
    api_key_env: str | None = None,
) -> str | None:
    # 0. Explicit env var override for compare contestants
    if api_key_env:
        val = os.environ.get(api_key_env)
        if val:
            return val

    # 1. Provider-specific env var (e.g. GROQ_API_KEY)
    if prov_info and prov_info.env_var:
        val = os.environ.get(prov_info.env_var)
        if val:
            return val

    # 2. Generic env var
    val = os.environ.get(API_KEY_ENV_VAR)
    if val:
        return val

    # 3. Global config
    from .global_config import load_global_config

    gcfg = load_global_config()
    if gcfg and gcfg.api_key:
        return gcfg.api_key

    return None


__all__ = ["build_client", "LLMError"]
