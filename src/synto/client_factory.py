"""Factory + per-role router for building LLM clients from config.

A `ModelRouter` resolves each role (fast / heavy / embed) independently and builds one
client per unique connection — so fast and heavy can live on different providers/accounts
(#24) while roles that share a connection share a single client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .api_keys import resolve_api_key
from .config import Config, ResolvedModel
from .openai_compat_client import LLMError, OpenAICompatClient
from .protocols import LLMClientProtocol
from .providers import ProviderInfo

if TYPE_CHECKING:
    from .cache import LLMCache

Role = Literal["fast", "heavy", "embed"]


def _build_client_for(resolved: ResolvedModel, cache: LLMCache | None) -> LLMClientProtocol:
    """Construct the right client class for one resolved role. Three-way dispatch."""
    headers = resolved.headers or None
    if resolved.provider_kind == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient(
            base_url=resolved.url,
            timeout=resolved.timeout,
            cache=cache,
            extra_headers=headers,
        )

    if resolved.anthropic_compat:
        from .anthropic_compat_client import AnthropicCompatClient

        return AnthropicCompatClient(
            base_url=resolved.url,
            provider_name=resolved.provider_kind,
            api_key=resolved.api_key,
            timeout=resolved.timeout,
            cache=cache,
            extra_headers=headers,
        )

    return OpenAICompatClient(
        base_url=resolved.url,
        provider_name=resolved.provider_kind,
        api_key=resolved.api_key,
        timeout=resolved.timeout,
        supports_json_mode=resolved.supports_json_mode,
        supports_embeddings=resolved.supports_embeddings,
        azure=resolved.azure,
        azure_api_version=resolved.azure_api_version,
        cache=cache,
        extra_headers=headers,
    )


@dataclass
class RoleEndpoint:
    """A client bound to a role, plus that role's resolved per-call params."""

    client: LLMClientProtocol
    model: str
    ctx: int
    think: bool | None
    temperature: float | None
    options: dict


class ModelRouter:
    """Resolves roles to (deduplicated) clients and per-role params."""

    def __init__(
        self, config: Config, cache: LLMCache | None = None, api_key_env: str | None = None
    ) -> None:
        self._config = config
        self._cache = cache
        self._api_key_env = api_key_env
        self._clients: dict[tuple, LLMClientProtocol] = {}
        self._endpoints: dict[str, RoleEndpoint] = {}

    def endpoint(self, role: Role) -> RoleEndpoint:
        if role not in self._endpoints:
            resolved = self._config.resolve_role(role, api_key_env=self._api_key_env)
            key = resolved.connection_key
            client = self._clients.get(key)
            if client is None:
                client = _build_client_for(resolved, self._cache)
                self._clients[key] = client
            self._endpoints[role] = RoleEndpoint(
                client=client,
                model=resolved.model,
                ctx=resolved.ctx,
                think=resolved.think,
                temperature=resolved.temperature,
                options=resolved.options,
            )
        return self._endpoints[role]

    def require_healthy(self) -> None:
        """Check the connections backing fast + heavy (each unique one once).

        embed is intentionally excluded: RAG is optional, and a down embed endpoint
        must not block ingest/compile. embed connectivity surfaces lazily on first use.
        """
        seen: set[tuple] = set()
        for role in ("fast", "heavy"):
            resolved = self._config.resolve_role(role, api_key_env=self._api_key_env)
            if resolved.connection_key in seen:
                continue
            seen.add(resolved.connection_key)
            self.endpoint(role).client.require_healthy()

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()
        self._endpoints.clear()


def build_router(
    config: Config, cache: LLMCache | None = None, api_key_env: str | None = None
) -> ModelRouter:
    """Build a per-role router for the vault's config."""
    return ModelRouter(config, cache=cache, api_key_env=api_key_env)


def build_client(
    config: Config,
    api_key_env: str | None = None,
    cache: LLMCache | None = None,
) -> LLMClientProtocol:
    """Return a single default client (the heavy role) for callers that want just one.

    For per-role routing use build_router(). Kept for setup/doctor/serve/compare paths
    and backward compatibility; resolves through the same per-role machinery so the new
    [providers.*] format is honored.
    """
    resolved = config.resolve_role("heavy", api_key_env=api_key_env)
    return _build_client_for(resolved, cache)


def _resolve_api_key(
    provider_name: str,
    prov_info: ProviderInfo | None,
    api_key_env: str | None = None,
) -> str | None:
    """Backward-compatible shim. Resolution now lives in api_keys.resolve_api_key."""
    return resolve_api_key(provider_name, api_key_env_override=api_key_env)


__all__ = ["build_client", "build_router", "ModelRouter", "RoleEndpoint", "LLMError"]
