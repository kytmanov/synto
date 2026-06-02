"""
Anthropic-compatible LLM client.

Covers providers that implement the Anthropic Messages API (/v1/messages):
  Cloud:  Kimi (https://api.kimi.com/coding)

Auth: x-api-key header + anthropic-version header
Chat endpoint: {base_url}/v1/messages
System prompt is top-level (not in messages array).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import httpx

from .openai_compat_client import LLMBadRequestError, LLMError, LLMTruncatedError

if TYPE_CHECKING:
    from .cache import LLMCache

log = logging.getLogger(__name__)


class AnthropicCompatClient:
    def __init__(
        self,
        base_url: str,
        provider_name: str = "custom",
        api_key: str | None = None,
        timeout: float = 120.0,
        cache: LLMCache | None = None,
        extra_headers: dict[str, str] | None = None,
        cache_namespace: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.provider_name = provider_name
        self._api_key = api_key
        self._timeout = timeout
        self.supports_json_mode = False
        self.supports_embeddings = False
        self._client = httpx.Client(
            headers={**self._build_headers(), **(extra_headers or {})},
            timeout=timeout,
        )
        self._last_stats: dict = {}
        self._cache = cache
        # Account-aware cache namespace; base_url fallback for direct construction.
        self._cache_namespace = cache_namespace or self.base_url

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    def _chat_url(self) -> str:
        return f"{self.base_url}/v1/messages"

    def _wrap_error(self, exc: Exception, context: str = "") -> LLMError:
        prefix = f"{self.provider_name}: " if self.provider_name else ""
        if isinstance(exc, httpx.ConnectError):
            return LLMError(f"{prefix}Cannot reach {self.base_url}. Check your network connection.")
        if isinstance(exc, httpx.TimeoutException):
            return LLMError(f"{prefix}Request timed out ({self._timeout}s). {context}")
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code == 400:
                return LLMBadRequestError(f"{prefix}HTTP {code}: {exc.response.text[:200]}")
            if code == 401:
                return LLMError(f"{prefix}HTTP 401 Unauthorized. Check your API key.")
            if code == 429:
                return LLMError(f"{prefix}HTTP 429 Rate limit exceeded. Wait and retry.")
            return LLMError(f"{prefix}HTTP {code}: {exc.response.text[:200]}")
        return LLMError(f"{prefix}{exc}")

    def _post_chat(self, payload: dict) -> httpx.Response:
        """POST to messages endpoint with 429 exponential backoff (max ~60s cumulative)."""
        delay = 1.0
        waited = 0.0
        while True:
            resp = self._client.post(self._chat_url(), json=payload)
            if resp.status_code != 429:
                return resp
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else delay
            except ValueError:
                wait = delay
            if waited + wait > 60.0:
                return resp
            resp.close()  # release connection before sleeping
            log.debug("%s: HTTP 429, backing off %.1fs", self.provider_name, wait)
            time.sleep(wait)
            waited += wait
            delay = min(delay * 2, 16.0)

    # ── Health ────────────────────────────────────────────────────────────────

    def healthcheck(self) -> bool:
        try:
            resp = self._client.get(self.base_url, timeout=5)
            # Any HTTP response proves the server is reachable.
            return resp.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        except Exception:
            return False

    def require_healthy(self) -> None:
        if not self.healthcheck():
            raise LLMError(
                f"Cannot reach {self.provider_name} at {self.base_url}. "
                f"Check your network and API key."
            )

    def list_models(self) -> list[str]:
        return []

    def list_models_detailed(self) -> list[dict]:
        return []

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        model: str,
        system: str = "",
        format: str | None = None,
        num_ctx: int = 8192,
        num_predict: int = -1,
        temperature: float | None = None,
        think: bool | None = None,
        options: dict | None = None,
    ) -> str:
        """
        Call /v1/messages (Anthropic Messages API format).

        Signature matches LLMClientProtocol (identical to OllamaClient.generate).
        num_ctx is silently ignored.
        num_predict > 0 maps to max_tokens; -1 defaults to 4096.
        format is silently ignored (JSON mode not supported).
        `think` is a no-op here (Ollama-specific flag); use `options` (e.g. a `thinking`
        budget object) for Anthropic-style reasoning control.
        """
        messages: list[dict] = [{"role": "user", "content": prompt}]

        # Build cache key once for both get and put
        cache_messages: list[dict] | None = None
        if self._cache is not None:
            cache_messages = []
            if system:
                cache_messages.append({"role": "system", "content": system})
            cache_messages.append({"role": "user", "content": prompt})
            cached = self._cache.get(model, cache_messages, namespace=self._cache_namespace)
            if cached is not None:
                self._last_stats = {"latency_ms": 0, "cache_hit": True}
                return cached

        max_tokens = num_predict if num_predict > 0 else 4096
        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if options:
            # Provider-native params (e.g. a `thinking` budget); merged last to override.
            payload.update(options)

        t0 = time.monotonic()
        try:
            resp = self._post_chat(payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._last_stats = {"latency_ms": int((time.monotonic() - t0) * 1000)}
            raise self._wrap_error(e) from e
        except httpx.TimeoutException as e:
            self._last_stats = {"latency_ms": int((time.monotonic() - t0) * 1000)}
            raise self._wrap_error(e) from e
        except httpx.RequestError as e:
            self._last_stats = {"latency_ms": int((time.monotonic() - t0) * 1000)}
            raise self._wrap_error(e) from e

        try:
            body = resp.json()
            content = body["content"][0]["text"]
            stop_reason = body.get("stop_reason")
        except (KeyError, IndexError, ValueError) as e:
            self._last_stats = {"latency_ms": int((time.monotonic() - t0) * 1000)}
            raise LLMError(
                f"{self.provider_name}: unexpected response format: {resp.text[:200]}"
            ) from e

        usage = body.get("usage") or {}
        self._last_stats = {
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
        }

        # Detect truncation: stop_reason == "max_tokens" or empty content
        is_truncated = stop_reason == "max_tokens"
        is_empty = not (content or "").strip()
        if is_truncated or is_empty:
            raise LLMTruncatedError(
                provider=self.provider_name,
                max_tokens=max_tokens,
                completion_tokens=usage.get("output_tokens"),
                finish_reason=stop_reason or ("empty_content" if is_empty else None),
            )

        if self._cache is not None and cache_messages is not None:
            self._cache.put(model, cache_messages, content, namespace=self._cache_namespace)

        return content

    # ── Embeddings (unsupported) ──────────────────────────────────────────────

    def embed_batch(self, texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
        raise LLMError(
            f"{self.provider_name} does not support embeddings. "
            f"Disable RAG or use a provider that supports it "
            f"(Ollama, Together AI, Mistral AI, Fireworks AI, SiliconFlow)."
        )

    def embed(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        raise LLMError(
            f"{self.provider_name} does not support embeddings. "
            f"Disable RAG or use a provider that supports it "
            f"(Ollama, Together AI, Mistral AI, Fireworks AI, SiliconFlow)."
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AnthropicCompatClient:
        return self

    def __exit__(self, *_) -> None:
        self.close()
