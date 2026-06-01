"""
OpenAI-compatible LLM client.

Covers all providers that implement the /v1/chat/completions spec:
  Local:  LM Studio, vLLM, llama.cpp, LocalAI, TGI, SGLang, Llamafile, Lemonade
  Cloud:  Groq, Together AI, Fireworks, DeepInfra, OpenRouter, Mistral, DeepSeek,
          SiliconFlow, Perplexity, xAI, Azure OpenAI

URL construction: endpoints are appended directly to base_url, which must
already include any path prefix (e.g. "https://api.groq.com/openai/v1").
Azure base_url ends at the deployment level, so /chat/completions appends
correctly without an extra /v1 segment.

Auth:
  - Standard providers: Authorization: Bearer {api_key}
  - Azure:              api-key: {api_key}  +  ?api-version= query param
  - Local no-auth:      no header

JSON mode: if supports_json_mode=True, format="json" injects
  response_format: {"type": "json_object"}.
  If the provider returns HTTP 400, the request is retried once without it
  (transparent auto-downgrade for models that reject the field).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .cache import LLMCache

log = logging.getLogger(__name__)

_LOCAL_MODEL_LOAD_RETRY_SIGNALS = (
    "model unloaded",
    "has been unloaded",
    "has not started loading",
    "failed to load model",
    "error loading model",
    "operation canceled",
    "operation was canceled",
)
_LOCAL_SERVER_ERROR_RETRY_SIGNALS = ("internal server error",)
_LOCAL_MODEL_LOAD_RETRY_DELAYS = (2.0, 4.0, 8.0, 16.0, 32.0)

# Some providers (notably OpenRouter free tier) return rate-limit / overloaded
# errors as an {"error": {...}} body with an HTTP-2xx status, bypassing the
# status-code backoff in _post_chat. These message substrings mark such a body
# as transient (worth a bounded retry) rather than a permanent failure.
_TRANSIENT_CLOUD_ERROR_SIGNALS = (
    "rate limit",
    "rate-limit",
    "ratelimit",
    "too many requests",
    "temporarily",
    "overloaded",
    "try again",
    "unavailable",
)
_TRANSIENT_CLOUD_RETRY_BUDGET_S = 60.0


class LLMError(Exception):
    """Base error for all LLM client failures (OllamaError inherits from this)."""


class LLMBadRequestError(LLMError):
    """HTTP 400 from the provider — usually bad input (prompt/context too long, etc.).

    Unlike transient connection or rate-limit errors this is per-request and non-retryable
    at the pipeline level, so compile_concepts catches it per-concept rather than aborting
    the whole run.
    """


class LLMTruncatedError(LLMError):
    """Model stopped at the max_tokens cap (finish_reason="length"/"max_tokens") and
    either returned no usable content or content known to be truncated.

    Carries enough context for the pipeline to render an actionable error message
    that points the user at the exact config knob to adjust.
    """

    def __init__(
        self,
        provider: str,
        max_tokens: int,
        completion_tokens: int | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.provider = provider
        self.max_tokens = max_tokens
        self.completion_tokens = completion_tokens
        self.finish_reason = finish_reason

        if finish_reason in ("length", "max_tokens") and max_tokens > 0:
            suggested = max(max_tokens * 2, 32768)
            detail = (
                f"output truncated at max_tokens={max_tokens} "
                f"(finish_reason={finish_reason or 'unknown'}). "
                f"Raise pipeline.article_max_tokens in your synto.toml "
                f"(suggested: {suggested}) or reduce source size."
            )
        elif finish_reason in ("length", "max_tokens"):
            detail = (
                f"output hit provider/model context limit "
                f"(finish_reason={finish_reason}; no max_tokens sent). "
                "Check that your loaded model n_ctx matches heavy_ctx in synto.toml, "
                "or reduce source size."
            )
        else:
            detail = (
                f"model returned no usable content (finish_reason={finish_reason or 'unknown'}). "
                "Likely causes: model context exhausted, provider/model incompatibility, or "
                "an excessively large requested output budget. Check that heavy_ctx matches "
                "the loaded model context, consider lowering pipeline.article_max_tokens, and "
                "check model logs."
            )
        super().__init__(f"{provider}: {detail}")


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        provider_name: str = "custom",
        api_key: str | None = None,
        timeout: float = 300.0,
        supports_json_mode: bool = True,
        supports_embeddings: bool = False,
        azure: bool = False,
        azure_api_version: str = "2024-02-15-preview",
        cache: LLMCache | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.provider_name = provider_name
        self._api_key = api_key
        self._timeout = timeout
        self.supports_json_mode = supports_json_mode
        self.supports_embeddings = supports_embeddings
        self._azure = azure
        self._azure_api_version = azure_api_version
        self._client = httpx.Client(
            headers={**self._build_headers(), **(extra_headers or {})},
            timeout=timeout,
        )
        self._last_stats: dict = {}
        self._cache = cache

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        if self._azure:
            return {"api-key": self._api_key}
        return {"Authorization": f"Bearer {self._api_key}"}

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _api_url(self, path: str) -> str:
        """Like _url() but appends ?api-version= for Azure endpoints."""
        url = self._url(path)
        if self._azure:
            url = f"{url}?api-version={self._azure_api_version}"
        return url

    def _chat_url(self) -> str:
        return self._api_url("chat/completions")

    def _models_url(self) -> str:
        """Return the correct models/health endpoint URL.

        Azure base_url ends at the deployment level, so /models appended there
        gives an invalid path. Derive the resource-level URL by stripping
        everything from /openai/ onwards, then append /openai/models.
        """
        if self._azure:
            idx = self.base_url.find("/openai/")
            resource = self.base_url[:idx] if idx >= 0 else self.base_url
            return f"{resource}/openai/models?api-version={self._azure_api_version}"
        return self._api_url("models")

    def _wrap_error(self, exc: Exception, context: str = "") -> LLMError:
        prefix = f"{self.provider_name}: " if self.provider_name else ""
        if isinstance(exc, httpx.ConnectError):
            if self._is_local():
                return LLMError(
                    f"{prefix}Cannot connect to {self.base_url}. Make sure the service is running."
                )
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

    def _is_local(self) -> bool:
        return self.base_url.startswith("http://localhost") or self.base_url.startswith(
            "http://127.0.0.1"
        )

    def _should_retry_local_model_load_400(self, resp: httpx.Response) -> bool:
        if not self._is_local() or resp.status_code != 400:
            return False
        err_text = resp.text.lower()
        return any(signal in err_text for signal in _LOCAL_MODEL_LOAD_RETRY_SIGNALS)

    def _local_transient_retry_reason(self, resp: httpx.Response) -> str | None:
        if self._should_retry_local_model_load_400(resp):
            return "model-load HTTP 400"
        if self._is_local() and resp.status_code == 500:
            err_text = resp.text.lower()
            if any(signal in err_text for signal in _LOCAL_SERVER_ERROR_RETRY_SIGNALS):
                return "HTTP 500"
        return None

    @staticmethod
    def _error_envelope_message(body: object) -> str | None:
        """Human-readable message from a provider {"error": {...}} envelope, if any."""
        if not isinstance(body, dict):
            return None
        err = body.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            code = err.get("code")
            if msg:
                return f"{msg} (code={code})" if code is not None else str(msg)
            return str(err)
        if isinstance(err, str) and err:
            return err
        return None

    def _transient_error_reason(self, body: object) -> str | None:
        """Reason string if an error envelope looks transient (rate limit / overload)."""
        msg = self._error_envelope_message(body)
        if not msg:
            return None
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            code = body["error"].get("code")
            if code in (429, "429"):
                return msg
        if any(sig in msg.lower() for sig in _TRANSIENT_CLOUD_ERROR_SIGNALS):
            return msg
        return None

    def _transient_2xx_error(self, resp: httpx.Response) -> str | None:
        """Reason if a 2xx response actually carries a transient error envelope.

        Returns None for normal completions and for non-2xx responses (those are
        handled by raise_for_status / _wrap_error).
        """
        if not (200 <= resp.status_code < 300):
            return None
        try:
            body = resp.json()
        except ValueError:
            return None
        return self._transient_error_reason(body)

    def _post_chat(self, payload: dict) -> httpx.Response:
        """POST to chat endpoint with 429 exponential backoff (max ~60s cumulative)."""
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
            log.debug("%s: HTTP 429, backing off %.1fs", self.provider_name, wait)
            time.sleep(wait)
            waited += wait
            delay = min(delay * 2, 16.0)

    def _apply_chat_downgrades(
        self,
        resp: httpx.Response,
        payload: dict,
        *,
        use_json_mode: bool,
    ) -> tuple[httpx.Response, dict]:
        current_payload = payload

        if resp.status_code == 400 and use_json_mode and "response_format" in current_payload:
            log.debug(
                "%s: HTTP 400 with response_format, retrying without json mode",
                self.provider_name,
            )
            current_payload = {k: v for k, v in current_payload.items() if k != "response_format"}
            resp = self._post_chat(current_payload)

        if resp.status_code == 400 and "max_tokens" in current_payload:
            err_text = resp.text.lower()
            if "tokens to keep" in err_text or "n_keep" in err_text:
                log.warning(
                    "%s: HTTP 400 n_keep error, retrying without max_tokens "
                    "(model n_ctx may be smaller than configured heavy_ctx; "
                    "output is now uncapped for this request)",
                    self.provider_name,
                )
                current_payload = {k: v for k, v in current_payload.items() if k != "max_tokens"}
                resp = self._post_chat(current_payload)

        if resp.status_code == 400 and "max_tokens" in current_payload:
            err_text = resp.text.lower()
            cloud_cap_signals = (
                "max_tokens",
                "max tokens",
                "completion_tokens",
                "completion tokens",
                "output tokens",
            )
            exceed_signals = ("exceed", "too large", "maximum", "greater than", "is too high")
            if any(s in err_text for s in cloud_cap_signals) and any(
                s in err_text for s in exceed_signals
            ):
                current_max_tokens = int(current_payload["max_tokens"])
                if current_max_tokens > 512:
                    halved = max(512, current_max_tokens // 2)
                    log.warning(
                        "%s: HTTP 400 max_tokens exceeds provider limit, halving %d → %d",
                        self.provider_name,
                        current_max_tokens,
                        halved,
                    )
                    current_payload = {**current_payload, "max_tokens": halved}
                    resp = self._post_chat(current_payload)
                else:
                    log.warning(
                        "%s: HTTP 400 max_tokens exceeds provider limit, but skipping "
                        "auto-downgrade because max_tokens=%d is already at or below "
                        "the 512 retry floor",
                        self.provider_name,
                        current_max_tokens,
                    )

        return resp, current_payload

    # ── Health ────────────────────────────────────────────────────────────────

    def healthcheck(self) -> bool:
        try:
            resp = self._client.get(self._models_url(), timeout=5)
            # Any HTTP response proves the server is reachable.
            # 404/405 are common for providers that lack a /models endpoint.
            return resp.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        except Exception:
            return False

    def require_healthy(self) -> None:
        if not self.healthcheck():
            if self._is_local():
                raise LLMError(
                    f"Cannot reach {self.provider_name} at {self.base_url}. "
                    f"Make sure the service is running."
                )
            raise LLMError(
                f"Cannot reach {self.provider_name} at {self.base_url}. "
                f"Check your network and API key."
            )

    def list_models(self) -> list[str]:
        try:
            resp = self._client.get(self._models_url())
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except (httpx.HTTPError, KeyError, ValueError):
            return []

    def list_models_detailed(self) -> list[dict]:
        """Return list of {'name': str, 'size_gb': str} — matches OllamaClient shape."""
        models = self.list_models()
        return [{"name": m, "size_gb": "(cloud)"} for m in models]

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
        Call /v1/chat/completions. Signature is identical to OllamaClient.generate().

        num_ctx is silently ignored (server-managed for cloud providers).
        num_predict > 0 maps to max_tokens; -1 omits the field (provider default).
        format="json" injects response_format when supports_json_mode=True.
        `think` is a no-op here (Ollama-specific flag); reasoning control for OpenAI-style
        providers is provider-specific — set it via `options` instead.
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self._cache is not None:
            cached = self._cache.get(model, messages, namespace=self.base_url)
            if cached is not None:
                self._last_stats = {"latency_ms": 0, "cache_hit": True}
                return cached

        payload: dict = {"model": model, "messages": messages, "stream": False}
        if temperature is not None:
            payload["temperature"] = temperature

        use_json_mode = format == "json" and self.supports_json_mode
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}

        if num_predict > 0:
            payload["max_tokens"] = num_predict

        if options:
            # Provider-native params (top_p, reasoning_effort, ...); merged last to override.
            payload.update(options)

        t0 = time.monotonic()
        try:
            resp = self._post_chat(payload)
            current_payload = payload
            resp, current_payload = self._apply_chat_downgrades(
                resp,
                current_payload,
                use_json_mode=use_json_mode,
            )

            for wait in _LOCAL_MODEL_LOAD_RETRY_DELAYS:
                retry_reason = self._local_transient_retry_reason(resp)
                if not retry_reason:
                    break
                log.warning(
                    "%s: transient local %s, retrying in %.1fs",
                    self.provider_name,
                    retry_reason,
                    wait,
                )
                time.sleep(wait)
                resp = self._post_chat(current_payload)
                resp, current_payload = self._apply_chat_downgrades(
                    resp,
                    current_payload,
                    use_json_mode=use_json_mode,
                )

            # Cloud throttle returned as an HTTP-2xx error envelope: retry with a
            # bounded budget so a transient free-tier rate limit doesn't fail the
            # caller. If the budget is exhausted the loop falls through and the
            # parse block below surfaces the provider message as LLMBadRequestError.
            waited = 0.0
            delay = 1.0
            while waited < _TRANSIENT_CLOUD_RETRY_BUDGET_S:
                reason = self._transient_2xx_error(resp)
                if not reason:
                    break
                wait = min(delay, _TRANSIENT_CLOUD_RETRY_BUDGET_S - waited)
                log.warning(
                    "%s: transient provider throttle (%s), retrying in %.1fs",
                    self.provider_name,
                    reason,
                    wait,
                )
                time.sleep(wait)
                waited += wait
                delay = min(delay * 2, 16.0)
                resp = self._post_chat(current_payload)
                resp, current_payload = self._apply_chat_downgrades(
                    resp,
                    current_payload,
                    use_json_mode=use_json_mode,
                )

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
        except ValueError as e:
            self._last_stats = {"latency_ms": int((time.monotonic() - t0) * 1000)}
            snippet = resp.text.strip()[:200] or "<empty>"
            raise LLMBadRequestError(f"{self.provider_name}: non-JSON response: {snippet}") from e

        choice = None
        if isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]

        if not isinstance(choice, dict):
            # No usable choices on a 2xx body. Surface the provider's own error
            # message (resp.text is unreliable — providers pad it with keep-alive
            # whitespace) and raise LLMBadRequestError so callers isolate this
            # per-unit instead of crashing the run.
            self._last_stats = {"latency_ms": int((time.monotonic() - t0) * 1000)}
            err_msg = self._error_envelope_message(body)
            snippet = resp.text.strip()[:200]
            detail = err_msg or f"no choices in response: {snippet or '<empty>'}"
            raise LLMBadRequestError(f"{self.provider_name}: {detail}")

        message = choice.get("message") or {}
        content = message.get("content")
        finish_reason = choice.get("finish_reason")

        usage = body.get("usage") or {}
        self._last_stats = {
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

        # Detect truncation: explicit length signal OR empty content (covers
        # providers that omit finish_reason but emit empty body when capped).
        is_length_signal = finish_reason in ("length", "max_tokens")
        is_empty_content = not (content or "").strip()
        if is_length_signal or is_empty_content:
            cap = int(current_payload.get("max_tokens", 0)) if current_payload else 0
            raise LLMTruncatedError(
                provider=self.provider_name,
                max_tokens=cap,
                completion_tokens=usage.get("completion_tokens"),
                finish_reason=finish_reason or ("empty_content" if is_empty_content else None),
            )

        if self._cache is not None:
            self._cache.put(model, messages, content, namespace=self.base_url)

        return content

    # ── Embeddings ────────────────────────────────────────────────────────────

    def embed_batch(self, texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
        if not texts:
            return []
        if not self.supports_embeddings:
            raise LLMError(
                f"{self.provider_name} does not support embeddings. "
                f"Disable RAG or use a provider that supports it "
                f"(Ollama, Together AI, Mistral AI, Fireworks AI, SiliconFlow)."
            )
        try:
            resp = self._client.post(
                self._api_url("embeddings"),
                json={"model": model, "input": texts},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise self._wrap_error(e) from e
        except httpx.TimeoutException as e:
            raise self._wrap_error(e) from e
        except httpx.RequestError as e:
            raise self._wrap_error(e) from e

        # OpenAI API may return embeddings out of order — sort by index
        try:
            data = resp.json().get("data", [])
            data.sort(key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in data]
        except (ValueError, KeyError) as e:
            raise LLMError(
                f"{self.provider_name}: unexpected embeddings response: {resp.text[:200]}"
            ) from e

    def embed(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        return self.embed_batch([text], model=model)[0]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatClient:
        return self

    def __exit__(self, *_) -> None:
        self.close()
