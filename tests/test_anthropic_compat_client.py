"""Tests for Anthropic-compatible client (Kimi, etc.)."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from synto.anthropic_compat_client import AnthropicCompatClient
from synto.openai_compat_client import LLMError, LLMTruncatedError


def _make_client(api_key: str = "test-key") -> AnthropicCompatClient:
    return AnthropicCompatClient(
        base_url="https://api.kimi.com/coding",
        provider_name="kimi",
        api_key=api_key,
    )


def _ok_response(
    content: str, stop_reason: str = "end_turn", input_tokens: int = 100, output_tokens: int = 50
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "content": [{"type": "text", "text": content}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    resp.text = ""
    return resp


def _error_response(status_code: int, text: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = {}
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        str(status_code), request=MagicMock(), response=resp
    )
    return resp


# ── Headers ──────────────────────────────────────────────────────────────────


def test_headers_use_x_api_key():
    client = _make_client(api_key="sk-kimi-123")
    headers = client._build_headers()
    assert headers["x-api-key"] == "sk-kimi-123"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in headers


def test_headers_no_api_key():
    client = AnthropicCompatClient(base_url="https://example.com", api_key=None)
    headers = client._build_headers()
    assert "x-api-key" not in headers
    assert headers["anthropic-version"] == "2023-06-01"


# ── Chat URL ─────────────────────────────────────────────────────────────────


def test_chat_url_is_v1_messages():
    client = _make_client()
    assert client._chat_url() == "https://api.kimi.com/coding/v1/messages"


def test_chat_url_strips_trailing_slash():
    client = AnthropicCompatClient(base_url="https://api.kimi.com/coding/")
    assert client._chat_url() == "https://api.kimi.com/coding/v1/messages"


# ── Generate ─────────────────────────────────────────────────────────────────


def test_generate_parses_anthropic_response():
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("Hello from Kimi!"))
    result = client.generate(prompt="hi", model="kimi-k2")
    assert result == "Hello from Kimi!"


def test_generate_system_prompt_top_level():
    """System prompt must be a top-level field, not in the messages array."""
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("ok"))

    client.generate(prompt="hi", model="m", system="You are helpful.")

    payload = client._post_chat.call_args.args[0]
    assert payload["system"] == "You are helpful."
    # Messages should only contain the user message, no system role
    assert all(m["role"] != "system" for m in payload["messages"])
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_generate_no_system_prompt_omits_field():
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("ok"))

    client.generate(prompt="hi", model="m")

    payload = client._post_chat.call_args.args[0]
    assert "system" not in payload


def test_generate_default_max_tokens():
    """When num_predict is -1 (default), max_tokens should default to 4096."""
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("ok"))

    client.generate(prompt="hi", model="m", num_predict=-1)

    payload = client._post_chat.call_args.args[0]
    assert payload["max_tokens"] == 4096


def test_generate_custom_max_tokens():
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("ok"))

    client.generate(prompt="hi", model="m", num_predict=8192)

    payload = client._post_chat.call_args.args[0]
    assert payload["max_tokens"] == 8192


def test_generate_stream_false():
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("ok"))

    client.generate(prompt="hi", model="m")

    payload = client._post_chat.call_args.args[0]
    assert payload["stream"] is False


def test_generate_temperature_passed():
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("ok"))

    client.generate(prompt="hi", model="m", temperature=0.7)

    payload = client._post_chat.call_args.args[0]
    assert payload["temperature"] == 0.7


def test_generate_records_usage_stats():
    client = _make_client()
    client._post_chat = MagicMock(
        return_value=_ok_response("ok", input_tokens=200, output_tokens=75)
    )

    client.generate(prompt="hi", model="m")

    assert client._last_stats["prompt_tokens"] == 200
    assert client._last_stats["completion_tokens"] == 75


# ── Truncation ───────────────────────────────────────────────────────────────


def test_truncation_on_max_tokens_stop_reason():
    client = _make_client()
    client._post_chat = MagicMock(
        return_value=_ok_response("partial...", stop_reason="max_tokens")
    )
    with pytest.raises(LLMTruncatedError) as exc_info:
        client.generate(prompt="hi", model="m", num_predict=4096)
    err = exc_info.value
    assert err.max_tokens == 4096
    assert err.finish_reason == "max_tokens"


def test_truncation_on_empty_content():
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("", stop_reason="end_turn"))
    with pytest.raises(LLMTruncatedError):
        client.generate(prompt="hi", model="m")


def test_no_truncation_on_end_turn():
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("done", stop_reason="end_turn"))
    result = client.generate(prompt="hi", model="m")
    assert result == "done"


# ── Healthcheck ──────────────────────────────────────────────────────────────


def test_healthcheck_any_response_is_healthy():
    client = _make_client()
    resp = MagicMock()
    resp.status_code = 404
    with pytest.MonkeyPatch.context() as m:
        m.setattr(client._client, "get", MagicMock(return_value=resp))
        assert client.healthcheck() is True


def test_healthcheck_server_error_is_unhealthy():
    client = _make_client()
    resp = MagicMock()
    resp.status_code = 500
    with pytest.MonkeyPatch.context() as m:
        m.setattr(client._client, "get", MagicMock(return_value=resp))
        assert client.healthcheck() is False


def test_healthcheck_connect_error_is_unhealthy():
    client = _make_client()
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            client._client,
            "get",
            MagicMock(side_effect=httpx.ConnectError("fail")),
        )
        assert client.healthcheck() is False


# ── Models ───────────────────────────────────────────────────────────────────


def test_list_models_returns_empty():
    client = _make_client()
    assert client.list_models() == []


def test_list_models_detailed_returns_empty():
    client = _make_client()
    assert client.list_models_detailed() == []


# ── Embeddings ───────────────────────────────────────────────────────────────


def test_embed_raises_unsupported():
    client = _make_client()
    with pytest.raises(LLMError, match="does not support embeddings"):
        client.embed("hello")


def test_embed_batch_raises_unsupported():
    client = _make_client()
    with pytest.raises(LLMError, match="does not support embeddings"):
        client.embed_batch(["hello"])


# ── Rate limit backoff ───────────────────────────────────────────────────────


def test_rate_limit_backoff(monkeypatch):
    client = _make_client()
    rate_limited = _error_response(429, "rate limited")
    rate_limited.raise_for_status.side_effect = None  # don't raise during _post_chat loop
    ok = _ok_response("recovered")

    # Simulate: first call gets 429, second succeeds
    call_count = 0

    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return rate_limited
        return ok

    sleeps: list[float] = []
    monkeypatch.setattr("synto.anthropic_compat_client.time.sleep", sleeps.append)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(client._client, "post", mock_post)
        result = client.generate(prompt="hi", model="m")

    assert result == "recovered"
    assert len(sleeps) == 1
    assert sleeps[0] == 1.0


# ── Error handling ───────────────────────────────────────────────────────────


def test_http_401_raises_auth_error():
    client = _make_client()
    client._post_chat = MagicMock(return_value=_error_response(401, "unauthorized"))
    with pytest.raises(LLMError, match="401 Unauthorized"):
        client.generate(prompt="hi", model="m")


def test_http_400_raises_bad_request():
    client = _make_client()
    client._post_chat = MagicMock(return_value=_error_response(400, "bad request"))
    from synto.openai_compat_client import LLMBadRequestError

    with pytest.raises(LLMBadRequestError):
        client.generate(prompt="hi", model="m")


# ── Context manager ─────────────────────────────────────────────────────────


def test_context_manager():
    client = _make_client()
    with client as c:
        assert c is client
