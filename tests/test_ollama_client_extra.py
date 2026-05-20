"""Additional tests for OllamaClient uncovered paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from synto.cache import LLMCache
from synto.ollama_client import OllamaClient, OllamaError
from synto.openai_compat_client import LLMTruncatedError


def _make_client() -> OllamaClient:
    return OllamaClient(base_url="http://localhost:11434", timeout=5.0)


# ── healthcheck ──────────────────────────────────────────────────────────────


def test_healthcheck_returns_false_on_connect_error():
    client = _make_client()
    with patch.object(client._client, "get", side_effect=httpx.ConnectError("refused")):
        assert client.healthcheck() is False


def test_require_healthy_raises_on_unhealthy():
    client = _make_client()
    with patch.object(client, "healthcheck", return_value=False):
        with pytest.raises(OllamaError, match="Ollama not running"):
            client.require_healthy()


# ── generate with format parameter ───────────────────────────────────────────


def test_generate_with_format_json():
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": '{"key": "value"}', "done_reason": "stop"}
    captured = {}

    def fake_post(url, json=None, **kw):
        captured["payload"] = json
        return mock_resp

    with patch.object(client._client, "post", side_effect=fake_post):
        client.generate(prompt="hi", model="gemma4:e4b", format="json")
    assert captured["payload"]["format"] == "json"


# ── generate with cache ──────────────────────────────────────────────────────


def test_generate_cache_hit():
    client = OllamaClient(
        base_url="http://localhost:11434",
        timeout=5.0,
        cache=MagicMock(spec=LLMCache),
    )
    client._cache.get.return_value = "cached response"
    with patch.object(client._client, "post") as mock_post:
        result = client.generate(prompt="hi", model="gemma4:e4b", system="sys")
        assert result == "cached response"
        assert client._last_stats["cache_hit"] is True
        mock_post.assert_not_called()


def test_generate_cache_miss_stores_response():
    client = OllamaClient(
        base_url="http://localhost:11434",
        timeout=5.0,
        cache=MagicMock(spec=LLMCache),
    )
    client._cache.get.return_value = None
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": "fresh", "done_reason": "stop"}
    with patch.object(client._client, "post", return_value=mock_resp):
        client.generate(prompt="hi", model="gemma4:e4b", system="sys")
    client._cache.put.assert_called_once()
    call_args = client._cache.put.call_args[0]
    assert call_args[0] == "gemma4:e4b"
    assert call_args[1] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert call_args[2] == "fresh"


def test_generate_cache_without_system():
    """Cache messages should not include system role when system is empty."""
    client = OllamaClient(
        base_url="http://localhost:11434",
        timeout=5.0,
        cache=MagicMock(spec=LLMCache),
    )
    client._cache.get.return_value = None
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": "x", "done_reason": "stop"}
    with patch.object(client._client, "post", return_value=mock_resp):
        client.generate(prompt="hi", model="gemma4:e4b")
    client._cache.put.assert_called_once()
    call_args = client._cache.put.call_args[0]
    assert call_args[1] == [{"role": "user", "content": "hi"}]


# ── generate error paths ─────────────────────────────────────────────────────


def test_generate_raises_on_timeout():
    client = _make_client()
    with patch.object(client._client, "post", side_effect=httpx.TimeoutException("timed out")):
        with pytest.raises(OllamaError, match="timed out"):
            client.generate(prompt="hi", model="gemma4:e4b")
    assert "latency_ms" in client._last_stats


def test_generate_raises_on_http_status_error():
    client = _make_client()
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "bad request"
    err = httpx.HTTPStatusError("error", request=MagicMock(), response=mock_response)
    with patch.object(client._client, "post", side_effect=err):
        with pytest.raises(OllamaError, match="HTTP error: 400"):
            client.generate(prompt="hi", model="gemma4:e4b")
    assert "latency_ms" in client._last_stats


# ── embed_batch ──────────────────────────────────────────────────────────────


def test_embed_batch_raises_on_connect_error():
    client = _make_client()
    with patch.object(client._client, "post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(OllamaError, match="Ollama not running"):
            client.embed_batch(["text"])


# ── context manager ──────────────────────────────────────────────────────────


def test_context_manager_exit_closes_client():
    client = _make_client()
    with patch.object(client._client, "close") as mock_close:
        with client:
            pass
        mock_close.assert_called_once()


# ── generate with num_predict ────────────────────────────────────────────────


def test_generate_truncated_error_includes_num_predict_cap():
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": "", "done_reason": "length"}
    with patch.object(client._client, "post", return_value=mock_resp):
        with pytest.raises(LLMTruncatedError) as exc_info:
            client.generate(prompt="hi", model="gemma4:e4b", num_predict=100)
    assert exc_info.value.max_tokens == 100
    assert exc_info.value.provider == "ollama"


def test_generate_truncated_empty_response_no_done_reason():
    """Empty response without done_reason should also raise LLMTruncatedError."""
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": "   "}  # whitespace-only
    with patch.object(client._client, "post", return_value=mock_resp):
        with pytest.raises(LLMTruncatedError) as exc_info:
            client.generate(prompt="hi", model="gemma4:e4b", num_predict=2048)
    assert exc_info.value.finish_reason == "empty_content"
