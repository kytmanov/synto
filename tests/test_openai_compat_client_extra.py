"""Additional tests for openai_compat_client.py uncovered paths."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from synto.openai_compat_client import (
    LLMError,
    OpenAICompatClient,
)


def _make_client() -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="https://api.example.com/v1",
        provider_name="test",
        api_key="sk-test",
    )


# ── Azure URL construction ───────────────────────────────────────────────────


def test_azure_models_url():
    """Azure models URL is derived from resource-level base_url."""
    client = OpenAICompatClient(
        base_url="https://example.openai.azure.com/openai/deployments/gpt4",
        provider_name="azure",
        api_key="azure-key",
        azure=True,
        azure_api_version="2024-01-01",
    )
    url = client._models_url()
    assert "openai/models" in url
    assert "api-version=2024-01-01" in url


def test_azure_models_url_no_openai_in_base():
    """Azure base_url without /openai/ falls back to base + /openai/models."""
    client = OpenAICompatClient(
        base_url="https://example.openai.azure.com",
        provider_name="azure",
        api_key="azure-key",
        azure=True,
        azure_api_version="2024-01-01",
    )
    url = client._models_url()
    assert url == "https://example.openai.azure.com/openai/models?api-version=2024-01-01"


# ── _wrap_error ──────────────────────────────────────────────────────────────


def test_wrap_error_connect_error_local():
    """Connect error for local provider gives local-specific message."""
    client = OpenAICompatClient(
        base_url="http://localhost:1234/v1",
        provider_name="lm_studio",
    )
    err = client._wrap_error(httpx.ConnectError("refused"))
    assert "localhost" in str(err)
    assert "service is running" in str(err)


def test_wrap_error_connect_error_cloud():
    """Connect error for cloud provider gives cloud-specific message."""
    client = _make_client()
    err = client._wrap_error(httpx.ConnectError("refused"))
    assert "network connection" in str(err)


def test_wrap_error_timeout():
    """Timeout error includes timeout duration."""
    client = _make_client()
    err = client._wrap_error(httpx.TimeoutException("timed out"), context="during generate")
    assert "timed out" in str(err)
    assert "during generate" in str(err)


def test_wrap_error_401():
    """401 error mentions API key."""
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"
    err = client._wrap_error(httpx.HTTPStatusError("401", request=MagicMock(), response=mock_resp))
    assert "401" in str(err)
    assert "API key" in str(err)


def test_wrap_error_429():
    """429 error mentions rate limit."""
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Rate limited"
    err = client._wrap_error(httpx.HTTPStatusError("429", request=MagicMock(), response=mock_resp))
    assert "429" in str(err)
    assert "Rate limit" in str(err)


def test_wrap_error_generic():
    """Non-httpx exception is wrapped with prefix."""
    client = _make_client()
    err = client._wrap_error(RuntimeError("something broke"))
    assert "test: something broke" in str(err)


def test_wrap_error_no_provider_name():
    """No provider name → no prefix."""
    client = OpenAICompatClient(
        base_url="http://localhost:8000/v1",
        provider_name="",
    )
    err = client._wrap_error(RuntimeError("error"))
    assert str(err) == "error"


# ── 429 retry logic ──────────────────────────────────────────────────────────


def test_post_chat_retries_on_429(monkeypatch):
    """429 triggers exponential backoff retry."""
    client = _make_client()
    sleeps = []
    monkeypatch.setattr("synto.openai_compat_client.time.sleep", sleeps.append)

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {}

    resp_200 = MagicMock()
    resp_200.status_code = 200

    call_count = 0

    def fake_post(url, json=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return resp_429
        return resp_200

    client._client.post = fake_post
    result = client._post_chat({"model": "m"})
    assert result.status_code == 200
    assert call_count == 2
    assert sleeps == [1.0]


def test_post_chat_429_with_retry_after_header(monkeypatch):
    """429 with Retry-After header uses that value."""
    client = _make_client()
    sleeps = []
    monkeypatch.setattr("synto.openai_compat_client.time.sleep", sleeps.append)

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "5"}

    resp_200 = MagicMock()
    resp_200.status_code = 200

    call_count = 0

    def fake_post(url, json=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return resp_429
        return resp_200

    client._client.post = fake_post
    client._post_chat({"model": "m"})
    assert sleeps == [5.0]


def test_post_chat_429_exceeds_60s_budget():
    """429 retry stops when cumulative wait exceeds 60s."""
    client = _make_client()
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "61"}

    client._client.post = MagicMock(return_value=resp_429)
    result = client._post_chat({"model": "m"})
    assert result.status_code == 429


def test_post_chat_429_invalid_retry_after(monkeypatch):
    """429 with invalid Retry-After falls back to exponential delay."""
    client = _make_client()
    sleeps = []
    monkeypatch.setattr("synto.openai_compat_client.time.sleep", sleeps.append)

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "invalid"}

    resp_200 = MagicMock()
    resp_200.status_code = 200

    call_count = 0

    def fake_post(url, json=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return resp_429
        return resp_200

    client._client.post = fake_post
    client._post_chat({"model": "m"})
    assert sleeps == [1.0]


# ── embed_batch ──────────────────────────────────────────────────────────────


def test_embed_batch_connect_error():
    client = OpenAICompatClient(
        base_url="https://api.example.com/v1",
        provider_name="test",
        api_key="sk-test",
        supports_embeddings=True,
    )
    client._client.post = MagicMock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(LLMError, match="Cannot reach"):
        client.embed_batch(["hello"])


# ── list_models ──────────────────────────────────────────────────────────────


def test_list_models_connect_error():
    """list_models returns [] on connection error (fail-soft)."""
    client = _make_client()
    client._client.get = MagicMock(side_effect=httpx.ConnectError("refused"))
    result = client.list_models()
    assert result == []


# ── require_healthy ──────────────────────────────────────────────────────────


def test_require_healthy_raises_on_unhealthy():
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503", request=MagicMock(), response=mock_resp
    )
    client._client.get = MagicMock(return_value=mock_resp)
    with pytest.raises(LLMError):
        client.require_healthy()
