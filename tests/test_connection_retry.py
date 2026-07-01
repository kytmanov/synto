"""Transient transport-drop retry across all LLM clients (issue #82).

A server that drops the HTTP connection mid-request raises httpx.RemoteProtocolError
("Server disconnected without sending a response."). This is transient — the request
succeeds on a re-issue — but before this fix it bypassed every status/body-based retry
loop and aborted the whole note. These tests pin the bounded retry and its boundaries:
genuine slow-generation (ReadTimeout) must NOT be retried.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from synto.anthropic_compat_client import AnthropicCompatClient
from synto.ollama_client import OllamaClient, OllamaError
from synto.openai_compat_client import (
    _CONNECTION_RETRY_DELAYS,
    LLMError,
    OpenAICompatClient,
)

# Backoff sleeps are no-op'd globally by the autouse fixture in conftest.py.

_MAX_POSTS = len(_CONNECTION_RETRY_DELAYS) + 1  # initial attempt + retries


def _disconnect() -> httpx.RemoteProtocolError:
    return httpx.RemoteProtocolError("Server disconnected without sending a response.")


# ── OpenAI-compat (the client in issue #82) ──────────────────────────────────


def _openai_ok(content: str = "hello") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    resp.text = ""
    return resp


def _openai_client() -> OpenAICompatClient:
    return OpenAICompatClient(base_url="http://localhost:1234/v1", provider_name="custom")


def test_openai_retries_transient_disconnect_then_succeeds():
    client = _openai_client()
    client._client.post = MagicMock(side_effect=[_disconnect(), _disconnect(), _openai_ok()])
    assert client.generate(prompt="hi", model="m") == "hello"
    assert client._client.post.call_count == 3


def test_openai_exhausted_disconnect_raises_llm_error():
    client = _openai_client()
    client._client.post = MagicMock(side_effect=[_disconnect() for _ in range(_MAX_POSTS)])
    with pytest.raises(LLMError) as exc_info:
        client.generate(prompt="hi", model="m")
    assert "Server disconnected" in str(exc_info.value)
    assert client._client.post.call_count == _MAX_POSTS


def test_openai_read_timeout_is_not_retried():
    """A ReadTimeout means the model is genuinely slow — re-issuing wastes the whole
    timeout window, so it must surface immediately, not loop the retry budget."""
    client = _openai_client()
    client._client.post = MagicMock(side_effect=httpx.ReadTimeout("slow generation"))
    with pytest.raises(LLMError):
        client.generate(prompt="hi", model="m")
    assert client._client.post.call_count == 1


# ── Ollama ───────────────────────────────────────────────────────────────────


def _ollama_ok(content: str = "hello") -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"response": content, "done_reason": "stop"}
    return resp


def test_ollama_retries_transient_disconnect_then_succeeds():
    client = OllamaClient(base_url="http://localhost:11434")
    client._client.post = MagicMock(side_effect=[_disconnect(), _ollama_ok()])
    assert client.generate(prompt="hi", model="m") == "hello"
    assert client._client.post.call_count == 2


def test_ollama_exhausted_disconnect_wraps_as_ollama_error():
    """RemoteProtocolError is not one of OllamaClient's named excepts; without the
    broadened RequestError arm it would escape unwrapped. Guard the clean wrap."""
    client = OllamaClient(base_url="http://localhost:11434")
    client._client.post = MagicMock(side_effect=[_disconnect() for _ in range(_MAX_POSTS)])
    with pytest.raises(OllamaError):
        client.generate(prompt="hi", model="m")
    assert client._client.post.call_count == _MAX_POSTS


def _ollama_embed_ok() -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    return resp


def test_ollama_embed_retries_transient_disconnect_then_succeeds():
    """RAG embedding gets the same transport resilience as generate (review issue #3)."""
    client = OllamaClient(base_url="http://localhost:11434")
    client._client.post = MagicMock(side_effect=[_disconnect(), _ollama_embed_ok()])
    assert client.embed_batch(["hello"]) == [[0.1, 0.2, 0.3]]
    assert client._client.post.call_count == 2


def test_ollama_embed_exhausted_disconnect_wraps_as_ollama_error():
    client = OllamaClient(base_url="http://localhost:11434")
    client._client.post = MagicMock(side_effect=[_disconnect() for _ in range(_MAX_POSTS)])
    with pytest.raises(OllamaError):
        client.embed_batch(["hello"])
    assert client._client.post.call_count == _MAX_POSTS


# ── Anthropic-compat ─────────────────────────────────────────────────────────


def _anthropic_ok(content: str = "hello") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    resp.text = ""
    return resp


def test_anthropic_retries_transient_disconnect_then_succeeds():
    client = AnthropicCompatClient(base_url="https://api.anthropic.com", provider_name="anthropic")
    client._client.post = MagicMock(side_effect=[_disconnect(), _anthropic_ok()])
    assert client.generate(prompt="hi", model="m") == "hello"
    assert client._client.post.call_count == 2
