"""think + options passthrough across all three LLM clients.

Why this matters: thinking models must be controllable (issue #31) and any new
provider/model-native param must be settable without code changes. `think` is an
Ollama-only flag; `options` is the universal escape hatch.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from synto.anthropic_compat_client import AnthropicCompatClient
from synto.ollama_client import OllamaClient
from synto.openai_compat_client import OpenAICompatClient


def _capture(client, post_attr="_client", json_body=None):
    captured = {}
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = json_body

    def fake_post(url, json=None, **kw):
        captured["payload"] = json
        return resp

    return captured, fake_post


# ── Ollama: think is top-level, options merge into options{} ──────────────────


def test_ollama_think_false_sent_top_level():
    client = OllamaClient(base_url="http://x", timeout=5)
    captured, fake = _capture(client, json_body={"response": "{}", "done_reason": "stop"})
    with patch.object(client._client, "post", side_effect=fake):
        client.generate(prompt="hi", model="qwen3.5:2b", think=False)
    assert captured["payload"]["think"] is False


def test_ollama_think_none_omits_key():
    client = OllamaClient(base_url="http://x", timeout=5)
    captured, fake = _capture(client, json_body={"response": "{}", "done_reason": "stop"})
    with patch.object(client._client, "post", side_effect=fake):
        client.generate(prompt="hi", model="m", think=None)
    assert "think" not in captured["payload"]


def test_ollama_options_merge_into_options_object_and_override():
    client = OllamaClient(base_url="http://x", timeout=5)
    captured, fake = _capture(client, json_body={"response": "{}", "done_reason": "stop"})
    with patch.object(client._client, "post", side_effect=fake):
        client.generate(
            prompt="hi", model="m", temperature=0, options={"top_k": 40, "temperature": 0.7}
        )
    opts = captured["payload"]["options"]
    assert opts["top_k"] == 40
    assert opts["temperature"] == 0.7  # options merged last -> overrides first-class temperature


# ── OpenAI-compat: think ignored, options merge top-level ─────────────────────


def test_openai_compat_think_ignored_options_top_level():
    client = OpenAICompatClient(base_url="http://x/v1", provider_name="custom")
    body = {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {},
    }
    captured, fake = _capture(client, json_body=body)
    with patch.object(client._client, "post", side_effect=fake):
        client.generate(prompt="hi", model="m", think=False, options={"top_p": 0.9})
    assert "think" not in captured["payload"]
    assert captured["payload"]["top_p"] == 0.9


# ── Anthropic-compat: think ignored, options merge top-level ──────────────────


def test_anthropic_compat_think_ignored_options_top_level():
    client = AnthropicCompatClient(base_url="http://x", provider_name="kimi", api_key="k")
    body = {"content": [{"text": "{}"}], "stop_reason": "end_turn", "usage": {}}
    captured, fake = _capture(client, json_body=body)
    with patch.object(client._client, "post", side_effect=fake):
        client.generate(prompt="hi", model="m", think=True, options={"thinking": {"budget": 1}})
    assert "think" not in captured["payload"]
    assert captured["payload"]["thinking"] == {"budget": 1}
