"""Tests for OpenAI-compat client truncation detection + auto-downgrade."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from synto.openai_compat_client import (
    LLMBadRequestError,
    LLMError,
    LLMTruncatedError,
    OpenAICompatClient,
)


def _make_client() -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="https://api.example.com/v1",
        provider_name="test",
        api_key="sk-test",
    )


def _make_local_client() -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="http://localhost:1234/v1",
        provider_name="lm_studio",
    )


def _ok_response(content: str, finish_reason: str | None = "stop") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
    resp.text = ""
    return resp


def _bad_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 400
    resp.text = text
    resp.json.return_value = {"error": {"message": text}}
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "400", request=MagicMock(), response=resp
    )
    return resp


def _server_error_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 500
    resp.text = text
    resp.json.return_value = {"error": {"message": text}}
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=resp
    )
    return resp


def test_generate_returns_content_on_finish_stop():
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("hello", finish_reason="stop"))
    assert client.generate(prompt="hi", model="m", num_predict=2048) == "hello"


def test_generate_raises_truncated_on_finish_length():
    """finish_reason='length' raises with the cap surfaced for actionable error."""
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("partial...", finish_reason="length"))
    with pytest.raises(LLMTruncatedError) as exc_info:
        client.generate(prompt="hi", model="m", num_predict=4096)
    err = exc_info.value
    assert err.max_tokens == 4096
    assert err.finish_reason == "length"
    assert "article_max_tokens" in str(err)


def test_generate_raises_truncated_on_finish_max_tokens():
    """Anthropic-via-OpenAI-compat uses 'max_tokens' as the truncation signal."""
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("partial", finish_reason="max_tokens"))
    with pytest.raises(LLMTruncatedError):
        client.generate(prompt="hi", model="m", num_predict=8192)


def test_generate_raises_truncated_on_empty_content():
    """Empty content with no length signal — defensive raise to surface silent
    failure modes from providers that don't set finish_reason properly."""
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("", finish_reason="stop"))
    with pytest.raises(LLMTruncatedError):
        client.generate(prompt="hi", model="m", num_predict=4096)


def test_cloud_auto_downgrade_halves_max_tokens_on_exceed_error():
    """When a cloud provider rejects max_tokens as too large, we should halve
    and retry once, not bubble the 400 to the user."""
    client = _make_client()
    bad = _bad_response(
        '{"error": {"message": "max_tokens exceeds the maximum allowed for this model"}}'
    )
    good = _ok_response("response after halving")
    client._post_chat = MagicMock(side_effect=[bad, good])

    result = client.generate(prompt="hi", model="m", num_predict=16384)

    assert result == "response after halving"
    # Second call should send half the original cap
    second_call = client._post_chat.call_args_list[1]
    assert second_call.args[0]["max_tokens"] == 8192


def test_cloud_auto_downgrade_does_not_fire_on_unrelated_400():
    """Ensure we don't strip max_tokens on 400s that aren't about cap exceeding."""
    client = _make_client()
    bad = _bad_response('{"error": {"message": "model not found"}}')
    client._post_chat = MagicMock(return_value=bad)

    with pytest.raises(LLMBadRequestError):
        client.generate(prompt="hi", model="m", num_predict=4096)
    # Only one call — no downgrade attempted
    assert client._post_chat.call_count == 1


def test_cloud_auto_downgrade_skips_when_max_tokens_already_below_floor():
    """Auto-downgrade must never increase the requested cap on a provider-limit 400."""
    client = _make_client()
    bad = _bad_response(
        '{"error": {"message": "max_tokens exceeds the maximum allowed for this model"}}'
    )
    client._post_chat = MagicMock(return_value=bad)

    with pytest.raises(LLMBadRequestError):
        client.generate(prompt="hi", model="m", num_predict=256)

    assert client._post_chat.call_count == 1


def test_lm_studio_auto_downgrade_strips_max_tokens_on_n_keep_error():
    """Existing auto-downgrade #2 (n_keep > n_ctx) still works after our changes."""
    client = _make_client()
    bad = _bad_response(
        '{"error": {"message": "tokens to keep from initial prompt exceeds n_ctx"}}'
    )
    good = _ok_response("response without max_tokens")
    client._post_chat = MagicMock(side_effect=[bad, good])

    result = client.generate(prompt="hi", model="m", num_predict=4096)

    assert result == "response without max_tokens"
    # Second call should not have max_tokens at all
    second_call = client._post_chat.call_args_list[1]
    assert "max_tokens" not in second_call.args[0]


def test_local_model_load_400_retries_and_recovers(monkeypatch):
    client = _make_local_client()
    bad = _bad_response('{"error":"Model unloaded."}')
    good = _ok_response("recovered")
    client._post_chat = MagicMock(side_effect=[bad, good])
    sleeps: list[float] = []

    monkeypatch.setattr("synto.openai_compat_client.time.sleep", sleeps.append)

    result = client.generate(prompt="hi", model="m")

    assert result == "recovered"
    assert client._post_chat.call_count == 2
    assert sleeps == [2.0]


def test_local_model_load_400_raises_after_retry_budget(monkeypatch):
    client = _make_local_client()
    bad = _bad_response(
        '{"error": {"message": "Failed to load model "m". Error: Operation canceled."}}'
    )
    client._post_chat = MagicMock(side_effect=[bad, bad, bad, bad, bad, bad])
    sleeps: list[float] = []

    monkeypatch.setattr("synto.openai_compat_client.time.sleep", sleeps.append)

    with pytest.raises(LLMBadRequestError):
        client.generate(prompt="hi", model="m")

    assert client._post_chat.call_count == 6
    assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0]


def test_local_model_load_400_retries_on_error_loading_model_variant(monkeypatch):
    client = _make_local_client()
    bad = _bad_response('{"error": {"message": "Error loading model "m": operation was canceled"}}')
    good = _ok_response("recovered")
    client._post_chat = MagicMock(side_effect=[bad, good])
    sleeps: list[float] = []

    monkeypatch.setattr("synto.openai_compat_client.time.sleep", sleeps.append)

    result = client.generate(prompt="hi", model="m")

    assert result == "recovered"
    assert client._post_chat.call_count == 2
    assert sleeps == [2.0]


def test_local_model_load_400_retries_on_not_started_loading_variant(monkeypatch):
    client = _make_local_client()
    bad = _bad_response('{"error":"Model has not started loading/has been unloaded."}')
    good = _ok_response("recovered")
    client._post_chat = MagicMock(side_effect=[bad, good])
    sleeps: list[float] = []

    monkeypatch.setattr("synto.openai_compat_client.time.sleep", sleeps.append)

    result = client.generate(prompt="hi", model="m")

    assert result == "recovered"
    assert client._post_chat.call_count == 2
    assert sleeps == [2.0]


def test_cloud_does_not_retry_local_model_load_style_400(monkeypatch):
    client = _make_client()
    bad = _bad_response('{"error":"Model unloaded."}')
    client._post_chat = MagicMock(return_value=bad)
    sleeps: list[float] = []

    monkeypatch.setattr("synto.openai_compat_client.time.sleep", sleeps.append)

    with pytest.raises(LLMBadRequestError):
        client.generate(prompt="hi", model="m")

    assert client._post_chat.call_count == 1
    assert sleeps == []


def test_local_internal_server_error_retries_and_recovers(monkeypatch):
    client = _make_local_client()
    bad = _server_error_response("<html><body><pre>Internal Server Error</pre></body></html>")
    good = _ok_response("recovered")
    client._post_chat = MagicMock(side_effect=[bad, good])
    sleeps: list[float] = []

    monkeypatch.setattr("synto.openai_compat_client.time.sleep", sleeps.append)

    result = client.generate(prompt="hi", model="m")

    assert result == "recovered"
    assert client._post_chat.call_count == 2
    assert sleeps == [2.0]


def test_local_500_then_response_format_400_downgrades_and_recovers(monkeypatch):
    client = _make_local_client()
    server_error = _server_error_response(
        "<html><body><pre>Internal Server Error</pre></body></html>"
    )
    response_format_error = _bad_response(
        "{\"error\":\"'response_format.type' must be 'json_schema' or 'text'\"}"
    )
    good = _ok_response('{"pages": ["Topic"]}')
    client._post_chat = MagicMock(side_effect=[server_error, response_format_error, good])
    sleeps: list[float] = []

    monkeypatch.setattr("synto.openai_compat_client.time.sleep", sleeps.append)

    result = client.generate(prompt="hi", model="m", format="json")

    assert result == '{"pages": ["Topic"]}'
    assert client._post_chat.call_count == 3
    assert sleeps == [2.0]
    first_payload = client._post_chat.call_args_list[0].args[0]
    second_payload = client._post_chat.call_args_list[1].args[0]
    third_payload = client._post_chat.call_args_list[2].args[0]
    assert first_payload["response_format"] == {"type": "json_object"}
    assert second_payload["response_format"] == {"type": "json_object"}
    assert "response_format" not in third_payload


def test_cloud_internal_server_error_does_not_retry(monkeypatch):
    client = _make_client()
    bad = _server_error_response("<html><body><pre>Internal Server Error</pre></body></html>")
    client._post_chat = MagicMock(return_value=bad)
    sleeps: list[float] = []

    monkeypatch.setattr("synto.openai_compat_client.time.sleep", sleeps.append)

    with pytest.raises(LLMError):
        client.generate(prompt="hi", model="m")

    assert client._post_chat.call_count == 1
    assert sleeps == []


def test_truncated_error_message_handles_no_cap_sent():
    """When num_predict was -1 (no cap), the error should reflect that —
    user has a model/context issue, not an article_max_tokens issue."""
    err = LLMTruncatedError(
        provider="lmstudio",
        max_tokens=0,
        finish_reason="length",
    )
    assert "context limit" in str(err) or "no max_tokens sent" in str(err)


def test_truncated_error_message_suggests_double():
    """Error message should suggest a higher value than current cap."""
    err = LLMTruncatedError(
        provider="ollama",
        max_tokens=4096,
        finish_reason="length",
    )
    msg = str(err)
    assert "article_max_tokens" in msg
    # suggested = max(cap*2, 32768) → 32768 here
    assert "32768" in msg


def test_truncated_error_message_for_stop_does_not_suggest_raising_cap():
    err = LLMTruncatedError(
        provider="ollama",
        max_tokens=251824,
        finish_reason="stop",
    )

    msg = str(err)
    assert "no usable content" in msg
    assert "lowering pipeline.article_max_tokens" in msg
    assert "Raise pipeline.article_max_tokens" not in msg


# ── Unsupported-parameter fixups (issue #88: GPT-5/o-series reject standard params) ─


_UNSUPPORTED_MAX_TOKENS_MSG = (
    "Unsupported parameter: 'max_tokens' is not supported with this model. "
    "Use 'max_completion_tokens' instead."
)
_UNSUPPORTED_TEMPERATURE_MSG = (
    "Unsupported value: 'temperature' does not support 0 with this model. "
    "Only the default (1) is supported."
)


def _unsupported_param_response(param: str, message: str) -> MagicMock:
    """The exact 400 shape OpenAI returns for GPT-5/o-series unsupported params."""
    body = {"error": {"message": message, "type": "invalid_request_error", "param": param}}
    resp = MagicMock()
    resp.status_code = 400
    resp.text = json.dumps(body)
    resp.json.return_value = body
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "400", request=MagicMock(), response=resp
    )
    return resp


def test_unsupported_max_tokens_renames_to_max_completion_tokens():
    """Issue #88: the provider's own 400 teaches the client which field to send."""
    client = _make_client()
    bad = _unsupported_param_response("max_tokens", _UNSUPPORTED_MAX_TOKENS_MSG)
    good = _ok_response("hello")
    client._post_chat = MagicMock(side_effect=[bad, good])

    assert client.generate(prompt="hi", model="gpt-5.5", num_predict=2048) == "hello"

    first, second = [c.args[0] for c in client._post_chat.call_args_list]
    assert first["max_tokens"] == 2048
    assert second["max_completion_tokens"] == 2048
    assert "max_tokens" not in second


def test_unsupported_max_tokens_rename_works_without_param_field():
    """Fallback for providers whose 400 body lacks error.param (e.g. Azure variants):
    the message text alone must trigger the rename."""
    client = _make_client()
    bad = _bad_response(_UNSUPPORTED_MAX_TOKENS_MSG)
    good = _ok_response("hello")
    client._post_chat = MagicMock(side_effect=[bad, good])

    assert client.generate(prompt="hi", model="my-deployment", num_predict=1024) == "hello"

    second = client._post_chat.call_args_list[1].args[0]
    assert second["max_completion_tokens"] == 1024
    assert "max_tokens" not in second


def test_unsupported_param_fixup_is_memoized_per_model():
    """Only the first request per model pays the learning 400; other models on the
    same client are unaffected."""
    client = _make_client()
    bad = _unsupported_param_response("max_tokens", _UNSUPPORTED_MAX_TOKENS_MSG)
    client._post_chat = MagicMock(side_effect=[bad, _ok_response("a"), _ok_response("b")])

    client.generate(prompt="hi", model="gpt-5.5", num_predict=2048)
    client.generate(prompt="again", model="gpt-5.5", num_predict=2048)

    third = client._post_chat.call_args_list[2].args[0]
    assert client._post_chat.call_count == 3  # 400 + retry, then single memoized call
    assert third["max_completion_tokens"] == 2048
    assert "max_tokens" not in third

    client._post_chat = MagicMock(return_value=_ok_response("c"))
    client.generate(prompt="hi", model="gpt-4.1", num_predict=2048)
    other = client._post_chat.call_args.args[0]
    assert other["max_tokens"] == 2048
    assert "max_completion_tokens" not in other


def test_unsupported_temperature_is_dropped_and_memoized():
    """o-series models reject non-default temperature; synto's ingest sends
    temperature=0 by default, so the drop must be automatic."""
    client = _make_client()
    bad = _unsupported_param_response("temperature", _UNSUPPORTED_TEMPERATURE_MSG)
    client._post_chat = MagicMock(side_effect=[bad, _ok_response("a"), _ok_response("b")])

    assert client.generate(prompt="hi", model="o3", num_predict=512, temperature=0) == "a"
    retry = client._post_chat.call_args_list[1].args[0]
    assert "temperature" not in retry

    client.generate(prompt="again", model="o3", num_predict=512, temperature=0)
    assert client._post_chat.call_count == 3
    assert "temperature" not in client._post_chat.call_args_list[2].args[0]


def test_model_rejecting_both_params_learns_both():
    """A model can reject max_tokens and temperature in successive 400s (OpenAI
    reports one offending param at a time)."""
    client = _make_client()
    bad_tokens = _unsupported_param_response("max_tokens", _UNSUPPORTED_MAX_TOKENS_MSG)
    bad_temp = _unsupported_param_response("temperature", _UNSUPPORTED_TEMPERATURE_MSG)
    client._post_chat = MagicMock(side_effect=[bad_tokens, bad_temp, _ok_response("a")])

    assert client.generate(prompt="hi", model="o3", num_predict=512, temperature=0) == "a"

    final = client._post_chat.call_args_list[2].args[0]
    assert final["max_completion_tokens"] == 512
    assert "max_tokens" not in final
    assert "temperature" not in final

    client._post_chat = MagicMock(return_value=_ok_response("b"))
    client.generate(prompt="again", model="o3", num_predict=512, temperature=0)
    memoized = client._post_chat.call_args.args[0]
    assert client._post_chat.call_count == 1
    assert memoized["max_completion_tokens"] == 512
    assert "max_tokens" not in memoized
    assert "temperature" not in memoized


def test_json_mode_survives_max_tokens_rename():
    """The unsupported-param fixup must run BEFORE the response_format downgrade
    (which fires on any 400), or GPT-5 JSON-mode calls silently lose json mode."""
    client = _make_client()
    bad = _unsupported_param_response("max_tokens", _UNSUPPORTED_MAX_TOKENS_MSG)
    good = _ok_response('{"ok": true}')
    client._post_chat = MagicMock(side_effect=[bad, good])

    result = client.generate(prompt="hi", model="gpt-5.5", num_predict=2048, format="json")

    assert result == '{"ok": true}'
    second = client._post_chat.call_args_list[1].args[0]
    assert second["response_format"] == {"type": "json_object"}
    assert second["max_completion_tokens"] == 2048


def test_options_max_tokens_override_survives_memoized_rename():
    """options is merged last to override; the rename must carry the user's value,
    not silently revert to num_predict."""
    client = _make_client()
    bad = _unsupported_param_response("max_tokens", _UNSUPPORTED_MAX_TOKENS_MSG)
    client._post_chat = MagicMock(side_effect=[bad, _ok_response("a"), _ok_response("b")])
    client.generate(prompt="hi", model="gpt-5.5", num_predict=2048)

    client.generate(prompt="again", model="gpt-5.5", num_predict=2048, options={"max_tokens": 8192})

    memoized = client._post_chat.call_args_list[2].args[0]
    assert memoized["max_completion_tokens"] == 8192
    assert "max_tokens" not in memoized


def test_explicit_max_completion_tokens_option_is_never_overwritten():
    """A user who already configured options={'max_completion_tokens': N} must win
    over the num_predict-derived value on the learning retry."""
    client = _make_client()
    bad = _unsupported_param_response("max_tokens", _UNSUPPORTED_MAX_TOKENS_MSG)
    good = _ok_response("a")
    client._post_chat = MagicMock(side_effect=[bad, good])

    client.generate(
        prompt="hi",
        model="gpt-5.5",
        num_predict=2048,
        options={"max_completion_tokens": 1234},
    )

    retry = client._post_chat.call_args_list[1].args[0]
    assert retry["max_completion_tokens"] == 1234
    assert "max_tokens" not in retry


def test_unrelated_unsupported_400_does_not_trigger_fixups():
    """An 'unsupported' 400 about some other parameter must not rename or drop
    anything — it raises like any other bad request."""
    client = _make_client()
    bad = _unsupported_param_response(
        "logit_bias", "Unsupported parameter: 'logit_bias' is not supported with this model."
    )
    client._post_chat = MagicMock(return_value=bad)

    with pytest.raises(LLMBadRequestError):
        client.generate(prompt="hi", model="gpt-5.5", num_predict=2048, temperature=0)

    assert client._post_chat.call_count == 1


def test_rename_then_cap_halving_cascade():
    """After the rename retry, a 'too large' 400 must still reach the cap-halving
    downgrade — it operates on max_completion_tokens now."""
    client = _make_client()
    bad_unsupported = _unsupported_param_response("max_tokens", _UNSUPPORTED_MAX_TOKENS_MSG)
    bad_cap = _bad_response(
        '{"error": {"message": "max_completion_tokens exceeds the maximum for this model"}}'
    )
    good = _ok_response("a")
    client._post_chat = MagicMock(side_effect=[bad_unsupported, bad_cap, good])

    assert client.generate(prompt="hi", model="gpt-5.5", num_predict=16384) == "a"

    third = client._post_chat.call_args_list[2].args[0]
    assert third["max_completion_tokens"] == 8192
    assert "max_tokens" not in third


def test_n_keep_downgrade_strips_max_completion_tokens():
    """The n_keep 400 recovery must work regardless of which output-cap field the
    payload carries."""
    client = _make_client()
    bad_unsupported = _unsupported_param_response("max_tokens", _UNSUPPORTED_MAX_TOKENS_MSG)
    bad_n_keep = _bad_response(
        '{"error": {"message": "tokens to keep from initial prompt exceeds n_ctx"}}'
    )
    good = _ok_response("a")
    client._post_chat = MagicMock(side_effect=[bad_unsupported, bad_n_keep, good])

    assert client.generate(prompt="hi", model="m", num_predict=4096) == "a"

    second = client._post_chat.call_args_list[1].args[0]
    assert second["max_completion_tokens"] == 4096
    third = client._post_chat.call_args_list[2].args[0]
    assert "max_completion_tokens" not in third
    assert "max_tokens" not in third


def test_truncation_cap_reported_from_max_completion_tokens():
    """LLMTruncatedError must surface the configured cap for either field spelling."""
    client = _make_client()
    bad = _unsupported_param_response("max_tokens", _UNSUPPORTED_MAX_TOKENS_MSG)
    truncated = _ok_response("partial...", finish_reason="length")
    client._post_chat = MagicMock(side_effect=[bad, truncated])

    with pytest.raises(LLMTruncatedError) as exc_info:
        client.generate(prompt="hi", model="gpt-5.5", num_predict=4096)

    assert exc_info.value.max_tokens == 4096


# ── HTTP-2xx error envelope (issue #25: OpenRouter returns errors with 200) ─────


def _error_body_2xx(message: str, code=None, *, text: str = "") -> MagicMock:
    """A 200 response whose JSON body carries an {"error": {...}} envelope and
    whose .text is keep-alive padding (the real failing case has whitespace text)."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    error: dict = {"message": message}
    if code is not None:
        error["code"] = code
    resp.json.return_value = {"error": error}
    resp.text = text
    return resp


def test_generate_2xx_error_body_surfaces_provider_message():
    """A 200 body with no usable choices must raise LLMBadRequestError carrying the
    provider's own error message — not the old blank 'unexpected response format'."""
    client = _make_client()
    client._post_chat = MagicMock(
        return_value=_error_body_2xx("Provider returned no completion", text="\n\n   \n")
    )
    with pytest.raises(LLMBadRequestError) as exc_info:
        client.generate(prompt="hi", model="m", num_predict=2048)
    msg = str(exc_info.value)
    assert "Provider returned no completion" in msg
    assert "unexpected response format" not in msg
    # Non-transient → no retry.
    assert client._post_chat.call_count == 1


def test_generate_transient_2xx_error_retries_then_succeeds(monkeypatch):
    """A transient rate-limit returned as a 200 error body is retried (issue #25:
    OpenRouter free tier bypasses the 429 status backoff)."""
    monkeypatch.setattr("synto.openai_compat_client.time.sleep", lambda _s: None)
    client = _make_client()
    transient = _error_body_2xx("Rate limit exceeded", code=429)
    good = _ok_response("recovered", finish_reason="stop")
    client._post_chat = MagicMock(side_effect=[transient, good])

    assert client.generate(prompt="hi", model="m", num_predict=2048) == "recovered"
    assert client._post_chat.call_count == 2


def test_generate_transient_2xx_error_exhausts_budget_then_raises(monkeypatch):
    """When the throttle never clears, the bounded retry gives up and raises a
    classified error whose message is the real reason (never blank)."""
    monkeypatch.setattr("synto.openai_compat_client.time.sleep", lambda _s: None)
    client = _make_client()
    client._post_chat = MagicMock(return_value=_error_body_2xx("Rate limit exceeded", code=429))

    with pytest.raises(LLMBadRequestError) as exc_info:
        client.generate(prompt="hi", model="m", num_predict=2048)
    assert "Rate limit exceeded" in str(exc_info.value)
    # Initial call + bounded retries (exponential 1,2,4,8,16,16,... within 60s).
    assert client._post_chat.call_count > 1


def test_transient_2xx_backoff_is_exponential_and_budget_bounded(monkeypatch):
    """Pins the budget arithmetic that call-count alone can't: back off exponentially
    (doubling, capped at 16s) and never sleep past the 60s cumulative budget. The final
    wait is clamped by min(delay, budget - waited) so the sum lands exactly on 60s."""
    waits: list[float] = []
    monkeypatch.setattr("synto.openai_compat_client.time.sleep", lambda s: waits.append(s))
    client = _make_client()
    client._post_chat = MagicMock(return_value=_error_body_2xx("Rate limit exceeded", code=429))

    with pytest.raises(LLMBadRequestError):
        client.generate(prompt="hi", model="m", num_predict=2048)

    assert waits[:5] == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert all(w <= 16.0 for w in waits)
    assert sum(waits) <= 60.0


def test_generate_non_dict_body_raises_bad_request():
    """A non-dict JSON body must not crash with an uncaught TypeError."""
    client = _make_client()
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = ["unexpected", "list"]
    resp.text = '["unexpected", "list"]'
    client._post_chat = MagicMock(return_value=resp)

    with pytest.raises(LLMBadRequestError):
        client.generate(prompt="hi", model="m", num_predict=2048)
