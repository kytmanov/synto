"""Additional tests for structured_output.py uncovered paths."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from synto.models import SingleArticle
from synto.ollama_client import OllamaClient
from synto.structured_output import (
    _extract_json,
    _fix_json_ctrl_escapes,
    _render_example,
    _try_parse,
    _unwrap,
    request_structured,
)


def _client(response: str) -> OllamaClient:
    c = MagicMock(spec=OllamaClient)
    c.generate.return_value = response
    return c


# ── _extract_json ─────────────────────────────────────────────────────────────


def test_extract_json_bare_code_block():
    """Bare ``` block without 'json' language tag."""
    text = 'Here it is:\n```\n{"key": "value"}\n```\nDone.'
    result = _extract_json(text)
    assert result == '{"key": "value"}'


def test_extract_json_no_json_found():
    """No JSON anywhere in text → None."""
    text = "This is just plain text with no JSON."
    assert _extract_json(text) is None


# ── _unwrap ───────────────────────────────────────────────────────────────────


def test_unwrap_single_key_dict():
    """Single-key dict wrapper is unwrapped."""
    data = {
        "AnalysisResult": {
            "summary": "s",
            "concepts": [],
            "suggested_topics": [],
            "quality": "high",
        }
    }
    result = _unwrap(data)
    assert "summary" in result


def test_unwrap_string_encoded_json():
    """Single-key with JSON string value is parsed."""
    data = {"result": '{"title": "T", "content": "body", "tags": []}'}
    result = _unwrap(data)
    assert result["title"] == "T"


def test_unwrap_string_encoded_json_invalid():
    """Single-key with invalid JSON string → returns original data."""
    data = {"result": "not json"}
    result = _unwrap(data)
    assert result == data


def test_unwrap_json_schema_echo():
    """JSON Schema echo with properties dict is unwrapped."""
    data = {
        "description": "A thing",
        "properties": {"title": "T", "content": "body", "tags": []},
    }
    result = _unwrap(data)
    assert result["title"] == "T"


def test_unwrap_json_schema_echo_with_nested_schema():
    """Properties with nested schema dicts are not unwrapped."""
    data = {
        "description": "A thing",
        "properties": {"title": {"type": "string"}},
    }
    result = _unwrap(data)
    # Not unwrapped because value is a schema dict with "type"
    assert result == data


# ── _fix_json_ctrl_escapes ───────────────────────────────────────────────────


def test_fix_ctrl_escapes_nested_dict():
    """Control chars in nested dict values are fixed."""
    data = {"content": "test\text"}
    result = _fix_json_ctrl_escapes(data)
    assert result["content"] == "test\\text"


def test_fix_ctrl_escapes_nested_list():
    """Control chars in nested list items are fixed."""
    data = ["item\twith\ttab"]
    result = _fix_json_ctrl_escapes(data)
    assert result == ["item\\twith\\ttab"]


# ── _try_parse ───────────────────────────────────────────────────────────────


def test_try_parse_invalid_json_returns_error():
    """Invalid JSON returns (None, error_string)."""
    result, error = _try_parse("not json", SingleArticle)
    assert result is None
    assert error


def test_try_parse_valid_json_wrong_schema():
    """Valid JSON but wrong schema returns (None, error_string)."""
    raw = json.dumps({"wrong": "schema"})
    result, error = _try_parse(raw, SingleArticle)
    assert result is None
    assert error


# ── _render_example ──────────────────────────────────────────────────────────


def test_render_example_anyof_with_null():
    """anyOf with null alternative skips to non-null."""
    schema_node = {
        "anyOf": [{"type": "null"}, {"type": "string", "description": "A string"}],
        "description": "Optional field",
    }
    result = _render_example(schema_node, {})
    assert result == "A string"


def test_render_example_anyof_all_null():
    """anyOf with only null → returns None."""
    schema_node = {"anyOf": [{"type": "null"}]}
    result = _render_example(schema_node, {})
    assert result is None


# ── request_structured with retry error feedback ─────────────────────────────


def test_retry_includes_error_feedback():
    """Retry attempts include the previous error in the prompt."""
    call_count = 0
    captured_prompts = []

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        captured_prompts.append(kwargs.get("prompt", ""))
        if call_count == 1:
            return "bad json"
        return json.dumps({"title": "T", "content": "body", "tags": []})

    c = MagicMock(spec=OllamaClient)
    c.generate.side_effect = side_effect

    result = request_structured(
        client=c,
        prompt="write article",
        model_class=SingleArticle,
        model="test",
        max_retries=1,
    )
    assert result.title == "T"
    assert call_count == 2
    # Second prompt should mention the error
    assert "error" in captured_prompts[1].lower() or "invalid" in captured_prompts[1].lower()
