"""Tests for issue #114: doctor and 401 messages must diagnose missing credentials,
not just reachability. All tests offline — no network, no Ollama.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from synto.anthropic_compat_client import AnthropicCompatClient
from synto.api_keys import credential_gap
from synto.cli import cli
from synto.openai_compat_client import LLMError, OpenAICompatClient
from synto.paths import CONFIG_FILE_NAME

ISSUE_114_TOML = """\
[providers.default]
name = "ollama"
url = "http://localhost:11434"
timeout = 1000.0

[providers.deepinfra]
name = "deepinfra"
url = "https://api.deepinfra.com/v1/openai"
timeout = 120.0
api_key_env = "DEEPINFRA_API_KEY"

[models.fast]
model = "gemma4:e4b"
provider = "default"
ctx = 16384

[models.heavy]
model = "openai/gpt-oss-120b"
provider = "deepinfra"
ctx = 32768
"""


@pytest.fixture
def cfg_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect global config to a temp dir so a real ~/.config/synto/config.toml
    cannot leak a provider_keys entry into these tests."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "xdg"))
    return tmp_path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _http_error(status_code: int, text: str = "unauthorized") -> httpx.HTTPStatusError:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = {}
    return httpx.HTTPStatusError(str(status_code), request=MagicMock(), response=resp)


def _anthropic_error_response(status_code: int, text: str = "unauthorized") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = {}
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        str(status_code), request=MagicMock(), response=resp
    )
    return resp


# ── credential_gap ───────────────────────────────────────────────────────────


def test_credential_gap_none_for_ollama():
    assert (
        credential_gap(
            "ollama",
            api_key=None,
            api_key_env=None,
            url="http://localhost:11434",
            has_custom_headers=False,
        )
        is None
    )


def test_credential_gap_none_for_lm_studio():
    assert (
        credential_gap(
            "lm_studio",
            api_key=None,
            api_key_env=None,
            url="http://localhost:1234/v1",
            has_custom_headers=False,
        )
        is None
    )


def test_credential_gap_none_for_custom_kind():
    assert (
        credential_gap(
            "custom",
            api_key=None,
            api_key_env=None,
            url="https://gateway.internal/v1",
            has_custom_headers=False,
        )
        is None
    )


def test_credential_gap_none_for_unknown_kind():
    assert (
        credential_gap(
            "nonexistent_provider_xyz",
            api_key=None,
            api_key_env=None,
            url="https://example.com/v1",
            has_custom_headers=False,
        )
        is None
    )


def test_credential_gap_none_when_custom_headers_present():
    """A corporate gateway authenticating via a custom header has api_key=None but works."""
    assert (
        credential_gap(
            "deepinfra",
            api_key=None,
            api_key_env=None,
            url="https://api.deepinfra.com/v1/openai",
            has_custom_headers=True,
        )
        is None
    )


def test_credential_gap_none_for_localhost_url_with_cloud_kind():
    """A cloud provider kind pointed at a keyless local proxy must not warn."""
    assert (
        credential_gap(
            "deepinfra",
            api_key=None,
            api_key_env=None,
            url="http://localhost:8000/v1",
            has_custom_headers=False,
        )
        is None
    )
    assert (
        credential_gap(
            "deepinfra",
            api_key=None,
            api_key_env=None,
            url="http://127.0.0.1:8000/v1",
            has_custom_headers=False,
        )
        is None
    )


def test_credential_gap_none_when_key_resolved():
    assert (
        credential_gap(
            "deepinfra",
            api_key="sk-real-key",
            api_key_env=None,
            url="https://api.deepinfra.com/v1/openai",
            has_custom_headers=False,
        )
        is None
    )


def test_credential_gap_declared_but_unset_issue_114_shape():
    result = credential_gap(
        "deepinfra",
        api_key=None,
        api_key_env="DEEPINFRA_API_KEY",
        url="https://api.deepinfra.com/v1/openai",
        has_custom_headers=False,
    )
    assert result == ("declared", "DEEPINFRA_API_KEY")


def test_credential_gap_missing_when_nothing_declared():
    result = credential_gap(
        "deepinfra",
        api_key=None,
        api_key_env=None,
        url="https://api.deepinfra.com/v1/openai",
        has_custom_headers=False,
    )
    assert result == ("missing", "DEEPINFRA_API_KEY")


# ── doctor CLI ───────────────────────────────────────────────────────────────


@pytest.fixture
def fake_healthy_client(monkeypatch: pytest.MonkeyPatch):
    """Patch the router's client builder so doctor's provider loop reports every
    connection reachable without touching the network (mirrors test_29_pack_doctor.py)."""

    class _Client:
        def require_healthy(self) -> None:
            return None

        def list_models(self) -> list[str]:
            return []

        def close(self) -> None:
            return None

    monkeypatch.setattr("synto.client_factory._build_client_for", lambda resolved, cache: _Client())


def test_doctor_reports_missing_credential_for_issue_114(
    tmp_path: Path,
    runner: CliRunner,
    fake_healthy_client,
    cfg_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.delenv("SYNTO_API_KEY", raising=False)

    result = runner.invoke(cli, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    (tmp_path / CONFIG_FILE_NAME).write_text(ISSUE_114_TOML)

    result = runner.invoke(cli, ["doctor", "--vault", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "heavy" in result.output
    assert "DEEPINFRA_API_KEY" in result.output
    assert "is not set" in result.output
    assert "export DEEPINFRA_API_KEY" in result.output
    # embed/fast stay on the keyless ollama connection — no false positive there.
    assert "Some checks need attention" in result.output


def test_doctor_uncredentialed_role_gets_no_green_checkmark(
    tmp_path: Path,
    runner: CliRunner,
    cfg_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The role must end up with ONE verdict, and it must not be green.

    DeepInfra's /models answers without auth, so the model list succeeds while the actual
    chat call 401s — that mismatch is issue #114 itself. A green "model found" line printed
    next to the credential failure reproduces the exact false signal the fix removes, so the
    fake client below returns the model to recreate that condition.
    """
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.delenv("SYNTO_API_KEY", raising=False)

    class _ListsTheModel:
        def require_healthy(self) -> None:
            return None

        def list_models(self) -> list[str]:
            return ["openai/gpt-oss-120b", "gemma4:e4b", "nomic-embed-text"]

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "synto.client_factory._build_client_for", lambda resolved, cache: _ListsTheModel()
    )

    result = runner.invoke(cli, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    (tmp_path / CONFIG_FILE_NAME).write_text(ISSUE_114_TOML)

    result = runner.invoke(cli, ["doctor", "--vault", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "✗ heavy" in result.output
    assert "$DEEPINFRA_API_KEY is not set" in result.output
    assert "✓ heavy" not in result.output
    # fast/embed sit on the keyless ollama block and keep their normal green verdict.
    assert "✓ fast" in result.output


def test_doctor_keyless_local_vault_prints_no_credential_warning(
    tmp_path: Path,
    runner: CliRunner,
    fake_healthy_client,
    cfg_dir: Path,
) -> None:
    """Regression guard: an all-Ollama vault must never print a credential warning."""
    result = runner.invoke(cli, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(cli, ["doctor", "--vault", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "is not set" not in result.output
    assert "no API key configured" not in result.output
    assert "Run: export" not in result.output


# ── self-diagnosing 401 ──────────────────────────────────────────────────────


def test_openai_compat_401_no_key_names_declared_var():
    client = OpenAICompatClient(
        base_url="https://api.deepinfra.com/v1/openai",
        provider_name="deepinfra",
        api_key=None,
        api_key_env="DEEPINFRA_API_KEY",
    )
    err = _http_error(401)
    with patch.object(client._client, "post", side_effect=err):
        with pytest.raises(LLMError) as exc_info:
            client.generate("p", model="m")
    msg = str(exc_info.value)
    assert "$DEEPINFRA_API_KEY" in msg
    assert "no API key was sent" in msg
    assert "rejected" not in msg


def test_openai_compat_401_with_key_says_rejected_and_omits_key_value():
    client = OpenAICompatClient(
        base_url="https://api.deepinfra.com/v1/openai",
        provider_name="deepinfra",
        api_key="sk-super-secret-value",
        api_key_env="DEEPINFRA_API_KEY",
    )
    err = _http_error(401)
    with patch.object(client._client, "post", side_effect=err):
        with pytest.raises(LLMError) as exc_info:
            client.generate("p", model="m")
    msg = str(exc_info.value)
    assert "$DEEPINFRA_API_KEY" in msg
    assert "rejected" in msg
    assert "sk-super-secret-value" not in msg


def test_openai_compat_401_falls_back_to_registry_env_var():
    """No api_key_env declared -> hint falls back to the provider registry's env_var."""
    client = OpenAICompatClient(
        base_url="https://api.deepinfra.com/v1/openai", provider_name="deepinfra", api_key=None
    )
    err = _http_error(401)
    with patch.object(client._client, "post", side_effect=err):
        with pytest.raises(LLMError) as exc_info:
            client.generate("p", model="m")
    assert "$DEEPINFRA_API_KEY" in str(exc_info.value)


def test_openai_compat_401_falls_back_to_synto_api_key_for_unknown_provider():
    client = OpenAICompatClient(
        base_url="https://example.com/v1", provider_name="custom", api_key=None
    )
    err = _http_error(401)
    with patch.object(client._client, "post", side_effect=err):
        with pytest.raises(LLMError) as exc_info:
            client.generate("p", model="m")
    assert "$SYNTO_API_KEY" in str(exc_info.value)


def test_anthropic_compat_401_no_key_names_declared_var():
    client = AnthropicCompatClient(
        base_url="https://api.kimi.com/coding",
        provider_name="kimi",
        api_key=None,
        api_key_env="KIMI_API_KEY",
    )
    client._post_chat = MagicMock(return_value=_anthropic_error_response(401))
    with pytest.raises(LLMError) as exc_info:
        client.generate("p", model="m")
    msg = str(exc_info.value)
    assert "$KIMI_API_KEY" in msg
    assert "no API key was sent" in msg
    assert "rejected" not in msg


def test_anthropic_compat_401_with_key_says_rejected_and_omits_key_value():
    client = AnthropicCompatClient(
        base_url="https://api.kimi.com/coding",
        provider_name="kimi",
        api_key="sk-anthropic-secret",
        api_key_env="KIMI_API_KEY",
    )
    client._post_chat = MagicMock(return_value=_anthropic_error_response(401))
    with pytest.raises(LLMError) as exc_info:
        client.generate("p", model="m")
    msg = str(exc_info.value)
    assert "$KIMI_API_KEY" in msg
    assert "rejected" in msg
    assert "sk-anthropic-secret" not in msg
