"""
Tests for the `synto vault` group (show / use / forget) and vault auto-registration.
All tests are offline — no Ollama instance required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from synto.cli import cli
from synto.config import ModelProfile, ProviderBlock
from synto.global_config import (
    GlobalConfig,
    _global_config_path,
    _known_vaults_path,
    load_global_config,
    load_known_vaults,
    save_global_config,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect global config to a temp dir."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # On Windows, patch APPDATA too
    monkeypatch.setenv("APPDATA", str(tmp_path))
    # Wide console so Rich can't wrap long tmp paths mid-token (same lesson as #110).
    monkeypatch.setenv("COLUMNS", "200")
    # The registry must not leak between tests via a real SYNTO_VAULT.
    monkeypatch.delenv("SYNTO_VAULT", raising=False)
    return tmp_path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_vault(base: Path, name: str) -> Path:
    """Minimal on-disk synto vault: a directory containing synto.toml."""
    vault = base / name
    vault.mkdir(parents=True)
    (vault / "synto.toml").write_text("[models]\n", encoding="utf-8")
    return vault


# ── vault use ─────────────────────────────────────────────────────────────────


def test_vault_use_sets_default_and_registers(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    vault = _make_vault(tmp_path, "my-wiki")
    result = runner.invoke(cli, ["vault", "use", str(vault)])
    assert result.exit_code == 0, result.output

    cfg = load_global_config()
    assert cfg is not None
    assert cfg.vault == str(vault.resolve())
    assert str(vault.resolve()) in load_known_vaults()


def test_vault_use_preserves_provider_config(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    """The point of #51: switching the default vault must not touch provider settings.

    A regression that rebuilds GlobalConfig from scratch instead of mutating the loaded one
    would silently discard provider blocks and API keys — exactly what the issue complains
    setup forces on users.
    """
    save_global_config(
        GlobalConfig(
            vault="/somewhere/old",
            provider_name="groq",
            provider_url="https://api.groq.com/openai/v1",
            api_key="sk-legacy-secret",
            azure_api_version="2024-02-15-preview",
            providers={"default": ProviderBlock(name="groq", api_key_env="GROQ_API_KEY")},
            models={
                "fast": ModelProfile(model="llama-3.1-8b-instant", provider="default"),
                "heavy": ModelProfile(model="llama-3.3-70b", provider="default", ctx=32768),
            },
            provider_keys={"default": "sk-raw-fallback"},
        )
    )
    vault = _make_vault(tmp_path, "new-wiki")

    result = runner.invoke(cli, ["vault", "use", str(vault)])
    assert result.exit_code == 0, result.output

    cfg = load_global_config()
    assert cfg is not None
    assert cfg.vault == str(vault.resolve())  # the one field that changed
    assert cfg.provider_name == "groq"
    assert cfg.provider_url == "https://api.groq.com/openai/v1"
    assert cfg.api_key == "sk-legacy-secret"
    assert cfg.azure_api_version == "2024-02-15-preview"
    assert cfg.providers["default"].name == "groq"
    assert cfg.providers["default"].api_key_env == "GROQ_API_KEY"
    assert cfg.models is not None
    assert cfg.models["fast"].model == "llama-3.1-8b-instant"
    assert cfg.models["heavy"].ctx == 32768
    assert cfg.provider_keys == {"default": "sk-raw-fallback"}


def test_vault_use_rejects_non_vault_dir(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    not_a_vault = tmp_path / "plain-dir"
    not_a_vault.mkdir()
    result = runner.invoke(cli, ["vault", "use", str(not_a_vault)])
    assert result.exit_code == 1
    assert "not a Synto vault" in result.output
    assert "--existing" in result.output
    assert load_global_config() is None
    assert load_known_vaults() == []


def test_vault_use_rejects_missing_path(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    result = runner.invoke(cli, ["vault", "use", str(tmp_path / "gone")])
    assert result.exit_code == 1
    assert "does not exist" in result.output
    assert load_global_config() is None


def test_vault_use_rejects_file_path(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    file_path = tmp_path / "note.md"
    file_path.write_text("hi", encoding="utf-8")
    result = runner.invoke(cli, ["vault", "use", str(file_path)])
    assert result.exit_code == 1
    assert "not a directory" in result.output


def test_vault_use_refuses_when_global_config_malformed(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """A present-but-unparseable config.toml may still hold provider settings and API keys;
    `vault use` must refuse rather than overwrite it with a vault-only config."""
    path = _global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("broken [ toml", encoding="utf-8")
    before = path.read_bytes()

    vault = _make_vault(tmp_path, "my-wiki")
    result = runner.invoke(cli, ["vault", "use", str(vault)])
    assert result.exit_code == 1
    assert "can't be parsed" in result.output
    assert path.read_bytes() == before


def test_vault_use_save_failure_exits_and_does_not_claim_success(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """Switching the default IS this command. If the save fails (permissions, disk full) the
    user must not be told it worked — every later command would still resolve the old vault."""
    vault = _make_vault(tmp_path, "my-wiki")

    with patch("synto.global_config.save_global_config", side_effect=OSError("disk full")):
        result = runner.invoke(cli, ["vault", "use", str(vault)])

    assert result.exit_code == 1
    assert "Default vault set to" not in result.output
    assert "disk full" in result.output
    assert load_global_config() is None


def test_vault_use_accepts_legacy_wiki_toml_vault(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    vault = tmp_path / "old-vault"
    vault.mkdir()
    (vault / "wiki.toml").write_text("[models]\n", encoding="utf-8")

    result = runner.invoke(cli, ["vault", "use", str(vault)])
    assert result.exit_code == 0, result.output
    # The default IS set, but most commands hard-fail on a legacy vault until migration —
    # the success message must not bury that.
    assert "migrate-olw" in result.output
    assert "most commands will fail" in result.output
    cfg = load_global_config()
    assert cfg is not None and cfg.vault == str(vault.resolve())


def test_vault_use_is_idempotent_no_duplicates(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    vault = _make_vault(tmp_path, "my-wiki")
    assert runner.invoke(cli, ["vault", "use", str(vault)]).exit_code == 0
    assert runner.invoke(cli, ["vault", "use", str(vault)]).exit_code == 0
    assert load_known_vaults() == [str(vault.resolve())]


# ── bare `synto vault` (list) ─────────────────────────────────────────────────


def test_bare_vault_lists_and_marks_default(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    vault_a = _make_vault(tmp_path, "vault-a")
    vault_b = _make_vault(tmp_path, "vault-b")
    runner.invoke(cli, ["vault", "use", str(vault_a)])
    runner.invoke(cli, ["vault", "use", str(vault_b)])

    result = runner.invoke(cli, ["vault"])
    assert result.exit_code == 0, result.output
    assert "Known vaults:" in result.output
    assert "vault-a" in result.output
    assert "vault-b" in result.output
    # vault-b is the default: marker + path + (default). Rich may soft-wrap long paths
    # across lines, so join the entry rather than requiring a single physical line.
    lines = result.output.splitlines()
    default_idx = next(i for i, line in enumerate(lines) if "(default)" in line)
    # Walk back to the list entry that starts with the marker (or its wrap siblings).
    entry = "\n".join(lines[max(0, default_idx - 2) : default_idx + 1])
    assert "vault-b" in entry
    assert "*" in entry
    # Non-default vault must not be marked as default.
    assert "(default)" not in "\n".join(line for line in lines if "vault-a" in line)


def test_bare_vault_flags_missing_and_not_a_vault(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    """A deleted dir and a dir that merely lost its config are different problems with
    different fixes (forget vs re-init) — the list must not lump both under [missing]."""
    import shutil

    vault_a = _make_vault(tmp_path, "vault-a")
    vault_b = _make_vault(tmp_path, "vault-b")
    vault_c = _make_vault(tmp_path, "vault-c")
    runner.invoke(cli, ["vault", "use", str(vault_b)])
    runner.invoke(cli, ["vault", "use", str(vault_c)])
    runner.invoke(cli, ["vault", "use", str(vault_a)])
    shutil.rmtree(vault_b)
    (vault_c / "synto.toml").unlink()

    result = runner.invoke(cli, ["vault"])
    assert result.exit_code == 0
    missing_line = next(line for line in result.output.splitlines() if "[missing]" in line)
    assert "vault-b" in missing_line
    not_vault_line = next(line for line in result.output.splitlines() if "[not a vault]" in line)
    assert "vault-c" in not_vault_line


def test_bare_vault_shows_unregistered_default(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    """A default that predates the registry (or was set by hand-editing config.toml) must
    still be listed."""
    vault = _make_vault(tmp_path, "pre-registry")
    save_global_config(GlobalConfig(vault=str(vault)))
    assert load_known_vaults() == []

    result = runner.invoke(cli, ["vault"])
    assert result.exit_code == 0
    assert "pre-registry" in result.output
    assert "(default)" in result.output


def test_bare_vault_survives_malformed_registry(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    vault = _make_vault(tmp_path, "my-wiki")
    save_global_config(GlobalConfig(vault=str(vault)))
    reg = _known_vaults_path()
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text("broken [ toml", encoding="utf-8")

    result = runner.invoke(cli, ["vault"])
    assert result.exit_code == 0
    assert "my-wiki" in result.output
    # The rewrite is destructive, so it must be announced before it happens.
    assert "unreadable" in result.output

    # The next successful registration rewrites the corrupt registry.
    assert runner.invoke(cli, ["vault", "use", str(vault)]).exit_code == 0
    assert load_known_vaults() == [str(vault.resolve())]


def test_malformed_registry_is_quarantined_not_clobbered(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """A registry that fails to parse still holds the only record of which vaults the user has.
    Overwriting it with the one vault being registered loses the rest silently."""
    vault_c = _make_vault(tmp_path, "vault-c")
    reg = _known_vaults_path()
    reg.parent.mkdir(parents=True, exist_ok=True)
    corrupt = 'vaults = ["/a/vault-a", "/b/vault-b"\n'  # truncated: valid paths, invalid TOML
    reg.write_bytes(corrupt.encode())

    assert runner.invoke(cli, ["vault", "use", str(vault_c)]).exit_code == 0

    quarantined = reg.with_name(reg.name + ".corrupt")
    assert quarantined.read_bytes() == corrupt.encode()
    assert load_known_vaults() == [str(vault_c.resolve())]


def test_bare_vault_empty_state(runner: CliRunner, cfg_dir: Path):
    result = runner.invoke(cli, ["vault"])
    assert result.exit_code == 0
    assert "No known vaults yet" in result.output


def test_bare_vault_notes_env_override(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """With SYNTO_VAULT set, the default is silently ignored by every command — the list
    must say so instead of letting the user believe switching worked."""
    vault = _make_vault(tmp_path, "my-wiki")
    runner.invoke(cli, ["vault", "use", str(vault)])
    monkeypatch.setenv("SYNTO_VAULT", str(tmp_path / "elsewhere"))

    result = runner.invoke(cli, ["vault"])
    assert result.exit_code == 0
    assert "SYNTO_VAULT" in result.output
    assert "overrides" in result.output


# ── vault forget ──────────────────────────────────────────────────────────────


def test_vault_forget_removes_entry(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    vault_a = _make_vault(tmp_path, "vault-a")
    vault_b = _make_vault(tmp_path, "vault-b")
    runner.invoke(cli, ["vault", "use", str(vault_a)])
    runner.invoke(cli, ["vault", "use", str(vault_b)])

    result = runner.invoke(cli, ["vault", "forget", str(vault_a)])
    assert result.exit_code == 0, result.output
    assert load_known_vaults() == [str(vault_b.resolve())]
    cfg = load_global_config()
    assert cfg is not None and cfg.vault == str(vault_b.resolve())


def test_vault_forget_default_warns_and_keeps_default(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """Forgetting is registry housekeeping; silently unsetting the default would change the
    vault resolution of every subsequent command."""
    vault = _make_vault(tmp_path, "my-wiki")
    runner.invoke(cli, ["vault", "use", str(vault)])

    result = runner.invoke(cli, ["vault", "forget", str(vault)])
    assert result.exit_code == 0
    assert "still the default" in result.output
    assert load_known_vaults() == []
    cfg = load_global_config()
    assert cfg is not None and cfg.vault == str(vault.resolve())


def test_vault_forget_default_warning_survives_non_canonical_default(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """The still-the-default check must compare canonical vault identity, not path strings.

    `config.toml` is user-editable and `synto setup` stores whatever the user typed, so the
    default can be a non-normalized spelling of the vault being forgotten. A raw string compare
    misses it and drops the warning, leaving the user to think forgetting also cleared the
    default.
    """
    vault = _make_vault(tmp_path, "my-wiki")
    runner.invoke(cli, ["vault", "use", str(vault)])

    # Same vault, different spelling — survives the TOML round-trip uncollapsed.
    cfg = load_global_config()
    assert cfg is not None
    cfg.vault = f"{tmp_path}/./my-wiki"
    save_global_config(cfg)
    assert load_global_config().vault != str(vault.resolve())

    result = runner.invoke(cli, ["vault", "forget", str(vault)])

    assert result.exit_code == 0, result.output
    assert "still the default" in result.output


def test_vault_forget_unknown_exits_1(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    result = runner.invoke(cli, ["vault", "forget", str(tmp_path / "never-registered")])
    assert result.exit_code == 1
    assert "Not in known vaults" in result.output


def test_vault_forget_write_failure_reports_registry_error(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """A registry the CLI could not write is not the same as a vault it never knew about —
    reporting the write failure as "Not in known vaults" tells the user the entry is gone when
    it is still there."""
    vault = _make_vault(tmp_path, "my-wiki")
    runner.invoke(cli, ["vault", "use", str(vault)])

    with patch("synto.global_config.save_known_vaults", side_effect=OSError("read-only fs")):
        result = runner.invoke(cli, ["vault", "forget", str(vault)])

    assert result.exit_code == 1
    assert "Not in known vaults" not in result.output
    assert "known-vaults registry" in result.output
    assert load_known_vaults() == [str(vault.resolve())]


# ── auto-registration: init / setup ───────────────────────────────────────────


def test_init_registers_vault(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    vault = tmp_path / "fresh-vault"
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0, result.output
    assert str(vault.resolve()) in load_known_vaults()
    # Without --default the global config must not be created/touched.
    assert load_global_config() is None


def test_init_default_registers_and_sets_default(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    vault = tmp_path / "fresh-vault"
    result = runner.invoke(cli, ["init", str(vault), "--default"])
    assert result.exit_code == 0, result.output
    assert str(vault.resolve()) in load_known_vaults()
    cfg = load_global_config()
    assert cfg is not None and cfg.vault == str(vault.resolve())


def test_init_default_refuses_overwrite_when_config_malformed(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """Same guard as `vault use`: a corrupt config.toml may still hold recoverable provider
    settings — init must create the vault but leave the file alone."""
    path = _global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("broken [ toml", encoding="utf-8")
    before = path.read_bytes()

    vault = tmp_path / "fresh-vault"
    result = runner.invoke(cli, ["init", str(vault), "--default"])
    assert result.exit_code == 0, result.output
    assert "can't be parsed" in result.output
    assert path.read_bytes() == before
    assert vault.is_dir()


def test_init_default_save_failure_does_not_claim_default(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """When the global-config save fails for a non-parse reason (permissions, disk full),
    init must not print the set-as-default success line or drop --vault from the next
    steps — the default was never saved and later commands would not resolve this vault."""
    vault = tmp_path / "fresh-vault"
    with patch("synto.global_config.save_global_config", side_effect=OSError("disk full")):
        result = runner.invoke(cli, ["init", str(vault), "--default"])

    assert result.exit_code == 0, result.output
    assert "Could not save default vault" in result.output
    assert "Set as default vault" not in result.output
    assert "--vault" in result.output
    assert load_global_config() is None


def test_setup_wizard_registers_vault(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    vault = _make_vault(tmp_path, "wizard-vault")
    with patch("synto.ollama_client.OllamaClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance

        result = runner.invoke(
            cli,
            ["setup"],
            # provider default, URL default, fast, heavy, vault path, citations off,
            # don't apply to the existing vault's synto.toml
            input=f"\n\ngemma4:e4b\n\nqwen2.5:14b\n{vault}\n\n\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert str(vault.resolve()) in load_known_vaults()
