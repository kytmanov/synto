"""
Tests for global_config module and synto setup command.
All tests are offline — no Ollama instance required.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from synto.cli import cli
from synto.config import Config
from synto.global_config import (
    GlobalConfig,
    _global_config_path,
    load_global_config,
    save_global_config,
)
from synto.models import RawNoteRecord
from synto.state import StateDB

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect global config to a temp dir."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # On Windows, patch APPDATA too
    monkeypatch.setenv("APPDATA", str(tmp_path))
    return tmp_path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ── string quoting / escaping (now owned by the to_toml serializer) ─────────────


def test_control_chars_round_trip_through_save_load(cfg_dir: Path):
    """Values with control chars (tabs, quotes, backslashes) must survive save → load."""
    cfg = GlobalConfig(fast_model='model\twith\t"quotes"\\and\\backslashes')
    save_global_config(cfg)
    loaded = load_global_config()
    assert loaded is not None
    assert loaded.fast_model == 'model\twith\t"quotes"\\and\\backslashes'


# ── save / load round-trip ────────────────────────────────────────────────────


def test_save_load_full_config(cfg_dir: Path):
    cfg = GlobalConfig(
        vault="/tmp/my-wiki",
        ollama_url="http://localhost:11434",
        fast_model="gemma4:e4b",
        heavy_model="qwen2.5:14b",
    )
    save_global_config(cfg)
    loaded = load_global_config()
    assert loaded == cfg


def test_save_load_experimental_inline_source_citations(cfg_dir: Path):
    cfg = GlobalConfig(experimental_inline_source_citations=True)
    save_global_config(cfg)
    loaded = load_global_config()
    assert loaded is not None
    assert loaded.experimental_inline_source_citations is True


def test_save_creates_parent_dirs(cfg_dir: Path):
    cfg = GlobalConfig(fast_model="gemma4:e4b")
    save_global_config(cfg)
    path = _global_config_path()
    assert path.exists()
    assert path.parent.is_dir()


def test_saved_file_is_valid_toml(cfg_dir: Path):
    cfg = GlobalConfig(
        vault="/tmp/wiki",
        fast_model="gemma4:e4b",
        heavy_model="qwen2.5:14b",
        ollama_url="http://localhost:11434",
    )
    save_global_config(cfg)
    path = _global_config_path()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    assert data["fast_model"] == "gemma4:e4b"
    assert data["heavy_model"] == "qwen2.5:14b"


def test_partial_config_no_null_keys(cfg_dir: Path):
    """None fields must not appear in the written TOML file."""
    cfg = GlobalConfig(fast_model="gemma4:e4b")
    save_global_config(cfg)
    path = _global_config_path()
    raw = path.read_text()
    assert "vault" not in raw
    assert "ollama_url" not in raw
    assert "heavy_model" not in raw
    assert "fast_model" in raw


def test_empty_config_writes_empty_file(cfg_dir: Path):
    save_global_config(GlobalConfig())
    path = _global_config_path()
    assert path.read_text() == ""


# ── load error handling ───────────────────────────────────────────────────────


def test_load_missing_file_returns_none(cfg_dir: Path):
    result = load_global_config()
    assert result is None


def test_load_malformed_toml_returns_none(cfg_dir: Path):
    path = _global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("this is not [ valid toml !!!!", encoding="utf-8")
    result = load_global_config()
    assert result is None


def test_load_unknown_fields_returns_none(cfg_dir: Path):
    """Extra keys not in GlobalConfig schema should cause load to return None."""
    path = _global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('unknown_key = "value"\n', encoding="utf-8")
    result = load_global_config()
    assert result is None


# ── _load_config fallback ─────────────────────────────────────────────────────


def test_load_config_uses_global_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg_dir: Path
):
    """_load_config(None) should fall back to global config vault."""
    vault = tmp_path / "wiki"
    vault.mkdir()
    (vault / "synto.toml").write_text(
        '[models]\nfast = "gemma4:e4b"\nheavy = "qwen2.5:14b"\n[ollama]\nurl = "http://localhost:11434"\n'
    )

    save_global_config(GlobalConfig(vault=str(vault)))

    # Import here to use real _load_config
    from synto.cli import _load_config

    config = _load_config(None)
    assert config.vault == vault


def test_load_config_no_vault_exits(monkeypatch: pytest.MonkeyPatch, cfg_dir: Path):
    """_load_config(None) with no global config should sys.exit(1)."""
    from synto.cli import _load_config

    monkeypatch.delenv("SYNTO_VAULT", raising=False)
    monkeypatch.chdir(cfg_dir)
    with pytest.raises(SystemExit) as exc:
        _load_config(None)
    assert exc.value.code == 1


def test_load_config_detects_vault_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg_dir: Path
):
    """_load_config(None) detects vault when CWD contains synto.toml."""
    vault = tmp_path / "myvault"
    vault.mkdir()
    (vault / "synto.toml").write_text(
        '[models]\nfast = "gemma4:e4b"\nheavy = "gemma4:e4b"\n[ollama]\nurl = "http://localhost:11434"\n'
    )
    monkeypatch.delenv("SYNTO_VAULT", raising=False)
    monkeypatch.chdir(vault)

    from synto.cli import _load_config

    config = _load_config(None)
    assert config.vault == vault


def test_load_config_detects_vault_from_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg_dir: Path
):
    """_load_config(None) walks up from a subdirectory to find synto.toml."""
    vault = tmp_path / "myvault"
    subdir = vault / "raw" / "notes"
    subdir.mkdir(parents=True)
    (vault / "synto.toml").write_text(
        '[models]\nfast = "gemma4:e4b"\nheavy = "gemma4:e4b"\n[ollama]\nurl = "http://localhost:11434"\n'
    )
    monkeypatch.delenv("SYNTO_VAULT", raising=False)
    monkeypatch.chdir(subdir)

    from synto.cli import _load_config

    config = _load_config(None)
    assert config.vault == vault


# ── synto setup --non-interactive ───────────────────────────────────────────────


def test_setup_non_interactive_no_config(runner: CliRunner, cfg_dir: Path):
    result = runner.invoke(cli, ["setup", "--non-interactive"])
    assert result.exit_code == 0
    assert "No global config" in result.output


def test_setup_non_interactive_with_config(runner: CliRunner, cfg_dir: Path):
    save_global_config(
        GlobalConfig(
            fast_model="gemma4:e4b",
            heavy_model="qwen2.5:14b",
            ollama_url="http://192.168.1.10:11434",
            experimental_inline_source_citations=True,
        )
    )
    result = runner.invoke(cli, ["setup", "--non-interactive"])
    assert result.exit_code == 0
    assert "gemma4:e4b" in result.output
    assert "qwen2.5:14b" in result.output
    assert "192.168.1.10" in result.output
    assert "Inline source citations" in result.output
    assert "on" in result.output


# ── synto setup --reset ─────────────────────────────────────────────────────────


def test_setup_reset_clears_config(runner: CliRunner, cfg_dir: Path):
    save_global_config(GlobalConfig(fast_model="old-model"))

    # Provide all wizard inputs via stdin so it runs non-interactively in tests
    # Inputs: URL, fast model, heavy model, vault, citations
    result = runner.invoke(
        cli,
        ["setup", "--reset"],
        input="\n\n\n\n\n\n\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    loaded = load_global_config()
    # After reset + wizard with defaults, old-model should be gone
    assert loaded is not None
    assert loaded.fast_model != "old-model"


# ── synto setup wizard (stdin input) ───────────────────────────────────────────


def test_setup_wizard_saves_config(runner: CliRunner, cfg_dir: Path):
    """Wizard with all-default inputs should create a valid config."""
    with patch("synto.ollama_client.OllamaClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance

        result = runner.invoke(
            cli,
            ["setup"],
            # provider default, URL default, fast, heavy, no vault, citations off
            input="\n\ngemma4:e4b\n\nqwen2.5:14b\n\n\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None
    assert cfg.fast_model == "gemma4:e4b"
    assert cfg.heavy_model == "qwen2.5:14b"
    assert cfg.experimental_inline_source_citations is False


def test_setup_wizard_saves_experimental_inline_source_citations(runner: CliRunner, cfg_dir: Path):
    with patch("synto.ollama_client.OllamaClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance

        result = runner.invoke(
            cli,
            ["setup"],
            input="\n\ngemma4:e4b\n\nqwen2.5:14b\n\ny\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None
    assert cfg.experimental_inline_source_citations is True


def test_setup_wizard_summary_says_uninitialized_vault_will_use_preference(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    vault = tmp_path / "new-vault"
    with patch("synto.ollama_client.OllamaClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance

        result = runner.invoke(
            cli,
            ["setup"],
            input=f"\n\ngemma4:e4b\n\nqwen2.5:14b\n{vault}\ny\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert "Inline source citations: on for new vaults" in result.output
    assert "Current vault: not initialized yet; will be on after init" in result.output
    assert "Current vault: not set" not in result.output


def test_setup_wizard_summary_mentions_support_and_metrics(runner: CliRunner, cfg_dir: Path):
    with patch("synto.ollama_client.OllamaClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance

        result = runner.invoke(
            cli,
            ["setup"],
            input="\n\ngemma4:e4b\n\nqwen2.5:14b\n\n\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert "Feedback:" in result.output
    assert "synto support" in result.output
    assert "local runtime and cost metrics" in result.output


def test_setup_wizard_model_number_selection(runner: CliRunner, cfg_dir: Path):
    """Selecting model by number from the list should resolve to model name."""
    with patch("synto.ollama_client.OllamaClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = True
        instance.list_models_detailed.return_value = [
            {"name": "gemma4:e4b", "size_gb": "4.3 GB"},
            {"name": "qwen2.5:14b", "size_gb": "8.7 GB"},
        ]
        MockClient.return_value = instance

        result = runner.invoke(
            cli,
            ["setup"],
            # provider default, URL default, pick #1 fast, pick #2 heavy, no vault, citations off
            input="\n\n1\n\n2\n\n\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None
    assert cfg.fast_model == "gemma4:e4b"
    assert cfg.heavy_model == "qwen2.5:14b"


def test_setup_wizard_whitespace_input_uses_default(runner: CliRunner, cfg_dir: Path):
    """Spaces-only model input should fall back to default, not save a blank model name."""
    with patch("synto.ollama_client.OllamaClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance

        # Send spaces for fast and heavy model prompts (no table, free-text path)
        result = runner.invoke(
            cli,
            ["setup"],
            # provider default, URL default, blank models, no vault, citations off
            input="\n\n   \n\n   \n\n\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None
    # Should have fallen back to defaults, not saved empty/whitespace strings
    assert cfg.fast_model and cfg.fast_model.strip() != ""
    assert cfg.heavy_model and cfg.heavy_model.strip() != ""


def test_setup_wizard_with_vault(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    """Setting a vault path in wizard should save it to global config."""
    vault = tmp_path / "my-wiki"

    with patch("synto.ollama_client.OllamaClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance

        result = runner.invoke(
            cli,
            ["setup"],
            input=f"\n\ngemma4:e4b\n\nqwen2.5:14b\n{vault}\n\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None
    assert cfg.vault == str(vault.resolve())


def test_setup_wizard_per_role_branch_reuses_primary_as_fast(runner: CliRunner, cfg_dir: Path):
    """Progressive disclosure (#24): no upfront question; the per-role split is offered after the
    fast model and reuses the already-configured primary provider as `fast`, collecting only the
    heavy provider. Answering "y" must persist a multi-provider config, not a flat one."""
    with (
        patch("synto.ollama_client.OllamaClient") as MockClient,
        patch("synto.openai_compat_client.OpenAICompatClient") as MockCloudClient,
    ):
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance
        # Heavy is groq → the unified routine now builds and probes an OpenAICompatClient.
        # Keep it offline; with no models the heavy model falls through to free-text entry.
        cloud_instance = MagicMock()
        cloud_instance.healthcheck.return_value = False
        cloud_instance.list_models_detailed.return_value = []
        MockCloudClient.return_value = cloud_instance

        # primary=ollama default + fast model, "y" to a different heavy provider, then the heavy
        # provider (groq, default key env, model), no vault, citations off.
        result = runner.invoke(
            cli,
            ["setup"],
            input="\n\ngemma4:e4b\ny\ngroq\n\n\nllama-3.3-70b\n\n\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    # The removed upfront prompt is gone; the contextual one is asked instead.
    assert "Use the same provider for all models" not in result.output
    assert "Use a different provider for the heavy (writing) model" in result.output

    cfg = load_global_config()
    assert cfg is not None and cfg.is_multi_provider
    # fast reuses the primary (ollama) as the "default" connection — no re-prompting.
    assert cfg.models["fast"].provider == "default"
    assert cfg.providers["default"].name == "ollama"
    assert cfg.models["fast"].model == "gemma4:e4b"
    # heavy is the separately-collected cloud provider.
    heavy_alias = cfg.models["heavy"].provider
    assert cfg.providers[heavy_alias].name == "groq"
    assert cfg.providers[heavy_alias].api_key_env == "GROQ_API_KEY"
    assert cfg.models["heavy"].model == "llama-3.3-70b"


def test_setup_wizard_per_role_heavy_lists_models(runner: CliRunner, cfg_dir: Path):
    """The unified routine routes Heavy through the same probe + model table as Fast: when the
    heavy provider is reachable and lists models, the user picks by number from the table instead
    of being forced to type a model name."""
    with (
        patch("synto.ollama_client.OllamaClient") as MockClient,
        patch("synto.openai_compat_client.OpenAICompatClient") as MockCloudClient,
    ):
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance
        # Heavy = LM Studio, reachable, advertises two models → the table is shown.
        cloud_instance = MagicMock()
        cloud_instance.healthcheck.return_value = True
        cloud_instance.list_models_detailed.return_value = [
            {"name": "qwen2.5:7b", "size_gb": "4.5 GB"},
            {"name": "llama-3.1-8b", "size_gb": "8.0 GB"},
        ]
        MockCloudClient.return_value = cloud_instance

        # primary=ollama default + fast model, "y" to different heavy, heavy=lm_studio (default URL;
        # local → no API-key-env prompt), then select heavy model #2 from the table, no vault,
        # citations off.
        result = runner.invoke(
            cli,
            ["setup"],
            input="\n\n\ny\nlm_studio\n\n2\n\n\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None and cfg.is_multi_provider
    heavy_alias = cfg.models["heavy"].provider
    assert cfg.providers[heavy_alias].name == "lm_studio"
    # Local heavy provider must not get a phantom API-key-env requirement.
    assert cfg.providers[heavy_alias].api_key_env is None
    # Picked by number from the table → the second listed model, proving the table path ran.
    assert cfg.models["heavy"].model == "llama-3.1-8b"


def test_setup_wizard_per_role_same_local_provider_dedupes_without_phantom_key(
    runner: CliRunner, cfg_dir: Path
):
    """Regression: picking the same local provider (LM Studio) for primary and heavy must collapse
    to a single key-less provider. The bug was that the heavy step prompted for an API-key env var
    on a local provider, defaulting to a phantom PROVIDER_API_KEY — which both demanded a needless
    key and broke dedup (key mismatch), leaving two identical provider blocks."""
    with patch("synto.openai_compat_client.OpenAICompatClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = True
        instance.list_models_detailed.return_value = [
            {"name": "google/gemma-4-e4b", "size_gb": "4.5 GB"},
        ]
        MockClient.return_value = instance

        # primary=lm_studio (default URL, blank raw key), fast model #1, "y" to a different heavy,
        # heavy=lm_studio again (default URL, no key prompt), heavy model #1, no vault, citations.
        result = runner.invoke(
            cli,
            ["setup"],
            input="lm_studio\n\n\n1\ny\nlm_studio\n\n1\n\n\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None and cfg.is_multi_provider
    # Identical local connections collapse to one provider — no redundant second block.
    assert len(cfg.providers) == 1
    alias = cfg.models["fast"].provider
    assert cfg.models["heavy"].provider == alias
    block = cfg.providers[alias]
    assert block.name == "lm_studio"
    assert block.api_key_env is None
    # And the summary must not tell the user to set an env var for a local no-auth server.
    assert "Set the API-key env var" not in result.output


def test_setup_wizard_no_upfront_same_provider_question(runner: CliRunner, cfg_dir: Path):
    """The default single-provider path must not show the old upfront question and must save a
    flat (non-multi) config when the heavy provider is left the same."""
    with patch("synto.ollama_client.OllamaClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance

        result = runner.invoke(
            cli,
            ["setup"],
            input="\n\ngemma4:e4b\n\nqwen2.5:14b\n\n\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert "Use the same provider for all models" not in result.output
    cfg = load_global_config()
    assert cfg is not None and not cfg.is_multi_provider
    assert cfg.fast_model == "gemma4:e4b" and cfg.heavy_model == "qwen2.5:14b"


# ── synto init uses global config models ───────────────────────────────────────


def test_init_uses_global_config_models(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    """synto init should pre-fill synto.toml from global config models."""
    save_global_config(
        GlobalConfig(
            fast_model="llama3.2:3b",
            heavy_model="llama3.1:8b",
            ollama_url="http://192.168.1.5:11434",
        )
    )
    vault = tmp_path / "test-vault"
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0

    toml_path = vault / "synto.toml"
    assert toml_path.exists()
    content = toml_path.read_text()
    assert "llama3.2:3b" in content
    assert "llama3.1:8b" in content
    assert "192.168.1.5" in content


def test_init_warns_when_no_global_config(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    """A fresh vault with no global config gets Ollama defaults — init must say so and point at
    `synto setup`, not silently wire the vault for a provider the user may not use."""
    vault = tmp_path / "no-config-vault"
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0
    assert "No global config found" in result.output
    assert "synto setup" in result.output


def test_init_no_warning_when_provider_configured(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    """When a provider is configured, init inherits it silently — no missing-config warning."""
    save_global_config(
        GlobalConfig(
            provider_name="ollama",
            fast_model="gemma4:e4b",
            heavy_model="gemma4:e4b",
            ollama_url="http://localhost:11434",
        )
    )
    vault = tmp_path / "configured-vault"
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0
    assert "No global config" not in result.output
    assert "No provider configured" not in result.output


def test_init_leaves_existing_vault_unchanged_without_global_config(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """Regression: with no global config there is nothing to sync from, so an already-configured
    vault must be left alone — never have Ollama's URL written into its (e.g. lm_studio) block."""
    vault = tmp_path / "lmstudio-vault"
    vault.mkdir()
    (vault / "synto.toml").write_text(
        _new_format_vault_toml("lm_studio", "http://localhost:1234/v1")
    )

    result = runner.invoke(cli, ["init", str(vault)])  # no save_global_config → gcfg is None
    assert result.exit_code == 0
    assert "leaving existing" in result.output

    parsed = tomllib.loads((vault / "synto.toml").read_text())
    assert parsed["providers"]["default"]["name"] == "lm_studio"
    assert parsed["providers"]["default"]["url"] == "http://localhost:1234/v1"


def test_init_non_default_vault_shows_vault_flag_and_default_tip(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """When no default vault is configured, next-steps must carry --vault and hint at --default
    so the user isn't left retyping the long absolute path on every command."""
    vault = tmp_path / "fresh-vault"
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0
    # Commands carry the flag; the tip points at --default. (Rich may wrap the long path, so match
    # on the contiguous command prefix rather than the full path.)
    assert "run --vault" in result.output
    assert "--default" in result.output


def test_init_when_vault_already_global_default_drops_vault_flag(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """If the global config already points here (e.g. set by `synto setup`), re-initialising must
    recognise it as the default and emit clean commands — no redundant --vault flags."""
    vault = tmp_path / "default-vault"
    save_global_config(GlobalConfig(vault=str(vault.resolve())))
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0
    assert "Already your default vault" in result.output
    # Commands must be clean — no --vault on any step (the only mention is the "needed" note).
    assert "run --vault" not in result.output
    assert "review --vault" not in result.output


def test_init_applies_experimental_inline_source_citations_preference(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    save_global_config(
        GlobalConfig(
            fast_model="gemma4:e4b",
            heavy_model="gemma4:e4b",
            experimental_inline_source_citations=True,
        )
    )
    vault = tmp_path / "test-vault"
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0

    content = (vault / "synto.toml").read_text()
    assert "inline_source_citations = true" in content


def test_init_defaults_without_global_config(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    """synto init without global config should use built-in defaults."""
    vault = tmp_path / "test-vault"
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0

    content = (vault / "synto.toml").read_text()
    assert "gemma4:e4b" in content
    assert "qwen2.5:14b" in content
    assert "# inline_source_citations = false" in content


def test_init_syncs_models_into_existing_wiki_toml(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """synto init on an existing vault should patch models from global config."""
    vault = tmp_path / "existing-vault"
    vault.mkdir()
    # Simulate old synto.toml with stale heavy model
    old_toml = (
        '[models]\nfast = "gemma4:e4b"\nheavy = "qwen2.5:14b"\n\n'
        '[ollama]\nurl = "http://localhost:11434"\ntimeout = 600\n\n'
        "[pipeline]\nauto_approve = false\nauto_commit = true\n"
    )
    (vault / "synto.toml").write_text(old_toml)

    save_global_config(
        GlobalConfig(
            fast_model="gemma4:e4b",
            heavy_model="gemma4:e4b",
            ollama_url="http://localhost:11434",
        )
    )
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0

    content = (vault / "synto.toml").read_text()
    # heavy should now be gemma4:e4b (patched from global config)
    assert 'heavy = "gemma4:e4b"' in content
    # pipeline settings must be preserved
    assert "auto_approve = false" in content


def _new_format_vault_toml(provider: str, url: str) -> str:
    from synto.config import ModelProfile, ProviderBlock, multi_provider_vault_toml

    return multi_provider_vault_toml(
        {"default": ProviderBlock(name=provider, url=url)},
        {
            "fast": ModelProfile(provider="default", model="m-fast", ctx=8192),
            "heavy": ModelProfile(provider="default", model="m-heavy", ctx=32768),
        },
        inline_source_citations=False,
    )


def test_init_sync_preserves_new_format_sections(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    """Regression: syncing a new-format vault must not destroy it. The old _replace_in_section
    used a greedy `.+` under re.DOTALL, so patching [providers.default].url matched through to the
    file's last quote (the `# language = "en"` comment), deleting every [models.*]/[pipeline]
    section in between. With a matching provider the sync runs — and must leave valid TOML."""
    vault = tmp_path / "newfmt-vault"
    vault.mkdir()
    # Provider matches the global default (ollama) so the sync runs and exercises the regex.
    (vault / "synto.toml").write_text(_new_format_vault_toml("ollama", "http://localhost:11434"))

    save_global_config(
        GlobalConfig(
            provider_name="ollama",
            fast_model="new-fast",
            heavy_model="new-heavy",
            ollama_url="http://localhost:9999",
        )
    )
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0

    content = (vault / "synto.toml").read_text()
    parsed = tomllib.loads(content)  # must still be valid TOML, not a mangled fragment
    assert "fast" in parsed["models"] and "heavy" in parsed["models"]
    assert parsed["pipeline"]["graph_quality_checks"] is True
    # The url + models were patched in place without swallowing the rest of the file.
    assert parsed["providers"]["default"]["url"] == "http://localhost:9999"
    assert parsed["models"]["fast"]["model"] == "new-fast"


def test_init_respects_vault_provider_on_mismatch(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    """If the vault is configured for a different provider than the global default, init must leave
    its provider/url/models untouched — never write the global provider's URL into a foreign block
    (the bug that pointed lm_studio at Ollama's port)."""
    vault = tmp_path / "lmstudio-vault"
    vault.mkdir()
    (vault / "synto.toml").write_text(
        _new_format_vault_toml("lm_studio", "http://localhost:1234/v1")
    )

    # Global default is Ollama — a different provider.
    save_global_config(
        GlobalConfig(provider_name="ollama", fast_model="g", heavy_model="g", ollama_url="x:11434")
    )
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0
    assert "configured for 'lm_studio'" in result.output

    parsed = tomllib.loads((vault / "synto.toml").read_text())
    # Vault config is respected: LM Studio URL and models intact, no Ollama bleed-through.
    assert parsed["providers"]["default"]["name"] == "lm_studio"
    assert parsed["providers"]["default"]["url"] == "http://localhost:1234/v1"
    assert parsed["models"]["fast"]["model"] == "m-fast"


def test_init_respects_vault_provider_against_multi_provider_global(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """Respect-the-vault also guards the multi-provider 'reproduce' path: a vault set up for one
    provider must not be silently rewritten to the global multi-provider default."""
    from synto.config import ModelProfile, ProviderBlock

    vault = tmp_path / "lmstudio-vault"
    vault.mkdir()
    (vault / "synto.toml").write_text(
        _new_format_vault_toml("lm_studio", "http://localhost:1234/v1")
    )

    # Global default is a multi-provider Ollama split — a different provider than the vault.
    save_global_config(
        GlobalConfig(
            providers={"default": ProviderBlock(name="ollama", url="http://localhost:11434")},
            models={"fast": ModelProfile(provider="default", model="gemma4:e4b")},
        )
    )
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0
    assert "configured for 'lm_studio'" in result.output

    parsed = tomllib.loads((vault / "synto.toml").read_text())
    assert parsed["providers"]["default"]["name"] == "lm_studio"
    assert parsed["providers"]["default"]["url"] == "http://localhost:1234/v1"


# ── default_wiki_toml pipeline fields ────────────────────────────────────────


def test_default_wiki_toml_contains_auto_maintain():
    """auto_maintain must appear in generated synto.toml so synto init exposes it."""
    from synto.config import default_wiki_toml

    content = default_wiki_toml()
    assert "auto_maintain" in content
    # Must be valid TOML
    parsed = tomllib.loads(content)
    assert parsed["pipeline"]["auto_maintain"] is False


def test_config_inline_source_citations_status_not_set(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "synto.toml").write_text('[models]\nfast="a"\nheavy="b"\n')

    result = runner.invoke(
        cli, ["config", "inline-source-citations", "status", "--vault", str(vault)]
    )

    assert result.exit_code == 0
    assert "not set (default: disabled)" in result.output


def test_config_inline_source_citations_status_invalid_toml(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "synto.toml").write_text("[models\n")

    result = runner.invoke(
        cli, ["config", "inline-source-citations", "status", "--vault", str(vault)]
    )

    assert result.exit_code == 1
    assert "Invalid TOML" in result.output


def test_config_inline_source_citations_status_rejects_integer(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "synto.toml").write_text(
        '[models]\nfast="a"\nheavy="b"\n\n[pipeline]\ninline_source_citations = 1\n'
    )

    result = runner.invoke(
        cli, ["config", "inline-source-citations", "status", "--vault", str(vault)]
    )

    assert result.exit_code == 1
    assert "expected boolean true/false" in result.output


def test_config_inline_source_citations_on_off_preserves_comments(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    vault = tmp_path / "vault"
    vault.mkdir()
    toml = vault / "synto.toml"
    toml.write_text(
        '[models]\nfast = "a"\nheavy = "b"\n\n[pipeline]\n# keep me\nauto_commit = true\n'
    )

    on = runner.invoke(cli, ["config", "inline-source-citations", "on", "--vault", str(vault)])
    off = runner.invoke(cli, ["config", "inline-source-citations", "off", "--vault", str(vault)])

    content = toml.read_text()
    assert on.exit_code == 0
    assert off.exit_code == 0
    assert "# keep me" in content
    assert "auto_commit = true" in content
    assert "inline_source_citations = false" in content


def test_config_inline_source_citations_creates_pipeline_section(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    vault = tmp_path / "vault"
    vault.mkdir()
    toml = vault / "synto.toml"
    toml.write_text('[models]\nfast = "a"\nheavy = "b"\n')

    result = runner.invoke(cli, ["config", "inline-source-citations", "on", "--vault", str(vault)])

    assert result.exit_code == 0
    assert "[pipeline]" in toml.read_text()
    assert "inline_source_citations = true" in toml.read_text()


def test_clean_preserves_source_concept_seed_before_deleting_state(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    vault = tmp_path / "vault"
    runner.invoke(cli, ["init", str(vault)])
    config = Config.from_vault(vault)
    db = StateDB(config.state_db_path)
    db.upsert_raw(
        RawNoteRecord(
            path="raw/api.md",
            content_hash="body-hash",
            status="ingested",
            language="en",
        )
    )
    db.upsert_concepts("raw/api.md", ["Response Validation", "Testing Checklist"])
    db.close()

    result = runner.invoke(cli, ["clean", "--vault", str(vault), "--yes"])

    assert result.exit_code == 0
    assert not config.state_db_path.exists()
    index_payload = (config.synto_dir / "INDEX.json").read_text(encoding="utf-8")
    assert "Response Validation" in index_payload
    assert "Testing Checklist" in index_payload
    assert "body-hash" in index_payload


def test_doctor_prints_graph_guidance(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    vault = tmp_path / "vault"
    runner.invoke(cli, ["init", str(vault)])
    (vault / "Welcome.md").write_text("Welcome. [[create a link]]")

    result = runner.invoke(cli, ["doctor", "--vault", str(vault)])

    assert "Graph view" in result.output
    assert "Draft review graph filter" in result.output
    assert "Published-only graph filter" in result.output
    assert "-path:raw" in result.output


# ── .gitignore defaults ───────────────────────────────────────────────────────


def test_init_gitignore_includes_lock_and_exports(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
) -> None:
    vault = tmp_path / "new-vault"
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0
    gitignore = (vault / ".gitignore").read_text()
    assert ".synto/pipeline.lock" in gitignore
    assert ".synto/exports/" in gitignore


def test_init_gitignore_skipped_when_exists(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
) -> None:
    vault = tmp_path / "existing-vault"
    vault.mkdir()
    (vault / ".gitignore").write_text("# custom\n")
    runner.invoke(cli, ["init", str(vault)])
    # Existing .gitignore is preserved (not overwritten by init)
    assert (vault / ".gitignore").read_text() == "# custom\n"


def test_init_default_flag_sets_vault_in_global_config(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
) -> None:
    save_global_config(GlobalConfig(fast_model="gemma4:e4b", heavy_model="qwen2.5:14b"))
    vault = tmp_path / "new-vault"
    result = runner.invoke(cli, ["init", str(vault), "--default"])
    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None
    assert cfg.vault == str(vault)


def test_init_without_default_flag_does_not_change_global_config(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
) -> None:
    save_global_config(GlobalConfig(fast_model="gemma4:e4b", vault="/original/vault"))
    vault = tmp_path / "new-vault"
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None
    assert cfg.vault == "/original/vault"
