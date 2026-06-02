"""Tests for the simplified compare CLI."""

from __future__ import annotations

from click.testing import CliRunner

from synto.cli import _compare_config_summary, cli
from synto.compare.models import AdvisorVerdict
from synto.config import Config, ProviderBlock


def _make_vault(tmp_path):
    vault = tmp_path / "vault"
    raw = vault / "raw"
    wiki = vault / "wiki"
    synto_dir = vault / ".synto"
    raw.mkdir(parents=True)
    wiki.mkdir()
    synto_dir.mkdir()
    for i in range(3):
        (raw / f"n{i}.md").write_text(f"# Note {i}\n\nBody {i}.\n")
    (vault / "synto.toml").write_text(
        '[models]\nfast = "base-fast"\nheavy = "base-heavy"\n\n[ollama]\nurl = "http://localhost:11434"\n'
    )
    return vault


def test_compare_requires_override(tmp_path):
    vault = _make_vault(tmp_path)
    result = CliRunner().invoke(cli, ["compare", "--vault", str(vault)])
    assert result.exit_code == 1
    assert "Provide at least one challenger override" in result.output


def test_compare_rejects_identical_override(tmp_path):
    vault = _make_vault(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "compare",
            "--vault",
            str(vault),
            "--fast-model",
            "base-fast",
            "--heavy-model",
            "base-heavy",
        ],
    )
    assert result.exit_code == 1
    assert "identical to current config" in result.output


def test_compare_summary_detects_fast_only_provider_change():
    # The identical-check must not collapse to the heavy role: a fast-role provider/url change
    # is a real difference. (CLI flags can't express this yet, so guard the summary directly.)
    base = Config(
        vault="/tmp/v",
        providers={
            "local": ProviderBlock(name="ollama", url="http://localhost:11434"),
            "cloud": ProviderBlock(name="groq", url="https://api.groq.com/openai/v1"),
        },
        models={
            "fast": {"provider": "local", "model": "f"},
            "heavy": {"provider": "cloud", "model": "h"},
        },
    )
    # Same heavy, but fast moved from local ollama to the cloud provider.
    moved_fast = base.model_copy(
        update={
            "models": base.models.model_copy(
                update={"fast": base.models.fast.model_copy(update={"provider": "cloud"})}
            )
        }
    )
    assert _compare_config_summary(base) != _compare_config_summary(moved_fast)
    # Truly identical configs compare equal.
    assert _compare_config_summary(base) == _compare_config_summary(base.model_copy(deep=True))
    # Same model + kind + url but a different account/header is a real challenger, not "identical":
    # connection identity (api_key/headers), not just kind/url. (No CLI flag expresses this yet, so
    # the summary must guard it directly.)
    with_header = base.model_copy(
        update={
            "providers": {
                **base.providers,
                "cloud": base.providers["cloud"].model_copy(update={"headers": {"X-Org": "acme"}}),
            }
        }
    )
    assert _compare_config_summary(base) != _compare_config_summary(with_header)


def test_compare_rejects_out_inside_raw(tmp_path):
    vault = _make_vault(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "compare",
            "--vault",
            str(vault),
            "--heavy-model",
            "new-heavy",
            "--out",
            str(vault / "raw" / "x"),
        ],
    )
    assert result.exit_code == 2
    assert "must not be inside raw/ or wiki/" in result.output


def test_compare_rejects_symlinked_queries(tmp_path):
    vault = _make_vault(tmp_path)
    target = tmp_path / "real.toml"
    target.write_text("")
    link = tmp_path / "queries.toml"
    link.symlink_to(target)
    result = CliRunner().invoke(
        cli,
        [
            "compare",
            "--vault",
            str(vault),
            "--heavy-model",
            "new-heavy",
            "--queries",
            str(link),
        ],
    )
    assert result.exit_code == 2
    assert "must not be a symlink" in result.output


def test_compare_requires_cloud_ack(tmp_path):
    vault = _make_vault(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "compare",
            "--vault",
            str(vault),
            "--provider",
            "groq",
            "--provider-url",
            "https://api.groq.com/openai/v1",
            "--heavy-model",
            "llama-3.1-70b-versatile",
        ],
    )
    assert result.exit_code == 1
    assert "--allow-cloud-upload" in result.output


def test_compare_requires_cloud_ack_for_unknown_provider(tmp_path):
    vault = _make_vault(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "compare",
            "--vault",
            str(vault),
            "--provider",
            "myproxy",
            "--provider-url",
            "https://api.example.com/v1",
            "--heavy-model",
            "proxy-model",
        ],
    )
    assert result.exit_code == 1
    assert "--allow-cloud-upload" in result.output


def test_compare_rejects_negative_sample_n(tmp_path):
    vault = _make_vault(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "compare",
            "--vault",
            str(vault),
            "--heavy-model",
            "new-heavy",
            "--sample-n",
            "-1",
        ],
    )
    assert result.exit_code == 2
    assert "must be at least 1" in result.output


def test_compare_runs_and_prints_verdict(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)

    class DummyReport:
        run_id = "rid"
        verdict = type("V", (), {"value": "manual_review"})()
        reasons = ["test reason"]

    monkeypatch.setattr("synto.compare.runner.run_compare", lambda **kwargs: DummyReport())
    monkeypatch.setattr("synto.compare.report.resolve", lambda report: None)
    monkeypatch.setattr("synto.compare.report.render_markdown", lambda report: "md")
    monkeypatch.setattr("synto.compare.report.render_json", lambda report: "{}")
    monkeypatch.setattr(
        "synto.compare.report.render_summary_json",
        lambda report: '{"verdict":"manual_review"}',
    )
    (vault / ".synto" / "compare" / "rid" / "results").mkdir(parents=True)

    result = CliRunner().invoke(
        cli,
        ["compare", "--vault", str(vault), "--heavy-model", "new-heavy", "--format", "json"],
    )
    assert result.exit_code == 0
    assert "Verdict:" in result.output


def test_compare_switch_output_includes_provider_config(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)

    class DummyReport:
        run_id = "rid"
        verdict = AdvisorVerdict.SWITCH
        reasons = ["test reason"]
        # The CLI prints the report's precomputed switch snippet verbatim (named-provider format).
        switch_config_toml = (
            '[providers.default]\nname = "groq"\nurl = "https://api.groq.com/openai/v1"\n'
            'timeout = 600\n\n[models.fast]\nprovider = "default"\n'
            'model = "new-heavy"\nctx = 8192\n'
        )

    monkeypatch.setattr("synto.compare.runner.run_compare", lambda **kwargs: DummyReport())
    monkeypatch.setattr("synto.compare.report.resolve", lambda report: None)
    monkeypatch.setattr("synto.compare.report.render_markdown", lambda report: "md")
    monkeypatch.setattr("synto.compare.report.render_json", lambda report: "{}")
    monkeypatch.setattr(
        "synto.compare.report.render_summary_json",
        lambda report: '{"verdict":"switch"}',
    )
    (vault / ".synto" / "compare" / "rid" / "results").mkdir(parents=True)

    result = CliRunner().invoke(
        cli,
        [
            "compare",
            "--vault",
            str(vault),
            "--provider",
            "groq",
            "--provider-url",
            "https://api.groq.com/openai/v1",
            "--heavy-model",
            "new-heavy",
            "--allow-cloud-upload",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    assert "[providers.default]" in result.output
    assert 'name = "groq"' in result.output
    assert 'url = "https://api.groq.com/openai/v1"' in result.output
