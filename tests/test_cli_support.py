from __future__ import annotations

from click.testing import CliRunner

from synto.cli import cli


def test_support_command_prints_feedback_links():
    result = CliRunner().invoke(cli, ["support"])

    assert result.exit_code == 0
    assert "local runtime and cost metrics" in result.output
    assert "aggregate rollups only" in result.output
    assert "https://github.com/kytmanov/synto/issues" in result.output
    assert "https://github.com/kytmanov/synto/discussions" in result.output
    assert "https://github.com/kytmanov/synto" in result.output


def test_root_help_mentions_support_command():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "support" in result.output
    assert "Run `synto support`" in result.output
    assert "bug" in result.output
    assert "reports, suggestions, and feedback links" in result.output
