"""Tests for `[maintain] ack` (discussion #94) — display-only advisory acknowledgement.

Covers the pure partitioning helper (synto.pipeline.lint.partition_acked), the CLI
rendering in `synto maintain` (acked issues collapse, health score is unaffected), and
the unknown-check-name warning on Config load.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from click.testing import CliRunner

from synto.cli import cli
from synto.config import Config
from synto.models import LintIssue
from synto.pipeline.lint import partition_acked


def _issue(issue_type: str, path: str = "") -> LintIssue:
    return LintIssue(
        path=path,
        issue_type=issue_type,
        description="d",
        suggestion="s",
    )


# ── partition_acked ──────────────────────────────────────────────────────────


def test_bare_check_matches_all_paths():
    issues = [_issue("graph_noise", "Welcome.md"), _issue("graph_noise", "Other.md")]

    active, acked = partition_acked(issues, ["graph_noise"])

    assert active == []
    assert acked == issues


def test_check_path_matches_only_that_path():
    matching = _issue("stale_lock", "wiki/Old Draft.md")
    other_path = _issue("stale_lock", "wiki/Other.md")

    active, acked = partition_acked([matching, other_path], ["stale_lock:wiki/Old Draft.md"])

    assert acked == [matching]
    assert active == [other_path]


def test_non_matching_check_stays_active():
    issues = [_issue("orphan", "wiki/A.md")]

    active, acked = partition_acked(issues, ["graph_noise"])

    assert active == issues
    assert acked == []


def test_backslash_ack_path_matches_posix_issue_path():
    # synto.toml is hand-edited; a Windows user may write a backslash path even though
    # LintIssue.path is always posix (see _vault_rel_path in pipeline/lint.py).
    issue = _issue("stale_lock", "wiki/Old Draft.md")

    active, acked = partition_acked([issue], ["stale_lock:wiki\\Old Draft.md"])

    assert acked == [issue]
    assert active == []


def test_split_on_first_colon_only():
    # A path could itself contain a colon (rare, but the split must not truncate it).
    issue = _issue("stale_lock", "wiki/10:30.md")

    active, acked = partition_acked([issue], ["stale_lock:wiki/10:30.md"])

    assert acked == [issue]
    assert active == []


# ── CLI rendering ─────────────────────────────────────────────────────────────


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    for sub in ("raw", "wiki", "wiki/.drafts", ".synto"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    # Welcome.md at vault root deterministically triggers a graph_noise advisory —
    # see _add_graph_quality_issues in pipeline/lint.py.
    (tmp_path / "Welcome.md").write_text("Welcome. [[create a link]]")
    return tmp_path


def test_maintain_without_ack_shows_advisory_issue(vault: Path):
    result = CliRunner().invoke(cli, ["maintain", "--dry-run", "--vault", str(vault)])

    assert result.exit_code == 0, result.output
    assert "graph_noise" in result.output
    assert "Obsidian starter Welcome.md appears in graph view" in result.output
    assert "acked issue(s) hidden" not in result.output
    assert "(1 advisory issue(s))" in result.output


def test_maintain_with_ack_hides_issue_and_shows_collapsed_count(vault: Path):
    (vault / "synto.toml").write_text('[maintain]\nack = ["graph_noise"]\n')

    result = CliRunner().invoke(cli, ["maintain", "--dry-run", "--vault", str(vault)])

    assert result.exit_code == 0, result.output
    assert "Obsidian starter Welcome.md appears in graph view" not in result.output
    assert "1 acked issue(s) hidden ([maintain].ack in synto.toml)" in result.output
    assert "(1 advisory, 1 acked)" in result.output


def test_ack_does_not_change_health_score(vault: Path):
    without = CliRunner().invoke(cli, ["maintain", "--dry-run", "--vault", str(vault)])
    assert without.exit_code == 0, without.output

    (vault / "synto.toml").write_text('[maintain]\nack = ["graph_noise"]\n')
    with_ack = CliRunner().invoke(cli, ["maintain", "--dry-run", "--vault", str(vault)])
    assert with_ack.exit_code == 0, with_ack.output

    assert "Structural health: 100.0/100" in without.output
    assert "Structural health: 100.0/100" in with_ack.output


# ── Unknown check name ────────────────────────────────────────────────────────


def test_unknown_check_name_warns_not_raises(caplog):
    with caplog.at_level(logging.WARNING):
        config = Config(vault="/tmp/v", maintain={"ack": ["not_a_real_check"]})

    assert config.maintain.ack == ["not_a_real_check"]
    assert any("unknown check name" in r.message for r in caplog.records)


def test_unknown_check_name_in_toml_still_loads_vault(vault: Path, caplog):
    (vault / "synto.toml").write_text('[maintain]\nack = ["not_a_real_check"]\n')

    with caplog.at_level(logging.WARNING):
        result = CliRunner().invoke(cli, ["maintain", "--dry-run", "--vault", str(vault)])

    assert result.exit_code == 0, result.output
    assert "Structural health" in result.output
