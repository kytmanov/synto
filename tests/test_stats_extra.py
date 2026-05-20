"""Additional tests for stats.py uncovered paths."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from synto.stats import (
    _estimate_total_cost,
    compute_stats,
    parse_since,
    render_text,
)

# ── parse_since ──────────────────────────────────────────────────────────────


def test_parse_since_days_format():
    since_day, since_ts, label = parse_since("7d")
    assert since_day == date.today() - timedelta(days=7)
    assert label == "7d"


def test_parse_since_negative_days_raises():
    """Negative days don't match the Nd pattern, so it falls through to ISO date parsing."""
    with pytest.raises(ValueError, match="ISO date"):
        parse_since("-1d")


def test_parse_since_iso_date():
    since_day, since_ts, label = parse_since("2026-01-15")
    assert since_day == date(2026, 1, 15)
    assert label == "2026-01-15"


def test_parse_since_invalid_format_raises():
    with pytest.raises(ValueError, match="ISO date"):
        parse_since("yesterday")


def test_parse_since_invalid_days_format_raises():
    with pytest.raises(ValueError, match="ISO date"):
        parse_since("7days")


# ── _estimate_total_cost ─────────────────────────────────────────────────────


def test_estimate_cost_local_provider_no_model_totals():
    """Local provider with no model totals → 0.0."""
    assert (
        _estimate_total_cost(provider_name="ollama", rollup_calls=5, event_count=3, model_totals=[])
        == 0.0
    )


def test_estimate_cost_cloud_provider_no_model_totals():
    """Cloud provider with no model totals → None (unknown)."""
    assert (
        _estimate_total_cost(
            provider_name="anthropic", rollup_calls=5, event_count=3, model_totals=[]
        )
        is None
    )


def test_estimate_cost_unknown_model():
    """Unknown model → None."""
    assert (
        _estimate_total_cost(
            provider_name="anthropic",
            rollup_calls=0,
            event_count=1,
            model_totals=[("unknown-model", 100, 50)],
        )
        is None
    )


def test_estimate_cost_known_model():
    """Known model → computed cost."""
    result = _estimate_total_cost(
        provider_name="anthropic",
        rollup_calls=0,
        event_count=1,
        model_totals=[("claude-sonnet-4-6", 1000, 500)],
    )
    assert result is not None
    assert result > 0


# ── compute_stats ────────────────────────────────────────────────────────────


def test_compute_stats_no_db_file(tmp_path):
    """compute_stats returns empty report when state.db doesn't exist."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / ".synto").mkdir()
    (tmp_path / "synto.toml").write_text(
        '[ollama]\nurl = "http://localhost:11434"\nfast_ctx = 8192\nheavy_ctx = 16384\n'
    )
    from synto.config import Config

    report = compute_stats(Config(vault=tmp_path))
    assert report.vault.raw_notes == 0
    assert report.metrics.rollup_calls == 0


def test_compute_stats_with_since(tmp_path):
    """compute_stats passes since parameter through."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / ".synto").mkdir()
    (tmp_path / "synto.toml").write_text(
        '[ollama]\nurl = "http://localhost:11434"\nfast_ctx = 8192\nheavy_ctx = 16384\n'
    )
    from synto.config import Config
    from synto.state import StateDB

    db = StateDB(tmp_path / ".synto" / "state.db")
    db.close()

    report = compute_stats(Config(vault=tmp_path), since="7d")
    assert report.metrics.since == "7d"


# ── render_text ──────────────────────────────────────────────────────────────


def test_render_text_with_since_label():
    """render_text includes 'Since:' line when since is set."""
    from synto.stats import MetricsStats, StatsReport, VaultStats

    vault = VaultStats(
        raw_notes=10,
        drafts=2,
        published_articles=5,
        synthesis_articles=1,
        concepts=8,
        aliases=3,
        knowledge_items=0,
        failed_notes=0,
        failed_concepts=0,
        source_segments=0,
        schema_version=1,
        provider="ollama",
        provider_is_cloud=False,
        low_confidence_articles=0,
        single_source_articles=0,
        manual_edit_conflicts_avoided=None,
    )
    metrics = MetricsStats(
        since="7d",
        rollup_calls=5,
        rollup_prompt_tokens=1000,
        rollup_completion_tokens=500,
        rollup_latency_ms_total=2000,
        rollup_successes=5,
        rollup_failures=0,
        event_count=3,
        event_prompt_tokens=500,
        event_completion_tokens=200,
        event_latency_ms_total=1000,
        event_successes=3,
        event_failures=0,
        estimated_cost_usd=0.0,
    )
    report = StatsReport(vault=vault, metrics=metrics)
    text = render_text(report)
    assert "Since: 7d" in text


def test_render_text_without_since_label():
    """render_text omits 'Since:' line when since is None."""
    from synto.stats import MetricsStats, StatsReport, VaultStats

    vault = VaultStats(
        raw_notes=0,
        drafts=0,
        published_articles=0,
        synthesis_articles=0,
        concepts=0,
        aliases=0,
        knowledge_items=0,
        failed_notes=0,
        failed_concepts=0,
        source_segments=0,
        schema_version=0,
        provider="ollama",
        provider_is_cloud=False,
        low_confidence_articles=0,
        single_source_articles=0,
        manual_edit_conflicts_avoided=None,
    )
    metrics = MetricsStats(
        since=None,
        rollup_calls=0,
        rollup_prompt_tokens=0,
        rollup_completion_tokens=0,
        rollup_latency_ms_total=0,
        rollup_successes=0,
        rollup_failures=0,
        event_count=0,
        event_prompt_tokens=0,
        event_completion_tokens=0,
        event_latency_ms_total=0,
        event_successes=0,
        event_failures=0,
        estimated_cost_usd=0.0,
    )
    report = StatsReport(vault=vault, metrics=metrics)
    text = render_text(report)
    assert "Since:" not in text
