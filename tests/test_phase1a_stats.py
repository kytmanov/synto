from __future__ import annotations

import json
from datetime import date, timedelta

from click.testing import CliRunner

from synto.cli import cli
from synto.models import RawNoteRecord, WikiArticleRecord
from synto.pipeline.maintain import create_stubs
from synto.state import StateDB
from synto.stats import compute_stats_from_db, render_json, render_text
from synto.vault import write_note


def _init_stats_vault(tmp_path, provider_name: str = "ollama"):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / ".synto").mkdir()
    (tmp_path / "synto.toml").write_text(
        (
            "[models]\n"
            'fast = "gemma4:e4b"\n'
            'heavy = "qwen2.5:14b"\n\n'
            "[provider]\n"
            f'name = "{provider_name}"\n'
            'url = "http://localhost:11434"\n'
        ),
        encoding="utf-8",
    )


def test_stats_empty_vault_returns_zeroes(tmp_path):
    _init_stats_vault(tmp_path)
    db = StateDB(tmp_path / ".synto" / "state.db")

    from synto.config import Config

    report = compute_stats_from_db(Config(vault=tmp_path), db)

    assert report.vault.raw_notes == 0
    assert report.vault.drafts == 0
    assert report.vault.published_articles == 0
    assert report.vault.synthesis_articles == 0
    assert report.vault.low_confidence_articles == 0
    assert report.vault.single_source_articles == 0
    assert report.vault.manual_edit_conflicts_avoided is None
    assert report.metrics.rollup_calls == 0
    assert report.metrics.event_count == 0
    assert report.metrics.estimated_cost_usd == 0.0


def test_stats_reports_known_model_cost_and_since_filter(tmp_path):
    _init_stats_vault(tmp_path, provider_name="anthropic")
    db = StateDB(tmp_path / ".synto" / "state.db")
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="ha", status="failed"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="hb", status="ingested"))
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/.drafts/draft.md",
            title="Draft",
            sources=["raw/b.md"],
            content_hash="drafthash",
            status="draft",
        )
    )
    db.insert_synthesis_atomic(
        WikiArticleRecord(
            path="wiki/synthesis/answer.md",
            title="Answer",
            sources=[],
            content_hash="synthhash",
            status="published",
            kind="synthesis",
            question_hash="qhash",
        )
    )

    today = date.today().isoformat()
    old_day = (date.today() - timedelta(days=30)).isoformat()
    db._conn.execute(
        """
        INSERT INTO metric_daily_rollups
        (day, vault_id, event_type, tier, calls, prompt_tokens, completion_tokens,
         latency_ms_total, successes, failures)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (today, "vault", "llm_call", "fast", 4, 1000, 300, 1500, 4, 0),
    )
    db._conn.execute(
        """
        INSERT INTO metric_daily_rollups
        (day, vault_id, event_type, tier, calls, prompt_tokens, completion_tokens,
         latency_ms_total, successes, failures)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (old_day, "vault", "llm_call", "fast", 7, 7000, 2100, 7000, 5, 2),
    )
    db._conn.execute(
        """
        INSERT INTO metric_events
        (ts, vault_id, event_type, model, tier, prompt_tokens, completion_tokens,
         latency_ms, success, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"{today}T12:00:00",
            "vault",
            "llm_call",
            "claude-sonnet-4-6",
            "fast",
            1000,
            500,
            250,
            1,
            "{}",
        ),
    )
    db._conn.execute(
        """
        INSERT INTO metric_events
        (ts, vault_id, event_type, model, tier, prompt_tokens, completion_tokens,
         latency_ms, success, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"{old_day}T12:00:00",
            "vault",
            "llm_call",
            "claude-sonnet-4-6",
            "fast",
            4000,
            1000,
            500,
            0,
            "{}",
        ),
    )
    db._conn.commit()

    from synto.config import Config

    report = compute_stats_from_db(Config(vault=tmp_path), db, since="7d")

    assert report.vault.raw_notes == 2
    assert report.vault.drafts == 1
    assert report.vault.published_articles == 1
    assert report.vault.synthesis_articles == 1
    assert report.vault.failed_notes == 1
    assert report.vault.low_confidence_articles == 0
    assert report.vault.single_source_articles == 0
    assert report.vault.manual_edit_conflicts_avoided is None
    assert report.metrics.rollup_calls == 4
    assert report.metrics.event_count == 1
    assert report.metrics.event_successes == 1
    assert report.metrics.event_failures == 0
    assert report.metrics.estimated_cost_usd == 0.0105


def test_stats_unknown_model_yields_unknown_cost(tmp_path):
    _init_stats_vault(tmp_path, provider_name="openai")
    db = StateDB(tmp_path / ".synto" / "state.db")
    today = date.today().isoformat()
    db._conn.execute(
        """
        INSERT INTO metric_events
        (ts, vault_id, event_type, model, tier, prompt_tokens, completion_tokens,
         latency_ms, success, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (f"{today}T12:00:00", "vault", "llm_call", "unknown-model", "fast", 10, 5, 10, 1, "{}"),
    )
    db._conn.commit()

    from synto.config import Config

    report = compute_stats_from_db(Config(vault=tmp_path), db)
    assert report.metrics.estimated_cost_usd is None


def test_stats_json_output_is_parseable_and_key_safe(tmp_path):
    _init_stats_vault(tmp_path, provider_name="anthropic")
    result = CliRunner().invoke(cli, ["report", "--vault", str(tmp_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["vault"]["provider"] == "anthropic"
    assert payload["vault"]["manual_edit_conflicts_avoided"] is None
    assert payload["metrics"]["estimated_cost_usd"] == 0.0
    assert "sk-" not in result.output
    assert "api_key" not in result.output.lower()


def test_stats_reports_low_confidence_and_single_source_articles(tmp_path):
    _init_stats_vault(tmp_path)
    db = StateDB(tmp_path / ".synto" / "state.db")
    write_note(tmp_path / "wiki" / "Topic.md", {"title": "Topic", "confidence": 0.3}, "Body")
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="ha", status="compiled"))
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )

    from synto.config import Config

    report = compute_stats_from_db(Config(vault=tmp_path), db)

    assert report.vault.low_confidence_articles == 1
    assert report.vault.single_source_articles == 1
    assert report.vault.manual_edit_conflicts_avoided is None


def test_render_json_sorts_keys_and_round_trips(tmp_path):
    _init_stats_vault(tmp_path)
    db = StateDB(tmp_path / ".synto" / "state.db")

    from synto.config import Config

    rendered = render_json(compute_stats_from_db(Config(vault=tmp_path), db))
    payload = json.loads(rendered)
    assert sorted(payload.keys()) == ["metrics", "vault"]


def test_render_text_uses_synto_report_heading(tmp_path):
    _init_stats_vault(tmp_path)
    db = StateDB(tmp_path / ".synto" / "state.db")

    from synto.config import Config

    rendered = render_text(compute_stats_from_db(Config(vault=tmp_path), db))

    assert rendered.startswith("synto report")


def test_stub_body_uses_synto_compile_guidance(tmp_path):
    _init_stats_vault(tmp_path)
    db = StateDB(tmp_path / ".synto" / "state.db")

    from synto.config import Config
    from synto.models import LintIssue

    config = Config(vault=tmp_path)
    created = create_stubs(
        config,
        db,
        broken_link_issues=[
            LintIssue(
                path="wiki/Ref.md",
                issue_type="broken_link",
                description="[[New Topic]] not found",
                suggestion="Create stub",
            )
        ],
    )

    text = created[0].read_text(encoding="utf-8")
    assert "synto compile" in text
