from __future__ import annotations

from click.testing import CliRunner

from synto.cli import cli
from synto.state import StateDB


def _init_vault(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / ".synto").mkdir()
    (tmp_path / "synto.toml").write_text(
        (
            "[models]\n"
            'fast = "gemma4:e4b"\n'
            'heavy = "gemma4:e4b"\n\n'
            "[provider]\n"
            'name = "ollama"\n'
            'url = "http://localhost:11434"\n'
        ),
        encoding="utf-8",
    )


def test_metrics_clear_with_yes_deletes_all_rows(tmp_path) -> None:
    _init_vault(tmp_path)
    db = StateDB(tmp_path / ".synto" / "state.db")
    db.insert_metric_event(
        ts="2026-05-14T12:00:00",
        vault_id="vault",
        event_type="llm_call",
        model="model",
        tier="fast",
        prompt_tokens=1,
        completion_tokens=2,
        latency_ms=3,
        success=True,
        metadata_json="{}",
    )
    db.upsert_metric_rollup(
        day="2026-05-14",
        vault_id="vault",
        event_type="llm_call",
        tier="fast",
        calls=1,
        prompt_tokens=1,
        completion_tokens=2,
        latency_ms_total=3,
        successes=1,
        failures=0,
    )
    db.close()

    result = CliRunner().invoke(cli, ["report", "clear", "--vault", str(tmp_path), "--yes"])

    assert result.exit_code == 0
    assert "Metrics cleared (2 rows deleted)." in result.output

    reopened = StateDB(tmp_path / ".synto" / "state.db")
    assert reopened.metric_event_totals()["events"] == 0
    assert reopened.metric_rollup_totals()["calls"] == 0
    reopened.close()


def test_metrics_clear_cancel_leaves_data_intact(tmp_path) -> None:
    _init_vault(tmp_path)
    db = StateDB(tmp_path / ".synto" / "state.db")
    db.insert_metric_event(
        ts="2026-05-14T12:00:00",
        vault_id="vault",
        event_type="llm_call",
        model="model",
        tier="fast",
        prompt_tokens=1,
        completion_tokens=2,
        latency_ms=3,
        success=True,
        metadata_json="{}",
    )
    db.close()

    result = CliRunner().invoke(cli, ["report", "clear", "--vault", str(tmp_path)], input="n\n")

    assert result.exit_code != 0
    reopened = StateDB(tmp_path / ".synto" / "state.db")
    assert reopened.metric_event_totals()["events"] == 1
    reopened.close()


def test_metrics_clear_missing_db_does_not_create_one(tmp_path) -> None:
    _init_vault(tmp_path)
    state_db_path = tmp_path / ".synto" / "state.db"
    if state_db_path.exists():
        state_db_path.unlink()

    result = CliRunner().invoke(cli, ["report", "clear", "--vault", str(tmp_path), "--yes"])

    assert result.exit_code == 0
    assert "No metrics data found" in result.output
    assert not state_db_path.exists()
