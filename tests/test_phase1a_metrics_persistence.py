from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from synto.config import Config, MetricsConfig
from synto.metrics import (
    LLMCallEvent,
    PersistentMetricsSink,
    current_persistent_sink,
    metrics_sink,
    persistent_metrics_sink,
)
from synto.state import StateDB


def _event(*, model_role: str = "fast", error: str | None = None) -> LLMCallEvent:
    return LLMCallEvent(
        stage="ingest",
        model="gemma4:e4b",
        tier=3,
        retries=1,
        latency_ms=123,
        prompt_tokens=10,
        completion_tokens=5,
        num_ctx=8192,
        error=error,
        model_role=model_role,
        extra={"source_id": "raw/note.md"},
    )


def test_aggregate_mode_writes_rollup_only(tmp_path):
    config = Config(vault=tmp_path, metrics=MetricsConfig(persist=True, detailed=False))
    db = StateDB(config.state_db_path)
    sink = PersistentMetricsSink(db, config.metrics, config.vault)

    with persistent_metrics_sink(sink):
        from synto.metrics import emit

        emit(_event(model_role="fast"))

    rollups = db.list_metric_rollups()
    events = db.list_metric_events()

    assert len(rollups) == 1
    assert rollups[0]["event_type"] == "llm_call"
    assert rollups[0]["tier"] == "fast"
    assert rollups[0]["calls"] == 1
    assert rollups[0]["prompt_tokens"] == 10
    assert rollups[0]["completion_tokens"] == 5
    assert rollups[0]["successes"] == 1
    assert rollups[0]["failures"] == 0
    assert events == []


def test_detailed_mode_writes_event_row_with_metadata(tmp_path):
    config = Config(vault=tmp_path, metrics=MetricsConfig(persist=True, detailed=True))
    db = StateDB(config.state_db_path)
    sink = PersistentMetricsSink(db, config.metrics, config.vault)

    with persistent_metrics_sink(sink):
        from synto.metrics import emit

        emit(_event(model_role="heavy", error="bad json"))

    events = db.list_metric_events()

    assert len(events) == 1
    row = events[0]
    assert row["event_type"] == "llm_call"
    assert row["tier"] == "heavy"
    assert row["success"] == 0
    assert row["source_id_hash"]
    payload = json.loads(row["metadata_json"])
    assert payload["parse_tier"] == 3
    assert payload["retries"] == 1
    assert payload["stage"] == "ingest"
    assert payload["num_ctx"] == 8192
    assert payload["error"] == "bad json"


def test_persistent_sink_bypassed_when_compare_sink_active(tmp_path):
    config = Config(vault=tmp_path, metrics=MetricsConfig(persist=True, detailed=True))
    db = StateDB(config.state_db_path)
    sink = PersistentMetricsSink(db, config.metrics, config.vault)

    with persistent_metrics_sink(sink), metrics_sink() as events:
        from synto.metrics import emit

        emit(_event())

    assert len(events) == 1
    assert db.list_metric_events() == []
    assert db.list_metric_rollups() == []


def test_persistent_sink_can_be_disabled(tmp_path):
    config = Config(vault=tmp_path, metrics=MetricsConfig(persist=False, detailed=True))
    db = StateDB(config.state_db_path)
    sink = PersistentMetricsSink(db, config.metrics, config.vault)

    with persistent_metrics_sink(sink):
        from synto.metrics import emit

        emit(_event())

    assert db.list_metric_events() == []
    assert db.list_metric_rollups() == []


def test_persistent_sink_context_is_reset(tmp_path):
    config = Config(vault=tmp_path, metrics=MetricsConfig(persist=True, detailed=False))
    db = StateDB(config.state_db_path)
    sink = PersistentMetricsSink(db, config.metrics, config.vault)

    assert current_persistent_sink() is None
    with persistent_metrics_sink(sink):
        assert current_persistent_sink() is sink
    assert current_persistent_sink() is None


def test_retention_deletes_old_detailed_events(tmp_path):
    config = Config(vault=tmp_path, metrics=MetricsConfig(retention_days=1, detailed=True))
    db = StateDB(config.state_db_path)
    old_ts = (datetime.now(UTC) - timedelta(days=5)).isoformat()
    db.insert_metric_event(
        ts=old_ts,
        vault_id="vault",
        event_type="llm_call",
        model="gemma4:e4b",
        tier="fast",
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1,
        success=True,
    )

    deleted = PersistentMetricsSink(db, config.metrics, config.vault).enforce_retention()

    assert deleted == 1
    assert db.list_metric_events() == []
