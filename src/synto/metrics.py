"""
Per-call LLM metrics sink.

`request_structured` emits one `LLMCallEvent` per call into a ContextVar-scoped
sink when one is active. When no sink is set (normal pipeline runs), emission
is a cheap no-op.

The `synto compare` runner sets a sink around each contestant-seed pipeline run
to collect reliability/efficiency signals without touching pipeline code.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import MetricsConfig

log = logging.getLogger(__name__)


@dataclass
class LLMCallEvent:
    stage: str
    model: str
    tier: int  # 1=native-json parse, 2=extracted, 3=retry-success, -1=final-failure
    retries: int
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    num_ctx: int
    error: str | None = None
    contestant: str | None = None
    seed: int | None = None
    model_role: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class AppEvent:
    name: str
    payload: dict[str, Any]


_sink: ContextVar[list[LLMCallEvent] | None] = ContextVar("synto_llm_metrics", default=None)
_app_sink: ContextVar[list[AppEvent] | None] = ContextVar("synto_app_metrics", default=None)
_persistent_sink: ContextVar[PersistentMetricsSink | None] = ContextVar(
    "synto_persistent_metrics", default=None
)


class PersistentMetricsSink:
    def __init__(self, db, config: MetricsConfig, vault_root: Path) -> None:
        self._db = db
        self._config = config
        self._vault_id = hashlib.sha256(str(vault_root.resolve()).encode("utf-8")).hexdigest()[:16]
        self._maintenance_done = False

    def record_llm_call(self, event: LLMCallEvent) -> None:
        if current_sink() is not None or not self._config.persist:
            return
        self._ensure_maintenance()
        success = event.error is None
        if self._config.detailed:
            self._db.insert_metric_event(
                ts=datetime.now(UTC).isoformat(),
                vault_id=self._vault_id,
                event_type="llm_call",
                model=event.model,
                tier=event.model_role or "",
                prompt_tokens=event.prompt_tokens,
                completion_tokens=event.completion_tokens,
                latency_ms=event.latency_ms,
                success=success,
                source_id=event.extra.get("source_id") if isinstance(event.extra, dict) else None,
                hash_source_id=self._config.hash_source_ids,
                metadata_json=self._metadata_json(event),
            )
            return
        day = datetime.now(UTC).date().isoformat()
        self._db.upsert_metric_rollup(
            day=day,
            vault_id=self._vault_id,
            event_type="llm_call",
            tier=event.model_role or "",
            calls=1,
            prompt_tokens=int(event.prompt_tokens or 0),
            completion_tokens=int(event.completion_tokens or 0),
            latency_ms_total=int(event.latency_ms or 0),
            successes=1 if success else 0,
            failures=0 if success else 1,
        )

    def enforce_retention(self) -> int:
        cutoff_ts = (datetime.now(UTC) - timedelta(days=self._config.retention_days)).isoformat()
        cutoff_day = (
            (datetime.now(UTC) - timedelta(days=self._config.retention_days * 4)).date().isoformat()
        )
        return self._db.delete_metrics_before(cutoff_ts=cutoff_ts, cutoff_day=cutoff_day)

    def enforce_size_cap(self) -> int:
        max_size_bytes = self._config.max_size_mb * 1024 * 1024
        if self._db.database_size_bytes() < max_size_bytes:
            return 0
        return self._db.trim_oldest_metric_events()

    def _ensure_maintenance(self) -> None:
        if self._maintenance_done:
            return
        self.enforce_retention()
        self.enforce_size_cap()
        self._maintenance_done = True

    def _metadata_json(self, event: LLMCallEvent) -> str:
        payload = {
            "parse_tier": event.tier,
            "retries": event.retries,
            "stage": event.stage,
            "num_ctx": event.num_ctx,
            "error": event.error,
            "contestant": event.contestant,
            "seed": event.seed,
            "extra": event.extra or {},
        }
        return json.dumps(payload, sort_keys=True)


def emit(event: LLMCallEvent) -> None:
    sink = _sink.get()
    if sink is not None:
        sink.append(event)
        return
    persistent_sink = _persistent_sink.get()
    if persistent_sink is not None:
        try:
            persistent_sink.record_llm_call(event)
        except Exception as exc:  # noqa: BLE001
            log.warning("metrics persistence failed: %s", exc)


def current_sink() -> list[LLMCallEvent] | None:
    return _sink.get()


def emit_app_event(event: AppEvent) -> None:
    sink = _app_sink.get()
    if sink is not None:
        sink.append(event)


def current_app_sink() -> list[AppEvent] | None:
    return _app_sink.get()


def current_persistent_sink() -> PersistentMetricsSink | None:
    return _persistent_sink.get()


@contextmanager
def metrics_sink() -> Iterator[list[LLMCallEvent]]:
    events: list[LLMCallEvent] = []
    token = _sink.set(events)
    try:
        yield events
    finally:
        _sink.reset(token)


@contextmanager
def app_event_sink() -> Iterator[list[AppEvent]]:
    events: list[AppEvent] = []
    token = _app_sink.set(events)
    try:
        yield events
    finally:
        _app_sink.reset(token)


@contextmanager
def persistent_metrics_sink(sink: PersistentMetricsSink | None) -> Iterator[None]:
    token = _persistent_sink.set(sink)
    try:
        yield
    finally:
        _persistent_sink.reset(token)
