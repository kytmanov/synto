"""Offline vault and runtime metrics statistics."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

from .config import Config
from .pricing import estimate_cost_usd, is_cloud_provider
from .readers import VaultReader
from .state import StateDB


@dataclass(frozen=True)
class VaultStats:
    raw_notes: int
    drafts: int
    published_articles: int
    synthesis_articles: int
    concepts: int
    aliases: int
    knowledge_items: int
    failed_notes: int
    failed_concepts: int
    source_segments: int
    schema_version: int
    provider: str
    provider_is_cloud: bool
    low_confidence_articles: int
    single_source_articles: int
    manual_edit_conflicts_avoided: int | None


@dataclass(frozen=True)
class MetricsStats:
    since: str | None
    rollup_calls: int
    rollup_prompt_tokens: int
    rollup_completion_tokens: int
    rollup_latency_ms_total: int
    rollup_successes: int
    rollup_failures: int
    event_count: int
    event_prompt_tokens: int
    event_completion_tokens: int
    event_latency_ms_total: int
    event_successes: int
    event_failures: int
    estimated_cost_usd: float | None


@dataclass(frozen=True)
class StatsReport:
    vault: VaultStats
    metrics: MetricsStats


def compute_stats(config: Config, *, since: str | None = None) -> StatsReport:
    try:
        db = StateDB.open_readonly(config.state_db_path)
    except FileNotFoundError:
        return _empty_stats_report(config, since=since)
    try:
        return compute_stats_from_db(config, db, since=since)
    finally:
        db.close()


def _empty_stats_report(config: Config, *, since: str | None = None) -> StatsReport:
    _, _, since_label = parse_since(since)
    provider_name = config.effective_provider.name
    low_confidence_articles = 0
    try:
        low_confidence_articles = sum(
            1
            for ref in VaultReader(config.vault).list_articles()
            if ref.confidence in ("low", None)
        )
    except Exception:
        low_confidence_articles = 0

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
        provider=provider_name,
        provider_is_cloud=is_cloud_provider(provider_name),
        low_confidence_articles=low_confidence_articles,
        single_source_articles=0,
        manual_edit_conflicts_avoided=None,
    )
    metrics = MetricsStats(
        since=since_label,
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
    return StatsReport(vault=vault, metrics=metrics)


def compute_stats_from_db(config: Config, db: StateDB, *, since: str | None = None) -> StatsReport:
    since_day, since_ts, since_label = parse_since(since)
    base_stats = db.stats(config.vault)
    raw = base_stats.get("raw", {})
    provider_name = config.effective_provider.name
    synthesis_articles = len(db.list_synthesis_articles_brief())
    low_confidence_articles = 0
    try:
        low_confidence_articles = sum(
            1
            for ref in VaultReader(config.vault).list_articles()
            if ref.confidence in ("low", None)
        )
    except Exception:
        low_confidence_articles = 0
    single_source_articles = sum(
        1 for record in db.list_articles() if not record.is_draft and len(record.sources) == 1
    )

    vault = VaultStats(
        raw_notes=sum(int(v) for v in raw.values()),
        drafts=int(base_stats["drafts"]),
        published_articles=int(base_stats["published"]),
        synthesis_articles=synthesis_articles,
        concepts=db.count_concepts(),
        aliases=db.count_aliases(),
        knowledge_items=db.count_knowledge_items(),
        failed_notes=db.count_failed_notes(),
        failed_concepts=db.count_failed_concepts(),
        source_segments=db.count_source_segments(),
        schema_version=db.schema_version(),
        provider=provider_name,
        provider_is_cloud=is_cloud_provider(provider_name),
        low_confidence_articles=low_confidence_articles,
        single_source_articles=single_source_articles,
        manual_edit_conflicts_avoided=None,
    )

    rollup_totals = db.metric_rollup_totals(since_day=since_day)
    event_totals = db.metric_event_totals(since_ts=since_ts)
    estimated_cost = _estimate_total_cost(
        provider_name=provider_name,
        rollup_calls=rollup_totals["calls"],
        event_count=event_totals["events"],
        model_totals=db.metric_event_model_totals(since_ts=since_ts),
    )

    metrics = MetricsStats(
        since=since_label,
        rollup_calls=rollup_totals["calls"],
        rollup_prompt_tokens=rollup_totals["prompt_tokens"],
        rollup_completion_tokens=rollup_totals["completion_tokens"],
        rollup_latency_ms_total=rollup_totals["latency_ms_total"],
        rollup_successes=rollup_totals["successes"],
        rollup_failures=rollup_totals["failures"],
        event_count=event_totals["events"],
        event_prompt_tokens=event_totals["prompt_tokens"],
        event_completion_tokens=event_totals["completion_tokens"],
        event_latency_ms_total=event_totals["latency_ms_total"],
        event_successes=event_totals["successes"],
        event_failures=event_totals["failures"],
        estimated_cost_usd=estimated_cost,
    )

    return StatsReport(vault=vault, metrics=metrics)


def parse_since(value: str | None) -> tuple[date | None, str | None, str | None]:
    if value is None:
        return None, None, None

    text = value.strip()
    if not text:
        return None, None, None

    days_match = re.fullmatch(r"(\d+)d", text)
    if days_match:
        days = int(days_match.group(1))
        if days < 0:
            raise ValueError("--since days must be non-negative")
        since_day = date.today() - timedelta(days=days)
        since_dt = datetime.combine(since_day, datetime.min.time())
        return since_day, since_dt.isoformat(), text

    try:
        since_day = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("--since must be ISO date (YYYY-MM-DD) or Nd like 7d") from exc
    since_dt = datetime.combine(since_day, datetime.min.time())
    return since_day, since_dt.isoformat(), text


def render_json(report: StatsReport) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True)


def render_text(report: StatsReport) -> str:
    lines = [
        "synto report",
        "",
        "Vault",
        f"  Raw notes: {report.vault.raw_notes}",
        f"  Drafts: {report.vault.drafts}",
        f"  Published articles: {report.vault.published_articles}",
        f"  Synthesis articles: {report.vault.synthesis_articles}",
        f"  Concepts: {report.vault.concepts}",
        f"  Aliases: {report.vault.aliases}",
        f"  Knowledge items: {report.vault.knowledge_items}",
        f"  Failed notes: {report.vault.failed_notes}",
        f"  Failed concepts: {report.vault.failed_concepts}",
        f"  Source segments: {report.vault.source_segments}",
        f"  Schema version: {report.vault.schema_version}",
        f"  Provider: {report.vault.provider}",
        f"  Cloud provider: {'yes' if report.vault.provider_is_cloud else 'no'}",
        f"  Low-confidence articles: {report.vault.low_confidence_articles}",
        f"  Single-source articles: {report.vault.single_source_articles}",
        "  Manual-edit conflicts avoided: "
        + (
            "n/a"
            if report.vault.manual_edit_conflicts_avoided is None
            else str(report.vault.manual_edit_conflicts_avoided)
        ),
        "",
        "Runtime and Cost Metrics",
    ]
    if report.metrics.since is not None:
        lines.append(f"  Since: {report.metrics.since}")
    lines.extend(
        [
            f"  Rollup calls: {report.metrics.rollup_calls}",
            f"  Rollup input tokens: {report.metrics.rollup_prompt_tokens}",
            f"  Rollup output tokens: {report.metrics.rollup_completion_tokens}",
            f"  Rollup latency ms: {report.metrics.rollup_latency_ms_total}",
            f"  Rollup successes: {report.metrics.rollup_successes}",
            f"  Rollup failures: {report.metrics.rollup_failures}",
            f"  Detailed events: {report.metrics.event_count}",
            f"  Event input tokens: {report.metrics.event_prompt_tokens}",
            f"  Event output tokens: {report.metrics.event_completion_tokens}",
            f"  Event latency ms: {report.metrics.event_latency_ms_total}",
            f"  Event successes: {report.metrics.event_successes}",
            f"  Event failures: {report.metrics.event_failures}",
            f"  Estimated cost (USD): {_format_cost(report.metrics.estimated_cost_usd)}",
        ]
    )
    return "\n".join(lines)


def _estimate_total_cost(
    *,
    provider_name: str,
    rollup_calls: int,
    event_count: int,
    model_totals: list[tuple[str, int, int]],
) -> float | None:
    if event_count == 0 and rollup_calls == 0:
        return 0.0

    if not model_totals:
        return 0.0 if not is_cloud_provider(provider_name) else None

    total = 0.0
    for model, input_tokens, output_tokens in model_totals:
        cost = estimate_cost_usd(model, input_tokens, output_tokens)
        if cost is None:
            return None
        total += cost
    return round(total, 6)


def _format_cost(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.6f}"
