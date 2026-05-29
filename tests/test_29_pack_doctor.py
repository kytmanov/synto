"""Tests for Feature 29 Stage 5 — MCP demand-vs-coverage backlog.

Covers the audit-path changes (result_count + resolved_label threaded through
the MCP handlers, hashed-by-default / raw under audit_detailed) and the five
StateDB report helpers plus the `synto doctor --backlog` rendering.

All tests are offline — in-memory / tmp_path SQLite, no Ollama required.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from synto.cli import cli
from synto.config import Config, McpConfig
from synto.readers import Article, ConceptRef
from synto.serve import _audit, build_tool_handlers
from synto.state import StateDB

# ── Helpers ──────────────────────────────────────────────────────────────────


def _audit_row(
    db: StateDB,
    *,
    ts: str,
    tool: str,
    vault_id: str | None = "v1",
    success: bool = True,
    result_count: int | None = None,
    resolved_label: str | None = None,
    query: str | None = None,
    concept_name: str | None = None,
) -> None:
    """Insert one synthesized mcp_call row exactly as insert_mcp_audit_event would.

    Uses insert_metric_event directly so tests can set vault_id=None (the
    audit path never does, but the report helper must handle it defensively).
    """
    args: dict[str, object] = {}
    if query is not None:
        args["query"] = query
    if concept_name is not None:
        args["concept_name"] = concept_name
    meta: dict[str, object] = {"tool": tool, "args": args}
    if result_count is not None:
        meta["result_count"] = result_count
    if resolved_label is not None:
        meta["resolved_label"] = resolved_label
    db.insert_metric_event(
        ts=ts,
        vault_id=vault_id,
        event_type="mcp_call",
        model=None,
        tier=None,
        prompt_tokens=None,
        completion_tokens=None,
        latency_ms=1,
        success=success,
        metadata_json=json.dumps(meta, sort_keys=True),
    )


def _add_occurrence(db: StateDB, concept_name: str, seg_id: str) -> None:
    db._conn.execute(
        """INSERT INTO concept_occurrences
           (concept_name, source_segment_id, ordinal, confidence)
           VALUES (?, ?, 0, 1.0)""",
        (concept_name, seg_id),
    )
    db._conn.commit()


def _last_mcp_meta(db: StateDB) -> dict:
    """Return the parsed metadata_json of the most recent mcp_call row."""
    rows = [r for r in db.list_metric_events() if r["event_type"] == "mcp_call"]
    return json.loads(rows[-1]["metadata_json"])


def _mock_reader_with_concept(name: str) -> MagicMock:
    """A reader whose find_concept resolves `name` to a visible published article."""
    reader = MagicMock()
    reader.find_concept.return_value = ConceptRef(
        name=name, canonical_article_id="art-1", aliases=()
    )
    reader.read_article.return_value = Article(
        id="art-1",
        name=name,
        path="wiki/concept.md",
        body="body",
        frontmatter={"visibility": "public"},
    )
    return reader


def _make_find_concept_handler(db: StateDB, *, detailed: bool, reader: MagicMock | None = None):
    mcp = McpConfig(audit=True, audit_detailed=detailed)
    config = Config(vault=Path("/tmp/vault"), mcp=mcp)
    if reader is None:
        reader = MagicMock()
        reader.find_concept.return_value = None
    handlers = build_tool_handlers(reader, config, db, vault_key="vk")
    return handlers["find_concept"]


# ── Audit-path: result_count ──────────────────────────────────────────────────


def test_audit_records_result_count(tmp_path: Path) -> None:
    """find_concept records result_count=0 on a miss and 1 on a hit.

    Why it matters: zero-result detection is the whole point of the backlog —
    a miss must be distinguishable from a hit, and from a raised failure.
    """
    db = StateDB(tmp_path / "state.db")

    miss = _make_find_concept_handler(db, detailed=True)
    assert miss("nope") is None
    assert _last_mcp_meta(db)["result_count"] == 0

    hit = _make_find_concept_handler(db, detailed=True, reader=_mock_reader_with_concept("Yoneda"))
    result = hit("Yoneda")
    assert result is not None
    assert _last_mcp_meta(db)["result_count"] == 1


def test_audit_records_resolved_label_hashed_by_default(tmp_path: Path) -> None:
    """With audit_detailed=False, resolved_label is stored as an 8-char hash."""
    db = StateDB(tmp_path / "state.db")
    handler = _make_find_concept_handler(
        db, detailed=False, reader=_mock_reader_with_concept("Yoneda lemma")
    )
    handler("yoneda")
    meta = _last_mcp_meta(db)
    label = meta["resolved_label"]
    assert label != "Yoneda lemma"
    assert len(label) == 8 and all(c in "0123456789abcdef" for c in label)
    # the query arg is hashed too (default privacy posture)
    assert meta["args"]["query"] != "yoneda"


def test_audit_detailed_stores_raw_args_and_label(tmp_path: Path) -> None:
    """With audit_detailed=True, the literal query and canonical name are stored."""
    db = StateDB(tmp_path / "state.db")
    handler = _make_find_concept_handler(
        db, detailed=True, reader=_mock_reader_with_concept("Yoneda lemma")
    )
    handler("yoneda")
    meta = _last_mcp_meta(db)
    assert meta["resolved_label"] == "Yoneda lemma"
    assert meta["args"]["query"] == "yoneda"


def test_audit_signature_backwards_compatible(tmp_path: Path) -> None:
    """_audit called without result_count/resolved_label still writes a valid row.

    Why it matters: existing serve/F42 tests call _audit with the old kwargs only;
    the new params must be optional so those tests keep passing untouched.
    """
    db = StateDB(tmp_path / "state.db")
    _audit(
        db,
        vault_id="vk",
        tool="list_sources",
        arguments={},
        success=True,
        latency_ms=5,
        mcp_config=McpConfig(audit=True),
    )
    rows = [r for r in db.list_metric_events() if r["event_type"] == "mcp_call"]
    assert len(rows) == 1
    meta = json.loads(rows[0]["metadata_json"])
    assert meta["tool"] == "list_sources"
    assert "result_count" not in meta
    assert "resolved_label" not in meta


# ── Report helpers: zero-result ───────────────────────────────────────────────


def test_backlog_zero_result_section(tmp_path: Path) -> None:
    """Rows with result_count=0 and success=1 appear; success=0 rows do not.

    Why it matters: success=0 means the tool raised — that is an error, not a
    coverage gap, and must not pollute the demand signal.
    """
    db = StateDB(tmp_path / "state.db")
    now = datetime.now(UTC).isoformat()
    _audit_row(
        db,
        ts=now,
        tool="search_source_segments",
        success=True,
        result_count=0,
        query="monad transformer",
    )
    _audit_row(
        db,
        ts=now,
        tool="search_source_segments",
        success=True,
        result_count=0,
        query="monad transformer",
    )
    # A raised failure on the same query — must be excluded.
    _audit_row(db, ts=now, tool="find_concept", success=False, result_count=0, query="boom")
    # A successful non-empty call — must be excluded.
    _audit_row(db, ts=now, tool="find_concept", success=True, result_count=3, query="known")

    rows = db.zero_result_query_counts("", top_n=20)
    assert rows == [("search_source_segments", "monad transformer", 2)]


def test_backlog_lookback_inclusive(tmp_path: Path) -> None:
    """--since=30d returns at least as many rows as --since=7d on the same set."""
    db = StateDB(tmp_path / "state.db")
    recent = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    older = (datetime.now(UTC) - timedelta(days=20)).isoformat()
    _audit_row(db, ts=recent, tool="find_concept", success=True, result_count=0, query="recent")
    _audit_row(db, ts=older, tool="find_concept", success=True, result_count=0, query="older")

    since_7d = (datetime.now(UTC) - timedelta(days=7)).isoformat()
    since_30d = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    n7 = len(db.zero_result_query_counts(since_7d))
    n30 = len(db.zero_result_query_counts(since_30d))
    assert n7 == 1
    assert n30 == 2
    assert n30 >= n7


def test_backlog_tiebreaker_stable(tmp_path: Path) -> None:
    """Two queries with identical hit counts sort by label ASC, deterministically."""
    db = StateDB(tmp_path / "state.db")
    now = datetime.now(UTC).isoformat()
    for _ in range(2):
        _audit_row(db, ts=now, tool="find_concept", success=True, result_count=0, query="zebra")
        _audit_row(db, ts=now, tool="find_concept", success=True, result_count=0, query="apple")

    rows = db.zero_result_query_counts("")
    labels = [label for _tool, label, _count in rows]
    assert labels == ["apple", "zebra"]
    # stable across repeated invocations
    assert [label for _t, label, _c in db.zero_result_query_counts("")] == labels


# ── Report helpers: single-source concepts ───────────────────────────────────


def test_backlog_single_source_concepts_section(tmp_path: Path) -> None:
    """A concept with one occurrence + matching resolved_label appears; >1 does not.

    Why it matters: the signal is "the agent keeps reaching for a concept that
    rests on a single source" — multi-source concepts are well-covered already.
    """
    db = StateDB(tmp_path / "state.db")
    _add_occurrence(db, "Yoneda lemma", "seg-1")  # single source
    _add_occurrence(db, "Monad", "seg-1")
    _add_occurrence(db, "Monad", "seg-2")  # two sources → excluded

    now = datetime.now(UTC).isoformat()
    _audit_row(db, ts=now, tool="find_concept", resolved_label="Yoneda lemma")
    _audit_row(db, ts=now, tool="get_source_passages", resolved_label="Yoneda lemma")
    _audit_row(db, ts=now, tool="find_concept", resolved_label="Monad")

    rows = db.single_source_concepts_in_demand("")
    assert rows == [("Yoneda lemma", 1, 2)]


def test_backlog_single_source_matches_hashed_label_default(tmp_path: Path) -> None:
    """Under the DEFAULT audit_detailed=False path, resolved_label is stored as a hash,
    yet the concept must still surface — matched via _hash8(name), not plaintext.

    Regression guard for a silent blinder: the section joined the hashed stored label
    against the plaintext concept_occurrences.concept_name, so it always returned (none)
    under the default config. The _audit_row helper writes plaintext and never caught it;
    this test drives the real _audit() hashing path that production uses.
    """
    db = StateDB(tmp_path / "state.db")
    _add_occurrence(db, "Yoneda lemma", "seg-1")  # single source

    now = datetime.now(UTC).isoformat()
    mcp = McpConfig(audit=True, audit_detailed=False)  # the default privacy posture
    for _ in range(2):
        _audit(
            db,
            vault_id="vk",
            tool="get_source_passages",
            arguments={"concept_name": "yoneda", "ts": now},
            success=True,
            latency_ms=1,
            mcp_config=mcp,
            result_count=3,
            resolved_label="Yoneda lemma",
        )

    # The stored label is the hash, not the plaintext name (proves we hit the real path).
    assert _last_mcp_meta(db)["resolved_label"] != "Yoneda lemma"

    rows = db.single_source_concepts_in_demand("")
    assert rows == [("Yoneda lemma", 1, 2)]


def test_backlog_repeat_weak_queries_matches_hashed_label_default(tmp_path: Path) -> None:
    """repeat_weak_queries must also work under the default hashed path, and map the
    target_concept back to its plaintext name for display."""
    db = StateDB(tmp_path / "state.db")
    mcp = McpConfig(audit=True, audit_detailed=False)
    for _ in range(2):
        _audit(
            db,
            vault_id="vk",
            tool="get_source_passages",
            arguments={"concept_name": "yoneda"},
            success=True,
            latency_ms=1,
            mcp_config=mcp,
            result_count=1,
            resolved_label="Yoneda lemma",
        )

    rows = db.repeat_weak_queries("", {"Yoneda lemma"}, min_hits=2)
    assert len(rows) == 1
    _label, hits, target = rows[0]
    assert hits == 2
    assert target == "Yoneda lemma"  # mapped back from the stored hash


def test_backlog_repeat_weak_queries_section(tmp_path: Path) -> None:
    """Only labels resolving to single-source concepts are included; min_hits respected."""
    db = StateDB(tmp_path / "state.db")
    now = datetime.now(UTC).isoformat()
    # "yoneda for dummies" resolves to the single-source "Yoneda lemma", 3 hits.
    for _ in range(3):
        _audit_row(
            db,
            ts=now,
            tool="find_concept",
            query="yoneda for dummies",
            resolved_label="Yoneda lemma",
        )
    # A query resolving to a non-single-source concept — excluded by the set filter.
    for _ in range(3):
        _audit_row(db, ts=now, tool="find_concept", query="what is a monad", resolved_label="Monad")
    # A single-hit query against the single-source concept — excluded by min_hits.
    _audit_row(db, ts=now, tool="find_concept", query="yoneda once", resolved_label="Yoneda lemma")

    rows = db.repeat_weak_queries("", {"Yoneda lemma"}, min_hits=2)
    assert rows == [("yoneda for dummies", 3, "Yoneda lemma")]


def test_repeat_weak_queries_empty_set_returns_empty(tmp_path: Path) -> None:
    """An empty single-source set short-circuits (no invalid IN () SQL)."""
    db = StateDB(tmp_path / "state.db")
    now = datetime.now(UTC).isoformat()
    _audit_row(db, ts=now, tool="find_concept", query="x", resolved_label="Y")
    assert db.repeat_weak_queries("", set(), min_hits=1) == []


# ── Report helpers: tool-mix sessionization ──────────────────────────────────


def test_backlog_tool_mix_sessionization(tmp_path: Path) -> None:
    """Rows across a 35-min gap split into two sessions; <5-call sessions excluded.

    Why it matters: the roadmap's adoption question ("verbatim vs synthesis?")
    is answered per-session; a stale gap must start a fresh session, and noise
    bursts under 5 calls are not real sessions.
    """
    db = StateDB(tmp_path / "state.db")
    base = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)

    # Session A: 6 contiguous calls (3 verbatim, 2 answer_question, 1 other).
    tools_a = [
        "search_source_segments",
        "get_source_passages",
        "read_source_segment",
        "answer_question",
        "answer_question",
        "list_articles",
    ]
    for i, tool in enumerate(tools_a):
        _audit_row(db, ts=(base + timedelta(minutes=i)).isoformat(), tool=tool)

    # Session B: starts 35 min after session A's last call, 5 calls.
    b_start = base + timedelta(minutes=len(tools_a) - 1 + 35)
    for i in range(5):
        _audit_row(db, ts=(b_start + timedelta(minutes=i)).isoformat(), tool="answer_question")

    # A 3-call burst far later — under min_calls, excluded.
    c_start = b_start + timedelta(hours=5)
    for i in range(3):
        _audit_row(db, ts=(c_start + timedelta(minutes=i)).isoformat(), tool="answer_question")

    sessions = db.tool_mix_sessions("")
    assert len(sessions) == 2
    # Session ids are sequential.
    assert [s[0] for s in sessions] == [0, 1]
    # Session A: total 6, verbatim 3, answer 2, other 1.
    a = next(s for s in sessions if s[1] == 6)
    assert a[2:] == (3, 2, 1)
    # Session B: total 5, all answer_question.
    b = next(s for s in sessions if s[1] == 5)
    assert b[2:] == (0, 5, 0)


def test_backlog_null_vault_id_bucketed_not_dropped(tmp_path: Path) -> None:
    """vault_id IS NULL rows form their own session bucket; counted, not lost."""
    db = StateDB(tmp_path / "state.db")
    base = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    # 5 calls with vault_id None and 5 with a real vault, same time window.
    for i in range(5):
        _audit_row(
            db, ts=(base + timedelta(minutes=i)).isoformat(), tool="answer_question", vault_id=None
        )
    for i in range(5):
        _audit_row(
            db,
            ts=(base + timedelta(minutes=i)).isoformat(),
            tool="answer_question",
            vault_id="real",
        )

    sessions = db.tool_mix_sessions("")
    assert len(sessions) == 2  # one per partition; the NULL bucket survived
    assert sorted(s[1] for s in sessions) == [5, 5]
    total_calls = sum(s[1] for s in sessions)
    assert total_calls == 10  # nothing dropped


# ── CLI: synto doctor --backlog ───────────────────────────────────────────────


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def vault(tmp_path: Path, runner: CliRunner) -> Path:
    """An initialised vault, ready for `synto doctor`."""
    result = runner.invoke(cli, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    return tmp_path


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch):
    """Stub build_client so doctor's provider/model checks don't hit Ollama."""

    class _FakeClient:
        def require_healthy(self) -> None:
            return None

        def list_models(self) -> list[str]:
            return []

    monkeypatch.setattr("synto.client_factory.build_client", lambda cfg: _FakeClient())


def _open_vault_db(vault: Path) -> StateDB:
    config = Config.from_vault(vault)
    return StateDB(config.state_db_path)


def test_backlog_empty_window(vault: Path, runner: CliRunner, fake_provider) -> None:
    """No audit rows → 'no activity' line, exit 0, no crash."""
    result = runner.invoke(cli, ["doctor", "--vault", str(vault), "--backlog", "--since", "7d"])
    assert result.exit_code == 0, result.output
    assert "MCP demand-vs-coverage (last 7d)" in result.output
    assert "no MCP activity in the last 7d" in result.output


def test_doctor_without_backlog_omits_section(
    vault: Path, runner: CliRunner, fake_provider
) -> None:
    """Default `synto doctor` (no --backlog) never prints the backlog section."""
    result = runner.invoke(cli, ["doctor", "--vault", str(vault)])
    assert result.exit_code == 0, result.output
    assert "MCP demand-vs-coverage" not in result.output
    # The v0.4.0 sections are still present and ordered.
    assert "Verbatim source index" in result.output
    assert "Graph view" in result.output
    assert result.output.index("Verbatim source index") < result.output.index("Graph view")


def test_backlog_populated_renders_all_sections(
    vault: Path, runner: CliRunner, fake_provider
) -> None:
    """A populated audit set renders all four sections with raw labels."""
    db = _open_vault_db(vault)
    _add_occurrence(db, "Yoneda lemma", "seg-1")
    now = datetime.now(UTC).isoformat()
    _audit_row(
        db,
        ts=now,
        tool="search_source_segments",
        success=True,
        result_count=0,
        query="monad transformer",
    )
    for _ in range(2):
        _audit_row(
            db,
            ts=now,
            tool="find_concept",
            query="yoneda for dummies",
            resolved_label="Yoneda lemma",
        )
    for i in range(5):
        _audit_row(db, ts=now, tool="answer_question")
    db._conn.close()

    result = runner.invoke(cli, ["doctor", "--vault", str(vault), "--backlog", "--since", "all"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "MCP demand-vs-coverage (all time)" in out
    assert "Zero-result queries" in out
    assert "monad transformer" in out
    assert "Single-source concepts in active demand" in out
    assert "Yoneda lemma" in out
    assert "Repeat weak queries" in out
    assert "yoneda for dummies" in out
    assert "Tool-mix per session" in out
    assert "Sessions:" in out
    # Raw labels → no degradation footer.
    assert "audit_detailed=true" not in out


def test_backlog_degrades_without_audit_detailed(
    vault: Path, runner: CliRunner, fake_provider
) -> None:
    """Hash-only labels render as <hash:8> and trigger the degradation footer."""
    db = _open_vault_db(vault)
    now = datetime.now(UTC).isoformat()
    # An 8-char hex label, as produced by the default (hashed) audit path.
    _audit_row(
        db, ts=now, tool="search_source_segments", success=True, result_count=0, query="a1b2c3d4"
    )
    db._conn.close()

    result = runner.invoke(cli, ["doctor", "--vault", str(vault), "--backlog", "--since", "all"])
    assert result.exit_code == 0, result.output
    assert "<a1b2c3d4>" in result.output
    assert "audit_detailed=true" in result.output


def test_doctor_warns_when_grandfather_exposes_sources(
    vault: Path, runner: CliRunner, fake_provider
) -> None:
    """A vault with no declared license must warn loudly that all raw text is exposed.

    Why it matters: the legacy-vault grandfather relaxes permissive_only to "all"; a
    privacy-sensitive user must never discover that silently. This is the "warn loudly"
    decision for v0.4.0.
    """
    result = runner.invoke(cli, ["doctor", "--vault", str(vault)])
    assert result.exit_code == 0, result.output
    assert 'effective "all"' in result.output
    assert "readable by MCP clients" in result.output


def test_doctor_shows_plain_mode_once_license_declared(
    vault: Path, runner: CliRunner, fake_provider
) -> None:
    """Once any source declares a license, the gate engages and the warning disappears."""
    db = _open_vault_db(vault)
    db._conn.execute(
        """INSERT OR REPLACE INTO source_documents
           (id, source_type, origin_uri, title, imported_at, redistribution, license)
           VALUES ('s1', 'pdf', '/raw/s1.pdf', 's1', '2024-01-01T00:00:00', 'unknown', 'CC-BY')"""
    )
    db._conn.commit()
    db._conn.close()

    result = runner.invoke(cli, ["doctor", "--vault", str(vault)])
    assert result.exit_code == 0, result.output
    assert 'source-access mode: "permissive_only"' in result.output
    assert 'effective "all"' not in result.output
