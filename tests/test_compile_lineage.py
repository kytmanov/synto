"""Tests for Feature 04: Compile Lineage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from conftest import as_router

from synto.models import PipelineVersion
from synto.state import StateDB

# ---------------------------------------------------------------------------
# Stage 1: migration v11 and compile_runs table
# ---------------------------------------------------------------------------


def test_migration_v11(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    # Table should exist after schema migration
    tables = {
        row[0]
        for row in db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "compile_runs" in tables


def test_compile_runs_schema(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    cols = {row[1] for row in db._conn.execute("PRAGMA table_info(compile_runs)").fetchall()}
    expected = {
        "run_ulid",
        "pipeline_json",
        "fast_model",
        "heavy_model",
        "started_at",
        "finished_at",
        "article_count",
        "total_tokens",
        "total_cost_usd",
    }
    assert expected.issubset(cols)


def test_current_schema_version() -> None:
    from synto.state import _CURRENT_SCHEMA_VERSION

    assert _CURRENT_SCHEMA_VERSION == 28


# ---------------------------------------------------------------------------
# Stage 2: PipelineVersion.fingerprint() determinism
# ---------------------------------------------------------------------------


def test_fingerprint_determinism() -> None:
    pv1 = PipelineVersion(fast_model="gemma4:e4b", heavy_model="qwen2.5:14b")
    pv2 = PipelineVersion(fast_model="gemma4:e4b", heavy_model="qwen2.5:14b")
    assert pv1.fingerprint() == pv2.fingerprint()


def test_fingerprint_changes_with_model() -> None:
    pv1 = PipelineVersion(fast_model="gemma4:e4b", heavy_model="qwen2.5:14b")
    pv2 = PipelineVersion(fast_model="gemma4:e4b", heavy_model="claude-opus-4-7")
    assert pv1.fingerprint() != pv2.fingerprint()


def test_fingerprint_is_hex_string() -> None:
    fp = PipelineVersion(fast_model="a", heavy_model="b").fingerprint()
    assert isinstance(fp, str)
    assert len(fp) >= 8
    int(fp[:8], 16)  # must be valid hex


# ---------------------------------------------------------------------------
# Stage 3: start/finish compile run DB methods
# ---------------------------------------------------------------------------


def test_start_compile_run(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    pv = PipelineVersion(fast_model="fast", heavy_model="heavy")
    db.start_compile_run("ULID001", pv.model_dump_json(), "fast", "heavy")
    row = db.get_compile_run("ULID001")
    assert row is not None
    assert row["run_ulid"] == "ULID001"
    assert row["fast_model"] == "fast"
    assert row["finished_at"] is None


def test_finish_compile_run(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    pv = PipelineVersion(fast_model="f", heavy_model="h")
    db.start_compile_run("ULID002", pv.model_dump_json(), "f", "h")
    db.finish_compile_run("ULID002", article_count=3)
    row = db.get_compile_run("ULID002")
    assert row["finished_at"] is not None
    assert row["article_count"] == 3


def test_compile_run_recorded(tmp_path: Path, config, db) -> None:
    """A full compile_concepts call must create a compile_runs row."""
    from synto.models import RawNoteRecord, SingleArticle
    from synto.pipeline.compile import compile_concepts

    # Set up a note with a concept needing compile
    (config.vault / "raw").mkdir(exist_ok=True)
    note = config.vault / "raw" / "test.md"
    note.write_text("# Test Note\nThis is a test note about TestConcept.\n")

    rel = "raw/test.md"
    db.upsert_raw(
        RawNoteRecord(
            path=rel,
            content_hash="abc123",
            status="ingested",
        )
    )
    db.upsert_concepts(rel, ["TestConcept"])

    mock_result = SingleArticle(
        title="TestConcept",
        content="Test content about TestConcept.",
        tags=["test"],
        summary="A test article.",
    )

    client = as_router(MagicMock())
    with patch("synto.pipeline.compile.request_structured", return_value=mock_result):
        compile_concepts(config, client, db)

    count = db._conn.execute("SELECT COUNT(*) FROM compile_runs").fetchone()[0]
    assert count == 1, "compile_runs row should be created"

    row = db._conn.execute("SELECT * FROM compile_runs").fetchone()
    assert row["finished_at"] is not None
    assert row["article_count"] >= 1


# ---------------------------------------------------------------------------
# Stage 4: lineage frontmatter in articles
# ---------------------------------------------------------------------------


def test_frontmatter_lineage_key(tmp_path: Path, config, db) -> None:
    """Compiled article frontmatter must contain a 'lineage' key."""
    from synto.models import RawNoteRecord, SingleArticle
    from synto.pipeline.compile import compile_concepts
    from synto.vault import parse_note

    rel = "raw/lineage_test.md"
    (config.vault / "raw").mkdir(exist_ok=True)
    note = config.vault / "raw" / "lineage_test.md"
    note.write_text("# Lineage test\nTest content.\n")
    db.upsert_raw(RawNoteRecord(path=rel, content_hash="def456", status="ingested"))
    db.upsert_concepts(rel, ["LineageConcept"])

    mock_result = SingleArticle(
        title="LineageConcept",
        content="Content.",
        tags=["test"],
        summary="Summary.",
    )

    client = as_router(MagicMock())
    with patch("synto.pipeline.compile.request_structured", return_value=mock_result):
        draft_paths, _, _ = compile_concepts(config, client, db)

    assert draft_paths, "At least one draft should be written"
    meta, _ = parse_note(draft_paths[0])
    assert "lineage" in meta, "lineage key missing from frontmatter"
    lineage = meta["lineage"]
    assert isinstance(lineage, list) and len(lineage) > 0
    entry = lineage[0]
    assert "compile_run" in entry
    assert "pipeline" in entry
    assert "timestamp" in entry


def test_stub_frontmatter_lineage_key(tmp_path: Path, config, db) -> None:
    """Stub compiles are LLM runs too — their drafts must record lineage.

    Regression: stub drafts were written without run_ulid, so published stubs
    traced to "No lineage recorded" and would be invisible to staleness checks.
    """
    from synto.models import SingleArticle
    from synto.pipeline.compile import compile_concepts
    from synto.vault import parse_note

    db.add_stub("StubConcept")

    mock_result = SingleArticle(
        title="StubConcept", content="Stub content.", tags=[], summary="Stub."
    )
    client = as_router(MagicMock())
    with patch("synto.pipeline.compile.request_structured", return_value=mock_result):
        draft_paths, _, _ = compile_concepts(config, client, db)

    assert draft_paths, "Stub draft should be written"
    meta, _ = parse_note(draft_paths[0])
    assert "lineage" in meta, "lineage key missing from stub frontmatter"
    entry = meta["lineage"][0]
    assert "compile_run" in entry
    assert "pipeline" in entry
    assert "timestamp" in entry


def test_legacy_frontmatter_lineage_key(tmp_path: Path, config, db) -> None:
    """Legacy (--legacy) compiles must record lineage and a compile_runs row.

    Regression: compile_notes had no run identity at all, so legacy-born
    articles traced to "No lineage recorded".
    """
    from synto.models import ArticlePlan, CompilePlan, RawNoteRecord, SingleArticle
    from synto.pipeline.compile import compile_notes
    from synto.vault import parse_note

    rel = "raw/legacy_test.md"
    (config.vault / "raw").mkdir(exist_ok=True)
    (config.vault / "raw" / "legacy_test.md").write_text("# Legacy test\nContent.\n")
    db.upsert_raw(RawNoteRecord(path=rel, content_hash="jkl012", status="ingested"))

    plan = CompilePlan(
        articles=[
            ArticlePlan(
                title="LegacyArticle",
                action="create",
                path="LegacyArticle.md",
                reasoning="test",
                source_paths=[rel],
            )
        ]
    )
    article = SingleArticle(title="LegacyArticle", content="Content.", tags=[], summary="S.")
    client = as_router(MagicMock())
    with patch("synto.pipeline.compile.request_structured", side_effect=[plan, article]):
        draft_paths, _ = compile_notes(config, client, db)

    assert draft_paths, "Legacy draft should be written"
    meta, _ = parse_note(draft_paths[0])
    assert "lineage" in meta, "lineage key missing from legacy frontmatter"
    entry = meta["lineage"][0]
    assert "compile_run" in entry
    assert "pipeline" in entry
    assert "timestamp" in entry

    row = db._conn.execute("SELECT * FROM compile_runs").fetchone()
    assert row is not None, "legacy compile should record a compile_runs row"
    assert row["run_ulid"] == entry["compile_run"]
    assert row["finished_at"] is not None


# ---------------------------------------------------------------------------
# Stage 5: synto trace article CLI command
# ---------------------------------------------------------------------------


def test_trace_article_command(tmp_path: Path, config, db) -> None:
    from click.testing import CliRunner

    from synto.cli import cli
    from synto.models import RawNoteRecord, SingleArticle
    from synto.pipeline.compile import compile_concepts

    # Compile to produce a draft with lineage
    rel = "raw/trace_test.md"
    (config.vault / "raw").mkdir(exist_ok=True)
    note = config.vault / "raw" / "trace_test.md"
    note.write_text("# Trace test\nContent.\n")
    db.upsert_raw(RawNoteRecord(path=rel, content_hash="ghi789", status="ingested"))
    db.upsert_concepts(rel, ["TraceConcept"])

    mock_result = SingleArticle(
        title="TraceConcept", content="Content.", tags=[], summary="Summary."
    )
    client = as_router(MagicMock())
    with patch("synto.pipeline.compile.request_structured", return_value=mock_result):
        compile_concepts(config, client, db)

    runner = CliRunner()
    result = runner.invoke(cli, ["trace", "article", "--vault", str(config.vault), "TraceConcept"])
    assert result.exit_code == 0, result.output
    assert "TraceConcept" in result.output or "Compile" in result.output or "run" in result.output


def test_trace_article_missing(config) -> None:
    from click.testing import CliRunner

    from synto.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli, ["trace", "article", "--vault", str(config.vault), "NonExistentArticle"]
    )
    assert result.exit_code != 0
