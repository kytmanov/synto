from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from synto.cli import cli
from synto.config import Config
from synto.indexer import generate_index_json
from synto.models import RawNoteRecord, WikiArticleRecord
from synto.pipeline.eval import render_json, run_offline
from synto.vault import write_note


def _init_eval_vault(tmp_path: Path) -> Config:
    (tmp_path / "raw").mkdir(exist_ok=True)
    (tmp_path / "wiki").mkdir(exist_ok=True)
    (tmp_path / "wiki" / ".drafts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "wiki" / "sources").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".synto").mkdir(exist_ok=True)
    (tmp_path / "synto.toml").write_text(
        (
            "[models]\n"
            'fast = "gemma4:e4b"\n'
            'heavy = "gemma4:e4b"\n\n'
            "[provider]\n"
            'name = "lm_studio"\n'
            'url = "http://localhost:1234/v1"\n'
        ),
        encoding="utf-8",
    )
    return Config(vault=tmp_path)


def _write_query_fixture(path: Path) -> Path:
    path.write_text(
        (
            "[[query]]\n"
            'id = "q1"\n'
            'question = "What is Topic?"\n'
            'expected_concepts = ["Topic"]\n'
            "expected_contains = []\n"
        ),
        encoding="utf-8",
    )
    return path


def test_eval_empty_vault_is_deterministic(tmp_path: Path) -> None:
    config = _init_eval_vault(tmp_path)
    queries = _write_query_fixture(tmp_path / "queries.toml")

    first = run_offline(config, queries)
    second = run_offline(config, queries)

    assert first == second
    assert first.article_coverage == 0.0
    assert first.term_recall is None
    assert first.index_json_validity is None
    assert first.details["index_json_validity"] == {"reason": "missing", "skipped": True}
    assert first.wikilink_resolution == 1.0
    assert first.harmonic_mean == 0.0


def test_eval_missing_index_json_returns_missing(tmp_path: Path) -> None:
    config = _init_eval_vault(tmp_path)
    result = run_offline(config, _write_query_fixture(tmp_path / "queries.toml"))
    assert result.index_json_validity is None
    assert result.details["index_json_validity"] == {"reason": "missing", "skipped": True}
    assert not (config.synto_dir / "INDEX.json").exists()


def test_eval_render_text_shows_missing_index_as_na(tmp_path: Path) -> None:
    config = _init_eval_vault(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "eval",
            "--vault",
            str(config.vault),
            "--queries",
            str(_write_query_fixture(tmp_path / "q.toml")),
        ],
    )
    assert result.exit_code == 0
    assert "INDEX.json validity: n/a" in result.output


def test_eval_invalid_index_json_returns_zero(tmp_path: Path) -> None:
    config = _init_eval_vault(tmp_path)
    (config.synto_dir / "INDEX.json").write_text("{broken", encoding="utf-8")
    result = run_offline(config, _write_query_fixture(tmp_path / "queries.toml"))
    assert result.index_json_validity == 0.0
    assert result.details["index_json_validity"] == {"reason": "invalid_json"}


def test_eval_broken_wikilinks_reduce_resolution(tmp_path: Path, db) -> None:
    config = _init_eval_vault(tmp_path)
    db = db.__class__(config.state_db_path)
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="ha", status="compiled"))
    db.upsert_concepts("raw/a.md", ["Topic"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            is_draft=False,
        )
    )
    write_note(
        config.wiki_dir / "Topic.md",
        {"title": "Topic", "sources": ["raw/a.md"]},
        "See [[Missing Topic]].",
    )
    generate_index_json(config, db)

    result = run_offline(config, _write_query_fixture(tmp_path / "queries.toml"))
    assert result.index_json_validity == 1.0
    assert result.wikilink_resolution is not None
    assert result.wikilink_resolution < 1.0


def test_eval_valid_sources_wikilinks_are_counted_as_resolved(tmp_path: Path) -> None:
    config = _init_eval_vault(tmp_path)
    from synto.state import StateDB

    db = StateDB(config.state_db_path)
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="ha", status="compiled"))
    db.upsert_concepts("raw/a.md", ["Topic"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            is_draft=False,
        )
    )
    write_note(
        config.wiki_dir / "Topic.md",
        {"title": "Topic", "sources": ["raw/a.md"]},
        "See [[sources/Alpha Source|S1]].",
    )
    write_note(
        config.sources_dir / "Alpha Source.md",
        {"title": "Alpha Source", "tags": ["source"]},
        "Body",
    )
    generate_index_json(config, db)

    result = run_offline(config, _write_query_fixture(tmp_path / "queries.toml"))

    assert result.wikilink_resolution == 1.0


def test_eval_source_wikilink_title_match_is_case_insensitive(tmp_path: Path) -> None:
    config = _init_eval_vault(tmp_path)
    from synto.state import StateDB

    db = StateDB(config.state_db_path)
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="ha", status="compiled"))
    db.upsert_concepts("raw/a.md", ["Topic"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            is_draft=False,
        )
    )
    write_note(
        config.wiki_dir / "Topic.md",
        {"title": "Topic", "sources": ["raw/a.md"]},
        "See [[sources/Reference Note|S1]].",
    )
    write_note(
        config.sources_dir / "Reference note.md",
        {"title": "Reference note", "tags": ["source"]},
        "Body",
    )
    generate_index_json(config, db)

    result = run_offline(config, _write_query_fixture(tmp_path / "queries.toml"))

    assert result.wikilink_resolution == 1.0


def test_eval_bare_source_wikilink_title_is_counted_as_resolved(tmp_path: Path) -> None:
    config = _init_eval_vault(tmp_path)
    from synto.state import StateDB

    db = StateDB(config.state_db_path)
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="ha", status="compiled"))
    db.upsert_concepts("raw/a.md", ["Topic"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            is_draft=False,
        )
    )
    write_note(
        config.wiki_dir / "Topic.md",
        {"title": "Topic", "sources": ["raw/a.md"]},
        "See [[Reference Note]].",
    )
    write_note(
        config.sources_dir / "Reference note.md",
        {"title": "Reference note", "tags": ["source"]},
        "Body",
    )
    generate_index_json(config, db)

    result = run_offline(config, _write_query_fixture(tmp_path / "queries.toml"))

    assert result.wikilink_resolution == 1.0


def test_eval_valid_index_and_article_coverage(tmp_path: Path) -> None:
    config = _init_eval_vault(tmp_path)
    from synto.state import StateDB

    db = StateDB(config.state_db_path)
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="ha", status="compiled"))
    db.upsert_concepts("raw/a.md", ["Topic"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            is_draft=False,
        )
    )
    write_note(config.wiki_dir / "Topic.md", {"title": "Topic", "sources": ["raw/a.md"]}, "Body")
    generate_index_json(config, db)

    result = run_offline(config, _write_query_fixture(tmp_path / "queries.toml"))
    assert result.article_coverage == 1.0
    assert result.citation_coverage == 1.0
    assert result.index_json_validity == 1.0
    assert result.term_recall is None
    assert result.harmonic_mean == 1.0


def test_eval_cli_live_exits_code_two(tmp_path: Path) -> None:
    config = _init_eval_vault(tmp_path)
    result = CliRunner().invoke(cli, ["eval", "--vault", str(config.vault), "--live"])
    assert result.exit_code == 2
    assert "not implemented in Phase 1A" in result.output
    assert "synto eval --live" in result.output


def test_eval_cli_json_is_parseable_and_offline(tmp_path: Path) -> None:
    config = _init_eval_vault(tmp_path)
    queries = _write_query_fixture(tmp_path / "queries.toml")
    with patch("synto.cli._load_deps", side_effect=AssertionError("no LLM calls")):
        result = CliRunner().invoke(
            cli,
            ["eval", "--vault", str(config.vault), "--queries", str(queries), "--json"],
        )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "article_coverage" in payload
    assert "details" in payload


def test_render_json_round_trips(tmp_path: Path) -> None:
    config = _init_eval_vault(tmp_path)
    rendered = render_json(run_offline(config, _write_query_fixture(tmp_path / "queries.toml")))
    payload = json.loads(rendered)
    assert "harmonic_mean" in payload
