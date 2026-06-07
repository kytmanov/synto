"""Feature 34: verified-status lifecycle.

Tests encode *why* each behavior matters so silent logic regressions
surface as failures rather than green-but-wrong tests.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from synto.cli import cli
from synto.models import WikiArticleRecord
from synto.pipeline.compile import approve_drafts, verify_drafts
from synto.state import StateDB
from synto.vault import parse_note

# ── Model contract ────────────────────────────────────────────────────────────


def test_verified_record_is_not_draft_so_agents_trust_it() -> None:
    """Agents that filter on .is_draft must treat verified articles as
    trusted; otherwise the new lifecycle adds a state nobody can use."""
    record = WikiArticleRecord(path="p", title="t", sources=[], content_hash="h", status="verified")
    assert record.is_draft is False
    assert record.status == "verified"


def test_invalid_status_value_is_rejected() -> None:
    """Typos like 'approved' silently passing would leave articles in an
    undefined limbo that downstream code cannot reason about."""
    with pytest.raises(ValidationError):
        WikiArticleRecord(path="p", title="t", sources=[], content_hash="h", status="approved")


def test_model_dump_omits_is_draft_after_column_removal() -> None:
    """model_dump() feeding serializers must not name a column that no
    longer exists on the wire."""
    record = WikiArticleRecord(path="p", title="t", sources=[], content_hash="h", status="draft")
    dump = record.model_dump()
    assert "is_draft" not in dump
    assert dump["status"] == "draft"


# ── v15 migration ─────────────────────────────────────────────────────────────


def _build_v14_db(db_path: Path) -> None:
    """Hand-build a pre-v15 wiki_articles schema for migration testing."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_version (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            version INTEGER NOT NULL
        );
        INSERT INTO schema_version (id, version) VALUES (1, 14);

        CREATE TABLE wiki_articles (
            path           TEXT PRIMARY KEY,
            title          TEXT NOT NULL,
            sources        TEXT NOT NULL,
            content_hash   TEXT NOT NULL,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL,
            is_draft       INTEGER NOT NULL DEFAULT 1,
            approved_at    TEXT,
            approval_notes TEXT,
            kind           TEXT NOT NULL DEFAULT 'concept',
            question_hash  TEXT,
            synthesis_sources TEXT,
            synthesis_source_hashes TEXT,
            article_id     TEXT,
            last_compile_pipeline TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def test_v15_migration_preserves_draft_intent_across_upgrade(tmp_path: Path) -> None:
    """If draft rows silently become 'published' on upgrade, every
    in-flight LLM draft gets promoted into the active wiki overnight.
    This test is the load-bearing guarantee against that data corruption."""
    db_path = tmp_path / "state.db"
    _build_v14_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO wiki_articles "
        "(path, title, sources, content_hash, created_at, updated_at, "
        "is_draft, kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "wiki/.drafts/d.md",
            "D",
            "[]",
            "h",
            "2024-01-01T00:00:00",
            "2024-01-01T00:00:00",
            1,
            "concept",
        ),
    )
    conn.commit()
    conn.close()

    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT status FROM wiki_articles WHERE path='wiki/.drafts/d.md'").fetchone()
    assert row[0] == "draft"
    cols = {r[1] for r in conn.execute("PRAGMA table_info(wiki_articles)").fetchall()}
    assert "is_draft" not in cols
    assert "status" in cols
    conn.close()


def test_v15_migration_preserves_published_intent_across_upgrade(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _build_v14_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO wiki_articles "
        "(path, title, sources, content_hash, created_at, updated_at, "
        "is_draft, kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("wiki/p.md", "P", "[]", "h", "2024-01-01T00:00:00", "2024-01-01T00:00:00", 0, "concept"),
    )
    conn.commit()
    conn.close()

    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT status FROM wiki_articles WHERE path='wiki/p.md'").fetchone()
    assert row[0] == "published"
    conn.close()


def test_v15_migration_is_idempotent_after_partial_failure(tmp_path: Path) -> None:
    """A v15 DB re-opened (e.g. after a crash) must not error or
    double-apply — the PRAGMA probe is the safety net."""
    db_path = tmp_path / "state.db"
    StateDB(db_path).close()
    StateDB(db_path).close()
    StateDB(db_path).close()

    conn = sqlite3.connect(db_path)
    version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    assert version == 17
    cols = {r[1] for r in conn.execute("PRAGMA table_info(wiki_articles)").fetchall()}
    assert "is_draft" not in cols
    conn.close()


def test_fresh_db_has_status_check_constraint(tmp_path: Path) -> None:
    """SQL-layer defense in depth — a bug in Python that bypasses the
    Pydantic model must still be caught before bad rows persist."""
    db_path = tmp_path / "state.db"
    StateDB(db_path).close()

    conn = sqlite3.connect(db_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO wiki_articles "
            "(path, title, sources, content_hash, created_at, updated_at, "
            "status, kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "wiki/x.md",
                "X",
                "[]",
                "h",
                "2024-01-01T00:00:00",
                "2024-01-01T00:00:00",
                "bogus",
                "concept",
            ),
        )
    conn.close()


# ── verify_article + approve_article COALESCE ────────────────────────────────


def test_verify_article_on_missing_row_is_silent_noop(db: StateDB) -> None:
    """Parity with publish_article — verify on a missing row should not
    crash callers that may race with a concurrent delete."""
    db.verify_article("wiki/.drafts/nonexistent.md")  # no row at this path


def test_verify_then_publish_preserves_first_approval_timestamp(
    vault: Path, config, db: StateDB
) -> None:
    """A verified-then-published article must retain the verify timestamp
    so the audit trail records when the human actually signed off, not
    when they later promoted the file."""
    from synto.vault import write_note

    draft_path = config.drafts_dir / "Article.md"
    write_note(draft_path, {"title": "Article", "status": "draft", "tags": []}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Article",
            sources=[],
            content_hash="h",
            status="draft",
        )
    )

    verify_drafts(config, db, [draft_path])
    verified = db.get_article(str(draft_path.relative_to(vault)))
    assert verified is not None
    first_approval = verified.approved_at
    assert first_approval is not None

    approve_drafts(config, db, [draft_path])
    published = db.get_article("wiki/Article.md")
    assert published is not None
    assert published.status == "published"
    assert published.approved_at == first_approval  # COALESCE preserved it


def test_publish_without_prior_verify_stamps_approved_at_at_publish_time(
    vault: Path, config, db: StateDB
) -> None:
    from synto.vault import write_note

    draft_path = config.drafts_dir / "Direct.md"
    write_note(draft_path, {"title": "Direct", "status": "draft", "tags": []}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Direct",
            sources=[],
            content_hash="h",
            status="draft",
        )
    )

    approve_drafts(config, db, [draft_path])
    published = db.get_article("wiki/Direct.md")
    assert published is not None
    assert published.approved_at is not None


# ── approve_drafts: verify branch ─────────────────────────────────────────────


def test_verify_default_verifies_in_place_to_enable_batch_review(
    vault: Path, config, db: StateDB
) -> None:
    """A curator must be able to verify drafts one-by-one without each
    one immediately promoting into the active wiki — that's the whole
    point of the new state."""
    from synto.vault import write_note

    draft_path = config.drafts_dir / "Article.md"
    write_note(draft_path, {"title": "Article", "status": "draft", "tags": []}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Article",
            sources=[],
            content_hash="h",
            status="draft",
        )
    )

    affected = verify_drafts(config, db, [draft_path])

    assert affected == [draft_path]
    assert draft_path.exists()
    assert not (config.wiki_dir / "Article.md").exists()

    meta, _ = parse_note(draft_path)
    assert meta["status"] == "verified"

    record = db.get_article(str(draft_path.relative_to(vault)))
    assert record is not None
    assert record.status == "verified"


def test_approve_default_promotes_to_wiki_and_strips_draft_file(
    vault: Path, config, db: StateDB
) -> None:
    from synto.vault import write_note

    draft_path = config.drafts_dir / "Article.md"
    write_note(draft_path, {"title": "Article", "status": "draft", "tags": []}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Article",
            sources=[],
            content_hash="h",
            status="draft",
        )
    )

    affected = approve_drafts(config, db, [draft_path])
    target = config.wiki_dir / "Article.md"
    assert affected == [target]
    assert target.exists()
    assert not draft_path.exists()

    record = db.get_article("wiki/Article.md")
    assert record is not None
    assert record.status == "published"


def test_reverify_is_noop_so_verify_is_safe_to_rerun_in_scripts(
    vault: Path, config, db: StateDB
) -> None:
    """A curator script that runs `synto approve --all` on every tick
    must not churn the DB or repeatedly rewrite frontmatter."""
    from synto.vault import write_note

    draft_path = config.drafts_dir / "Article.md"
    write_note(draft_path, {"title": "Article", "status": "draft", "tags": []}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Article",
            sources=[],
            content_hash="h",
            status="draft",
        )
    )

    verify_drafts(config, db, [draft_path])
    first = db.get_article(str(draft_path.relative_to(vault)))
    assert first is not None
    first_approved = first.approved_at
    first_updated = first.updated_at

    affected = verify_drafts(config, db, [draft_path])
    assert affected == []  # nothing to do

    second = db.get_article(str(draft_path.relative_to(vault)))
    assert second is not None
    assert second.approved_at == first_approved
    assert second.updated_at == first_updated


def test_verify_skips_draft_when_published_twin_already_exists(
    vault: Path, config, db: StateDB
) -> None:
    """If publish already won, verify must not recreate a verified draft row/file."""
    from synto.vault import write_note

    draft_path = config.drafts_dir / "Article.md"
    published_path = config.wiki_dir / "Article.md"
    write_note(draft_path, {"title": "Article", "status": "draft", "tags": []}, "Body.")
    write_note(published_path, {"title": "Article", "status": "published", "tags": []}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Article.md",
            title="Article",
            sources=[],
            content_hash="h",
            status="published",
        )
    )

    affected = verify_drafts(config, db, [draft_path])

    assert affected == []
    assert not draft_path.exists()
    assert db.get_article("wiki/.drafts/Article.md") is None
    published = db.get_article("wiki/Article.md")
    assert published is not None
    assert published.status == "published"


def test_reject_works_on_verified_article_so_curators_can_change_their_mind(
    vault: Path, config, db: StateDB
) -> None:
    """Verified is a curator-controlled state — they must be able to
    walk it back without manual SQL surgery."""
    from synto.pipeline.compile import reject_draft
    from synto.vault import write_note

    draft_path = config.drafts_dir / "Article.md"
    write_note(draft_path, {"title": "Article", "status": "draft", "tags": []}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Article",
            sources=[],
            content_hash="h",
            status="draft",
        )
    )
    verify_drafts(config, db, [draft_path])
    assert db.get_article(str(draft_path.relative_to(vault))).status == "verified"

    reject_draft(draft_path, config, db, feedback="changed my mind")

    assert not draft_path.exists()
    assert db.get_article(str(draft_path.relative_to(vault))) is None


def test_list_articles_drafts_only_excludes_verified(vault: Path, config, db: StateDB) -> None:
    """`drafts_only=True` is the system's "what's unreviewed?" question.
    Verified articles are reviewed, so they must not appear there or the
    queue view becomes misleading."""
    from synto.vault import write_note

    draft = config.drafts_dir / "D.md"
    verified = config.drafts_dir / "V.md"
    for p, title in [(draft, "D"), (verified, "V")]:
        write_note(p, {"title": title, "status": "draft", "tags": []}, "Body.")
        db.upsert_article(
            WikiArticleRecord(
                path=str(p.relative_to(vault)),
                title=title,
                sources=[],
                content_hash="h",
                status="draft",
            )
        )
    verify_drafts(config, db, [verified])

    draft_paths = [a.path for a in db.list_articles(drafts_only=True)]
    assert str(draft.relative_to(vault)) in draft_paths
    assert str(verified.relative_to(vault)) not in draft_paths


def test_verify_marks_concept_compiled_so_next_compile_skips_verified_draft(
    vault: Path, config, db: StateDB
) -> None:
    """A verified draft must not be regenerated by the next `synto
    compile` — that would silently clobber human-reviewed content. The
    concept compile-state transition is the gate."""
    from synto.vault import write_note

    draft_path = config.drafts_dir / "Alpha.md"
    write_note(draft_path, {"title": "Alpha", "status": "draft", "tags": []}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Alpha",
            sources=["raw/a.md"],
            content_hash="h",
            status="draft",
        )
    )
    db.mark_concept_compile_state("Alpha", ["raw/a.md"], "deferred_draft")

    verify_drafts(config, db, [draft_path])

    state = db.get_compile_state("Alpha", "raw/a.md")
    assert state["status"] == "compiled"


# ── CLI ──────────────────────────────────────────────────────────────────────


def _seed_draft(vault: Path, config, db: StateDB, name: str) -> Path:
    from synto.vault import write_note

    draft_path = config.drafts_dir / f"{name}.md"
    write_note(draft_path, {"title": name, "status": "draft", "tags": []}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title=name,
            sources=[],
            content_hash="h",
            status="draft",
        )
    )
    return draft_path


def test_cli_approve_default_publishes(vault: Path, config, db: StateDB) -> None:
    _seed_draft(vault, config, db, "Article")
    runner = CliRunner()
    result = runner.invoke(cli, ["approve", "--vault", str(vault), "--all"])
    assert result.exit_code == 0, result.output
    assert "Published" in result.output


def test_cli_verify_command_marks_reviewed(vault: Path, config, db: StateDB) -> None:
    draft_path = _seed_draft(vault, config, db, "VerifiedArticle")
    runner = CliRunner()
    result = runner.invoke(cli, ["verify", "--vault", str(vault), str(draft_path)])
    assert result.exit_code == 0, result.output
    assert "Verified" in result.output
    meta, _ = parse_note(draft_path)
    assert meta["status"] == "verified"


def test_cli_verify_refuses_when_lock_held(vault: Path, monkeypatch) -> None:
    @contextlib.contextmanager
    def _held(_vault):
        yield False

    monkeypatch.setattr("synto.pipeline.lock.pipeline_lock", _held)

    runner = CliRunner()
    result = runner.invoke(cli, ["verify", "--vault", str(vault), "--all"])

    assert result.exit_code == 1
    assert "lock held" in result.output.lower()


def test_cli_approve_refuses_when_lock_held(vault: Path, monkeypatch) -> None:
    @contextlib.contextmanager
    def _held(_vault):
        yield False

    monkeypatch.setattr("synto.pipeline.lock.pipeline_lock", _held)

    runner = CliRunner()
    result = runner.invoke(cli, ["approve", "--vault", str(vault), "--all"])

    assert result.exit_code == 1
    assert "lock held" in result.output.lower()


def test_cli_verify_help_is_available(vault: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["verify", "--help"])
    assert result.exit_code == 0
    assert "--all" in result.output
