"""Tests for the v10 schema migration path.

Verifies:
- Fresh DB creation sets schema_version to 10 with metric tables and indexes.
- A v8 DB shape upgrades to v10 cleanly.
- All v8-era indexes survive the upgrade.
- Existing and future wiki articles get stable article_id values.
- metric_events.success enforces 0/1 only.
- Old pre-public telemetry tables are removed during v10 upgrade.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from synto.models import WikiArticleRecord
from synto.state import _CURRENT_SCHEMA_VERSION, StateDB


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _table_indexes(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?", (table,)
    ).fetchall()
    return {row[0] for row in rows if not row[0].startswith("sqlite_")}


V7_ERA_INDEXES = {
    "idx_raw_hash",
    "idx_raw_status",
    "idx_concept_name",
    "idx_ingest_chunks_source",
    "idx_concept_compile_status",
    "idx_concept_compile_name",
    "idx_rejections_concept",
    "idx_items_kind",
    "idx_items_status",
    "idx_mentions_item",
    "idx_mentions_source",
    "idx_wiki_articles_kind",
    "idx_wiki_articles_question_hash",
}

V10_TABLES = {
    "source_documents",
    "source_segments",
    "source_warnings",
    "metric_events",
    "metric_daily_rollups",
    "generated_assets",
}

V10_INDEXES = {
    "idx_source_segments_source",
    "idx_source_segments_identity",
    "idx_source_warnings_source",
    "idx_metric_events_ts",
    "idx_metric_events_type_ts",
    "idx_metric_daily_rollups_day",
    "idx_generated_assets_source",
    "idx_wiki_articles_article_id",
}

ARTICLE_ID_RE = re.compile(r"^[0-9A-Z]{26,28}$")


def _build_v8_db(db_path: Path) -> None:
    _build_v7_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        ALTER TABLE raw_notes ADD COLUMN source_type TEXT NOT NULL DEFAULT 'notes';
        ALTER TABLE raw_notes ADD COLUMN origin_uri TEXT;
        ALTER TABLE raw_notes ADD COLUMN imported_at TEXT;
        ALTER TABLE raw_notes ADD COLUMN normalized_hash TEXT;
        ALTER TABLE raw_notes ADD COLUMN extractor_version TEXT;
        ALTER TABLE raw_notes ADD COLUMN prompt_version TEXT;
        UPDATE schema_version SET version = 8 WHERE id = 1;
        """
    )
    conn.commit()
    conn.close()


def _build_v7_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_version (
            id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL
        );
        CREATE TABLE raw_notes (
            path        TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'new',
            summary     TEXT,
            quality     TEXT,
            language    TEXT,
            ingested_at TEXT,
            compiled_at TEXT,
            error       TEXT
        );
        CREATE TABLE concepts (
            name TEXT NOT NULL, source_path TEXT NOT NULL,
            PRIMARY KEY (name, source_path)
        );
        CREATE TABLE wiki_articles (
            path TEXT PRIMARY KEY, title TEXT NOT NULL, sources TEXT NOT NULL,
            content_hash TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, is_draft INTEGER NOT NULL DEFAULT 1,
            approved_at TEXT, approval_notes TEXT,
            kind TEXT NOT NULL DEFAULT 'concept', question_hash TEXT,
            synthesis_sources TEXT, synthesis_source_hashes TEXT
        );
        CREATE TABLE rejections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT NOT NULL, feedback TEXT NOT NULL,
            rejected_body TEXT, rejected_at TEXT NOT NULL
        );
        CREATE TABLE stubs (
            concept TEXT PRIMARY KEY, created_at TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'auto'
        );
        CREATE TABLE blocked_concepts (
            concept TEXT PRIMARY KEY, blocked_at TEXT NOT NULL
        );
        CREATE TABLE concept_aliases (
            concept_name TEXT NOT NULL, alias TEXT NOT NULL,
            PRIMARY KEY (concept_name, alias)
        );
        CREATE TABLE knowledge_items (
            name TEXT PRIMARY KEY, kind TEXT NOT NULL DEFAULT 'ambiguous',
            subtype TEXT, status TEXT NOT NULL DEFAULT 'candidate',
            confidence REAL NOT NULL DEFAULT 0.5,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE item_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT NOT NULL, source_path TEXT NOT NULL,
            mention_text TEXT NOT NULL, context TEXT,
            evidence_level TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            UNIQUE(item_name, source_path, mention_text, evidence_level)
        );
        CREATE TABLE ingest_chunks (
            source_path TEXT NOT NULL, content_hash TEXT NOT NULL,
            chunk_index INTEGER NOT NULL, chunk_count INTEGER NOT NULL,
            chunk_size INTEGER NOT NULL, checkpoint_schema INTEGER NOT NULL,
            result_json TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (
                source_path,
                content_hash,
                chunk_index,
                chunk_count,
                chunk_size,
                checkpoint_schema
            )
        );
        CREATE TABLE concept_compile_state (
            concept_name TEXT NOT NULL, source_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', error TEXT,
            compiled_at TEXT, updated_at TEXT NOT NULL,
            PRIMARY KEY (concept_name, source_path),
            CHECK (
                status IN (
                    'pending', 'failed', 'compiled', 'deferred_draft', 'deferred_manual_edit'
                )
            )
        );

        CREATE INDEX idx_raw_hash ON raw_notes(content_hash);
        CREATE INDEX idx_raw_status ON raw_notes(status);
        CREATE INDEX idx_concept_name ON concepts(name);
        CREATE INDEX idx_ingest_chunks_source ON ingest_chunks(source_path, content_hash);
        CREATE INDEX idx_concept_compile_status ON concept_compile_state(status, source_path);
        CREATE INDEX idx_concept_compile_name ON concept_compile_state(lower(concept_name));
        CREATE INDEX idx_rejections_concept ON rejections(concept);
        CREATE INDEX idx_alias_lookup ON concept_aliases(lower(alias));
        CREATE INDEX idx_items_kind ON knowledge_items(kind);
        CREATE INDEX idx_items_status ON knowledge_items(status);
        CREATE INDEX idx_mentions_item ON item_mentions(item_name);
        CREATE INDEX idx_mentions_source ON item_mentions(source_path);
        CREATE INDEX idx_wiki_articles_kind ON wiki_articles(kind);
        CREATE UNIQUE INDEX idx_wiki_articles_question_hash
            ON wiki_articles(question_hash) WHERE question_hash IS NOT NULL;

        INSERT INTO schema_version (id, version) VALUES (1, 7);

        INSERT INTO raw_notes (path, content_hash, status, summary, quality, language)
            VALUES ('raw/old.md', 'h1', 'ingested', 'pre-existing note', 'high', 'en');
        INSERT INTO concepts (name, source_path) VALUES ('Test Concept', 'raw/old.md');
        INSERT INTO wiki_articles
            (path, title, sources, content_hash, created_at, updated_at, is_draft, kind)
            VALUES ('wiki/Test.md', 'Test Concept', '["raw/old.md"]', 'wh1',
                    '2024-01-01T00:00:00', '2024-01-01T00:00:00', 0, 'concept');
        """
    )
    conn.commit()
    conn.close()


def test_fresh_db_is_at_v10(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    assert version == _CURRENT_SCHEMA_VERSION == 25
    conn.close()


def test_fresh_db_has_all_v10_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    missing = V10_TABLES - tables
    assert not missing, f"missing v10 tables: {missing}"
    assert "telemetry_events" not in tables
    assert "telemetry_daily_rollups" not in tables
    conn.close()


def test_fresh_db_has_v10_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    all_indexes = set()
    for table in [
        "source_segments",
        "source_warnings",
        "metric_events",
        "metric_daily_rollups",
        "generated_assets",
        "wiki_articles",
    ]:
        all_indexes |= _table_indexes(conn, table)
    missing = V10_INDEXES - all_indexes
    assert not missing, f"missing v10 indexes: {missing}"
    conn.close()


def test_metric_events_success_check_constraint(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO metric_events (ts, event_type, success) VALUES (?, ?, ?)",
        ("2026-05-13T00:00:00", "llm_call", 0),
    )
    conn.execute(
        "INSERT INTO metric_events (ts, event_type, success) VALUES (?, ?, ?)",
        ("2026-05-13T00:00:01", "llm_call", 1),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO metric_events (ts, event_type, success) VALUES (?, ?, ?)",
            ("2026-05-13T00:00:02", "llm_call", 2),
        )
        conn.commit()
    conn.close()


def test_v8_to_v10_upgrade_preserves_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _build_v8_db(db_path)

    db = StateDB(db_path)

    conn = sqlite3.connect(db_path)
    version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    assert version == 25
    assert conn.execute("SELECT COUNT(*) FROM raw_notes").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM wiki_articles").fetchone()[0] == 1

    # v22 rebuild: the legacy concepts row ('Test Concept', 'raw/old.md') must be
    # re-keyed onto the entity backfilled at v18 — name carried as a cache, source
    # edge resolvable by entity identity, not the name string.
    crow = conn.execute("SELECT entity_id, source_path, name FROM concepts").fetchone()
    assert crow is not None
    entity_id, source_path, name = crow
    assert name == "Test Concept"
    assert source_path == "raw/old.md"
    assert entity_id and entity_id == db.entity_id_for_name("Test Concept")
    assert db.get_sources_for_concept("Test Concept") == ["raw/old.md"]
    conn.close()


def test_v8_to_v10_upgrade_backfills_article_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _build_v8_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        (
            "INSERT INTO wiki_articles "
            "(path, title, sources, content_hash, created_at, updated_at, is_draft, kind) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        [
            (
                f"wiki/Test-{i}.md",
                f"Test {i}",
                "[]",
                f"hash-{i}",
                "2024-01-01T00:00:00",
                "2024-01-01T00:00:00",
                0,
                "concept",
            )
            for i in range(5)
        ],
    )
    conn.commit()
    conn.close()

    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT article_id FROM wiki_articles ORDER BY path").fetchall()
    article_ids = [row[0] for row in rows]
    assert all(article_ids)
    assert len(article_ids) == len(set(article_ids))
    assert all(ARTICLE_ID_RE.match(article_id) for article_id in article_ids)
    conn.close()


def test_ulid_backfill_is_atomic_on_failure(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _build_v8_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        (
            "INSERT INTO wiki_articles "
            "(path, title, sources, content_hash, created_at, updated_at, is_draft, kind) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        [
            (
                f"wiki/Test-{i}.md",
                f"Test {i}",
                "[]",
                f"hash-{i}",
                "2024-01-01T00:00:00",
                "2024-01-01T00:00:00",
                0,
                "concept",
            )
            for i in range(5)
        ],
    )
    conn.commit()
    conn.close()

    values = iter(["01TESTULID0000000000000001", "01TESTULID0000000000000002"])

    def fail_on_third() -> str:
        try:
            return next(values)
        except StopIteration as exc:
            raise RuntimeError("boom") from exc

    with patch("synto.state._generate_article_id", side_effect=fail_on_third):
        with pytest.raises(RuntimeError, match="boom"):
            StateDB(db_path)

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT article_id FROM wiki_articles ORDER BY path").fetchall()
    assert all(row[0] is None for row in rows)
    conn.close()


def test_re_running_migration_does_not_create_duplicate_ulids(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _build_v8_db(db_path)

    StateDB(db_path)
    first = StateDB(db_path)
    first_ids = [row.article_id for row in first.list_articles()]
    first.close()

    second = StateDB(db_path)
    second_ids = [row.article_id for row in second.list_articles()]
    second.close()
    assert first_ids == second_ids


def test_article_id_is_ulid_unique(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    for i in range(100):
        db.upsert_article(
            WikiArticleRecord(
                path=f"wiki/A-{i}.md",
                title=f"A {i}",
                sources=["raw/x.md"],
                content_hash=f"hash-{i}",
                status="published",
            )
        )

    article_ids = [row.article_id for row in db.list_articles()]
    assert all(article_ids)
    assert len(article_ids) == len(set(article_ids))
    assert all(ARTICLE_ID_RE.match(article_id) for article_id in article_ids if article_id)


def test_article_id_persists_across_reopens(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    db = StateDB(db_path)
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/A.md",
            title="A",
            sources=["raw/x.md"],
            content_hash="hash",
            status="published",
        )
    )
    first = db.get_article("wiki/A.md")
    assert first is not None and first.article_id is not None
    db.close()

    reopened = StateDB(db_path)
    second = reopened.get_article("wiki/A.md")
    assert second is not None
    assert second.article_id == first.article_id


def test_article_id_unique_index_enforces(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    db = StateDB(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        (
            "INSERT INTO wiki_articles "
            "(path, title, sources, content_hash, created_at, updated_at, "
            "status, kind, article_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        (
            "wiki/A.md",
            "A",
            "[]",
            "h1",
            "2024-01-01T00:00:00",
            "2024-01-01T00:00:00",
            "published",
            "concept",
            "01TESTULID0000000000000001",
        ),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            (
                "INSERT INTO wiki_articles "
                "(path, title, sources, content_hash, created_at, updated_at, "
                "status, kind, article_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                "wiki/B.md",
                "B",
                "[]",
                "h2",
                "2024-01-01T00:00:00",
                "2024-01-01T00:00:00",
                0,
                "concept",
                "01TESTULID0000000000000001",
            ),
        )
        conn.commit()
    conn.close()
    db.close()


def test_v8_indexes_survive_upgrade(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _build_v8_db(db_path)
    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    all_indexes = set()
    for table in [
        "raw_notes",
        "concepts",
        "wiki_articles",
        "rejections",
        "stubs",
        "blocked_concepts",
        "concept_aliases",
        "knowledge_items",
        "item_mentions",
        "ingest_chunks",
        "concept_compile_state",
    ]:
        all_indexes |= _table_indexes(conn, table)
    missing = V7_ERA_INDEXES - all_indexes
    assert not missing, f"v10 upgrade dropped existing indexes: {missing}"
    conn.close()


def test_wiki_article_record_loads_from_v10_schema(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/A.md",
            title="A",
            sources=["raw/x.md"],
            content_hash="hash",
            status="published",
            last_compile_pipeline='{"version": 1}',
        )
    )

    record = db.get_article("wiki/A.md")
    assert record is not None
    assert record.article_id is not None
    assert record.last_compile_pipeline == '{"version": 1}'


def test_fresh_db_has_new_wiki_article_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    cols = _table_columns(conn, "wiki_articles")
    assert {"article_id", "last_compile_pipeline"}.issubset(cols)
    conn.close()


def test_v9_to_v10_upgrade_drops_legacy_telemetry_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    StateDB(db_path).close()

    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE schema_version SET version = 9 WHERE id = 1")
    conn.execute("DROP INDEX IF EXISTS idx_metric_events_ts")
    conn.execute("DROP INDEX IF EXISTS idx_metric_events_type_ts")
    conn.execute("DROP INDEX IF EXISTS idx_metric_daily_rollups_day")
    conn.execute("DROP TABLE metric_events")
    conn.execute("DROP TABLE metric_daily_rollups")
    conn.executescript(
        """
        CREATE TABLE telemetry_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            vault_id TEXT,
            event_type TEXT NOT NULL,
            model TEXT,
            tier TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            latency_ms INTEGER,
            success INTEGER CHECK(success IN (0, 1)),
            source_id_hash TEXT,
            metadata_json TEXT
        );
        CREATE TABLE telemetry_daily_rollups (
            day TEXT NOT NULL,
            vault_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            tier TEXT NOT NULL DEFAULT '',
            calls INTEGER NOT NULL DEFAULT 0,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            latency_ms_total INTEGER NOT NULL DEFAULT 0,
            successes INTEGER NOT NULL DEFAULT 0,
            failures INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (day, vault_id, event_type, tier)
        );
        CREATE INDEX idx_telemetry_events_ts ON telemetry_events(ts);
        CREATE INDEX idx_telemetry_events_type_ts ON telemetry_events(event_type, ts);
        CREATE INDEX idx_telemetry_daily_rollups_day ON telemetry_daily_rollups(day);
        INSERT INTO telemetry_daily_rollups (day, vault_id, event_type, tier, calls) VALUES (
            '2026-05-15', 'vault', 'llm_call', 'fast', 2
        );
        """
    )
    conn.commit()
    conn.close()

    StateDB(db_path).close()

    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "metric_events" in tables
    assert "metric_daily_rollups" in tables
    assert "telemetry_events" not in tables
    assert "telemetry_daily_rollups" not in tables
    conn.close()
