"""
SQLite-backed state tracking for the pipeline.

Tracks raw note processing status and wiki article lineage.
Handles: dedup via content hash, partial failure recovery, resume.

Schema versioning: schema_version table tracks migration level.
  v1 — initial (summary/quality columns on raw_notes)
  v2 — rejections, stubs, blocked_concepts tables; approved_at/approval_notes on wiki_articles
  v3 — language column on raw_notes
  v4 — concept_aliases table; backfill from existing concept titles
  v5 — knowledge_items + item_mentions tables; backfill existing concepts
  v6 — ingest_chunks + concept_compile_state tables; backfill compile state from articles
  v7 — synthesis article metadata on wiki_articles
  v8 — source metadata columns on raw_notes
  v9 — source segmenting, generated assets, and pre-public telemetry tables
  v10 — public metrics tables; drops old telemetry tables without preserving rows
  v11 — compile_runs lineage tracking
  v12 — llm_cache for semantic cache
  v13 — concept_occurrences for term extraction
  v14 — drop dead v8 metadata columns from raw_notes (superseded by
         source_documents in v9): source_type, origin_uri, imported_at,
         normalized_hash, extractor_version
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

from .models import ItemMentionRecord, KnowledgeItemRecord, RawNoteRecord, WikiArticleRecord

_CURRENT_SCHEMA_VERSION = 16
_CHECKPOINT_SCHEMA_VERSION = 2

_CROCKFORD32_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

# Full current schema — idempotent (CREATE IF NOT EXISTS).
# Fresh DBs get all tables + columns from here. Existing DBs use _VERSIONED_MIGRATIONS.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    id      INTEGER PRIMARY KEY CHECK(id = 1),
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_notes (
    path              TEXT PRIMARY KEY,
    content_hash      TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'new',
    summary           TEXT,
    quality           TEXT,
    language          TEXT,
    ingested_at       TEXT,
    compiled_at       TEXT,
    error             TEXT,
    prompt_version    TEXT
);

CREATE TABLE IF NOT EXISTS concepts (
    name        TEXT NOT NULL,
    source_path TEXT NOT NULL,
    PRIMARY KEY (name, source_path)
);

CREATE TABLE IF NOT EXISTS wiki_articles (
    path           TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    sources        TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'draft'
                       CHECK (status IN ('draft','verified','published')),
    approved_at    TEXT,
    approval_notes TEXT,
    kind           TEXT NOT NULL DEFAULT 'concept',
    question_hash  TEXT,
    synthesis_sources TEXT,
    synthesis_source_hashes TEXT,
    article_id     TEXT,
    last_compile_pipeline TEXT
);

CREATE TABLE IF NOT EXISTS source_documents (
    id                TEXT PRIMARY KEY,
    source_type       TEXT NOT NULL DEFAULT 'unknown_text',
    origin_uri        TEXT,
    title             TEXT,
    imported_at       TEXT,
    raw_hash          TEXT,
    normalized_hash   TEXT,
    extractor_version TEXT,
    license           TEXT,
    redistribution    TEXT NOT NULL DEFAULT 'unknown',
    metadata_json     TEXT
);

CREATE TABLE IF NOT EXISTS source_segments (
    id                  TEXT PRIMARY KEY,
    identity            TEXT NOT NULL,
    ordinal             INTEGER NOT NULL,
    source_id           TEXT NOT NULL,
    structural_locator  TEXT NOT NULL,
    content_hash        TEXT NOT NULL,
    text                TEXT NOT NULL,
    section_path_json   TEXT,
    page_start          INTEGER,
    page_end            INTEGER,
    char_start          INTEGER,
    char_end            INTEGER,
    metadata_json       TEXT,
    FOREIGN KEY (source_id) REFERENCES source_documents(id)
);

CREATE TABLE IF NOT EXISTS source_warnings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id    TEXT NOT NULL,
    severity     TEXT NOT NULL CHECK(severity IN ('info', 'warning', 'error')),
    category     TEXT NOT NULL,
    message      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES source_documents(id)
);

CREATE TABLE IF NOT EXISTS metric_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    vault_id          TEXT,
    event_type        TEXT NOT NULL,
    model             TEXT,
    tier              TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    latency_ms        INTEGER,
    success           INTEGER CHECK(success IN (0, 1)),
    source_id_hash    TEXT,
    metadata_json     TEXT
);

CREATE TABLE IF NOT EXISTS metric_daily_rollups (
    day               TEXT NOT NULL,
    vault_id          TEXT NOT NULL,
    event_type        TEXT NOT NULL,
    tier              TEXT NOT NULL DEFAULT '',
    calls             INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms_total  INTEGER NOT NULL DEFAULT 0,
    successes         INTEGER NOT NULL DEFAULT 0,
    failures          INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, vault_id, event_type, tier)
);

CREATE TABLE IF NOT EXISTS generated_assets (
    path                TEXT PRIMARY KEY,
    source_id           TEXT NOT NULL,
    asset_type          TEXT NOT NULL,
    master_path         TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    last_referenced_at  TEXT,
    referenced_by_json  TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (source_id) REFERENCES source_documents(id)
);

CREATE TABLE IF NOT EXISTS rejections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    concept       TEXT NOT NULL,
    feedback      TEXT NOT NULL,
    rejected_body TEXT,
    rejected_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stubs (
    concept    TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'auto'
);

CREATE TABLE IF NOT EXISTS blocked_concepts (
    concept    TEXT PRIMARY KEY,
    blocked_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS concept_aliases (
    concept_name TEXT NOT NULL,
    alias        TEXT NOT NULL,
    PRIMARY KEY (concept_name, alias)
);

CREATE TABLE IF NOT EXISTS knowledge_items (
    name       TEXT PRIMARY KEY,
    kind       TEXT NOT NULL DEFAULT 'ambiguous',
    subtype    TEXT,
    status     TEXT NOT NULL DEFAULT 'candidate',
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS item_mentions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name      TEXT NOT NULL,
    source_path    TEXT NOT NULL,
    mention_text   TEXT NOT NULL,
    context        TEXT,
    evidence_level TEXT NOT NULL,
    confidence     REAL NOT NULL DEFAULT 0.5,
    UNIQUE(item_name, source_path, mention_text, evidence_level)
);

CREATE TABLE IF NOT EXISTS ingest_chunks (
    source_path        TEXT NOT NULL,
    content_hash       TEXT NOT NULL,
    chunk_index        INTEGER NOT NULL,
    chunk_count        INTEGER NOT NULL,
    chunk_size         INTEGER NOT NULL,
    checkpoint_schema  INTEGER NOT NULL,
    result_json        TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    PRIMARY KEY (source_path, content_hash, chunk_index, chunk_count, chunk_size, checkpoint_schema)
);

CREATE TABLE IF NOT EXISTS concept_compile_state (
    concept_name TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    error        TEXT,
    compiled_at  TEXT,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (concept_name, source_path),
    CHECK (status IN ('pending', 'failed', 'compiled', 'deferred_draft', 'deferred_manual_edit'))
);

CREATE INDEX IF NOT EXISTS idx_raw_hash ON raw_notes(content_hash);
CREATE INDEX IF NOT EXISTS idx_raw_status ON raw_notes(status);
CREATE INDEX IF NOT EXISTS idx_concept_name ON concepts(name);
CREATE INDEX IF NOT EXISTS idx_ingest_chunks_source ON ingest_chunks(source_path, content_hash);
CREATE INDEX IF NOT EXISTS idx_concept_compile_status ON concept_compile_state(status, source_path);
CREATE INDEX IF NOT EXISTS idx_concept_compile_name ON concept_compile_state(lower(concept_name));
CREATE INDEX IF NOT EXISTS idx_rejections_concept ON rejections(concept);
CREATE INDEX IF NOT EXISTS idx_alias_lookup ON concept_aliases(lower(alias));
CREATE INDEX IF NOT EXISTS idx_items_kind ON knowledge_items(kind);
CREATE INDEX IF NOT EXISTS idx_items_status ON knowledge_items(status);
CREATE INDEX IF NOT EXISTS idx_mentions_item ON item_mentions(item_name);
CREATE INDEX IF NOT EXISTS idx_mentions_source ON item_mentions(source_path);
CREATE INDEX IF NOT EXISTS idx_source_segments_source ON source_segments(source_id);
CREATE INDEX IF NOT EXISTS idx_source_segments_identity ON source_segments(identity);
CREATE INDEX IF NOT EXISTS idx_source_warnings_source ON source_warnings(source_id);
CREATE INDEX IF NOT EXISTS idx_metric_events_ts ON metric_events(ts);
CREATE INDEX IF NOT EXISTS idx_metric_events_type_ts ON metric_events(event_type, ts);
CREATE INDEX IF NOT EXISTS idx_metric_daily_rollups_day ON metric_daily_rollups(day);
CREATE INDEX IF NOT EXISTS idx_generated_assets_source ON generated_assets(source_id);

CREATE TABLE IF NOT EXISTS compile_runs (
    run_ulid        TEXT PRIMARY KEY,
    pipeline_json   TEXT NOT NULL,
    fast_model      TEXT NOT NULL,
    heavy_model     TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    article_count   INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    total_cost_usd  REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key    TEXT PRIMARY KEY,
    model        TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    last_hit_at  TEXT,
    hit_count    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS concept_occurrences (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_name      TEXT NOT NULL,
    source_segment_id TEXT NOT NULL,
    ordinal           INTEGER NOT NULL DEFAULT 0,
    confidence        REAL NOT NULL DEFAULT 1.0,
    extraction_run    TEXT,
    UNIQUE(concept_name, source_segment_id)
);

CREATE INDEX IF NOT EXISTS idx_concept_occurrences_concept ON concept_occurrences(concept_name);
"""

# Migrations keyed by version they bring the DB to.
_VERSIONED_MIGRATIONS: dict[int, list[str]] = {
    1: [
        # v0.1: add summary/quality columns to raw_notes (were missing in earliest schema)
        "ALTER TABLE raw_notes ADD COLUMN summary TEXT",
        "ALTER TABLE raw_notes ADD COLUMN quality TEXT",
    ],
    2: [
        # v0.2: new tables and columns
        """CREATE TABLE IF NOT EXISTS rejections (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               concept TEXT NOT NULL,
               feedback TEXT NOT NULL,
               rejected_body TEXT,
               rejected_at TEXT NOT NULL
           )""",
        "CREATE INDEX IF NOT EXISTS idx_rejections_concept ON rejections(concept)",
        """CREATE TABLE IF NOT EXISTS stubs (
               concept TEXT PRIMARY KEY,
               created_at TEXT NOT NULL,
               source TEXT NOT NULL DEFAULT 'auto'
           )""",
        """CREATE TABLE IF NOT EXISTS blocked_concepts (
               concept TEXT PRIMARY KEY,
               blocked_at TEXT NOT NULL
           )""",
        "ALTER TABLE wiki_articles ADD COLUMN approved_at TEXT",
        "ALTER TABLE wiki_articles ADD COLUMN approval_notes TEXT",
    ],
    3: [
        "ALTER TABLE raw_notes ADD COLUMN language TEXT",
    ],
    4: [
        """CREATE TABLE IF NOT EXISTS concept_aliases (
               concept_name TEXT NOT NULL,
               alias        TEXT NOT NULL,
               PRIMARY KEY (concept_name, alias)
           )""",
        "CREATE INDEX IF NOT EXISTS idx_alias_lookup ON concept_aliases(lower(alias))",
    ],
    5: [
        """CREATE TABLE IF NOT EXISTS knowledge_items (
               name TEXT PRIMARY KEY,
               kind TEXT NOT NULL DEFAULT 'ambiguous',
               subtype TEXT,
               status TEXT NOT NULL DEFAULT 'candidate',
               confidence REAL NOT NULL DEFAULT 0.5,
               created_at TEXT NOT NULL,
               updated_at TEXT NOT NULL
           )""",
        """CREATE TABLE IF NOT EXISTS item_mentions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               item_name TEXT NOT NULL,
               source_path TEXT NOT NULL,
               mention_text TEXT NOT NULL,
               context TEXT,
               evidence_level TEXT NOT NULL,
               confidence REAL NOT NULL DEFAULT 0.5,
               UNIQUE(item_name, source_path, mention_text, evidence_level)
           )""",
        "CREATE INDEX IF NOT EXISTS idx_items_kind ON knowledge_items(kind)",
        "CREATE INDEX IF NOT EXISTS idx_items_status ON knowledge_items(status)",
        "CREATE INDEX IF NOT EXISTS idx_mentions_item ON item_mentions(item_name)",
        "CREATE INDEX IF NOT EXISTS idx_mentions_source ON item_mentions(source_path)",
    ],
    6: [
        """CREATE TABLE IF NOT EXISTS ingest_chunks (
               source_path        TEXT NOT NULL,
               content_hash       TEXT NOT NULL,
               chunk_index        INTEGER NOT NULL,
               chunk_count        INTEGER NOT NULL,
               chunk_size         INTEGER NOT NULL,
               checkpoint_schema  INTEGER NOT NULL,
               result_json        TEXT NOT NULL,
               created_at         TEXT NOT NULL,
               updated_at         TEXT NOT NULL,
               PRIMARY KEY (
                   source_path,
                   content_hash,
                   chunk_index,
                   chunk_count,
                   chunk_size,
                   checkpoint_schema
               )
           )""",
        (
            "CREATE INDEX IF NOT EXISTS idx_ingest_chunks_source "
            "ON ingest_chunks(source_path, content_hash)"
        ),
        """CREATE TABLE IF NOT EXISTS concept_compile_state (
               concept_name TEXT NOT NULL,
               source_path  TEXT NOT NULL,
               status       TEXT NOT NULL DEFAULT 'pending',
               error        TEXT,
               compiled_at  TEXT,
               updated_at   TEXT NOT NULL,
               PRIMARY KEY (concept_name, source_path),
               CHECK (
                   status IN (
                       'pending',
                       'failed',
                       'compiled',
                       'deferred_draft',
                       'deferred_manual_edit'
                   )
               )
           )""",
        (
            "CREATE INDEX IF NOT EXISTS idx_concept_compile_status "
            "ON concept_compile_state(status, source_path)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_concept_compile_name "
            "ON concept_compile_state(lower(concept_name))"
        ),
    ],
    7: [
        "ALTER TABLE wiki_articles ADD COLUMN kind TEXT NOT NULL DEFAULT 'concept'",
        "ALTER TABLE wiki_articles ADD COLUMN question_hash TEXT",
        "ALTER TABLE wiki_articles ADD COLUMN synthesis_sources TEXT",
        "ALTER TABLE wiki_articles ADD COLUMN synthesis_source_hashes TEXT",
        "CREATE INDEX IF NOT EXISTS idx_wiki_articles_kind ON wiki_articles(kind)",
        (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_wiki_articles_question_hash "
            "ON wiki_articles(question_hash) WHERE question_hash IS NOT NULL"
        ),
    ],
    8: [
        # V6 Phase 0: additive raw_notes columns for source-type metadata.
        # See CLAUDE-4.7-HIGH_ROADMAP_V6.md §10.7 step 1.
        "ALTER TABLE raw_notes ADD COLUMN source_type TEXT NOT NULL DEFAULT 'notes'",
        "ALTER TABLE raw_notes ADD COLUMN origin_uri TEXT",
        "ALTER TABLE raw_notes ADD COLUMN imported_at TEXT",
        "ALTER TABLE raw_notes ADD COLUMN normalized_hash TEXT",
        "ALTER TABLE raw_notes ADD COLUMN extractor_version TEXT",
        "ALTER TABLE raw_notes ADD COLUMN prompt_version TEXT",
    ],
    9: [
        """CREATE TABLE IF NOT EXISTS source_documents (
               id TEXT PRIMARY KEY,
               source_type TEXT NOT NULL DEFAULT 'unknown_text',
               origin_uri TEXT,
               title TEXT,
               imported_at TEXT,
               raw_hash TEXT,
               normalized_hash TEXT,
               extractor_version TEXT,
               license TEXT,
               redistribution TEXT NOT NULL DEFAULT 'unknown',
               metadata_json TEXT
           )""",
        """CREATE TABLE IF NOT EXISTS source_segments (
               id TEXT PRIMARY KEY,
               identity TEXT NOT NULL,
               ordinal INTEGER NOT NULL,
               source_id TEXT NOT NULL,
               structural_locator TEXT NOT NULL,
               content_hash TEXT NOT NULL,
               text TEXT NOT NULL,
               section_path_json TEXT,
               page_start INTEGER,
               page_end INTEGER,
               char_start INTEGER,
               char_end INTEGER,
               metadata_json TEXT,
               FOREIGN KEY (source_id) REFERENCES source_documents(id)
           )""",
        """CREATE TABLE IF NOT EXISTS source_warnings (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               source_id TEXT NOT NULL,
               severity TEXT NOT NULL CHECK(severity IN ('info', 'warning', 'error')),
               category TEXT NOT NULL,
               message TEXT NOT NULL,
               created_at TEXT NOT NULL,
               FOREIGN KEY (source_id) REFERENCES source_documents(id)
           )""",
        """CREATE TABLE IF NOT EXISTS telemetry_events (
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
           )""",
        """CREATE TABLE IF NOT EXISTS telemetry_daily_rollups (
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
           )""",
        """CREATE TABLE IF NOT EXISTS generated_assets (
               path TEXT PRIMARY KEY,
               source_id TEXT NOT NULL,
               asset_type TEXT NOT NULL,
               master_path TEXT NOT NULL,
               created_at TEXT NOT NULL,
               last_referenced_at TEXT,
               referenced_by_json TEXT NOT NULL DEFAULT '[]',
               FOREIGN KEY (source_id) REFERENCES source_documents(id)
           )""",
        "ALTER TABLE wiki_articles ADD COLUMN article_id TEXT",
        "ALTER TABLE wiki_articles ADD COLUMN last_compile_pipeline TEXT",
        "CREATE INDEX IF NOT EXISTS idx_source_segments_source ON source_segments(source_id)",
        "CREATE INDEX IF NOT EXISTS idx_source_segments_identity ON source_segments(identity)",
        "CREATE INDEX IF NOT EXISTS idx_source_warnings_source ON source_warnings(source_id)",
        "CREATE INDEX IF NOT EXISTS idx_telemetry_events_ts ON telemetry_events(ts)",
        (
            "CREATE INDEX IF NOT EXISTS idx_telemetry_events_type_ts "
            "ON telemetry_events(event_type, ts)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_telemetry_daily_rollups_day "
            "ON telemetry_daily_rollups(day)"
        ),
        "CREATE INDEX IF NOT EXISTS idx_generated_assets_source ON generated_assets(source_id)",
        (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_wiki_articles_article_id "
            "ON wiki_articles(article_id) WHERE article_id IS NOT NULL"
        ),
    ],
    10: [
        """CREATE TABLE IF NOT EXISTS metric_events (
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
           )""",
        """CREATE TABLE IF NOT EXISTS metric_daily_rollups (
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
           )""",
        "CREATE INDEX IF NOT EXISTS idx_metric_events_ts ON metric_events(ts)",
        ("CREATE INDEX IF NOT EXISTS idx_metric_events_type_ts ON metric_events(event_type, ts)"),
        ("CREATE INDEX IF NOT EXISTS idx_metric_daily_rollups_day ON metric_daily_rollups(day)"),
    ],
    11: [
        """CREATE TABLE IF NOT EXISTS compile_runs (
               run_ulid        TEXT PRIMARY KEY,
               pipeline_json   TEXT NOT NULL,
               fast_model      TEXT NOT NULL,
               heavy_model     TEXT NOT NULL,
               started_at      TEXT NOT NULL,
               finished_at     TEXT,
               article_count   INTEGER NOT NULL DEFAULT 0,
               total_tokens    INTEGER NOT NULL DEFAULT 0,
               total_cost_usd  REAL NOT NULL DEFAULT 0.0
           )""",
    ],
    12: [
        """CREATE TABLE IF NOT EXISTS llm_cache (
               cache_key     TEXT PRIMARY KEY,
               model         TEXT NOT NULL,
               response_json TEXT NOT NULL,
               created_at    TEXT NOT NULL,
               last_hit_at   TEXT,
               hit_count     INTEGER NOT NULL DEFAULT 0
           )""",
    ],
    13: [
        """CREATE TABLE IF NOT EXISTS concept_occurrences (
               id                INTEGER PRIMARY KEY AUTOINCREMENT,
               concept_name      TEXT NOT NULL,
               source_segment_id TEXT NOT NULL,
               ordinal           INTEGER NOT NULL DEFAULT 0,
               confidence        REAL NOT NULL DEFAULT 1.0,
               extraction_run    TEXT,
               UNIQUE(concept_name, source_segment_id)
           )""",
        "CREATE INDEX IF NOT EXISTS idx_concept_occurrences_concept "
        "ON concept_occurrences(concept_name)",
    ],
    14: [],  # all v14 work happens in the post-hook below for atomicity
    15: [],  # all v15 work happens in the post-hook below for atomicity
    16: [],  # all v16 work happens in the post-hook below for atomicity
}


def _b32_encode_48bit(value: int) -> str:
    if value < 0 or value >= 1 << 48:
        raise ValueError("value out of range for 48-bit base32 encoding")

    chars: list[str] = []
    for _ in range(10):
        chars.append(_CROCKFORD32_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def _generate_article_id() -> str:
    try:
        import ulid

        return str(ulid.ULID())
    except ImportError:
        timestamp_ms = int(time.time() * 1000)
        random_suffix = uuid.uuid4().hex[:16].upper()
        return f"{_b32_encode_48bit(timestamp_ms)}{random_suffix}"


def _database_needs_upgrade(db_path: Path) -> bool:
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        version = int(row[0]) if row else 0
        return version < _CURRENT_SCHEMA_VERSION
    except sqlite3.DatabaseError:
        return True
    finally:
        conn.close()


class SynthesisInsertConflictError(RuntimeError):
    """Base error for synthesis insert conflicts."""


class DuplicateSynthesisQuestionHashError(SynthesisInsertConflictError):
    """Raised when a synthesis question_hash already exists."""


class DuplicateArticlePathError(SynthesisInsertConflictError):
    """Raised when a synthesis article path already exists."""


class StateDB:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._tx_depth = 0
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()

    @classmethod
    def open_readonly(cls, db_path: Path) -> StateDB:
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(db_path)
        if _database_needs_upgrade(db_path):
            writable = cls(db_path)
            writable.close()

        self = cls.__new__(cls)
        uri = f"file:{db_path.resolve()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._tx_depth = 0
        return self

    def _migrate(self) -> None:
        """Apply schema migrations in version order. Idempotent."""
        # SQLite >= 3.35 is required for the v14 migration's DROP COLUMN.
        # Modern Python stdlib on macOS/Linux ships ≥ 3.40; this guard turns a
        # mid-migration SQL error into a clear startup message.
        if sqlite3.sqlite_version_info < (3, 35, 0):
            raise RuntimeError(
                f"synto requires SQLite >= 3.35.0 (found {sqlite3.sqlite_version}). "
                f"Upgrade Python or system SQLite."
            )
        # Upgrade schema_version table if it lacks the id column (pre-v0.2 DBs).
        sv_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(schema_version)").fetchall()}
        if sv_cols and "id" not in sv_cols:
            # Read the current version from the old single-column table, then
            # recreate it with the proper constraint.
            old_row = self._conn.execute(
                "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            old_version = old_row[0] if old_row else None
            self._conn.executescript(
                "DROP TABLE schema_version;"
                "CREATE TABLE schema_version "
                "(id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL);"
            )
            if old_version is not None:
                self._conn.execute(
                    "INSERT INTO schema_version (id, version) VALUES (1, ?)", (old_version,)
                )
            self._conn.commit()

        # Use ORDER BY rowid DESC LIMIT 1 to be robust against legacy DBs that
        # accumulated multiple rows before the id=1 uniqueness constraint was added.
        row = self._conn.execute(
            "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
        ).fetchone()

        if row is None:
            # No version record yet. Determine starting state by inspecting schema:
            # Check that all columns from the current schema version exist so we
            # don't skip migrations on a partially-upgraded DB (e.g. v2 DB with
            # approved_at but no language column).
            wiki_cols = {
                r[1] for r in self._conn.execute("PRAGMA table_info(wiki_articles)").fetchall()
            }
            note_cols = {
                r[1] for r in self._conn.execute("PRAGMA table_info(raw_notes)").fetchall()
            }
            if "approved_at" in wiki_cols and "language" in note_cols:
                # DB has v3 features but no version record — stamp as v3 so the v4
                # migration (backfill) still runs through the loop below.
                with self._tx():
                    self._conn.execute(
                        "INSERT OR REPLACE INTO schema_version (id, version) VALUES (1, 3)"
                    )
                current_version = 3
            else:
                # Existing DB with no version tracking — start from 0, apply all migrations.
                with self._tx():
                    self._conn.execute(
                        "INSERT OR REPLACE INTO schema_version (id, version) VALUES (1, 0)"
                    )
                current_version = 0
        else:
            current_version = row[0]

        if current_version > _CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"On-disk DB schema_version={current_version} is newer than this "
                f"synto binary (supports v{_CURRENT_SCHEMA_VERSION}). Upgrade synto."
            )
        if current_version >= _CURRENT_SCHEMA_VERSION:
            return

        for version, stmts in sorted(_VERSIONED_MIGRATIONS.items()):
            if current_version >= version:
                continue
            for stmt in stmts:
                try:
                    self._conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            if version == 4:
                self._backfill_aliases_v4()
            if version == 5:
                self._backfill_items_v5()
            if version == 6:
                self._validate_v6_tables()
                self._backfill_compile_state_v6()
            if version == 9:
                self._backfill_article_ids_v9()
            if version == 10:
                self._drop_legacy_telemetry_tables_v10()
            if version == 14:
                self._drop_zombie_v8_columns_v14()
            if version == 15:
                self._apply_status_column_v15()
            if version == 16:
                self._create_source_segments_fts_v16()
            with self._tx():
                self._conn.execute(
                    "INSERT OR REPLACE INTO schema_version (id, version) VALUES (1, ?)",
                    (version,),
                )
            current_version = version

    def _backfill_article_ids_v9(self) -> None:
        rows = self._conn.execute(
            "SELECT path FROM wiki_articles WHERE article_id IS NULL ORDER BY path"
        ).fetchall()
        if not rows:
            return

        with self._tx():
            for row in rows:
                self._conn.execute(
                    "UPDATE wiki_articles SET article_id = ? WHERE path = ?",
                    (_generate_article_id(), row["path"]),
                )

    def _drop_legacy_telemetry_tables_v10(self) -> None:
        with self._tx():
            for index_name in [
                "idx_telemetry_events_ts",
                "idx_telemetry_events_type_ts",
                "idx_telemetry_daily_rollups_day",
            ]:
                self._conn.execute(f"DROP INDEX IF EXISTS {index_name}")
            self._conn.execute("DROP TABLE IF EXISTS telemetry_events")
            self._conn.execute("DROP TABLE IF EXISTS telemetry_daily_rollups")

    def _drop_zombie_v8_columns_v14(self) -> None:
        """Drop the v8 source-metadata columns from raw_notes.

        These columns were duplicated by source_documents (v9) and never
        populated on raw_notes. The cleanup was missed at v9 and the dead
        columns lured PR #11 into reading raw_notes.origin_uri (always
        NULL) instead of source_documents.origin_uri (the real store).

        Atomic via _tx(); idempotent via PRAGMA table_info probe so a
        re-run after partial failure is a no-op rather than an error.
        DROP COLUMN requires SQLite >= 3.35.0; the version guard at the
        top of _migrate enforces this.
        """
        existing = {r[1] for r in self._conn.execute("PRAGMA table_info(raw_notes)").fetchall()}
        targets = [
            "origin_uri",
            "imported_at",
            "normalized_hash",
            "extractor_version",
            "source_type",
        ]
        to_drop = [c for c in targets if c in existing]
        if not to_drop:
            return
        with self._tx():
            for col in to_drop:
                self._conn.execute(f"ALTER TABLE raw_notes DROP COLUMN {col}")

    def _apply_status_column_v15(self) -> None:
        """Add wiki_articles.status, backfill from is_draft, drop is_draft.

        Atomic via _tx(); idempotent via PRAGMA probe so a re-run after a
        partial failure is a no-op. DROP COLUMN requires SQLite >= 3.35
        (enforced at the top of _migrate). The CHECK constraint from the
        fresh CREATE TABLE is not added here — SQLite cannot add CHECK via
        ALTER. The Pydantic model is the enforcement point for migrated DBs.
        """
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(wiki_articles)").fetchall()}
        if "status" in cols and "is_draft" not in cols:
            return
        with self._tx():
            if "status" not in cols:
                self._conn.execute(
                    "ALTER TABLE wiki_articles ADD COLUMN status TEXT NOT NULL DEFAULT 'draft'"
                )
            if "is_draft" in cols:
                self._conn.execute(
                    "UPDATE wiki_articles SET status = "
                    "CASE WHEN is_draft = 1 THEN 'draft' ELSE 'published' END"
                )
                self._conn.execute("ALTER TABLE wiki_articles DROP COLUMN is_draft")

    def _create_source_segments_fts_v16(self) -> None:
        """Create FTS5 virtual table and sync triggers for source_segments.

        External-content FTS5 (content='source_segments') keeps the index
        small — only the indexed text column is stored in the FTS table;
        other columns are joined back via rowid at query time.

        Triggers maintain the FTS index for INSERT/UPDATE/DELETE after the
        initial backfill. This covers every current and future extractor
        automatically without touching extractor code.

        Atomic via _tx(); idempotent via sqlite_master probe.
        """
        exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_segments_fts'"
        ).fetchone()
        if exists:
            return
        with self._tx():
            self._conn.execute("""
                CREATE VIRTUAL TABLE source_segments_fts USING fts5(
                    text,
                    content='source_segments',
                    content_rowid='rowid'
                )
            """)
            self._conn.execute("""
                CREATE TRIGGER source_segments_fts_ai AFTER INSERT ON source_segments BEGIN
                    INSERT INTO source_segments_fts(rowid, text) VALUES (new.rowid, new.text);
                END
            """)
            self._conn.execute("""
                CREATE TRIGGER source_segments_fts_ad AFTER DELETE ON source_segments BEGIN
                    INSERT INTO source_segments_fts(source_segments_fts, rowid, text)
                        VALUES ('delete', old.rowid, old.text);
                END
            """)
            self._conn.execute("""
                CREATE TRIGGER source_segments_fts_au AFTER UPDATE ON source_segments BEGIN
                    INSERT INTO source_segments_fts(source_segments_fts, rowid, text)
                        VALUES ('delete', old.rowid, old.text);
                    INSERT INTO source_segments_fts(rowid, text) VALUES (new.rowid, new.text);
                END
            """)
            # Backfill rows that existed before this migration ran.
            # Triggers only fire for future writes; historical rows need explicit insert.
            self._conn.execute("""
                INSERT INTO source_segments_fts(rowid, text)
                SELECT rowid, text FROM source_segments
            """)

    def _validate_v6_tables(self) -> None:
        expected_ingest = {
            "source_path",
            "content_hash",
            "chunk_index",
            "chunk_count",
            "chunk_size",
            "checkpoint_schema",
            "result_json",
            "created_at",
            "updated_at",
        }
        expected_compile = {
            "concept_name",
            "source_path",
            "status",
            "error",
            "compiled_at",
            "updated_at",
        }
        self._validate_or_recreate_table("ingest_chunks", expected_ingest)
        self._validate_or_recreate_table("concept_compile_state", expected_compile)

    def _validate_or_recreate_table(self, table: str, expected_cols: set[str]) -> None:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        cols = {row["name"] for row in rows}
        if cols == expected_cols:
            return
        row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        count = row[0] if row else 0
        if count != 0:
            raise sqlite3.OperationalError(
                f"Existing table '{table}' has incompatible schema. Back up your state.db and "
                "migrate manually."
            )
        with self._tx():
            self._conn.execute(f"DROP TABLE IF EXISTS {table}")
            if table == "ingest_chunks":
                self._conn.execute(_VERSIONED_MIGRATIONS[6][0])
                self._conn.execute(_VERSIONED_MIGRATIONS[6][1])
            elif table == "concept_compile_state":
                self._conn.execute(_VERSIONED_MIGRATIONS[6][2])
                self._conn.execute(_VERSIONED_MIGRATIONS[6][3])
                self._conn.execute(_VERSIONED_MIGRATIONS[6][4])

    def _backfill_compile_state_v6(self) -> None:
        self._ensure_compile_state_rows()
        alias_rows = self._conn.execute(
            "SELECT concept_name, alias FROM concept_aliases ORDER BY concept_name, alias"
        ).fetchall()
        alias_map: dict[str, set[str]] = {}
        for row in alias_rows:
            alias_map.setdefault(row["concept_name"], set()).add(row["alias"])

        articles = self.list_articles()
        for row in self._conn.execute("SELECT name, source_path FROM concepts").fetchall():
            concept_name = row["name"]
            source_path = row["source_path"]
            article = self._match_article_for_concept_v6(
                concept_name, source_path, articles, alias_map
            )
            if article is None:
                continue
            self.mark_concept_compile_state(concept_name, [source_path], "compiled")

        self._refresh_all_raw_compile_statuses()

    def _match_article_for_concept_v6(
        self,
        concept_name: str,
        source_path: str,
        articles: list[WikiArticleRecord],
        alias_map: dict[str, set[str]],
    ) -> WikiArticleRecord | None:
        concept_lower = concept_name.casefold()
        alias_lowers = {alias.casefold() for alias in alias_map.get(concept_name, set())}
        candidates: list[WikiArticleRecord] = []

        for article in articles:
            title_lower = article.title.casefold()
            path_stem = Path(article.path).stem.casefold()
            if title_lower == concept_lower or path_stem == concept_lower:
                candidates.append(article)
                continue
            if title_lower in alias_lowers or path_stem in alias_lowers:
                candidates.append(article)

        if not candidates:
            return None

        with_source_overlap = [a for a in candidates if source_path in a.sources]
        if len(with_source_overlap) == 1:
            return with_source_overlap[0]
        if len(with_source_overlap) > 1:
            return None

        without_sources = [a for a in candidates if not a.sources]
        if len(without_sources) == 1:
            return without_sources[0]
        return None

    def _ensure_compile_state_rows(self, source_path: str | None = None) -> None:
        query = "SELECT name, source_path FROM concepts"
        params: tuple[str, ...] = ()
        if source_path is not None:
            query += " WHERE source_path = ?"
            params = (source_path,)
        rows = self._conn.execute(query, params).fetchall()
        now = datetime.now().isoformat()
        with self._tx():
            for row in rows:
                self._conn.execute(
                    """INSERT OR IGNORE INTO concept_compile_state
                           (concept_name, source_path, status, error, compiled_at, updated_at)
                       VALUES (?, ?, 'pending', NULL, NULL, ?)""",
                    (row["name"], row["source_path"], now),
                )

    def _refresh_all_raw_compile_statuses(self) -> None:
        rows = self._conn.execute("SELECT path FROM raw_notes").fetchall()
        for row in rows:
            self.refresh_raw_compile_status(row["path"])

    def _backfill_aliases_v4(self) -> None:
        """Populate concept_aliases with deterministic aliases for all existing concepts.

        Uses the same logic as vault.generate_aliases: add lowercase variant + ALL_CAPS
        abbreviations from parenthetical notation (e.g. 'Program Counter (PC)' → 'PC').
        No LLM calls — fast and deterministic.
        """
        import re as _re

        abbr_pattern = _re.compile(r"\(([A-Z]{2,})\)")
        rows = self._conn.execute("SELECT DISTINCT name FROM concepts").fetchall()
        for (name,) in rows:
            aliases: list[str] = []
            lower = name.lower()
            if lower != name:
                aliases.append(lower)
            for m in abbr_pattern.finditer(name):
                abbr = m.group(1)
                if abbr.lower() != name.lower():
                    aliases.append(abbr)
            for alias in aliases:
                alias = alias.strip()
                if alias and alias.lower() != name.lower():
                    self._conn.execute(
                        "INSERT OR IGNORE INTO concept_aliases (concept_name, alias) VALUES (?, ?)",
                        (name, alias),
                    )
        self._conn.commit()

    def _backfill_items_v5(self) -> None:
        """Backfill existing concepts into the neutral knowledge item ledger."""
        rows = self._conn.execute("SELECT DISTINCT name FROM concepts").fetchall()
        now = datetime.now().isoformat()
        for (name,) in rows:
            self._conn.execute(
                """INSERT OR IGNORE INTO knowledge_items
                   (name, kind, subtype, status, confidence, created_at, updated_at)
                   VALUES (?, 'concept', NULL, 'confirmed', 1.0, ?, ?)""",
                (name, now, now),
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def schema_version(self) -> int:
        row = self._conn.execute(
            "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return int(row[0]) if row else 0

    @contextmanager
    def _tx(self):
        savepoint_name = f"synto_tx_{self._tx_depth}"
        nested = self._tx_depth > 0
        self._tx_depth += 1
        try:
            if nested:
                self._conn.execute(f"SAVEPOINT {savepoint_name}")
            yield self._conn
            if nested:
                self._conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            else:
                self._conn.commit()
        except Exception:
            if nested:
                self._conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                self._conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            else:
                self._conn.rollback()
            raise
        finally:
            self._tx_depth -= 1

    def _has_table(self, table_name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    # ── Raw Notes ─────────────────────────────────────────────────────────────

    def upsert_raw(self, record: RawNoteRecord) -> None:
        with self._tx():
            self._conn.execute(
                """INSERT INTO raw_notes
                       (path, content_hash, status, summary, quality, language,
                        ingested_at, compiled_at, error, prompt_version)
                   VALUES
                       (:path, :content_hash, :status, :summary, :quality, :language,
                        :ingested_at, :compiled_at, :error, :prompt_version)
                   ON CONFLICT(path) DO UPDATE SET
                       content_hash=excluded.content_hash,
                       status=excluded.status,
                       summary=excluded.summary,
                       quality=excluded.quality,
                       language=excluded.language,
                       ingested_at=excluded.ingested_at,
                       compiled_at=excluded.compiled_at,
                       error=excluded.error,
                       prompt_version=excluded.prompt_version""",
                {
                    "path": record.path,
                    "content_hash": record.content_hash,
                    "status": record.status,
                    "summary": record.summary,
                    "quality": record.quality,
                    "language": record.language,
                    "ingested_at": record.ingested_at.isoformat() if record.ingested_at else None,
                    "compiled_at": record.compiled_at.isoformat() if record.compiled_at else None,
                    "error": record.error,
                    "prompt_version": record.prompt_version,
                },
            )

    def get_raw(self, path: str) -> RawNoteRecord | None:
        row = self._conn.execute("SELECT * FROM raw_notes WHERE path = ?", (path,)).fetchone()
        return _row_to_raw(row) if row else None

    def get_raw_by_hash(self, content_hash: str) -> RawNoteRecord | None:
        row = self._conn.execute(
            "SELECT * FROM raw_notes WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return _row_to_raw(row) if row else None

    def list_raw(self, status: str | None = None) -> list[RawNoteRecord]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM raw_notes WHERE status = ?", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM raw_notes").fetchall()
        return [_row_to_raw(r) for r in rows]

    def get_origin_uris_for_raw_notes(self, paths: list[str]) -> dict[str, str | None]:
        """Return {raw_note_path: origin_uri} via source_documents JOIN.

        Source of truth is source_documents (the v9 canonical store for
        document metadata). Link is by filename convention:
        `raw_notes.path` is `"raw/<source_id>.md"` where `<source_id>`
        matches `source_documents.id` — `synto add` writes the raw file
        at exactly this path.

        Returns None for raw notes with no matching source_documents row
        (e.g., notes the user dropped into raw/ manually, without going
        through `synto add`). Callers should fall back to path uniqueness
        in that case.
        """
        result: dict[str, str | None] = {p: None for p in paths}
        if not paths:
            return result
        placeholders = ",".join("?" * len(paths))
        rows = self._conn.execute(
            f"""
            SELECT 'raw/' || sd.id || '.md' AS raw_path, sd.origin_uri
            FROM source_documents sd
            WHERE 'raw/' || sd.id || '.md' IN ({placeholders})
            """,
            tuple(paths),
        ).fetchall()
        for row in rows:
            if row[1] is not None:
                result[row[0]] = row[1]
        return result

    def get_note_language(self, path: str) -> str | None:
        row = self._conn.execute(
            "SELECT language FROM raw_notes WHERE path = ?", (path,)
        ).fetchone()
        return row[0] if row else None

    def list_note_languages(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT language FROM raw_notes WHERE language IS NOT NULL ORDER BY language"
        ).fetchall()
        return [str(row[0]) for row in rows if row[0]]

    def mark_raw_status(self, path: str, status: str, error: str | None = None) -> None:
        now = datetime.now().isoformat()
        with self._tx():
            if status == "ingested":
                self._conn.execute(
                    "UPDATE raw_notes SET status=?, ingested_at=?, error=NULL WHERE path=?",
                    (status, now, path),
                )
            elif status == "compiled":
                self._conn.execute(
                    "UPDATE raw_notes SET status=?, compiled_at=?, error=NULL WHERE path=?",
                    (status, now, path),
                )
            else:
                self._conn.execute(
                    "UPDATE raw_notes SET status=?, error=? WHERE path=?",
                    (status, error, path),
                )

    # ── Concepts ──────────────────────────────────────────────────────────────

    def upsert_concepts(self, source_path: str, concept_names: list[str]) -> None:
        """Link concept names to a source note (idempotent)."""
        with self._tx():
            for name in concept_names:
                name = name.strip()
                if not name:
                    continue
                self._conn.execute(
                    "INSERT OR IGNORE INTO concepts (name, source_path) VALUES (?, ?)",
                    (name, source_path),
                )
                now = datetime.now().isoformat()
                self._conn.execute(
                    """INSERT OR IGNORE INTO knowledge_items
                       (name, kind, subtype, status, confidence, created_at, updated_at)
                       VALUES (?, 'concept', NULL, 'confirmed', 1.0, ?, ?)""",
                    (name, now, now),
                )
        self._ensure_compile_state_rows(source_path)

    def replace_concepts_for_source(self, source_path: str, concept_names: list[str]) -> None:
        """Replace concept links for a source and reset compile state for current concepts."""
        normalized = []
        seen: set[str] = set()
        for name in concept_names:
            cleaned = name.strip()
            if not cleaned or cleaned.casefold() in seen:
                continue
            seen.add(cleaned.casefold())
            normalized.append(cleaned)

        existing_rows = self._conn.execute(
            "SELECT name FROM concepts WHERE source_path = ?", (source_path,)
        ).fetchall()
        existing_names = {row["name"] for row in existing_rows}
        new_names = set(normalized)
        removed = existing_names - new_names

        now = datetime.now().isoformat()
        with self._tx():
            if removed:
                placeholders = ",".join("?" * len(removed))
                params = [source_path, *removed]
                self._conn.execute(
                    f"DELETE FROM concepts WHERE source_path = ? AND name IN ({placeholders})",
                    params,
                )
                self._conn.execute(
                    (
                        "DELETE FROM concept_compile_state "
                        f"WHERE source_path = ? AND concept_name IN ({placeholders})"
                    ),
                    params,
                )
            for name in normalized:
                self._conn.execute(
                    "INSERT OR IGNORE INTO concepts (name, source_path) VALUES (?, ?)",
                    (name, source_path),
                )
                self._conn.execute(
                    """INSERT OR IGNORE INTO knowledge_items
                           (name, kind, subtype, status, confidence, created_at, updated_at)
                       VALUES (?, 'concept', NULL, 'confirmed', 1.0, ?, ?)""",
                    (name, now, now),
                )
                self._conn.execute(
                    """INSERT INTO concept_compile_state
                           (concept_name, source_path, status, error, compiled_at, updated_at)
                       VALUES (?, ?, 'pending', NULL, NULL, ?)
                       ON CONFLICT(concept_name, source_path) DO UPDATE SET
                           status='pending',
                           error=NULL,
                           compiled_at=NULL,
                           updated_at=excluded.updated_at""",
                    (name, source_path, now),
                )
        self.refresh_raw_compile_status(source_path)

    def list_all_concept_names(self) -> list[str]:
        """All unique canonical concept names, sorted."""
        rows = self._conn.execute("SELECT DISTINCT name FROM concepts ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def get_sources_for_concept(self, name: str) -> list[str]:
        """Raw note paths linked to a concept (case-insensitive match)."""
        rows = self._conn.execute(
            "SELECT DISTINCT source_path FROM concepts WHERE lower(name) = lower(?)",
            (name,),
        ).fetchall()
        return [r[0] for r in rows]

    def upsert_aliases(self, concept_name: str, aliases: list[str]) -> None:
        """Merge aliases for a concept. Skips self-matches (alias == canonical)."""
        canonical_lower = concept_name.lower()
        with self._tx():
            for alias in aliases:
                alias = alias.strip()
                if not alias or alias.lower() == canonical_lower:
                    continue
                self._conn.execute(
                    "INSERT OR IGNORE INTO concept_aliases (concept_name, alias) VALUES (?, ?)",
                    (concept_name, alias),
                )

    def get_aliases(self, concept_name: str) -> list[str]:
        """All aliases stored for a concept (case-insensitive match on concept_name)."""
        if not self._has_table("concept_aliases"):
            return []
        rows = self._conn.execute(
            "SELECT alias FROM concept_aliases WHERE lower(concept_name) = lower(?) ORDER BY alias",
            (concept_name,),
        ).fetchall()
        return [r[0] for r in rows]

    def aliases_for_concept(self, concept_name: str) -> list[str]:
        return self.get_aliases(concept_name)

    def resolve_alias(self, surface: str) -> str | None:
        """Return canonical concept name if surface unambiguously matches exactly one concept."""
        rows = self._conn.execute(
            "SELECT DISTINCT concept_name FROM concept_aliases WHERE lower(alias) = lower(?)",
            (surface,),
        ).fetchall()
        if len(rows) == 1:
            return rows[0][0]
        return None

    def list_alias_map(self) -> dict[str, str]:
        """Return {lower(alias): canonical_name} for all unambiguous aliases.

        Aliases claimed by more than one concept are excluded — they are unsafe to rewrite.
        """
        rows = self._conn.execute(
            "SELECT lower(alias) as al, concept_name FROM concept_aliases"
        ).fetchall()
        counts: dict[str, int] = {}
        mapping: dict[str, str] = {}
        for al, canonical in rows:
            counts[al] = counts.get(al, 0) + 1
            mapping[al] = canonical
        return {al: canonical for al, canonical in mapping.items() if counts[al] == 1}

    def load_concept_alias_map(self) -> dict[str, list[str]]:
        """Return {concept_name: [aliases]} for all concepts with aliases.

        Used by query routing to bridge task-vocabulary questions to source-vocabulary
        page titles. Empty dict on missing table (older DBs predate v4).

        Excludes ambiguous aliases (claimed by >= 2 distinct concepts) — hinting
        multiple concepts from one surface dilutes the routing signal.
        """
        if not self._has_table("concept_aliases"):
            return {}
        rows = self._conn.execute(
            """
            SELECT concept_name, alias
            FROM concept_aliases
            WHERE lower(alias) NOT IN (
                SELECT lower(alias)
                FROM concept_aliases
                GROUP BY lower(alias)
                HAVING count(DISTINCT lower(concept_name)) >= 2
            )
            ORDER BY concept_name, alias
            """
        ).fetchall()
        result: dict[str, list[str]] = {}
        for concept_name, alias in rows:
            result.setdefault(concept_name, []).append(alias)
        return result

    def list_frequent_aliases(self, threshold: int = 2) -> list[str]:
        """Aliases (lower-cased) claimed by >= threshold distinct concepts.

        Used at export time to filter ambiguous aliases that can't be used for
        concept lookup. Language-agnostic — works for any language.
        """
        if not self._has_table("concept_aliases"):
            return []
        rows = self._conn.execute(
            """
            SELECT lower(alias)
            FROM concept_aliases
            GROUP BY lower(alias)
            HAVING count(DISTINCT lower(concept_name)) >= ?
            """,
            (threshold,),
        ).fetchall()
        return [r[0] for r in rows]

    def delete_aliases_for_concept(self, concept_name: str) -> None:
        """Remove all aliases for a concept (call when concept is removed)."""
        with self._tx():
            self._conn.execute(
                "DELETE FROM concept_aliases WHERE lower(concept_name) = lower(?)",
                (concept_name,),
            )

    def get_concepts_for_sources(self, source_paths: list[str]) -> list[str]:
        """Concept names linked to any of the given source paths."""
        if not source_paths:
            return []
        placeholders = ",".join("?" * len(source_paths))
        rows = self._conn.execute(
            f"SELECT DISTINCT name FROM concepts WHERE source_path IN ({placeholders})",
            source_paths,
        ).fetchall()
        return [r[0] for r in rows]

    def list_source_concept_seeds(self) -> list[tuple[str, str, list[str]]]:
        """Return content-hash-guarded source-to-concept links for rebuild seeds."""
        if not self._has_table("raw_notes") or not self._has_table("concepts"):
            return []
        rows = self._conn.execute(
            """
            SELECT c.source_path, r.content_hash, c.name
            FROM concepts c
            JOIN raw_notes r ON r.path = c.source_path
            ORDER BY lower(c.source_path), c.source_path, lower(c.name), c.name
            """
        ).fetchall()

        grouped: list[tuple[str, str, list[str]]] = []
        current_path: str | None = None
        current_hash = ""
        current_concepts: list[str] = []
        for row in rows:
            source_path = str(row["source_path"])
            if current_path is not None and source_path != current_path:
                grouped.append((current_path, current_hash, current_concepts))
                current_concepts = []
            current_path = source_path
            current_hash = str(row["content_hash"])
            current_concepts.append(str(row["name"]))
        if current_path is not None:
            grouped.append((current_path, current_hash, current_concepts))
        return grouped

    def list_failed_concepts(self) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT DISTINCT concept_name
            FROM concept_compile_state
            WHERE status = 'failed'
              AND lower(concept_name) NOT IN (SELECT lower(concept) FROM blocked_concepts)
            ORDER BY lower(concept_name)
            """
        ).fetchall()
        return [row["concept_name"] for row in rows]

    def find_concept_by_name_or_alias(self, query: str) -> tuple[str, list[str]] | None:
        q = query.strip()
        q_lower = q.casefold()
        q_escaped = q_lower.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        row = self._conn.execute(
            "SELECT DISTINCT name FROM concepts WHERE lower(name) = ? LIMIT 1",
            (q_lower,),
        ).fetchone()
        if row:
            name = row["name"]
            return name, self.aliases_for_concept(name)

        row = self._conn.execute(
            "SELECT concept_name FROM concept_aliases WHERE lower(alias) = ? LIMIT 1",
            (q_lower,),
        ).fetchone()
        if row:
            name = row["concept_name"]
            return name, self.aliases_for_concept(name)

        row = self._conn.execute(
            "SELECT DISTINCT name FROM concepts WHERE lower(name) LIKE ? ESCAPE '\\' LIMIT 1",
            (f"%{q_escaped}%",),
        ).fetchone()
        if row:
            name = row["name"]
            return name, self.aliases_for_concept(name)

        return None

    def get_compile_state(self, concept_name: str, source_path: str) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT * FROM concept_compile_state
            WHERE lower(concept_name) = lower(?) AND source_path = ?
            """,
            (concept_name, source_path),
        ).fetchone()

    def mark_concept_compile_state(
        self,
        concept_name: str,
        source_paths: list[str],
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        now = datetime.now().isoformat()
        with self._tx():
            for source_path in source_paths:
                self._conn.execute(
                    """INSERT INTO concept_compile_state
                           (concept_name, source_path, status, error, compiled_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(concept_name, source_path) DO UPDATE SET
                           status=excluded.status,
                           error=excluded.error,
                           compiled_at=excluded.compiled_at,
                           updated_at=excluded.updated_at""",
                    (
                        concept_name,
                        source_path,
                        status,
                        error,
                        now if status == "compiled" else None,
                        now,
                    ),
                )
        for source_path in source_paths:
            self.refresh_raw_compile_status(source_path)

    def clear_deferred_state(
        self, concept_name: str, source_paths: list[str] | None = None
    ) -> None:
        params: list[str] = [concept_name]
        query = (
            "UPDATE concept_compile_state SET status='pending', error=NULL, compiled_at=NULL, "
            "updated_at=? WHERE lower(concept_name)=lower(?) AND "
            "status IN ('deferred_draft', 'deferred_manual_edit')"
        )
        now = datetime.now().isoformat()
        params.insert(0, now)
        if source_paths:
            placeholders = ",".join("?" * len(source_paths))
            query += f" AND source_path IN ({placeholders})"
            params.extend(source_paths)
        with self._tx():
            self._conn.execute(query, params)
        if source_paths:
            for source_path in source_paths:
                self.refresh_raw_compile_status(source_path)

    def refresh_raw_compile_status(self, source_path: str) -> None:
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM concepts WHERE source_path = ?", (source_path,)
        ).fetchone()
        concept_count = row["cnt"] if row else 0
        if concept_count == 0:
            return

        compiled_count = self._conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM concept_compile_state
            WHERE source_path = ? AND status = 'compiled'
            """,
            (source_path,),
        ).fetchone()["cnt"]

        if compiled_count == concept_count:
            self.mark_raw_status(source_path, "compiled")
        else:
            self.mark_raw_status(source_path, "ingested")

    # ── Knowledge Items ───────────────────────────────────────────────────────

    def upsert_item(self, record: KnowledgeItemRecord) -> None:
        now = datetime.now().isoformat()
        with self._tx():
            self._conn.execute(
                """INSERT INTO knowledge_items
                       (name, kind, subtype, status, confidence, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                       kind=excluded.kind,
                       subtype=excluded.subtype,
                       status=excluded.status,
                       confidence=max(knowledge_items.confidence, excluded.confidence),
                       updated_at=excluded.updated_at""",
                (
                    record.name,
                    record.kind,
                    record.subtype,
                    record.status,
                    record.confidence,
                    record.created_at.isoformat(),
                    now,
                ),
            )

    def get_item(self, name: str) -> KnowledgeItemRecord | None:
        row = self._conn.execute(
            "SELECT * FROM knowledge_items WHERE lower(name) = lower(?)", (name,)
        ).fetchone()
        return _row_to_item(row) if row else None

    def list_items(
        self, kind: str | None = None, status: str | None = None
    ) -> list[KnowledgeItemRecord]:
        clauses: list[str] = []
        params: list[str] = []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM knowledge_items{where} ORDER BY lower(name)", params
        ).fetchall()
        return [_row_to_item(row) for row in rows]

    def add_item_mention(self, record: ItemMentionRecord) -> None:
        with self._tx():
            self._conn.execute(
                """INSERT OR IGNORE INTO item_mentions
                       (item_name, source_path, mention_text, context, evidence_level, confidence)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    record.item_name,
                    record.source_path,
                    record.mention_text,
                    record.context,
                    record.evidence_level,
                    record.confidence,
                ),
            )

    def get_item_mentions(self, name: str) -> list[ItemMentionRecord]:
        rows = self._conn.execute(
            "SELECT * FROM item_mentions WHERE lower(item_name) = lower(?) ORDER BY source_path",
            (name,),
        ).fetchall()
        return [_row_to_item_mention(row) for row in rows]

    def concepts_needing_compile(self) -> list[str]:
        """Concepts with pending/failed compile state, plus stub concepts.

        Excludes blocked and deferred concepts from normal scheduling.
        """
        rows = self._conn.execute(
            """
            SELECT DISTINCT ccs.concept_name AS name
            FROM concept_compile_state ccs
            JOIN concepts c ON c.name = ccs.concept_name AND c.source_path = ccs.source_path
            WHERE ccs.status IN ('pending', 'failed')
              AND lower(ccs.concept_name) NOT IN (SELECT lower(concept) FROM blocked_concepts)

            UNION

            SELECT s.concept FROM stubs s
            WHERE s.concept NOT IN (
                SELECT DISTINCT c2.name FROM concepts c2
            )
            AND lower(s.concept) NOT IN (SELECT lower(concept) FROM blocked_concepts)

            ORDER BY 1
            """
        ).fetchall()
        return [r[0] for r in rows]

    # ── Wiki Articles ─────────────────────────────────────────────────────────

    def find_article_candidates(self, concept_name: str) -> list[WikiArticleRecord]:
        concept_lower = concept_name.casefold()
        alias_rows = self._conn.execute(
            "SELECT alias FROM concept_aliases WHERE lower(concept_name) = lower(?)",
            (concept_name,),
        ).fetchall()
        aliases = {row[0].casefold() for row in alias_rows}

        matches: list[WikiArticleRecord] = []
        for article in self.list_articles():
            title_lower = article.title.casefold()
            stem_lower = Path(article.path).stem.casefold()
            if title_lower == concept_lower or stem_lower == concept_lower:
                matches.append(article)
                continue
            if title_lower in aliases or stem_lower in aliases:
                matches.append(article)
        return matches

    def _upsert_article_row(self, record: WikiArticleRecord) -> None:
        article_id = self._resolve_article_id(record.path, record.article_id)
        self._conn.execute(
            """INSERT INTO wiki_articles
                   (
                        path, title, sources, content_hash, created_at, updated_at, status,
                        approved_at, approval_notes, kind, question_hash,
                        synthesis_sources, synthesis_source_hashes, article_id,
                        last_compile_pipeline
                    )
                VALUES (:path, :title, :sources, :content_hash,
                        :created_at, :updated_at, :status,
                        :approved_at, :approval_notes, :kind, :question_hash,
                        :synthesis_sources, :synthesis_source_hashes, :article_id,
                        :last_compile_pipeline)
                ON CONFLICT(path) DO UPDATE SET
                    title=excluded.title,
                    sources=excluded.sources,
                    content_hash=excluded.content_hash,
                   updated_at=excluded.updated_at,
                   status=excluded.status,
                   approved_at=excluded.approved_at,
                    approval_notes=excluded.approval_notes,
                    kind=excluded.kind,
                    question_hash=excluded.question_hash,
                    synthesis_sources=excluded.synthesis_sources,
                    synthesis_source_hashes=excluded.synthesis_source_hashes,
                    article_id=COALESCE(wiki_articles.article_id, excluded.article_id),
                    last_compile_pipeline=COALESCE(
                        excluded.last_compile_pipeline,
                        wiki_articles.last_compile_pipeline
                    )""",
            {
                "path": record.path,
                "title": record.title,
                "sources": json.dumps(record.sources),
                "content_hash": record.content_hash,
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat(),
                "status": record.status,
                "approved_at": record.approved_at.isoformat() if record.approved_at else None,
                "approval_notes": record.approval_notes,
                "kind": record.kind,
                "question_hash": record.question_hash,
                "synthesis_sources": json.dumps(record.synthesis_sources),
                "synthesis_source_hashes": json.dumps(record.synthesis_source_hashes),
                "article_id": article_id,
                "last_compile_pipeline": record.last_compile_pipeline,
            },
        )

    def _resolve_article_id(self, path: str, article_id: str | None) -> str:
        if article_id is not None:
            if not article_id:
                raise ValueError("article_id must not be empty")
            return article_id

        existing = self._conn.execute(
            "SELECT article_id FROM wiki_articles WHERE path = ?",
            (path,),
        ).fetchone()
        if existing and existing["article_id"]:
            return existing["article_id"]
        return _generate_article_id()

    def upsert_article(self, record: WikiArticleRecord) -> None:
        with self._tx():
            self._upsert_article_row(record)

    def insert_synthesis_atomic(self, record: WikiArticleRecord) -> None:
        article_id = self._resolve_article_id(record.path, record.article_id)
        try:
            with self._tx():
                self._conn.execute(
                    """INSERT INTO wiki_articles
                           (
                                path, title, sources, content_hash, created_at, updated_at,
                                status,
                                approved_at, approval_notes, kind, question_hash,
                                synthesis_sources, synthesis_source_hashes, article_id,
                                last_compile_pipeline
                            )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.path,
                        record.title,
                        json.dumps(record.sources),
                        record.content_hash,
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                        record.status,
                        record.approved_at.isoformat() if record.approved_at else None,
                        record.approval_notes,
                        record.kind,
                        record.question_hash,
                        json.dumps(record.synthesis_sources),
                        json.dumps(record.synthesis_source_hashes),
                        article_id,
                        record.last_compile_pipeline,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            message = str(exc)
            if "wiki_articles.question_hash" in message:
                raise DuplicateSynthesisQuestionHashError(message) from exc
            if "wiki_articles.path" in message:
                raise DuplicateArticlePathError(message) from exc
            raise SynthesisInsertConflictError(message) from exc

    def get_article(self, path: str) -> WikiArticleRecord | None:
        row = self._conn.execute("SELECT * FROM wiki_articles WHERE path = ?", (path,)).fetchone()
        return _row_to_article(row) if row else None

    def list_articles(self, drafts_only: bool = False) -> list[WikiArticleRecord]:
        if drafts_only:
            rows = self._conn.execute(
                "SELECT * FROM wiki_articles WHERE status = 'draft'"
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM wiki_articles").fetchall()
        return [_row_to_article(r) for r in rows]

    def list_synthesis_articles_brief(self) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT path, title FROM wiki_articles "
            "WHERE kind = 'synthesis' AND status = 'published' ORDER BY path"
        ).fetchall()
        return [(row["path"], row["title"]) for row in rows]

    def find_synthesis_by_question_hash(self, question_hash: str) -> WikiArticleRecord | None:
        row = self._conn.execute(
            "SELECT * FROM wiki_articles WHERE kind = 'synthesis' AND question_hash = ?",
            (question_hash,),
        ).fetchone()
        return _row_to_article(row) if row else None

    def publish_article(self, old_path: str, new_path: str) -> None:
        with self._tx():
            # Guard: draft row must exist before we touch anything.
            # Without this, the DELETE below would silently destroy the previously
            # published row when the draft was never recorded in wiki_articles.
            if not self._conn.execute(
                "SELECT 1 FROM wiki_articles WHERE path = ?", (old_path,)
            ).fetchone():
                return
            # Remove existing published row at target path (re-publish scenario)
            if old_path != new_path:
                self._conn.execute("DELETE FROM wiki_articles WHERE path = ?", (new_path,))
            self._conn.execute(
                "UPDATE wiki_articles SET path=?, status='published', updated_at=? WHERE path=?",
                (new_path, datetime.now().isoformat(), old_path),
            )

    def verify_article(self, path: str, notes: str = "") -> None:
        """Mark a draft as human-verified in place. Does not move the file.

        approved_at / approval_notes are set only when NULL so a later
        publish does not overwrite the first-approval audit trail.

        Also transitions concept_compile_state to "compiled" so the next
        `synto compile` run does not regenerate (and clobber) a draft a
        human has already signed off on.
        """
        with self._tx():
            # Parity with publish_article: silent no-op if the row is missing.
            if not self._conn.execute(
                "SELECT 1 FROM wiki_articles WHERE path = ?", (path,)
            ).fetchone():
                return
            now = datetime.now().isoformat()
            self._conn.execute(
                "UPDATE wiki_articles SET status='verified', "
                "approved_at=COALESCE(approved_at, ?), "
                "approval_notes=COALESCE(approval_notes, ?), "
                "updated_at=? WHERE path=?",
                (now, notes or None, now, path),
            )
        art = self.get_article(path)
        if art:
            self.mark_concept_compile_state(art.title, art.sources, "compiled")

    def count_articles_by_status(self, status: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM wiki_articles WHERE status = ?",
            (status,),
        ).fetchone()
        return int(row[0]) if row else 0

    def approve_article(self, path: str, notes: str = "") -> None:
        """Record approval timestamp on a published article.

        COALESCE preserves the first-approval timestamp when an article
        was previously verified — the verify→publish path must not lose
        audit history.
        """
        with self._tx():
            self._conn.execute(
                "UPDATE wiki_articles SET "
                "approved_at=COALESCE(approved_at, ?), "
                "approval_notes=COALESCE(approval_notes, ?) "
                "WHERE path=?",
                (datetime.now().isoformat(), notes or None, path),
            )
        art = self.get_article(path)
        if art:
            self.mark_concept_compile_state(art.title, art.sources, "compiled")

    def delete_article(self, path: str) -> None:
        with self._tx():
            self._conn.execute("DELETE FROM wiki_articles WHERE path = ?", (path,))

    # ── Rejections ────────────────────────────────────────────────────────────

    _REJECTION_CAP = 5

    def add_rejection(self, concept: str, feedback: str, body: str = "") -> None:
        """Store a rejection record. Auto-blocks concept after _REJECTION_CAP rejections."""
        with self._tx():
            self._conn.execute(
                """INSERT INTO rejections (concept, feedback, rejected_body, rejected_at)
                   VALUES (?, ?, ?, ?)""",
                (concept, feedback, body or None, datetime.now().isoformat()),
            )
        if self.rejection_count(concept) >= self._REJECTION_CAP:
            self.mark_concept_blocked(concept)

    def get_rejections(self, concept: str, limit: int = 3) -> list[dict]:
        """Return most recent rejections for a concept, newest first."""
        rows = self._conn.execute(
            """SELECT feedback, rejected_body, rejected_at
               FROM rejections WHERE concept = ?
               ORDER BY rejected_at DESC LIMIT ?""",
            (concept, limit),
        ).fetchall()
        return [
            {"feedback": r["feedback"], "body": r["rejected_body"], "rejected_at": r["rejected_at"]}
            for r in rows
        ]

    def rejection_count(self, concept: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM rejections WHERE concept = ?", (concept,)
        ).fetchone()
        return row[0] if row else 0

    # ── Blocked Concepts ──────────────────────────────────────────────────────

    def mark_concept_blocked(self, concept: str) -> None:
        with self._tx():
            self._conn.execute(
                "INSERT OR REPLACE INTO blocked_concepts (concept, blocked_at) VALUES (?, ?)",
                (concept, datetime.now().isoformat()),
            )

    def is_concept_blocked(self, concept: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM blocked_concepts WHERE lower(concept) = lower(?)", (concept,)
        ).fetchone()
        return row is not None

    def unblock_concept(self, concept: str) -> None:
        with self._tx():
            self._conn.execute("DELETE FROM blocked_concepts WHERE concept = ?", (concept,))

    def list_blocked_concepts(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT concept FROM blocked_concepts ORDER BY concept"
        ).fetchall()
        return [r[0] for r in rows]

    # ── Stubs ─────────────────────────────────────────────────────────────────

    def add_stub(self, concept: str, source: str = "auto") -> None:
        with self._tx():
            self._conn.execute(
                "INSERT OR IGNORE INTO stubs (concept, created_at, source) VALUES (?, ?, ?)",
                (concept, datetime.now().isoformat(), source),
            )

    # ── Ingest Checkpoints ────────────────────────────────────────────────────

    def list_ingest_chunks(
        self,
        source_path: str,
        content_hash: str,
        chunk_count: int,
        chunk_size: int,
        checkpoint_schema: int = _CHECKPOINT_SCHEMA_VERSION,
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT * FROM ingest_chunks
            WHERE source_path = ?
              AND content_hash = ?
              AND chunk_count = ?
              AND chunk_size = ?
              AND checkpoint_schema = ?
            ORDER BY chunk_index
            """,
            (source_path, content_hash, chunk_count, chunk_size, checkpoint_schema),
        ).fetchall()

    def upsert_ingest_chunk(
        self,
        source_path: str,
        content_hash: str,
        chunk_index: int,
        chunk_count: int,
        chunk_size: int,
        result_json: str,
        checkpoint_schema: int = _CHECKPOINT_SCHEMA_VERSION,
    ) -> None:
        now = datetime.now().isoformat()
        with self._tx():
            self._conn.execute(
                """INSERT INTO ingest_chunks
                       (source_path, content_hash, chunk_index, chunk_count, chunk_size,
                        checkpoint_schema, result_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(
                       source_path,
                       content_hash,
                       chunk_index,
                       chunk_count,
                       chunk_size,
                       checkpoint_schema
                   )
                   DO UPDATE SET
                       result_json=excluded.result_json,
                       updated_at=excluded.updated_at""",
                (
                    source_path,
                    content_hash,
                    chunk_index,
                    chunk_count,
                    chunk_size,
                    checkpoint_schema,
                    result_json,
                    now,
                    now,
                ),
            )

    def purge_ingest_chunks(self, source_path: str, *, keep_hash: str | None = None) -> None:
        with self._tx():
            if keep_hash is None:
                self._conn.execute(
                    "DELETE FROM ingest_chunks WHERE source_path = ?", (source_path,)
                )
            else:
                self._conn.execute(
                    "DELETE FROM ingest_chunks WHERE source_path = ? AND content_hash <> ?",
                    (source_path, keep_hash),
                )

    def delete_ingest_chunks(
        self,
        source_path: str,
        content_hash: str,
        chunk_count: int,
        chunk_size: int,
        checkpoint_schema: int = _CHECKPOINT_SCHEMA_VERSION,
    ) -> None:
        with self._tx():
            self._conn.execute(
                """
                DELETE FROM ingest_chunks
                WHERE source_path = ?
                  AND content_hash = ?
                  AND chunk_count = ?
                  AND chunk_size = ?
                  AND checkpoint_schema = ?
                """,
                (source_path, content_hash, chunk_count, chunk_size, checkpoint_schema),
            )

    def delete_stub(self, concept: str) -> None:
        with self._tx():
            self._conn.execute("DELETE FROM stubs WHERE concept = ?", (concept,))

    def has_stub(self, concept: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM stubs WHERE concept = ?", (concept,)).fetchone()
        return row is not None

    def get_stubs(self) -> list[str]:
        rows = self._conn.execute("SELECT concept FROM stubs ORDER BY concept").fetchall()
        return [r[0] for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self, vault: Path | None = None) -> dict:
        raw_counts = {
            row["status"]: row["cnt"]
            for row in self._conn.execute(
                "SELECT status, COUNT(*) as cnt FROM raw_notes GROUP BY status"
            ).fetchall()
        }
        if vault is not None:
            raw_dir = vault / "raw"
            if raw_dir.exists():
                tracked_rows = self._conn.execute("SELECT path FROM raw_notes").fetchall()
                tracked_paths = {row["path"] for row in tracked_rows}
                untracked_raw = sum(
                    1
                    for path in raw_dir.rglob("*.md")
                    if "processed" not in path.parts
                    and not path.name.startswith(".")
                    and str(path.relative_to(vault)) not in tracked_paths
                )
                if untracked_raw:
                    raw_counts["new"] = raw_counts.get("new", 0) + untracked_raw
        db_draft_count = self.count_articles_by_status("draft")
        db_verified_count = self.count_articles_by_status("verified")
        orphan_draft_count = 0
        orphan_verified_count = 0
        if vault is not None:
            drafts_dir = vault / "wiki" / ".drafts"
            if drafts_dir.exists():
                tracked_rows = self._conn.execute(
                    "SELECT path FROM wiki_articles WHERE path LIKE 'wiki/.drafts/%'"
                ).fetchall()
                tracked_paths = {row["path"] for row in tracked_rows}
                for path in drafts_dir.rglob("*.md"):
                    rel_path = str(path.relative_to(vault))
                    if rel_path in tracked_paths:
                        continue
                    try:
                        meta = self._infer_orphan_draft_status(path)
                    except Exception:
                        meta = "draft"
                    if meta == "verified":
                        orphan_verified_count += 1
                    else:
                        orphan_draft_count += 1
        pub_count = self.count_articles_by_status("published")
        return {
            "raw": raw_counts,
            "drafts": db_draft_count + orphan_draft_count,
            "verified": db_verified_count + orphan_verified_count,
            "published": pub_count,
        }

    def _infer_orphan_draft_status(self, path: Path) -> str:
        from .vault import parse_note

        meta, _ = parse_note(path)
        status = meta.get("status")
        if status == "verified":
            return "verified"
        return "draft"

    def quality_stats(self) -> dict[str, int]:
        """Distribution of source quality levels."""
        rows = self._conn.execute(
            "SELECT quality, COUNT(*) as cnt FROM raw_notes "
            "WHERE quality IS NOT NULL GROUP BY quality"
        ).fetchall()
        result: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for row in rows:
            if row["quality"] in result:
                result[row["quality"]] = row["cnt"]
        return result

    def count_concepts(self) -> int:
        if not self._has_table("concepts"):
            return 0
        return int(self._conn.execute("SELECT COUNT(DISTINCT name) FROM concepts").fetchone()[0])

    def count_aliases(self) -> int:
        if not self._has_table("concept_aliases"):
            return 0
        return int(self._conn.execute("SELECT COUNT(*) FROM concept_aliases").fetchone()[0])

    def count_knowledge_items(self) -> int:
        if not self._has_table("knowledge_items"):
            return 0
        return int(self._conn.execute("SELECT COUNT(*) FROM knowledge_items").fetchone()[0])

    def count_failed_notes(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM raw_notes WHERE status = 'failed'"
        ).fetchone()
        return int(row[0])

    def count_failed_concepts(self) -> int:
        if not self._has_table("concept_compile_state"):
            return 0
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT concept_name) FROM concept_compile_state WHERE status = 'failed'"
        ).fetchone()
        return int(row[0])

    def count_source_segments(self) -> int:
        if not self._has_table("source_segments"):
            return 0
        return int(self._conn.execute("SELECT COUNT(*) FROM source_segments").fetchone()[0])

    # ── Compile run tracking ──────────────────────────────────────────────

    def start_compile_run(
        self,
        run_ulid: str,
        pipeline_json: str,
        fast_model: str,
        heavy_model: str,
    ) -> None:
        now = datetime.now().isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO compile_runs
               (run_ulid, pipeline_json, fast_model, heavy_model, started_at)
               VALUES (?, ?, ?, ?, ?)""",
            (run_ulid, pipeline_json, fast_model, heavy_model, now),
        )
        self._conn.commit()

    def finish_compile_run(
        self,
        run_ulid: str,
        article_count: int = 0,
        total_tokens: int = 0,
        total_cost_usd: float = 0.0,
    ) -> None:
        now = datetime.now().isoformat()
        self._conn.execute(
            """UPDATE compile_runs
               SET finished_at = ?, article_count = ?, total_tokens = ?, total_cost_usd = ?
               WHERE run_ulid = ?""",
            (now, article_count, total_tokens, total_cost_usd, run_ulid),
        )
        self._conn.commit()

    def update_article_compile_run(self, article_path: str, run_ulid: str) -> None:
        self._conn.execute(
            "UPDATE wiki_articles SET last_compile_pipeline = ? WHERE path = ?",
            (run_ulid, article_path),
        )
        self._conn.commit()

    def get_compile_run(self, run_ulid: str) -> sqlite3.Row | None:
        if not self._has_table("compile_runs"):
            return None
        return self._conn.execute(
            "SELECT * FROM compile_runs WHERE run_ulid = ?", (run_ulid,)
        ).fetchone()

    # ── Term extraction (concept occurrences) ──────────────────────────────

    def upsert_concept_occurrences(
        self,
        terms: list,
        source_segment_id: str,
        extraction_run: str | None = None,
    ) -> None:
        """Persist TermRecord list to concept_occurrences. Idempotent."""
        if not self._has_table("concept_occurrences"):
            return
        with self._tx():
            for ordinal, term in enumerate(terms):
                self._conn.execute(
                    """INSERT OR REPLACE INTO concept_occurrences
                       (concept_name, source_segment_id, ordinal, confidence, extraction_run)
                       VALUES (?, ?, ?, ?, ?)""",
                    (term.name, source_segment_id, ordinal, term.confidence, extraction_run),
                )

    def list_concept_occurrences(self) -> list[sqlite3.Row]:
        if not self._has_table("concept_occurrences"):
            return []
        return self._conn.execute(
            "SELECT * FROM concept_occurrences ORDER BY concept_name, source_segment_id"
        ).fetchall()

    def upsert_source_document(self, doc: object) -> None:
        """Insert or replace a SourceDocument record."""
        import json
        from datetime import UTC, datetime

        imported_at = getattr(doc, "imported_at", None)
        if imported_at is None:
            imported_at = datetime.now(UTC).isoformat()
        elif hasattr(imported_at, "isoformat"):
            imported_at = imported_at.isoformat()

        meta = dict(getattr(doc, "metadata", {}) or {})
        biblio = getattr(doc, "bibliographic_metadata", None)
        if biblio is not None:
            meta["bibliographic_metadata"] = biblio.model_dump(exclude_none=True)

        self._conn.execute(
            """INSERT OR REPLACE INTO source_documents
               (id, source_type, origin_uri, title, imported_at, raw_hash,
                normalized_hash, extractor_version, license, redistribution, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc.id,
                getattr(doc, "source_type", "unknown_text"),
                getattr(doc, "origin_uri", None),
                getattr(doc, "title", None),
                imported_at,
                getattr(doc, "raw_hash", None),
                getattr(doc, "normalized_hash", None),
                getattr(doc, "extractor_version", None),
                getattr(doc, "license", None),
                getattr(doc, "redistribution", "unknown"),
                json.dumps(meta) if meta else None,
            ),
        )
        self._conn.commit()

    def get_source_document(self, source_id: str) -> sqlite3.Row | None:
        """Fetch a source document by ID, or None if not found."""
        if not self._has_table("source_documents"):
            return None
        return self._conn.execute(
            "SELECT * FROM source_documents WHERE id = ?", (source_id,)
        ).fetchone()

    def get_source_document_by_raw_hash(self, raw_hash: str) -> sqlite3.Row | None:
        """Fetch the first source document with the given raw hash, or None."""
        if not self._has_table("source_documents"):
            return None
        return self._conn.execute(
            "SELECT * FROM source_documents WHERE raw_hash = ? "
            "ORDER BY imported_at DESC, id DESC LIMIT 1",
            (raw_hash,),
        ).fetchone()

    def list_source_documents(self) -> list[tuple[str, str | None, str]]:
        if not self._has_table("source_documents"):
            return []
        rows = self._conn.execute(
            "SELECT id, title, source_type FROM source_documents ORDER BY id"
        ).fetchall()
        return [(row["id"], row["title"], row["source_type"]) for row in rows]

    def delete_source_import_data(self, source_id: str) -> None:
        """Remove source import records for a single source document."""
        with self._tx():
            if self._has_table("generated_assets"):
                self._conn.execute("DELETE FROM generated_assets WHERE source_id = ?", (source_id,))
            if self._has_table("source_segments"):
                self._conn.execute("DELETE FROM source_segments WHERE source_id = ?", (source_id,))
            if self._has_table("source_documents"):
                self._conn.execute("DELETE FROM source_documents WHERE id = ?", (source_id,))

    def list_source_segments_brief(self) -> list[tuple[str, str, str, str]]:
        if not self._has_table("source_segments"):
            return []
        rows = self._conn.execute(
            "SELECT id, identity, source_id, content_hash FROM source_segments ORDER BY ordinal, id"
        ).fetchall()
        return [(row["id"], row["identity"], row["source_id"], row["content_hash"]) for row in rows]

    def list_metric_rollups(self) -> list[sqlite3.Row]:
        if not self._has_table("metric_daily_rollups"):
            return []
        return self._conn.execute(
            "SELECT * FROM metric_daily_rollups ORDER BY day, vault_id, event_type, tier"
        ).fetchall()

    def list_metric_events(self) -> list[sqlite3.Row]:
        if not self._has_table("metric_events"):
            return []
        return self._conn.execute("SELECT * FROM metric_events ORDER BY ts, id").fetchall()

    def insert_metric_event(
        self,
        *,
        ts: str,
        vault_id: str | None,
        event_type: str,
        model: str | None,
        tier: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency_ms: int | None,
        success: bool | None,
        source_id: str | None = None,
        hash_source_id: bool = True,
        metadata_json: str | None = None,
    ) -> None:
        if not self._has_table("metric_events"):
            return
        source_id_hash = None
        if source_id:
            source_id_hash = (
                hashlib.sha256(source_id.encode("utf-8")).hexdigest()
                if hash_source_id
                else source_id
            )
        with self._tx():
            self._conn.execute(
                """INSERT INTO metric_events
                   (ts, vault_id, event_type, model, tier, prompt_tokens,
                     completion_tokens, latency_ms, success, source_id_hash,
                     metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ts,
                    vault_id,
                    event_type,
                    model,
                    tier,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                    None if success is None else (1 if success else 0),
                    source_id_hash,
                    metadata_json,
                ),
            )

    def insert_mcp_audit_event(
        self,
        *,
        ts: str,
        vault_id: str,
        tool: str,
        args_summary: dict[str, str | None],
        latency_ms: int,
        success: bool,
    ) -> None:
        self.insert_metric_event(
            ts=ts,
            vault_id=vault_id,
            event_type="mcp_call",
            model=None,
            tier=None,
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=latency_ms,
            success=success,
            metadata_json=json.dumps({"tool": tool, "args": args_summary}, sort_keys=True),
        )

    def upsert_metric_rollup(
        self,
        *,
        day: str,
        vault_id: str,
        event_type: str,
        tier: str,
        calls: int,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms_total: int,
        successes: int,
        failures: int,
    ) -> None:
        if not self._has_table("metric_daily_rollups"):
            return
        with self._tx():
            self._conn.execute(
                """INSERT INTO metric_daily_rollups
                   (day, vault_id, event_type, tier, calls, prompt_tokens,
                     completion_tokens, latency_ms_total, successes, failures)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(day, vault_id, event_type, tier) DO UPDATE SET
                     calls = calls + excluded.calls,
                     prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                     completion_tokens = completion_tokens + excluded.completion_tokens,
                     latency_ms_total = latency_ms_total + excluded.latency_ms_total,
                     successes = successes + excluded.successes,
                     failures = failures + excluded.failures""",
                (
                    day,
                    vault_id,
                    event_type,
                    tier,
                    calls,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms_total,
                    successes,
                    failures,
                ),
            )

    def delete_metrics_before(self, *, cutoff_ts: str, cutoff_day: str | None = None) -> int:
        deleted = 0
        with self._tx():
            if self._has_table("metric_events"):
                cur = self._conn.execute("DELETE FROM metric_events WHERE ts < ?", (cutoff_ts,))
                deleted += cur.rowcount or 0
            if cutoff_day is not None and self._has_table("metric_daily_rollups"):
                self._conn.execute("DELETE FROM metric_daily_rollups WHERE day < ?", (cutoff_day,))
        return deleted

    def trim_oldest_metric_events(self, *, divisor: int = 10) -> int:
        if divisor <= 0 or not self._has_table("metric_events"):
            return 0
        with self._tx():
            cur = self._conn.execute(
                "DELETE FROM metric_events WHERE id IN ("
                " SELECT id FROM metric_events ORDER BY ts ASC, id ASC"
                " LIMIT (SELECT COUNT(*) / ? FROM metric_events)"
                ")",
                (divisor,),
            )
        return cur.rowcount or 0

    def clear_metrics(self) -> int:
        """Delete all metric rows. Returns total rows deleted."""
        deleted = 0
        with self._tx():
            if self._has_table("metric_events"):
                deleted += self._conn.execute("DELETE FROM metric_events").rowcount or 0
            if self._has_table("metric_daily_rollups"):
                deleted += self._conn.execute("DELETE FROM metric_daily_rollups").rowcount or 0
        return deleted

    def database_size_bytes(self) -> int:
        row = self._conn.execute("PRAGMA page_count").fetchone()
        page_count = int(row[0]) if row else 0
        row = self._conn.execute("PRAGMA page_size").fetchone()
        page_size = int(row[0]) if row else 0
        return page_count * page_size

    def metric_rollup_totals(self, *, since_day: date | None = None) -> dict[str, int]:
        if not self._has_table("metric_daily_rollups"):
            return {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms_total": 0,
                "successes": 0,
                "failures": 0,
            }

        query = (
            "SELECT "
            "COALESCE(SUM(calls), 0) AS calls, "
            "COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, "
            "COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
            "COALESCE(SUM(latency_ms_total), 0) AS latency_ms_total, "
            "COALESCE(SUM(successes), 0) AS successes, "
            "COALESCE(SUM(failures), 0) AS failures "
            "FROM metric_daily_rollups"
        )
        params: tuple[str, ...] = ()
        if since_day is not None:
            query += " WHERE day >= ?"
            params = (since_day.isoformat(),)
        row = self._conn.execute(query, params).fetchone()
        return {
            "calls": int(row["calls"]),
            "prompt_tokens": int(row["prompt_tokens"]),
            "completion_tokens": int(row["completion_tokens"]),
            "latency_ms_total": int(row["latency_ms_total"]),
            "successes": int(row["successes"]),
            "failures": int(row["failures"]),
        }

    def metric_event_totals(self, *, since_ts: str | None = None) -> dict[str, int]:
        if not self._has_table("metric_events"):
            return {
                "events": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms_total": 0,
                "successes": 0,
                "failures": 0,
            }

        query = (
            "SELECT "
            "COUNT(*) AS events, "
            "COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, "
            "COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
            "COALESCE(SUM(latency_ms), 0) AS latency_ms_total, "
            "COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) AS successes, "
            "COALESCE(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END), 0) AS failures "
            "FROM metric_events"
        )
        params: tuple[str, ...] = ()
        if since_ts is not None:
            query += " WHERE ts >= ?"
            params = (since_ts,)
        row = self._conn.execute(query, params).fetchone()
        return {
            "events": int(row["events"]),
            "prompt_tokens": int(row["prompt_tokens"]),
            "completion_tokens": int(row["completion_tokens"]),
            "latency_ms_total": int(row["latency_ms_total"]),
            "successes": int(row["successes"]),
            "failures": int(row["failures"]),
        }

    def metric_event_model_totals(
        self, *, since_ts: str | None = None
    ) -> list[tuple[str, int, int]]:
        if not self._has_table("metric_events"):
            return []

        query = (
            "SELECT model, "
            "COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, "
            "COALESCE(SUM(completion_tokens), 0) AS completion_tokens "
            "FROM metric_events WHERE model IS NOT NULL AND model <> ''"
        )
        params: tuple[str, ...] = ()
        if since_ts is not None:
            query += " AND ts >= ?"
            params = (since_ts,)
        query += " GROUP BY model ORDER BY model"
        rows = self._conn.execute(query, params).fetchall()
        return [
            (str(row["model"]), int(row["prompt_tokens"]), int(row["completion_tokens"]))
            for row in rows
        ]


# ── Row converters ────────────────────────────────────────────────────────────


def _row_to_raw(row: sqlite3.Row) -> RawNoteRecord:
    keys = row.keys()
    return RawNoteRecord(
        path=row["path"],
        content_hash=row["content_hash"],
        status=row["status"],
        summary=row["summary"] if "summary" in keys else None,
        quality=row["quality"] if "quality" in keys else None,
        language=row["language"] if "language" in keys else None,
        prompt_version=row["prompt_version"] if "prompt_version" in keys else None,
        ingested_at=datetime.fromisoformat(row["ingested_at"]) if row["ingested_at"] else None,
        compiled_at=datetime.fromisoformat(row["compiled_at"]) if row["compiled_at"] else None,
        error=row["error"],
    )


def _row_to_article(row: sqlite3.Row) -> WikiArticleRecord:
    keys = row.keys()
    return WikiArticleRecord(
        path=row["path"],
        title=row["title"],
        sources=json.loads(row["sources"]),
        content_hash=row["content_hash"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        status=row["status"],
        approved_at=(
            datetime.fromisoformat(row["approved_at"])
            if "approved_at" in keys and row["approved_at"]
            else None
        ),
        approval_notes=row["approval_notes"] if "approval_notes" in keys else None,
        kind=row["kind"] if "kind" in keys else "concept",
        question_hash=row["question_hash"] if "question_hash" in keys else None,
        synthesis_sources=(
            json.loads(row["synthesis_sources"])
            if "synthesis_sources" in keys and row["synthesis_sources"]
            else []
        ),
        synthesis_source_hashes=(
            json.loads(row["synthesis_source_hashes"])
            if "synthesis_source_hashes" in keys and row["synthesis_source_hashes"]
            else []
        ),
        article_id=row["article_id"] if "article_id" in keys else None,
        last_compile_pipeline=(
            row["last_compile_pipeline"] if "last_compile_pipeline" in keys else None
        ),
    )


def _row_to_item(row: sqlite3.Row) -> KnowledgeItemRecord:
    return KnowledgeItemRecord(
        name=row["name"],
        kind=row["kind"],
        subtype=row["subtype"],
        status=row["status"],
        confidence=float(row["confidence"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_item_mention(row: sqlite3.Row) -> ItemMentionRecord:
    return ItemMentionRecord(
        id=row["id"],
        item_name=row["item_name"],
        source_path=row["source_path"],
        mention_text=row["mention_text"],
        context=row["context"],
        evidence_level=row["evidence_level"],
        confidence=float(row["confidence"]),
    )
