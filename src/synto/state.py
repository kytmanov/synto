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
  v18 — concept_entities + concept_labels (entity identity layer; backfill from
         concepts + concept_aliases; resolve_label seam; INDEX.json seed extended)
  v19 — concept_occurrences extended (entity_id, surface, resolution_status, source_path,
         nullable source_segment_id); concept_occurrence_candidates child table
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import sqlite3
import threading
import time
import types
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from .concept_text import concept_key as _ck
from .concept_text import match_key as _mk
from .models import ItemMentionRecord, KnowledgeItemRecord, RawNoteRecord, WikiArticleRecord
from .paths import rel_posix, to_posix

log = logging.getLogger(__name__)

# Vault-relative paths are stored with POSIX separators so a vault is portable across OSes
# (#55). `_normalizes_db_paths` is applied to StateDB so its methods never carry inline
# normalization: any argument named below — typed str / list[str] — is POSIX-normalized at the
# call boundary. Filesystem-path args (e.g. `db_path`, the `Path`-typed `path` of
# `_infer_orphan_draft_status`) are skipped by the runtime isinstance(str) guard, and a
# completeness test pins the convention so a future path-named param can't slip through.
_DB_PATH_PARAMS = frozenset({"path", "source_path", "old_path", "new_path", "article_path"})
_DB_PATH_LIST_PARAMS = frozenset({"paths", "source_paths"})


def _normalizes_db_paths(cls: type) -> type:
    """Wrap StateDB methods so str path arguments are POSIX-normalized at the boundary."""
    for name, func in list(vars(cls).items()):
        if not isinstance(func, types.FunctionType):
            continue  # skip classmethod/staticmethod descriptors, etc.
        params = list(inspect.signature(func).parameters.values())
        # Precompute (positional_index, name, is_list) once per method, so call-time work is a
        # tiny fixed loop rather than a Signature.bind. Index counts `self` at 0.
        targets = [
            (i, p.name, p.name in _DB_PATH_LIST_PARAMS)
            for i, p in enumerate(params)
            if p.name in _DB_PATH_PARAMS or p.name in _DB_PATH_LIST_PARAMS
        ]
        if not targets:
            continue
        setattr(cls, name, _wrap_path_normalizer(func, targets))
    return cls


def _wrap_path_normalizer(func, targets):
    def _norm(value, is_list):
        if is_list and isinstance(value, list):
            return [to_posix(v) if isinstance(v, str) else v for v in value]
        return to_posix(value) if isinstance(value, str) else value

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_list = None
        for idx, pname, is_list in targets:
            if pname in kwargs:
                kwargs[pname] = _norm(kwargs[pname], is_list)
            elif idx < len(args):
                if args_list is None:
                    args_list = list(args)
                args_list[idx] = _norm(args_list[idx], is_list)
        return func(*(args_list if args_list is not None else args), **kwargs)

    return wrapper


def _hash8(value: str) -> str:
    """8-char SHA256 prefix used for hashed MCP audit labels.

    Single source of truth: serve._audit() writes resolved labels through this
    when audit_detailed is off, and the backlog report matches against it. The
    two must stay identical, so serve.py imports this rather than redefining it.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def _fts5_available(conn: sqlite3.Connection) -> bool:
    """True if this SQLite build supports FTS5.

    Probes with a throwaway temp virtual table — the only reliable check that
    also catches FTS5 loaded as a runtime extension (PRAGMA compile_options
    misses that case). Side-effect-free: the probe table lives in the
    connection-scoped temp schema and is dropped immediately. DDL runs in
    autocommit under sqlite3's default isolation, so this never opens or
    disturbs a transaction.
    """
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.__synto_fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS temp.__synto_fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


_CURRENT_SCHEMA_VERSION = 26


@dataclass
class ResolveResult:
    """Result from resolve_label: a list of entity IDs + ambiguity flag."""

    ids: list[str] = field(default_factory=list)
    ambiguous: bool = False


_CHECKPOINT_SCHEMA_VERSION = 2
_SESSION_IDLE_GAP_SECONDS = 1800  # 30-min idle gap that splits MCP sessions in the backlog report

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
    entity_id   TEXT NOT NULL,
    source_path TEXT NOT NULL,
    name        TEXT NOT NULL,
    PRIMARY KEY (entity_id, source_path)
);
-- idx_concepts_entity is created in the v22 post-hook, NOT here: _SCHEMA runs on
-- every open before migrations, and a pre-v22 `concepts` table (PK name) has no
-- entity_id column yet, so indexing it here would fail on the upgrade path.

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
    last_compile_pipeline TEXT,
    entity_id      TEXT
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
    source_segment_id TEXT,
    source_path       TEXT,
    ordinal           INTEGER NOT NULL DEFAULT 0,
    confidence        REAL NOT NULL DEFAULT 1.0,
    extraction_run    TEXT,
    entity_id         TEXT,
    surface           TEXT,
    resolution_status TEXT NOT NULL DEFAULT 'unresolved'
                      CHECK (resolution_status IN ('resolved', 'ambiguous', 'unresolved'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_occ_seg
    ON concept_occurrences(concept_name, source_segment_id)
    WHERE source_segment_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_concept_occurrences_concept ON concept_occurrences(concept_name);
-- idx_occ_path (source_path) and idx_concept_occurrences_entity (entity_id) are NOT created
-- here: those columns are added by the v19 migration, and this base schema runs (via
-- executescript) BEFORE migrations. On a pre-v19 vault the columns don't exist yet, so
-- creating the indexes here would raise "no such column" and block the upgrade. They are
-- created in _extend_occurrences_v19 instead, which runs for both fresh and upgraded DBs.

CREATE TABLE IF NOT EXISTS concept_occurrence_candidates (
    occurrence_id INTEGER NOT NULL REFERENCES concept_occurrences(id) ON DELETE CASCADE,
    entity_id     TEXT NOT NULL,
    PRIMARY KEY (occurrence_id, entity_id)
);

CREATE TABLE IF NOT EXISTS concept_entities (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL DEFAULT 'concept',
    status      TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'merged')),
    merged_into TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS concept_labels (
    entity_id  TEXT NOT NULL REFERENCES concept_entities(id),
    label      TEXT NOT NULL,
    label_key  TEXT NOT NULL,
    match_key  TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('preferred', 'alias')),
    source     TEXT NOT NULL
                   CHECK (source IN ('extracted', 'user', 'rename', 'legacy_backfill')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (entity_id, label_key)
);

CREATE INDEX IF NOT EXISTS idx_concept_labels_label_key ON concept_labels(label_key);
CREATE INDEX IF NOT EXISTS idx_concept_labels_match_key ON concept_labels(match_key);
CREATE UNIQUE INDEX IF NOT EXISTS idx_concept_labels_preferred_global
    ON concept_labels(label_key) WHERE role = 'preferred';
CREATE UNIQUE INDEX IF NOT EXISTS idx_concept_labels_preferred_per_entity
    ON concept_labels(entity_id) WHERE role = 'preferred';

CREATE TABLE IF NOT EXISTS concept_identity_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    op         TEXT NOT NULL CHECK (op IN ('merge', 'split', 'rename', 'unmerge')),
    entity_ids TEXT NOT NULL,
    labels     TEXT NOT NULL,
    meta       TEXT,
    ts         TEXT NOT NULL
);

-- Advisory worklist (v26): pairs the identity rule promoted apart but that a human may want
-- merged (e.g. "GD" extracted as a concept after it was a weak alias of "Gradient Descent").
-- The pair is ordered by preferred label_key (NOT entity_id, which is random) so the same
-- logical pair dedups to one row regardless of ingest order. Advisory only; re-derived on
-- re-ingest, so it stays out of the .synto/INDEX.json durability seed.
CREATE TABLE IF NOT EXISTS concept_merge_candidates (
    entity_a   TEXT NOT NULL,
    entity_b   TEXT NOT NULL,
    surface    TEXT NOT NULL,
    reason     TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (entity_a, entity_b, surface)
);
"""

# v17 separator-collision tables resolved by recency, not "POSIX wins" (issue #55 follow-up).
# Each entry: (table, path_col, other_key_cols, recency_expr, state_cols). These tables key on
# a vault-relative path and carry mutable state, so a vault used on two OSes can hold both
# ``X\Y`` and ``X/Y`` forms of one logical row; dropping the backslash row blindly would
# discard newer state. recency_expr is a per-row SQL expression with an ``{a}`` alias
# placeholder; state_cols are the columns whose divergence is worth logging.
# wiki_articles is resolved separately (_resolve_article_separator_collisions_v17) because it
# also merges approval audit. concepts/generated_assets are NOT here: concepts carries only its
# PK (a duplicate is lossless) and generated_assets.path is always POSIX (never collides).
_RECENCY_PATH_COLLISIONS: list[tuple[str, str, tuple[str, ...], str, tuple[str, ...]]] = [
    (
        "concept_compile_state",
        "source_path",
        ("concept_name",),
        "{a}.updated_at",
        ("status", "error", "compiled_at"),
    ),
    (
        "ingest_chunks",
        "source_path",
        ("content_hash", "chunk_index", "chunk_count", "chunk_size", "checkpoint_schema"),
        "{a}.updated_at",
        ("result_json",),
    ),
    (
        "raw_notes",
        "path",
        (),
        "MAX(COALESCE({a}.compiled_at, ''), COALESCE({a}.ingested_at, ''))",
        ("content_hash", "status", "prompt_version"),
    ),
    (
        "item_mentions",
        "source_path",
        ("item_name", "mention_text", "evidence_level"),
        # No timestamp on mentions; only exact-duplicate mentions collide, so the higher
        # confidence is a principled, non-POSIX-biased tiebreak.
        "{a}.confidence",
        ("confidence", "context"),
    ),
]

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
    17: [],  # all v17 work happens in the post-hook below for atomicity
    18: [
        # Entity identity layer: stable opaque IDs + label sets (feature 45, Phase 1).
        # Backfill from concepts + concept_aliases happens in the post-hook.
        """CREATE TABLE IF NOT EXISTS concept_entities (
               id          TEXT PRIMARY KEY,
               kind        TEXT NOT NULL DEFAULT 'concept',
               status      TEXT NOT NULL DEFAULT 'active'
                               CHECK (status IN ('active', 'merged')),
               merged_into TEXT,
               created_at  TEXT NOT NULL,
               updated_at  TEXT NOT NULL
           )""",
        """CREATE TABLE IF NOT EXISTS concept_labels (
               entity_id  TEXT NOT NULL REFERENCES concept_entities(id),
               label      TEXT NOT NULL,
               label_key  TEXT NOT NULL,
               match_key  TEXT NOT NULL,
               role       TEXT NOT NULL CHECK (role IN ('preferred', 'alias')),
               source     TEXT NOT NULL
                              CHECK (source IN ('extracted', 'user', 'rename', 'legacy_backfill')),
               created_at TEXT NOT NULL,
               PRIMARY KEY (entity_id, label_key)
           )""",
        "CREATE INDEX IF NOT EXISTS idx_concept_labels_label_key ON concept_labels(label_key)",
        "CREATE INDEX IF NOT EXISTS idx_concept_labels_match_key ON concept_labels(match_key)",
        (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_concept_labels_preferred_global"
            " ON concept_labels(label_key) WHERE role = 'preferred'"
        ),
        (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_concept_labels_preferred_per_entity"
            " ON concept_labels(entity_id) WHERE role = 'preferred'"
        ),
    ],
    19: [],  # all v19 work happens in the post-hook (_extend_occurrences_v19) for atomicity
    20: ["DROP TABLE IF EXISTS concept_aliases"],
    21: [
        """CREATE TABLE IF NOT EXISTS concept_identity_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            op         TEXT NOT NULL CHECK (op IN ('merge', 'split', 'rename')),
            entity_ids TEXT NOT NULL,
            labels     TEXT NOT NULL,
            meta       TEXT,
            ts         TEXT NOT NULL
        )"""
    ],
    22: [],  # rebuild `concepts` onto PK (entity_id, source_path) in the post-hook for atomicity
    23: [],  # rebuild `concept_compile_state` onto PK (entity_id, source_path) in the post-hook
    24: ["ALTER TABLE wiki_articles ADD COLUMN entity_id TEXT"],  # backfill in the post-hook
    25: [],  # rebuild concept_identity_log to allow op='unmerge' in the post-hook for atomicity
    26: [
        # Advisory merge-candidate worklist (single CREATE — atomic on its own, no backfill).
        """CREATE TABLE IF NOT EXISTS concept_merge_candidates (
            entity_a   TEXT NOT NULL,
            entity_b   TEXT NOT NULL,
            surface    TEXT NOT NULL,
            reason     TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (entity_a, entity_b, surface)
        )"""
    ],
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


@_normalizes_db_paths
class StateDB:
    """SQLite state store backed by a single Connection shared across threads.

    Threading contract (the connection is opened with check_same_thread=False, so callers — not
    the driver — are responsible for safe concurrent use):

    - Every write / transaction-control statement (BEGIN/COMMIT/SAVEPOINT, and therefore every
      INSERT/UPDATE/DELETE) MUST go through ``_tx()``, which holds ``_lock`` for the whole
      transaction. This is what makes parallel ingest safe (#75): concurrent writers — e.g. the
      LLM cache on a worker thread vs. checkpoint writes on the main thread — are serialized, so no
      thread can commit inside another's open transaction.
    - A read that can run concurrently with a writer MUST use ``_read()`` (same re-entrant lock, no
      commit). It then serializes against writers — it waits, then reads committed state — rather
      than racing. Do not assume a bare ``self._conn`` SELECT is safe under concurrency: CPython's
      sqlite3 Connection shares a statement cache + transaction bookkeeping across threads and
      releases the GIL inside ``sqlite3_step``.
    - Bare ``self._conn`` reads are permitted ONLY on single-threaded / main-thread paths (the bulk
      of query methods here, plus a few external read-only callers). They are not wrapped because no
      concurrent-reader path exists today and locking every query would tax the common single-thread
      case for no correctness gain.
    - The only writes outside ``_tx()`` are the construction/migration bootstrap commits below,
      which run on the creating thread before any worker thread exists.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._tx_depth = 0
        # The connection is shared across threads (check_same_thread=False) — parallel ingest
        # runs chunk-analysis workers that write via the LLM cache while the main thread writes
        # checkpoints. Re-entrant because _tx() nests (savepoints) and StateDB methods call other
        # _tx() methods on the same thread. Every transaction is serialized through this lock so
        # no two threads can corrupt the connection's transaction state (#75).
        self._lock = threading.RLock()
        self._conn.executescript(_SCHEMA)
        # Construction-only commit: runs on the creating thread before any worker exists, so it is
        # intentionally outside the _tx()/_lock contract (no concurrency to serialize against yet).
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
        # open_readonly bypasses __init__ (cls.__new__), so _lock must be set here too: a
        # read-only StateDB can still have an LLMCache attached (stats.py) whose ops route
        # through _tx(), which acquires this lock.
        self._lock = threading.RLock()
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
            # Migration runs during __init__ on the creating thread; bare commit is intentional
            # (single-threaded, before workers exist — outside the _tx()/_lock contract).
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
            if version == 17:
                self._normalize_path_separators_v17()
            if version == 18:
                self._backfill_entities_v18()
            if version == 19:
                self._extend_occurrences_v19()
            if version == 22:
                self._rebuild_concepts_on_entity_id_v22()
            if version == 23:
                self._rebuild_compile_state_on_entity_id_v23()
            if version == 24:
                self._backfill_article_entities_v24()
            if version == 25:
                self._expand_identity_log_ops_v25()
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

        Skips gracefully on SQLite builds without FTS5: the table/triggers are
        not created and the migration still returns normally so the schema
        version advances and every non-MCP command keeps working. Only
        `search_source_segments` is unavailable in that case — the other three
        verbatim tools query source_segments directly and are unaffected.

        Atomic via _tx(); idempotent via sqlite_master probe.
        """
        exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_segments_fts'"
        ).fetchone()
        if exists:
            return
        if not _fts5_available(self._conn):
            log.warning(
                "This SQLite build lacks the FTS5 module; the verbatim search index was "
                "not created. All other synto features work normally. Rebuild Python/SQLite "
                "with FTS5 to enable the search_source_segments MCP tool."
            )
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

    def _normalize_path_separators_v17(self) -> None:
        r"""Rewrite Windows-style path separators in stored paths to POSIX (issue #55).

        A vault built on Windows stored vault-relative paths like ``raw\note.md``; moved to
        Linux, the same note resolves to ``raw/note.md``, so dedup/lookup treated every note
        as a duplicate and skipped it. This repairs already-built DBs on first open.

        Atomic via _tx(); idempotent — ``REPLACE`` and re-encoding are no-ops once paths are
        already POSIX, so re-runs and Linux/macOS-built DBs (no backslashes) are unaffected.
        Tradeoff: a backslash is a legal POSIX filename char and would also be rewritten, but
        synto vault paths (Obsidian/markdown filenames) never contain one.

        JSON list columns (``sources``/``synthesis_sources``) are decoded and re-encoded, not
        REPLACE'd: a Windows path is JSON-escaped on disk as ``raw\\note.md``, so a naive
        ``REPLACE('\', '/')`` would corrupt it to ``raw//note.md``.

        Collision-safe without silent loss: a vault used on two OSes can hold both
        ``wiki\Qubit.md`` and ``wiki/Qubit.md`` for one logical row. Any path column that
        carries mutable state is resolved by recency (newer wins, last-write-wins like upserts)
        BEFORE the REPLACE below, so the newer of a ``X\Y`` / ``X/Y`` pair is never silently
        dropped: ``wiki_articles`` via ``_resolve_article_separator_collisions_v17`` (which
        also preserves approval audit) and the ``_RECENCY_PATH_COLLISIONS`` tables via
        ``_resolve_separator_collisions_by_recency_v17``. ``concepts`` carries only its PK so a
        separator-duplicate is lossless, and ``generated_assets.path`` is always POSIX; for
        those the REPLACE pass's ``UPDATE OR IGNORE`` keeps the POSIX row and drops the
        backslash duplicate, so the migration can't abort on a uniqueness violation.
        """
        # Vault-relative path KEYS only — these are looked up across OSes, so their separators
        # must be normalized. Absolute/external columns are deliberately excluded because
        # rewriting their separators would corrupt the stored value, not portability-fix it:
        #   - generated_assets.master_path: absolute imported-source path, e.g.
        #     C:\Users\alice\paper.pdf (extractors/pdf.py writes str(path)).
        #   - source_documents.origin_uri: external URI / origin of an imported document.
        #
        # Every path column lives in exactly one bucket so none silently falls back to
        # "POSIX wins". Resolved tables (recency + wiki_articles) have their twins removed
        # before the REPLACE pass, so a leftover backslash row there is unexpected (an exotic
        # mixed-separator path with no clean-POSIX twin — synto writes homogeneous separators
        # via str(path.relative_to(vault)), so this should never happen) and is logged at
        # ERROR. Lossless tables (concepts/generated_assets) can't lose meaningful state, so a
        # leftover there is only WARNed.
        recency_columns = [(table, col) for (table, col, *_rest) in _RECENCY_PATH_COLLISIONS]
        resolved_columns = recency_columns + [("wiki_articles", "path")]
        lossless_columns = [("concepts", "source_path"), ("generated_assets", "path")]
        all_columns = resolved_columns + lossless_columns
        with self._tx():
            # Resolve collisions by recency first so the REPLACE below can't collide and can't
            # silently drop the newer of a ``X\Y`` / ``X/Y`` pair.
            self._resolve_article_separator_collisions_v17()
            for (
                table,
                path_col,
                other_key_cols,
                recency_expr,
                state_cols,
            ) in _RECENCY_PATH_COLLISIONS:
                self._resolve_separator_collisions_by_recency_v17(
                    table, path_col, other_key_cols, recency_expr, state_cols
                )
            for table, col in all_columns:
                self._conn.execute(
                    f"UPDATE OR IGNORE {table} SET {col} = REPLACE({col}, '\\', '/')"
                )
                # Rows whose normalized path collided with an existing POSIX row keep their
                # backslash value after UPDATE OR IGNORE; drop them, but log first so the drop
                # is never silent. ERROR for resolved tables (unexpected — see comment above),
                # WARNING for lossless ones.
                leftover = self._conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE INSTR({col}, '\\') > 0"
                ).fetchone()[0]
                if leftover:
                    emit = log.error if (table, col) in resolved_columns else log.warning
                    emit(
                        "v17 migration: dropping %d duplicate %s.%s row(s) whose normalized "
                        "path collided with an existing POSIX row",
                        leftover,
                        table,
                        col,
                    )
                self._conn.execute(f"DELETE FROM {table} WHERE INSTR({col}, '\\') > 0")
            for col in ("sources", "synthesis_sources"):
                rows = self._conn.execute(
                    f"SELECT rowid AS rid, {col} AS val FROM wiki_articles"
                ).fetchall()
                for row in rows:
                    raw = row["val"]
                    if not raw or "\\" not in raw:
                        continue
                    fixed = [to_posix(s) if isinstance(s, str) else s for s in json.loads(raw)]
                    self._conn.execute(
                        f"UPDATE wiki_articles SET {col} = ? WHERE rowid = ?",
                        (json.dumps(fixed), row["rid"]),
                    )

    def _resolve_article_separator_collisions_v17(self) -> None:
        r"""Resolve wiki_articles rows that exist under both ``wiki\X`` and ``wiki/X``.

        A vault compiled on two OSes (compile has no dedup guard) can hold both forms. Keep
        the row with the newer ``updated_at`` (last-write-wins, matching upsert semantics)
        rather than always preferring POSIX — the backslash row may carry newer status /
        approval / content_hash. Approval audit is preserved from either row via COALESCE
        (the invariant `verify_article`/`approve_article` already uphold), and a real
        divergence is logged so the dropped state is never silent. Caller runs inside _tx().
        """
        pairs = self._conn.execute(
            r"""
            SELECT b.path AS bpath, b.updated_at AS bupd, b.status AS bstatus,
                   b.content_hash AS bhash, b.approved_at AS bappr, b.approval_notes AS bnotes,
                   p.path AS ppath, p.updated_at AS pupd, p.status AS pstatus,
                   p.content_hash AS phash, p.approved_at AS pappr, p.approval_notes AS pnotes
            FROM wiki_articles b
            JOIN wiki_articles p ON p.path = REPLACE(b.path, '\', '/')
            WHERE INSTR(b.path, '\') > 0
            """
        ).fetchall()
        for r in pairs:
            # (path, updated_at, status, content_hash, approved_at, approval_notes)
            backslash = (r["bpath"], r["bupd"], r["bstatus"], r["bhash"], r["bappr"], r["bnotes"])
            posix = (r["ppath"], r["pupd"], r["pstatus"], r["phash"], r["pappr"], r["pnotes"])
            # ISO `updated_at` strings sort chronologically; tie keeps POSIX.
            winner, loser = (backslash, posix) if backslash[1] > posix[1] else (posix, backslash)
            if backslash[2] != posix[2] or backslash[3] != posix[3]:
                log.warning(
                    "v17 migration: separator-duplicate article diverged; keeping newer %r "
                    "(status=%s) and dropping %r (status=%s)",
                    winner[0],
                    winner[2],
                    loser[0],
                    loser[2],
                )
            # Preserve approval audit from the loser if the winner lacks it.
            self._conn.execute(
                "UPDATE wiki_articles SET approved_at = COALESCE(approved_at, ?), "
                "approval_notes = COALESCE(approval_notes, ?) WHERE path = ?",
                (loser[4], loser[5], winner[0]),
            )
            self._conn.execute("DELETE FROM wiki_articles WHERE path = ?", (loser[0],))

    def _resolve_separator_collisions_by_recency_v17(
        self,
        table: str,
        key_col: str,
        other_key_cols: tuple[str, ...],
        recency_expr: str,
        state_cols: tuple[str, ...],
    ) -> None:
        r"""Resolve ``table`` rows that exist under both ``X\Y`` and ``X/Y`` separator forms.

        Generalizes ``_resolve_article_separator_collisions_v17`` for the path-keyed tables
        that carry mutable state (``_RECENCY_PATH_COLLISIONS``). A vault used on two OSes can
        hold both forms of the same logical key; keep the row whose ``recency_expr`` is newer
        (last-write-wins) instead of always preferring POSIX, which would silently discard
        newer status / checkpoint / extraction state. Tie keeps POSIX, matching the article
        resolver. A divergence in ``state_cols`` is logged so dropped state is never silent;
        a byte-identical re-ingest dedup resolves quietly. Caller runs inside _tx().

        Identifiers come from the hardcoded ``_RECENCY_PATH_COLLISIONS`` table (no user input),
        so f-string interpolation is safe.
        """
        # Null-safe key matching (``IS``, not ``=``): a NULL secondary key must still pair a
        # twin, else the row would bypass this resolver and be dropped by the generic cleanup.
        # Consistent with the ``IS NOT`` divergence clause below. Current key columns are all
        # NOT NULL, so this only hardens the helper for future _RECENCY_PATH_COLLISIONS entries.
        key_eq = "".join(f" AND b.{c} IS p.{c}" for c in other_key_cols)
        # ``p`` is the clean-POSIX twin of backslash row ``b`` (and is distinct from ``b``,
        # since REPLACE strips the backslash that ``b`` is required to have).
        join = (
            f"FROM {table} b JOIN {table} p "
            f"ON p.{key_col} = REPLACE(b.{key_col}, '\\', '/'){key_eq} "
            f"WHERE INSTR(b.{key_col}, '\\') > 0"
        )
        rb, rp = recency_expr.format(a="b"), recency_expr.format(a="p")
        divergence = " OR ".join(f"b.{c} IS NOT p.{c}" for c in state_cols)
        diverged = self._conn.execute(f"SELECT COUNT(*) {join} AND ({divergence})").fetchone()[0]
        if diverged:
            log.warning(
                "v17 migration: resolving %d separator-duplicate %s row(s) with divergent "
                "state by recency (newer wins, older dropped)",
                diverged,
                table,
            )
        # Step 1: drop the POSIX twin of each backslash row that is strictly newer. Its
        # backslash winner now has no twin, so it survives step 2 and the REPLACE pass.
        self._conn.execute(
            f"DELETE FROM {table} WHERE rowid IN (SELECT p.rowid {join} AND {rb} > {rp})"
        )
        # Step 2: drop the remaining backslash rows that still have a POSIX twin (POSIX won, or
        # a recency tie). The REPLACE pass then normalizes the surviving rows.
        self._conn.execute(f"DELETE FROM {table} WHERE rowid IN (SELECT b.rowid {join})")

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

    def _backfill_entities_v18(self) -> None:
        """Populate concept_entities + concept_labels from existing concepts + concept_aliases.

        Idempotent: skips the whole backfill if concept_entities already has rows.
        One entity per distinct canonical concept name; aliases from concept_aliases
        land as role='alias', source='legacy_backfill' (untrusted, not blessed).
        match_key collisions among preferred labels are expected (User/Users, issue #54)
        and are logged as the dedup worklist — they do not abort the migration.
        """
        existing = self._conn.execute("SELECT COUNT(*) FROM concept_entities").fetchone()[0]
        if existing > 0:
            return

        now = datetime.now().isoformat()
        names = self._conn.execute("SELECT DISTINCT name FROM concepts ORDER BY name").fetchall()

        # Pass 1: create entities + preferred labels, build name→entity_id mapping.
        name_to_id: dict[str, str] = {}
        seen_preferred_keys: set[str] = set()
        with self._tx():
            for row in names:
                name = row[0]
                lk = _ck(name)
                mk = _mk(name)
                if not lk:
                    continue
                if lk in seen_preferred_keys:
                    # Collision: two concept names share a label_key — log and skip.
                    log.warning(
                        "v18 backfill: label_key collision for %r, skipping entity creation",
                        name,
                    )
                    continue
                seen_preferred_keys.add(lk)
                entity_id = _generate_article_id()
                name_to_id[name] = entity_id
                self._conn.execute(
                    "INSERT INTO concept_entities (id, kind, status, created_at, updated_at)"
                    " VALUES (?, 'concept', 'active', ?, ?)",
                    (entity_id, now, now),
                )
                self._conn.execute(
                    "INSERT INTO concept_labels"
                    " (entity_id, label, label_key, match_key, role, source, created_at)"
                    " VALUES (?, ?, ?, ?, 'preferred', 'extracted', ?)",
                    (entity_id, name, lk, mk, now),
                )

        # Pass 2: import aliases from concept_aliases (legacy_backfill = untrusted).
        if not self._has_table("concept_aliases"):
            return
        alias_rows = self._conn.execute(
            "SELECT concept_name, alias FROM concept_aliases ORDER BY concept_name, alias"
        ).fetchall()

        # Also add knowledge_items names not yet in concepts (orphan KI rows).
        ki_names = self._conn.execute(
            "SELECT name FROM knowledge_items WHERE kind = 'concept'"
            " AND name NOT IN (SELECT DISTINCT name FROM concepts)"
        ).fetchall()
        with self._tx():
            for ki_row in ki_names:
                name = ki_row[0]
                lk = _ck(name)
                mk = _mk(name)
                if not lk or lk in seen_preferred_keys:
                    continue
                seen_preferred_keys.add(lk)
                entity_id = _generate_article_id()
                name_to_id[name] = entity_id
                self._conn.execute(
                    "INSERT INTO concept_entities (id, kind, status, created_at, updated_at)"
                    " VALUES (?, 'concept', 'active', ?, ?)",
                    (entity_id, now, now),
                )
                self._conn.execute(
                    "INSERT INTO concept_labels"
                    " (entity_id, label, label_key, match_key, role, source, created_at)"
                    " VALUES (?, ?, ?, ?, 'preferred', 'extracted', ?)",
                    (entity_id, name, lk, mk, now),
                )

        with self._tx():
            for alias_row in alias_rows:
                concept_name, alias = alias_row[0], alias_row[1]
                entity_id = name_to_id.get(concept_name)
                if entity_id is None:
                    continue
                alias_lk = _ck(alias)
                if not alias_lk or alias_lk == _ck(concept_name):
                    continue  # self-alias or empty
                if alias_lk in seen_preferred_keys:
                    # The alias is already a preferred label of another concept; importing it would
                    # make that concept resolve ambiguously (and silently drop its article). Skip it
                    # — the pair surfaces as a merge candidate via preferred-label match_key, not a
                    # polluting alias. Same invariant the extraction path enforces.
                    continue
                try:
                    self._conn.execute(
                        "INSERT INTO concept_labels"
                        " (entity_id, label, label_key, match_key, role, source, created_at)"
                        " VALUES (?, ?, ?, ?, 'alias', 'legacy_backfill', ?)",
                        (entity_id, alias, alias_lk, _mk(alias), now),
                    )
                except sqlite3.IntegrityError:
                    pass  # (entity_id, label_key) duplicate — already have this alias

        match_key_counts: dict[str, int] = {}
        mk_rows = self._conn.execute(
            "SELECT match_key FROM concept_labels WHERE role = 'preferred'"
        ).fetchall()
        for r in mk_rows:
            match_key_counts[r[0]] = match_key_counts.get(r[0], 0) + 1
        collisions = [(mk_v, n) for mk_v, n in match_key_counts.items() if n > 1]
        if collisions:
            log.info(
                "v18 backfill: %d match_key collision(s) among preferred labels"
                " (merge candidates — run `synto doctor` for details)",
                len(collisions),
            )

    def _extend_occurrences_v19(self) -> None:
        """Extend concept_occurrences + add concept_occurrence_candidates.

        Idempotent: checks for the resolution_status column before running.
        Uses table-recreation because source_segment_id must become nullable
        and the UNIQUE constraint must be split into two partial indexes.
        """
        cols = {
            r[1] for r in self._conn.execute("PRAGMA table_info(concept_occurrences)").fetchall()
        }
        has_path_index = bool(
            self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_occ_path'"
            ).fetchone()
        )
        if (
            "resolution_status" in cols
            and self._has_table("concept_occurrence_candidates")
            and has_path_index
        ):
            return
        with self._tx():
            if "resolution_status" not in cols:
                self._conn.executescript(
                    """
                    ALTER TABLE concept_occurrences RENAME TO concept_occurrences_old;

                    CREATE TABLE concept_occurrences (
                        id                INTEGER PRIMARY KEY AUTOINCREMENT,
                        concept_name      TEXT NOT NULL,
                        source_segment_id TEXT,
                        source_path       TEXT,
                        ordinal           INTEGER NOT NULL DEFAULT 0,
                        confidence        REAL NOT NULL DEFAULT 1.0,
                        extraction_run    TEXT,
                        entity_id         TEXT,
                        surface           TEXT,
                        resolution_status TEXT NOT NULL DEFAULT 'unresolved'
                                          CHECK (resolution_status IN
                                                 ('resolved', 'ambiguous', 'unresolved'))
                    );

                    INSERT INTO concept_occurrences
                        (concept_name, source_segment_id, ordinal, confidence, extraction_run)
                    SELECT concept_name, source_segment_id, ordinal, confidence, extraction_run
                    FROM concept_occurrences_old;

                    DROP TABLE concept_occurrences_old;

                    CREATE UNIQUE INDEX idx_occ_seg
                        ON concept_occurrences(concept_name, source_segment_id)
                        WHERE source_segment_id IS NOT NULL;
                    CREATE UNIQUE INDEX idx_occ_path
                        ON concept_occurrences(concept_name, source_path)
                        WHERE source_path IS NOT NULL AND source_segment_id IS NULL;
                    CREATE INDEX idx_concept_occurrences_concept
                        ON concept_occurrences(concept_name);
                    CREATE INDEX idx_concept_occurrences_entity
                        ON concept_occurrences(entity_id)
                        WHERE entity_id IS NOT NULL;
                    """
                )
            if not self._has_table("concept_occurrence_candidates"):
                self._conn.execute(
                    """CREATE TABLE IF NOT EXISTS concept_occurrence_candidates (
                        occurrence_id INTEGER NOT NULL
                                      REFERENCES concept_occurrences(id) ON DELETE CASCADE,
                        entity_id     TEXT NOT NULL,
                        PRIMARY KEY (occurrence_id, entity_id)
                    )"""
                )
            # These two indexes reference v19-added columns, so they are NOT in the base
            # _SCHEMA (which runs before migrations and would crash a pre-v19 vault). Ensure
            # them here for both the fresh-DB path (table already extended above) and after a
            # recreation. The recreation block already creates them; IF NOT EXISTS makes this
            # a no-op there.
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_occ_path"
                " ON concept_occurrences(concept_name, source_path)"
                " WHERE source_path IS NOT NULL AND source_segment_id IS NULL"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_concept_occurrences_entity"
                " ON concept_occurrences(entity_id) WHERE entity_id IS NOT NULL"
            )

    def _rebuild_concepts_on_entity_id_v22(self) -> None:
        """Rebuild `concepts` onto PK (entity_id, source_path); keep `name` as a cache.

        Feature 45 Architecture X: identity must flow through entity_id, not the
        name string. The old PK (name, source_path) made name load-bearing. This
        rebuild re-keys every source edge on the entity it already resolves to
        (via the v18 entity layer), carrying `name` as a denormalized display
        cache refreshed on rename/merge.

        Atomic via _tx(); idempotent via PRAGMA probe (skips if `concepts`
        already carries entity_id). Two old rows whose distinct names resolve to
        the same entity for one source collapse to one row (INSERT OR IGNORE) —
        the dedup this feature intends.
        """
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(concepts)").fetchall()}
        if "entity_id" not in cols:
            now = datetime.now().isoformat()
            old_rows = self._conn.execute(
                "SELECT name, source_path FROM concepts ORDER BY name, source_path"
            ).fetchall()
            with self._tx():
                self._conn.execute(
                    """CREATE TABLE concepts_new (
                        entity_id   TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        name        TEXT NOT NULL,
                        PRIMARY KEY (entity_id, source_path)
                    )"""
                )
                for row in old_rows:
                    name = row["name"]
                    source_path = row["source_path"]
                    entity_id = self._ensure_entity_for_name(name, now)
                    if entity_id is None:
                        # Degenerate name (empty label_key) — cannot key on identity; drop.
                        log.warning("v22 rebuild: dropping concept with empty label_key: %r", name)
                        continue
                    self._conn.execute(
                        "INSERT OR IGNORE INTO concepts_new (entity_id, source_path, name)"
                        " VALUES (?, ?, ?)",
                        (entity_id, source_path, name),
                    )
                self._conn.execute("DROP TABLE concepts")
                self._conn.execute("ALTER TABLE concepts_new RENAME TO concepts")

        # Recreate indexes for both paths: the rebuild above (old DBs, which drops
        # the old table and its idx_concept_name) and fresh DBs whose `concepts`
        # already carries entity_id from _SCHEMA. _SCHEMA's idx_concept_name only
        # reappears on the next open, so recreate it here for the rebuild path.
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_entity ON concepts(entity_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_name ON concepts(name)")

        # Path-stability check (feature 45 risk): compile derives the output filename from
        # the entity's preferred label. On a clean pre-45 upgrade the cached name equals the
        # preferred label by construction, so a hit here is a pre-existing anomaly worth
        # surfacing rather than hiding — its published file path may shift on next compile.
        divergent = self._conn.execute(
            """SELECT COUNT(*) FROM concepts c
               JOIN concept_labels cl ON cl.entity_id = c.entity_id AND cl.role = 'preferred'
               WHERE lower(c.name) != lower(cl.label)"""
        ).fetchone()[0]
        if divergent:
            log.warning(
                "v22: %d concept row(s) whose cached name differs from the entity's preferred "
                "label; compile output paths derive from the preferred label",
                divergent,
            )

    def _rebuild_compile_state_on_entity_id_v23(self) -> None:
        """Rebuild `concept_compile_state` onto PK (entity_id, source_path).

        Feature 45 Architecture X: compile scheduling must key on entity identity,
        not the name string, so two homonyms schedule as distinct units. `concept_name`
        is carried as a denormalized cache (refreshed on rename/merge). Unlike `concepts`,
        the _SCHEMA copy of this table stays old-shaped (PK concept_name) because the v6
        `_validate_v6_tables` hook would reject an entity_id column on the upgrade path;
        every DB therefore gains entity_id here.

        Atomic via _tx(); idempotent via PRAGMA probe (skips the rebuild if entity_id is
        already present). Two old rows whose names resolve to one entity for a source
        collapse to one row (INSERT OR IGNORE).
        """
        cols = {
            r[1] for r in self._conn.execute("PRAGMA table_info(concept_compile_state)").fetchall()
        }
        if "entity_id" not in cols:
            now = datetime.now().isoformat()
            old_rows = self._conn.execute(
                "SELECT concept_name, source_path, status, error, compiled_at, updated_at"
                " FROM concept_compile_state"
            ).fetchall()
            with self._tx():
                self._conn.execute(
                    """CREATE TABLE concept_compile_state_new (
                        entity_id    TEXT NOT NULL,
                        source_path  TEXT NOT NULL,
                        concept_name TEXT NOT NULL,
                        status       TEXT NOT NULL DEFAULT 'pending',
                        error        TEXT,
                        compiled_at  TEXT,
                        updated_at   TEXT NOT NULL,
                        PRIMARY KEY (entity_id, source_path),
                        CHECK (status IN ('pending', 'failed', 'compiled',
                                          'deferred_draft', 'deferred_manual_edit'))
                    )"""
                )
                for row in old_rows:
                    concept_name = row["concept_name"]
                    entity_id = self._ensure_entity_for_name(concept_name, now)
                    if entity_id is None:
                        log.warning(
                            "v23 rebuild: dropping compile-state with empty label_key: %r",
                            concept_name,
                        )
                        continue
                    self._conn.execute(
                        "INSERT OR IGNORE INTO concept_compile_state_new"
                        " (entity_id, source_path, concept_name, status, error,"
                        "  compiled_at, updated_at)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            entity_id,
                            row["source_path"],
                            concept_name,
                            row["status"],
                            row["error"],
                            row["compiled_at"],
                            row["updated_at"],
                        ),
                    )
                self._conn.execute("DROP TABLE concept_compile_state")
                self._conn.execute(
                    "ALTER TABLE concept_compile_state_new RENAME TO concept_compile_state"
                )
        # Recreate indexes for both the rebuild path and fresh DBs (see v22 rationale).
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_concept_compile_status"
            " ON concept_compile_state(status, source_path)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_concept_compile_name"
            " ON concept_compile_state(lower(concept_name))"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_concept_compile_entity"
            " ON concept_compile_state(entity_id)"
        )

    def _backfill_article_entities_v24(self) -> None:
        """Bind existing concept articles to their entity via the title→entity bridge.

        Only 'concept' articles are bound; synthesis & disambiguation pages keep
        entity_id NULL (no single owning entity). Idempotent: only fills NULLs, and a
        title that is ambiguous/unknown is left NULL rather than guessed.
        """
        rows = self._conn.execute(
            "SELECT path, title FROM wiki_articles WHERE kind = 'concept' AND entity_id IS NULL"
        ).fetchall()
        with self._tx():
            for row in rows:
                entity_id = self.entity_id_for_name(row["title"])
                if entity_id is None:
                    continue
                self._conn.execute(
                    "UPDATE wiki_articles SET entity_id = ? WHERE path = ?",
                    (entity_id, row["path"]),
                )

    def _expand_identity_log_ops_v25(self) -> None:
        """Allow op='unmerge' in concept_identity_log (Stage 7 reversibility).

        SQLite cannot ALTER a CHECK constraint, so rebuild the table preserving every
        row. Atomic via _tx(); idempotent via a probe of the stored table SQL (skips if
        the definition already permits 'unmerge', e.g. fresh DBs created from _SCHEMA).
        """
        if not self._has_table("concept_identity_log"):
            return
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='concept_identity_log'"
        ).fetchone()
        if row and "unmerge" in (row[0] or ""):
            return
        with self._tx():
            self._conn.execute(
                """CREATE TABLE concept_identity_log_new (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    op         TEXT NOT NULL CHECK (op IN ('merge', 'split', 'rename', 'unmerge')),
                    entity_ids TEXT NOT NULL,
                    labels     TEXT NOT NULL,
                    meta       TEXT,
                    ts         TEXT NOT NULL
                )"""
            )
            self._conn.execute(
                "INSERT INTO concept_identity_log_new (id, op, entity_ids, labels, meta, ts)"
                " SELECT id, op, entity_ids, labels, meta, ts FROM concept_identity_log"
            )
            self._conn.execute("DROP TABLE concept_identity_log")
            self._conn.execute(
                "ALTER TABLE concept_identity_log_new RENAME TO concept_identity_log"
            )

    def _backfill_compile_state_v6(self) -> None:
        self._ensure_compile_state_rows()
        alias_rows = self._conn.execute(
            "SELECT concept_name, alias FROM concept_aliases ORDER BY concept_name, alias"
        ).fetchall()
        alias_map: dict[str, set[str]] = {}
        for row in alias_rows:
            alias_map.setdefault(row["concept_name"], set()).add(row["alias"])

        articles = self.list_articles()
        now = datetime.now().isoformat()
        # mark_concept_compile_state resolves through _ensure_entity_for_name then writes via
        # _mark_compile_state_for_entity, which INSERTs an entity_id. At this v6 hook that path is
        # broken: _SCHEMA (run on every open before migrations) has already created concept_labels,
        # so _ensure_entity_for_name does NOT short-circuit — it mints an entity — but
        # concept_compile_state is still the name-keyed v6 shape (the entity_id rebuild is v23), so
        # the INSERT raises "no such column: entity_id" and aborts the whole upgrade. Write the
        # name-keyed row directly here; v23's rebuild carries it forward onto entity_id.
        with self._tx():
            for row in self._conn.execute("SELECT name, source_path FROM concepts").fetchall():
                concept_name = row["name"]
                source_path = row["source_path"]
                article = self._match_article_for_concept_v6(
                    concept_name, source_path, articles, alias_map
                )
                if article is None:
                    continue
                self._conn.execute(
                    "UPDATE concept_compile_state SET status='compiled', compiled_at=?,"
                    " updated_at=? WHERE concept_name=? AND source_path=?",
                    (now, now, concept_name, source_path),
                )

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
        # Schema-aware on the INSERT target (concept_compile_state): this runs both at
        # runtime (compile_state keyed on entity_id, v23) and from the v6 backfill
        # mid-migration, where compile_state is still name-keyed. compile_state having
        # entity_id implies v22 ran too, so concepts.entity_id is then readable.
        ccs_cols = {
            r[1] for r in self._conn.execute("PRAGMA table_info(concept_compile_state)").fetchall()
        }
        has_entity = "entity_id" in ccs_cols
        cols = "entity_id, name, source_path" if has_entity else "name, source_path"
        query = f"SELECT {cols} FROM concepts"
        params: tuple[str, ...] = ()
        if source_path is not None:
            query += " WHERE source_path = ?"
            params = (source_path,)
        rows = self._conn.execute(query, params).fetchall()
        now = datetime.now().isoformat()
        with self._tx():
            for row in rows:
                if has_entity:
                    self._conn.execute(
                        """INSERT OR IGNORE INTO concept_compile_state
                               (entity_id, source_path, concept_name, status, error,
                                compiled_at, updated_at)
                           VALUES (?, ?, ?, 'pending', NULL, NULL, ?)""",
                        (row["entity_id"], row["source_path"], row["name"], now),
                    )
                else:
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
        # Migration backfill (called only from _migrate during __init__): single-threaded, before
        # workers exist, so a bare commit outside the _tx()/_lock contract is intentional.
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
        # Migration backfill (called only from _migrate during __init__): single-threaded, before
        # workers exist, so a bare commit outside the _tx()/_lock contract is intentional.
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
        # Held for the whole transaction (depth bookkeeping → yield → finally) so a concurrent
        # thread's write — e.g. an LLM-cache commit from a parallel-ingest worker — cannot land
        # inside this transaction and corrupt its state (#75). RLock allows same-thread nesting.
        with self._lock:
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

    @contextmanager
    def _read(self):
        """Lock-held read for callers that can run concurrently with a writer.

        Holds ``_lock`` (the same re-entrant lock as ``_tx()``) for the SELECT but never commits,
        so the read serializes against in-flight transactions — it waits for a writer to finish and
        then sees committed state — instead of racing on the shared connection. Single-threaded read
        paths don't need this and use ``self._conn`` directly.
        """
        with self._lock:
            yield self._conn

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
        now = datetime.now().isoformat()
        with self._tx():
            for name in concept_names:
                name = name.strip()
                if not name:
                    continue
                # Resolve to identity at the write seam: mint if absent, then key
                # the source edge on entity_id (concepts is entity-keyed since v22).
                entity_id = self._ensure_entity_for_name(name, now)
                if entity_id is None:
                    continue
                self._conn.execute(
                    "INSERT OR IGNORE INTO concepts (entity_id, source_path, name)"
                    " VALUES (?, ?, ?)",
                    (entity_id, source_path, name),
                )
                now = datetime.now().isoformat()
                self._conn.execute(
                    """INSERT OR IGNORE INTO knowledge_items
                       (name, kind, subtype, status, confidence, created_at, updated_at)
                       VALUES (?, 'concept', NULL, 'confirmed', 1.0, ?, ?)""",
                    (name, now, now),
                )
        self._ensure_compile_state_rows(source_path)

    def replace_concepts_for_source(
        self, source_path: str, concept_names: list[str]
    ) -> list[tuple[str, str]]:
        """Replace concept links for a source and reset compile state for current concepts.

        Returns (demoted_host_entity_id, surface) for each weak alias demoted to mint a name in
        this batch, so the caller can strip the stale alias from the host article's frontmatter.
        """
        normalized = []
        seen: set[str] = set()
        for name in concept_names:
            cleaned = name.strip()
            if not cleaned or cleaned.casefold() in seen:
                continue
            seen.add(cleaned.casefold())
            normalized.append(cleaned)

        now = datetime.now().isoformat()
        # (new_entity, demoted_host, surface) for surfaces minted here that demoted a weak alias
        # off another entity — recorded as merge candidates after the tx (seam (a), order #1:
        # the host aliased the surface first, then a later note extracts it as a concept).
        promoted_pairs: list[tuple[str, str, str]] = []
        with self._tx():
            # Resolve/mint each new name to its entity up front; concepts is keyed
            # on entity_id (v22), so removal is by entity_id, not by name string.
            name_to_id: dict[str, str] = {}
            kept_ids: set[str] = set()
            for name in normalized:
                demoted: list[str] = []
                entity_id = self._ensure_entity_for_name(name, now, demoted_hosts=demoted)
                if entity_id is None:
                    continue
                name_to_id[name] = entity_id
                kept_ids.add(entity_id)
                promoted_pairs.extend((entity_id, host, name) for host in demoted)

            existing_rows = self._conn.execute(
                "SELECT entity_id, name FROM concepts WHERE source_path = ?", (source_path,)
            ).fetchall()
            removed_ids = [r["entity_id"] for r in existing_rows if r["entity_id"] not in kept_ids]
            if removed_ids:
                placeholders = ",".join("?" * len(removed_ids))
                self._conn.execute(
                    f"DELETE FROM concepts WHERE source_path = ? AND entity_id IN ({placeholders})",
                    [source_path, *removed_ids],
                )
                self._conn.execute(
                    "DELETE FROM concept_compile_state "
                    f"WHERE source_path = ? AND entity_id IN ({placeholders})",
                    [source_path, *removed_ids],
                )
            for name in normalized:
                entity_id = name_to_id.get(name)
                if entity_id is None:
                    continue
                self._conn.execute(
                    "INSERT OR IGNORE INTO concepts (entity_id, source_path, name)"
                    " VALUES (?, ?, ?)",
                    (entity_id, source_path, name),
                )
                self._conn.execute(
                    """INSERT OR IGNORE INTO knowledge_items
                           (name, kind, subtype, status, confidence, created_at, updated_at)
                       VALUES (?, 'concept', NULL, 'confirmed', 1.0, ?, ?)""",
                    (name, now, now),
                )
                self._conn.execute(
                    """INSERT INTO concept_compile_state
                           (entity_id, source_path, concept_name, status, error,
                            compiled_at, updated_at)
                       VALUES (?, ?, ?, 'pending', NULL, NULL, ?)
                       ON CONFLICT(entity_id, source_path) DO UPDATE SET
                           concept_name=excluded.concept_name,
                           status='pending',
                           error=NULL,
                           compiled_at=NULL,
                           updated_at=excluded.updated_at""",
                    (entity_id, source_path, name, now),
                )
        for new_entity, host, surface in promoted_pairs:
            self.record_merge_candidate(host, new_entity, surface, reason="promoted-from-alias")
        self.refresh_raw_compile_status(source_path)
        return [(host, surface) for _new, host, surface in promoted_pairs]

    def list_all_concept_names(self) -> list[str]:
        """All unique canonical concept names, sorted."""
        rows = self._conn.execute("SELECT DISTINCT name FROM concepts ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def get_sources_for_concept(self, name: str) -> list[str]:
        """Raw note paths linked to a concept, resolved by entity identity.

        Resolves the label to its entity (handles aliases + the denormalized name
        cache uniformly) then reads source edges by entity_id. Returns [] for an
        unknown or ambiguous label.
        """
        entity_id = self.entity_id_for_name(name)
        if entity_id is None:
            return []
        return self.get_sources_for_entity(entity_id)

    def get_sources_for_entity(self, entity_id: str) -> list[str]:
        """Raw note paths linked to an entity."""
        rows = self._conn.execute(
            "SELECT DISTINCT source_path FROM concepts WHERE entity_id = ?",
            (entity_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def _ensure_entity_for_name(
        self, name: str, now: str, *, demoted_hosts: list[str] | None = None
    ) -> str | None:
        """Get or create an entity for a concept name (must be called inside a _tx).

        Returns entity_id, or None if name produces an empty label_key.
        Idempotent: a second call for the same name returns the existing entity_id.

        When ``demoted_hosts`` is provided, the entity_ids of any WEAK aliases demoted to
        mint this preferred label are appended to it, so the ingest write seam can record
        merge candidates at the call site. Other callers (rebuild/rename/restore) omit it
        and demote silently — recording there would manufacture spurious candidates.
        """
        if not self._has_table("concept_labels"):
            return None
        lk = _ck(name)
        if not lk:
            return None
        # The global unique index keeps at most one preferred row per label_key, regardless of
        # entity status — and a merge-loser KEEPS its preferred row (get_merged_entity_id/unmerge
        # depend on it). So a hit here can point at a RETIRED entity. Resolve to the live identity:
        #   - active            → return it.
        #   - merged (by merge) → chase merged_into to the active winner (the surface now belongs
        #                         to the winner, which holds this label as an alias).
        #   - merged (by split, merged_into NULL) → return None: the bare label is now a
        #                         disambiguation surface with no single owner.
        # Never return the dead id, and never fall through to mint — minting a preferred row with
        # this label_key would collide on idx_concept_labels_preferred_global.
        row = self._conn.execute(
            "SELECT cl.entity_id, ce.status, ce.merged_into FROM concept_labels cl"
            " JOIN concept_entities ce ON ce.id = cl.entity_id"
            " WHERE cl.label_key = ? AND cl.role = 'preferred'",
            (lk,),
        ).fetchone()
        if row is not None:
            entity_id, status, merged_into = row[0], row[1], row[2]
            if status != "merged":
                return entity_id
            return self._active_entity_after_merge(merged_into)
        # Minting a new preferred label: a WEAK alias (extracted/legacy_backfill) with the same
        # label_key on another entity is now a collision (an alias may not equal another entity's
        # preferred label — it would make resolve_label ambiguous and drop the new concept's
        # article at compile), so demote it. A human-blessed alias (source user/rename) is NEVER
        # silently deleted — the classifier links to it before reaching this mint, so a blessed
        # collision here is a genuine conflict surfaced loudly elsewhere, not data to destroy.
        if demoted_hosts is not None:
            host_rows = self._conn.execute(
                "SELECT DISTINCT entity_id FROM concept_labels"
                " WHERE role = 'alias' AND label_key = ?"
                "   AND source IN ('extracted', 'legacy_backfill')",
                (lk,),
            ).fetchall()
            demoted_hosts.extend(r[0] for r in host_rows)
        self._conn.execute(
            "DELETE FROM concept_labels WHERE role = 'alias' AND label_key = ?"
            "   AND source IN ('extracted', 'legacy_backfill')",
            (lk,),
        )
        entity_id = _generate_article_id()
        self._conn.execute(
            "INSERT INTO concept_entities (id, kind, status, created_at, updated_at)"
            " VALUES (?, 'concept', 'active', ?, ?)",
            (entity_id, now, now),
        )
        self._conn.execute(
            "INSERT INTO concept_labels"
            " (entity_id, label, label_key, match_key, role, source, created_at)"
            " VALUES (?, ?, ?, ?, 'preferred', 'extracted', ?)",
            (entity_id, name, lk, _mk(name), now),
        )
        return entity_id

    def _active_entity_after_merge(self, entity_id: str | None) -> str | None:
        """Follow the merged_into chain from entity_id to the first active entity.

        Returns None if the chain dead-ends at a NULL merged_into on a still-merged entity
        (e.g. a split-retired original, which has no single successor) or hits a cycle.
        """
        seen: set[str] = set()
        current = entity_id
        while current is not None and current not in seen:
            seen.add(current)
            row = self._conn.execute(
                "SELECT status, merged_into FROM concept_entities WHERE id = ?", (current,)
            ).fetchone()
            if row is None:
                return None
            if row[0] != "merged":
                return current
            current = row[1]
        return None

    # ── Entity identity (feature 45, Phase 1) ─────────────────────────────────

    def resolve_label(self, text: str) -> ResolveResult:
        """Resolve a label string to entity IDs. The single identity seam.

        Returns ids=[] (unknown), ids=[id] (unambiguous), ids=[...] (ambiguous).
        Phase 1 matches on exact label_key only. Phase 2 extends to match_key.
        """
        key = _ck(text)
        if not key:
            return ResolveResult()
        if not self._has_table("concept_labels"):
            return ResolveResult()
        rows = self._conn.execute(
            """
            SELECT DISTINCT cl.entity_id
            FROM concept_labels cl
            JOIN concept_entities ce ON ce.id = cl.entity_id
            WHERE cl.label_key = ? AND ce.status = 'active'
            """,
            (key,),
        ).fetchall()
        ids = [r[0] for r in rows]
        return ResolveResult(ids=ids, ambiguous=len(ids) > 1)

    def entity_id_for_name(self, name: str) -> str | None:
        """Return entity_id for concept name, or None if not found / ambiguous."""
        result = self.resolve_label(name)
        if result.ambiguous or not result.ids:
            return None
        return result.ids[0]

    def preferred_label_for_entity(self, entity_id: str) -> str | None:
        """Return the preferred label string for a given entity_id."""
        if not self._has_table("concept_labels"):
            return None
        row = self._conn.execute(
            "SELECT label FROM concept_labels WHERE entity_id = ? AND role = 'preferred'",
            (entity_id,),
        ).fetchone()
        return row[0] if row else None

    def count_ambiguous_occurrences_for_label(self, label: str) -> int:
        """Count ambiguous occurrences for a display label (encapsulates the legacy
        concept_name column query used by inspect paths).
        """
        if not self._has_table("concept_occurrences"):
            return 0
        row = self._conn.execute(
            "SELECT COUNT(*) FROM concept_occurrences "
            "WHERE lower(concept_name)=lower(?) AND resolution_status='ambiguous'",
            (label,),
        ).fetchone()
        return int(row[0]) if row else 0

    def get_compile_state_for_label(self, label: str) -> list[tuple[str, str | None]]:
        """Return (status, updated_at) rows for a label from the (still partially
        name-keyed) compile_state table. Thin wrapper so CLI does not do raw SQL.
        """
        if not self._has_table("concept_compile_state"):
            return []
        rows = self._conn.execute(
            "SELECT status, updated_at FROM concept_compile_state "
            "WHERE lower(concept_name)=lower(?)",
            (label,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def alias_collides_with_preferred(self, alias_lk: str, owner_entity_id: str) -> bool:
        """Return True if alias_lk is already the preferred label of a DIFFERENT active entity.

        Used by the extraction path to detect merge-candidate collisions (issue #54).
        Does not block cross-language aliases stored via explicit upsert_aliases calls.
        """
        if not self._has_table("concept_labels"):
            return False
        row = self._conn.execute(
            """SELECT 1 FROM concept_labels cl
               JOIN concept_entities ce ON ce.id = cl.entity_id
               WHERE cl.label_key = ? AND cl.role = 'preferred'
                 AND cl.entity_id != ? AND ce.status = 'active'""",
            (alias_lk, owner_entity_id),
        ).fetchone()
        return row is not None

    # ── Role-aware surface classification (v26, order-independent identity) ────
    # The ingest classifier asks these in order: a surface is a known concept iff it matches a
    # PREFERRED label or a human-BLESSED alias; matching only WEAK aliases means it should mint
    # its own entity (and demote those weak aliases). This makes identity a pure function of the
    # claim-set, independent of ingest order.

    def preferred_entity_for_surface(self, name: str) -> ResolveResult:
        """Resolve a surface against PREFERRED labels only (exact label_key, then match_key fold).

        Returns ids=[] / ids=[id] / ambiguous like resolve_label. match_key folding handles
        "Users"→"User"; it is consulted only when exact label_key finds nothing.
        """
        if not self._has_table("concept_labels"):
            return ResolveResult()
        lk = _ck(name)
        if not lk:
            return ResolveResult()
        rows = self._conn.execute(
            """SELECT DISTINCT cl.entity_id FROM concept_labels cl
               JOIN concept_entities ce ON ce.id = cl.entity_id
               WHERE cl.label_key = ? AND cl.role = 'preferred' AND ce.status = 'active'""",
            (lk,),
        ).fetchall()
        ids = [r[0] for r in rows]
        if ids:
            return ResolveResult(ids=ids, ambiguous=len(ids) > 1)
        mk = _mk(name)
        if not mk:
            return ResolveResult()
        rows = self._conn.execute(
            """SELECT DISTINCT cl.entity_id FROM concept_labels cl
               JOIN concept_entities ce ON ce.id = cl.entity_id
               WHERE cl.match_key = ? AND cl.role = 'preferred' AND ce.status = 'active'""",
            (mk,),
        ).fetchall()
        ids = [r[0] for r in rows]
        return ResolveResult(ids=ids, ambiguous=len(ids) > 1)

    def blessed_alias_entities_for_surface(self, name: str) -> list[str]:
        """Active entity_ids carrying ``name`` as a human-blessed alias (source user/rename)."""
        if not self._has_table("concept_labels"):
            return []
        lk = _ck(name)
        if not lk:
            return []
        rows = self._conn.execute(
            """SELECT DISTINCT cl.entity_id FROM concept_labels cl
               JOIN concept_entities ce ON ce.id = cl.entity_id
               WHERE cl.label_key = ? AND cl.role = 'alias'
                 AND cl.source IN ('user', 'rename') AND ce.status = 'active'""",
            (lk,),
        ).fetchall()
        return [r[0] for r in rows]

    def weak_alias_entities_for_surface(self, name: str) -> list[str]:
        """Active entity_ids carrying ``name`` as a weak alias (extracted/legacy_backfill)."""
        if not self._has_table("concept_labels"):
            return []
        lk = _ck(name)
        if not lk:
            return []
        rows = self._conn.execute(
            """SELECT DISTINCT cl.entity_id FROM concept_labels cl
               JOIN concept_entities ce ON ce.id = cl.entity_id
               WHERE cl.label_key = ? AND cl.role = 'alias'
                 AND cl.source IN ('extracted', 'legacy_backfill') AND ce.status = 'active'""",
            (lk,),
        ).fetchall()
        return [r[0] for r in rows]

    def has_disambiguation_stub(self, name: str) -> bool:
        """True if an active disambiguation stub exists at the bare ``name``.

        A managed homonym (post-split) shares its bare label as a weak alias across senses, which
        is structurally identical to the bug case (a weak alias on >=2 unrelated hosts). The stub's
        existence is the only signal distinguishing "route to disambiguation" from "mint + demote".
        """
        if not self._has_table("wiki_articles"):
            return False
        key = _ck(name)
        if not key:
            return False
        rows = self._conn.execute(
            "SELECT title FROM wiki_articles WHERE kind = 'disambiguation'"
        ).fetchall()
        return any(_ck(r[0]) == key for r in rows)

    # ── Merge-candidate worklist (v26) ────────────────────────────────────────

    def record_merge_candidate(
        self, entity_x: str, entity_y: str, surface: str, reason: str | None = None
    ) -> None:
        """Record an advisory merge candidate for two entities the identity rule kept apart.

        The pair is ordered by the entities' preferred label_key (NOT entity_id, which is a
        random value) so the same logical pair dedups to one row regardless of ingest order.
        """
        if not self._has_table("concept_merge_candidates"):
            return
        if not entity_x or not entity_y or entity_x == entity_y:
            return
        kx = _ck(self.preferred_label_for_entity(entity_x) or "")
        ky = _ck(self.preferred_label_for_entity(entity_y) or "")
        # Deterministic ordering: primary key is the preferred label_key; entity_id is only a
        # tiebreak for the degenerate same-label_key case (which preferred uniqueness precludes).
        if (ky, entity_y) < (kx, entity_x):
            entity_x, entity_y = entity_y, entity_x
        now = datetime.now().isoformat()
        with self._tx():
            self._conn.execute(
                "INSERT OR IGNORE INTO concept_merge_candidates"
                " (entity_a, entity_b, surface, reason, created_at) VALUES (?, ?, ?, ?, ?)",
                (entity_x, entity_y, surface, reason, now),
            )

    def list_merge_candidates(self) -> list[dict[str, str | None]]:
        """All merge candidates with both preferred labels resolved for display, newest first.

        Pairs whose entities are no longer both active are skipped (the candidate is stale —
        a merge/unmerge/rename has since changed the graph)."""
        if not self._has_table("concept_merge_candidates"):
            return []
        rows = self._conn.execute(
            "SELECT entity_a, entity_b, surface, reason FROM concept_merge_candidates"
            " ORDER BY created_at DESC, surface"
        ).fetchall()
        out: list[dict[str, str | None]] = []
        for r in rows:
            label_a = self.preferred_label_for_entity(r[0])
            label_b = self.preferred_label_for_entity(r[1])
            if label_a is None or label_b is None:
                continue
            out.append(
                {
                    "entity_a": r[0],
                    "entity_b": r[1],
                    "label_a": label_a,
                    "label_b": label_b,
                    "surface": r[2],
                    "reason": r[3],
                }
            )
        return out

    def clear_merge_candidates_for_entity(self, entity_id: str) -> None:
        """Drop every merge candidate touching ``entity_id`` (resolved by a curation op)."""
        if not self._has_table("concept_merge_candidates") or not entity_id:
            return
        with self._tx():
            self._conn.execute(
                "DELETE FROM concept_merge_candidates WHERE entity_a = ? OR entity_b = ?",
                (entity_id, entity_id),
            )

    def resolve_by_match_key(self, text: str) -> ResolveResult:
        """Resolve using match_key (plural/singular folded) — Phase 2 seam.

        "Users" and "User" share match_key "user", so this returns the User entity
        even when 'users' is not an alias. Returns the same shape as resolve_label.
        """
        mk = _mk(text)
        if not mk:
            return ResolveResult()
        if not self._has_table("concept_labels"):
            return ResolveResult()
        rows = self._conn.execute(
            """
            SELECT DISTINCT cl.entity_id
            FROM concept_labels cl
            JOIN concept_entities ce ON ce.id = cl.entity_id
            WHERE cl.match_key = ? AND ce.status = 'active'
            """,
            (mk,),
        ).fetchall()
        ids = [r[0] for r in rows]
        return ResolveResult(ids=ids, ambiguous=len(ids) > 1)

    def get_sticky_entity_for_source(
        self, source_path: str, candidate_ids: list[str]
    ) -> str | None:
        """Return the single candidate entity already linked to source_path, if exactly one.

        Used for sticky resolution (decision 18): when re-ingesting a source that
        already has an edge to one of the ambiguous candidates, keep that edge.
        """
        if not candidate_ids:
            return None
        placeholders = ",".join("?" * len(candidate_ids))
        rows = self._conn.execute(
            f"""
            SELECT DISTINCT c.entity_id
            FROM concepts c
            JOIN concept_entities ce ON ce.id = c.entity_id AND ce.status = 'active'
            WHERE c.source_path = ? AND c.entity_id IN ({placeholders})
            """,
            (source_path, *candidate_ids),
        ).fetchall()
        if len(rows) == 1:
            return rows[0][0]
        return None

    def record_resolved_occurrence(
        self,
        concept_name: str,
        entity_id: str,
        surface: str,
        *,
        source_segment_id: str | None = None,
        source_path: str | None = None,
        ordinal: int = 0,
        confidence: float = 1.0,
        extraction_run: str | None = None,
    ) -> None:
        """Record a resolved (1-entity) concept occurrence."""
        if not self._has_table("concept_occurrences"):
            return
        with self._tx():
            if source_segment_id is not None:
                self._conn.execute(
                    """INSERT INTO concept_occurrences
                       (concept_name, source_segment_id, ordinal, confidence, extraction_run,
                        entity_id, surface, resolution_status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'resolved')
                       ON CONFLICT(concept_name, source_segment_id)
                       WHERE source_segment_id IS NOT NULL
                       DO UPDATE SET entity_id=excluded.entity_id, surface=excluded.surface,
                           resolution_status='resolved'""",
                    (
                        concept_name,
                        source_segment_id,
                        ordinal,
                        confidence,
                        extraction_run,
                        entity_id,
                        surface,
                    ),
                )
            else:
                self._conn.execute(
                    """INSERT INTO concept_occurrences
                       (concept_name, source_path, ordinal, confidence, extraction_run,
                        entity_id, surface, resolution_status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'resolved')
                       ON CONFLICT(concept_name, source_path)
                       WHERE source_path IS NOT NULL AND source_segment_id IS NULL
                       DO UPDATE SET entity_id=excluded.entity_id, surface=excluded.surface,
                           resolution_status='resolved'""",
                    (
                        concept_name,
                        source_path,
                        ordinal,
                        confidence,
                        extraction_run,
                        entity_id,
                        surface,
                    ),
                )

    def record_ambiguous_occurrence(
        self,
        concept_name: str,
        candidate_ids: list[str],
        surface: str,
        *,
        source_segment_id: str | None = None,
        source_path: str | None = None,
        ordinal: int = 0,
        confidence: float = 1.0,
        extraction_run: str | None = None,
    ) -> None:
        """Record an ambiguous occurrence with N candidate entity IDs."""
        if not self._has_table("concept_occurrences") or not candidate_ids:
            return
        with self._tx():
            if source_segment_id is not None:
                cursor = self._conn.execute(
                    """INSERT INTO concept_occurrences
                       (concept_name, source_segment_id, ordinal, confidence, extraction_run,
                        surface, resolution_status)
                       VALUES (?, ?, ?, ?, ?, ?, 'ambiguous')
                       ON CONFLICT(concept_name, source_segment_id)
                       WHERE source_segment_id IS NOT NULL
                       DO UPDATE SET surface=excluded.surface, resolution_status='ambiguous'
                       RETURNING id""",
                    (concept_name, source_segment_id, ordinal, confidence, extraction_run, surface),
                )
            else:
                cursor = self._conn.execute(
                    """INSERT INTO concept_occurrences
                       (concept_name, source_path, ordinal, confidence, extraction_run,
                        surface, resolution_status)
                       VALUES (?, ?, ?, ?, ?, ?, 'ambiguous')
                       ON CONFLICT(concept_name, source_path)
                       WHERE source_path IS NOT NULL AND source_segment_id IS NULL
                       DO UPDATE SET surface=excluded.surface, resolution_status='ambiguous'
                       RETURNING id""",
                    (concept_name, source_path, ordinal, confidence, extraction_run, surface),
                )
            row = cursor.fetchone()
            if row is None:
                return
            occ_id = row[0]
            for eid in candidate_ids:
                self._conn.execute(
                    "INSERT OR IGNORE INTO concept_occurrence_candidates"
                    " (occurrence_id, entity_id) VALUES (?, ?)",
                    (occ_id, eid),
                )

    def count_ambiguous_occurrences(self) -> int:
        """Count unresolved ambiguous occurrence rows (for lint)."""
        if not self._has_table("concept_occurrences"):
            return 0
        row = self._conn.execute(
            "SELECT COUNT(*) FROM concept_occurrences WHERE resolution_status = 'ambiguous'"
        ).fetchone()
        return row[0] if row else 0

    def resolve_ambiguous_occurrences(self, surface: str, entity_id: str) -> int:
        """Assign all pending ambiguous occurrences for surface to entity_id.

        Returns the number of rows updated.
        """
        with self._tx():
            cursor = self._conn.execute(
                """UPDATE concept_occurrences
                   SET entity_id=?, resolution_status='resolved'
                   WHERE lower(concept_name)=lower(?) AND resolution_status='ambiguous'""",
                (entity_id, surface),
            )
        return cursor.rowcount

    # ── Identity operations (merge / split) ───────────────────────────────────

    def _log_identity_op(
        self,
        op: str,
        entity_ids: list[str],
        labels: dict[str, str],
        meta: dict | None = None,
    ) -> None:
        """Append one row to concept_identity_log. Must be called inside a _tx()."""
        if not self._has_table("concept_identity_log"):
            return

        self._conn.execute(
            "INSERT INTO concept_identity_log (op, entity_ids, labels, meta, ts)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                op,
                json.dumps(entity_ids),
                json.dumps(labels),
                json.dumps(meta) if meta else None,
                datetime.now().isoformat(),
            ),
        )

    def merge_entities(self, winner_name: str, loser_name: str) -> dict:
        """Merge loser entity into winner: move all edges, union labels, retire loser.

        Returns a summary dict with keys: winner, loser, sources_moved, labels_absorbed.
        Raises ValueError if either entity not found, or both are the same.
        """
        winner_id = self.entity_id_for_name(winner_name)
        loser_id = self.entity_id_for_name(loser_name)
        if winner_id is None:
            raise ValueError(f"Concept not found: {winner_name!r}")
        if loser_id is None:
            raise ValueError(f"Concept not found: {loser_name!r}")
        if winner_id == loser_id:
            raise ValueError(f"{winner_name!r} and {loser_name!r} are the same entity")

        now = datetime.now().isoformat()
        with self._tx():
            # Snapshot for log.
            loser_sources = self.get_sources_for_concept(loser_name)
            loser_aliases_rows = self._conn.execute(
                "SELECT label FROM concept_labels WHERE entity_id=? AND role='alias'",
                (loser_id,),
            ).fetchall()
            labels_snapshot = {
                winner_id: winner_name,
                loser_id: loser_name,
            }

            # Move source edges onto the winner entity. UPDATE OR IGNORE skips a
            # row that would collide on (winner_id, source_path) — i.e. both
            # entities cited the same source — and the leftover loser row is then
            # deleted, collapsing to a single winner edge.
            self._conn.execute(
                "UPDATE OR IGNORE concepts SET entity_id=?, name=? WHERE entity_id=?",
                (winner_id, winner_name, loser_id),
            )
            self._conn.execute("DELETE FROM concepts WHERE entity_id=?", (loser_id,))

            # Union labels: loser preferred → winner alias; loser aliases → winner aliases.
            # Absorbed labels are blessed (source='user'): the merge is an explicit human decision
            # that these surfaces map to the winner, so the order-independent ingest rule must
            # respect them (link, never re-mint) — this is what makes a resolved merge candidate
            # stick instead of re-fragmenting on the next ingest. Upgrade any pre-existing weak
            # alias to blessed on conflict for the same reason.
            winner_lk = _ck(winner_name)
            # Absorb the loser's canonical preferred label (its stored casing), not the raw
            # CLI input casing, so labels_absorbed/report and the winner's blessed alias carry
            # the loser's real label. All other loser_name uses below are lower()-keyed and
            # unaffected. loser_pref is the first element iterated, so the loop appends it once;
            # seeding the list with it too would double-count it (duplicate winner alias).
            loser_pref = self.preferred_label_for_entity(loser_id) or loser_name
            labels_absorbed: list[str] = []
            for row in [{"label": loser_pref}, *loser_aliases_rows]:
                lbl = row["label"] if isinstance(row, dict) else row[0]
                lk = _ck(lbl)
                if not lk or lk == winner_lk:
                    continue
                self._conn.execute(
                    "INSERT INTO concept_labels"
                    " (entity_id, label, label_key, match_key, role, source, created_at)"
                    " VALUES (?, ?, ?, ?, 'alias', 'user', ?)"
                    " ON CONFLICT(entity_id, label_key) DO UPDATE SET source='user'",
                    (winner_id, lbl, lk, _mk(lbl), now),
                )
                labels_absorbed.append(lbl)

            # Move compile state onto the winner entity (UPDATE OR IGNORE + delete the
            # leftover collides exactly like the concepts edge-move), refreshing the cache.
            self._conn.execute(
                "UPDATE OR IGNORE concept_compile_state SET entity_id=?, concept_name=?"
                " WHERE entity_id=?",
                (winner_id, winner_name, loser_id),
            )
            self._conn.execute("DELETE FROM concept_compile_state WHERE entity_id=?", (loser_id,))
            # Ensure pending rows exist for loser sources now attached to winner.
            for src in loser_sources:
                self._conn.execute(
                    """INSERT INTO concept_compile_state
                           (entity_id, source_path, concept_name, status, updated_at)
                       VALUES (?, ?, ?, 'pending', ?)
                       ON CONFLICT(entity_id, source_path) DO NOTHING""",
                    (winner_id, src, winner_name, now),
                )

            # Move behavioral state. Occurrences carry UNIQUE(concept_name, segment/path)
            # indexes, so a plain rename collides when a segment cited both concepts —
            # OR IGNORE skips the collisions, then the leftover loser rows are dropped
            # (same collapse pattern as the concepts / compile_state edge moves above).
            #
            # NOTE (legacy name-keyed debt): rejections, blocked_concepts, stubs, and
            # knowledge_items are still keyed by display name (case-insensitive match only).
            # Unmerge intentionally does not restore them. The planned migration that keys
            # these on entity_id is FOLLOWUP_PLAN1 Stage 3 (v24).
            self._conn.execute(
                "UPDATE OR IGNORE concept_occurrences SET concept_name=?"
                " WHERE lower(concept_name)=lower(?)",
                (winner_name, loser_name),
            )
            self._conn.execute(
                "DELETE FROM concept_occurrences WHERE lower(concept_name)=lower(?)",
                (loser_name,),
            )
            # Repoint resolved occurrences off the now-retired loser entity (no unique
            # index on entity_id, so this cannot collide).
            self._conn.execute(
                "UPDATE concept_occurrences SET entity_id=? WHERE entity_id=?",
                (winner_id, loser_id),
            )
            # Repoint wiki_articles.entity_id (v24) so that published concept articles
            # whose title matched the loser continue to carry a live entity_id in
            # INDEX.json and pack exports rather than a retired "merged" one.
            # (Articles are title-bound snapshots at publish time; explicit merge
            # advances the binding for durable identity, matching how source edges
            # and occurrences are moved.) Record the exact paths repointed so unmerge can
            # reverse precisely, mirroring sources_moved.
            articles_repointed = [
                str(r[0])
                for r in self._conn.execute(
                    "SELECT path FROM wiki_articles WHERE entity_id=?", (loser_id,)
                ).fetchall()
            ]
            self._conn.execute(
                "UPDATE wiki_articles SET entity_id=? WHERE entity_id=?",
                (winner_id, loser_id),
            )
            self._conn.execute(
                "UPDATE rejections SET concept=? WHERE lower(concept)=lower(?)",
                (winner_name, loser_name),
            )
            # Block winner if loser was blocked.
            was_blocked = self._conn.execute(
                "SELECT 1 FROM blocked_concepts WHERE lower(concept)=lower(?)",
                (loser_name,),
            ).fetchone()
            if was_blocked:
                self._conn.execute(
                    "INSERT OR IGNORE INTO blocked_concepts (concept, blocked_at) VALUES (?,?)",
                    (winner_name, now),
                )
            self._conn.execute(
                "DELETE FROM blocked_concepts WHERE lower(concept)=lower(?)", (loser_name,)
            )
            # stubs.concept is PRIMARY KEY: if the winner already has a stub row, a plain
            # rename collides. OR IGNORE skips the collision, then the leftover loser row is
            # deleted — same collapse pattern as blocked_concepts above.
            self._conn.execute(
                "UPDATE OR IGNORE stubs SET concept=? WHERE lower(concept)=lower(?)",
                (winner_name, loser_name),
            )
            self._conn.execute("DELETE FROM stubs WHERE lower(concept)=lower(?)", (loser_name,))
            # knowledge_items has a UNIQUE(name) constraint. If the winner row already
            # exists, renaming the loser would conflict — just delete the loser row.
            winner_ki = self._conn.execute(
                "SELECT 1 FROM knowledge_items WHERE lower(name)=lower(?)", (winner_name,)
            ).fetchone()
            if winner_ki:
                self._conn.execute(
                    "DELETE FROM knowledge_items WHERE lower(name)=lower(?)", (loser_name,)
                )
            else:
                self._conn.execute(
                    "UPDATE knowledge_items SET name=? WHERE lower(name)=lower(?)",
                    (winner_name, loser_name),
                )

            # Retire loser entity.
            self._conn.execute(
                "UPDATE concept_entities SET status='merged', merged_into=?, updated_at=?"
                " WHERE id=?",
                (winner_id, now, loser_id),
            )

            # The merge resolves any candidate pairing winner/loser and orphans pairs touching the
            # retired loser — drop both so the worklist reflects the new graph.
            if self._has_table("concept_merge_candidates"):
                self._conn.execute(
                    "DELETE FROM concept_merge_candidates"
                    " WHERE entity_a IN (?, ?) OR entity_b IN (?, ?)",
                    (winner_id, loser_id, winner_id, loser_id),
                )

            self._log_identity_op(
                "merge",
                [winner_id, loser_id],
                labels_snapshot,
                meta={
                    "sources_moved": loser_sources,
                    "labels_absorbed": labels_absorbed,
                    "articles_repointed": articles_repointed,
                },
            )

        return {
            "winner": winner_name,
            "loser": loser_name,
            "sources_moved": loser_sources,
            "labels_absorbed": labels_absorbed,
        }

    def split_entity(self, entity_name: str, senses: list[dict[str, object]]) -> dict:
        """Split one entity into multiple senses, each owning a subset of sources.

        senses is a list of {name: str, sources: list[str]}.
        Each sense gets the bare entity_name as a shared alias.
        Original entity is retired (status='merged').
        Returns summary with keys: original, senses (list of {name, entity_id, sources}),
        stub_needed.
        """
        original_id = self.entity_id_for_name(entity_name)
        if original_id is None:
            raise ValueError(f"Concept not found: {entity_name!r}")
        if len(senses) < 2:
            raise ValueError("split_entity requires at least 2 senses")

        now = datetime.now().isoformat()
        with self._tx():
            # Validate all claimed sources belong to this entity.
            owned = set(self.get_sources_for_concept(entity_name))
            claimed: set[str] = set()
            for sense in senses:
                for src in sense["sources"]:  # type: ignore[union-attr]
                    if src not in owned:
                        raise ValueError(
                            f"Source {src!r} does not belong to entity {entity_name!r}"
                        )
                    claimed.add(src)
            unclaimed = owned - claimed
            if unclaimed:
                raise ValueError(f"Sources not assigned to any sense: {sorted(unclaimed)}")

            bare_lk = _ck(entity_name)
            result_senses: list[dict] = []

            for sense in senses:
                new_name = str(sense["name"])
                sources = list(sense["sources"])  # type: ignore[arg-type]
                # The bare label is reserved for the disambiguation page; a sense that
                # reuses it would resolve back to the original entity (then be retired)
                # and its stub would be clobbered by the disambiguation stub.
                if bare_lk and _ck(new_name) == bare_lk:
                    raise ValueError(
                        f"Sense {new_name!r} cannot reuse the original label"
                        f" {entity_name!r} — that name is reserved for the"
                        " disambiguation page."
                    )
                # Mint new entity.
                new_id = self._ensure_entity_for_name(new_name, now)
                if new_id is None:
                    raise ValueError(f"Cannot mint entity for sense name: {new_name!r}")
                # Add bare label as a shared alias on the new entity.
                new_lk = _ck(new_name)
                if bare_lk and bare_lk != new_lk:
                    self._conn.execute(
                        "INSERT INTO concept_labels"
                        " (entity_id, label, label_key, match_key, role, source, created_at)"
                        " VALUES (?, ?, ?, ?, 'alias', 'extracted', ?)"
                        " ON CONFLICT(entity_id, label_key) DO NOTHING",
                        (new_id, entity_name, bare_lk, _mk(entity_name), now),
                    )
                # Move source edges onto the sense entity (sources are partitioned
                # across senses, so no (new_id, src) collision is expected).
                for src in sources:
                    self._conn.execute(
                        "UPDATE OR IGNORE concepts SET entity_id=?, name=?"
                        " WHERE entity_id=? AND source_path=?",
                        (new_id, new_name, original_id, src),
                    )
                    # Seed pending compile state on the sense entity.
                    self._conn.execute(
                        """INSERT INTO concept_compile_state
                               (entity_id, source_path, concept_name, status, updated_at)
                           VALUES (?, ?, ?, 'pending', ?)
                           ON CONFLICT(entity_id, source_path) DO UPDATE
                               SET concept_name=excluded.concept_name,
                                   status='pending', updated_at=excluded.updated_at""",
                        (new_id, src, new_name, now),
                    )
                result_senses.append({"name": new_name, "entity_id": new_id, "sources": sources})

            # Retire original entity.
            self._conn.execute(
                "UPDATE concept_entities SET status='merged', updated_at=? WHERE id=?",
                (now, original_id),
            )
            # OR IGNORE guards the UNIQUE(concept_name, segment/path) indexes (defensive —
            # the primary sense is a fresh name post-guard, so collisions are unexpected).
            self._conn.execute(
                "UPDATE OR IGNORE concept_occurrences SET concept_name=?"
                " WHERE lower(concept_name)=lower(?) AND resolution_status='resolved'",
                (result_senses[0]["name"], entity_name),
            )
            # Remove stale compile state rows for the original entity. Source edges
            # were moved to the senses above, so these rows are orphaned — the
            # scheduler now excludes them anyway (original entity is retired), but they
            # would inflate refresh_raw_compile_status counts.
            self._conn.execute(
                "DELETE FROM concept_compile_state WHERE entity_id=?",
                (original_id,),
            )

            self._log_identity_op(
                "split",
                [original_id, *[s["entity_id"] for s in result_senses]],
                {original_id: entity_name, **{s["entity_id"]: s["name"] for s in result_senses}},
                meta={
                    "senses": [{"name": s["name"], "sources": s["sources"]} for s in result_senses]
                },
            )

        return {
            "original": entity_name,
            "senses": result_senses,
            "stub_needed": len(result_senses) >= 2,
        }

    def get_merged_entity_id(self, name: str) -> str | None:
        """Return the id of a *merged* (retired) entity whose preferred label is ``name``.

        After a merge the loser's label is added as an alias on the winner, so
        ``resolve_label``/``entity_id_for_name`` resolve ``name`` to the WINNER. Unmerge
        needs the loser, found here directly via its surviving preferred-label row joined
        to the retired entity. Most recent merge wins if a label was reused.
        """
        if not (self._has_table("concept_entities") and self._has_table("concept_labels")):
            return None
        lk = _ck(name)
        if not lk:
            return None
        row = self._conn.execute(
            "SELECT ce.id FROM concept_entities ce "
            "JOIN concept_labels cl ON cl.entity_id = ce.id "
            "WHERE ce.status = 'merged' AND cl.role = 'preferred' AND cl.label_key = ? "
            "ORDER BY ce.updated_at DESC LIMIT 1",
            (lk,),
        ).fetchone()
        return row[0] if row else None

    def _alias_owed_by_other_merge(
        self, winner_id: str, excluding_loser_id: str, label_key: str
    ) -> bool:
        """True if a still-merged loser other than excluding_loser_id contributed label_key.

        Used by unmerge to avoid stripping an absorbed alias that another active merge still owes
        to the winner. Scans ALL merge rows for this winner (no row cap): a deep curation history
        must not let an older still-active merge's claim fall outside a fixed scan window, or
        unmerge would drop the still-owed alias — the exact bug this guard exists to prevent.
        """
        if not self._has_table("concept_identity_log"):
            return False

        rows = self._conn.execute(
            "SELECT entity_ids, meta FROM concept_identity_log WHERE op='merge'"
        ).fetchall()
        for entity_ids_json, meta_json in rows:
            ids = json.loads(entity_ids_json)
            if len(ids) != 2 or ids[0] != winner_id or ids[1] == excluding_loser_id:
                continue
            row = self._conn.execute(
                "SELECT status, merged_into FROM concept_entities WHERE id=?", (ids[1],)
            ).fetchone()
            if row is None or row[0] != "merged" or row[1] != winner_id:
                continue
            meta = json.loads(meta_json) if meta_json else {}
            if any(_ck(a) == label_key for a in (meta.get("labels_absorbed") or [])):
                return True
        return False

    def unmerge_entities(self, merged_entity_id: str) -> dict:
        """Reverse the most recent merge whose loser was ``merged_entity_id``.

        Restores the loser to active, moves its source edges back off the winner, drops
        the absorbed alias labels from the winner, reseeds the loser's compile state as
        pending, re-points occurrences scoped to the restored sources, and repoints back the
        exact wiki_articles the merge moved (recorded in meta). Identity is recovered from the
        merge's ``concept_identity_log`` entry.

        Best-effort and reverses exactly ONE merge (the most recent for this loser):
          - a source cited by BOTH winner and loser at merge time collapsed to one winner
            edge; unmerge returns it to the loser, so the winner loses that edge (the
            original split is unrecorded);
          - name-keyed behavioral ledgers (rejections, blocked_concepts, stubs,
            knowledge_items) are NOT restored — only source-scoped occurrences are.
        Raises ValueError if no reversible merge is found for this entity.
        """
        log_entry = None
        if self._has_table("concept_identity_log"):
            # Direct unbounded scan (DESC) for the *most recent* merge whose loser matches.
            # Mirrors the design of _alias_owed_by_other_merge (full WHERE op='merge' scan);
            # list_identity_log caps at a "recent" window for INDEX export and must not be used
            # for unmerge discovery (old merges would be invisible despite the entity still being
            # marked merged and get_merged_entity_id succeeding).
            rows = self._conn.execute(
                "SELECT op, entity_ids, labels, meta, ts FROM concept_identity_log "
                "WHERE op='merge' ORDER BY id DESC"
            ).fetchall()
            for op, eids_json, labels_json, meta_json, ts in rows:
                try:
                    ids = json.loads(eids_json)
                except Exception:
                    continue
                if len(ids) == 2 and ids[1] == merged_entity_id:
                    entry: dict = {"op": op, "ts": ts, "entity_ids": ids}
                    try:
                        entry["labels"] = json.loads(labels_json) if labels_json else {}
                    except Exception:
                        entry["labels"] = {}
                    if meta_json:
                        try:
                            entry["meta"] = json.loads(meta_json)
                        except Exception:
                            pass
                    log_entry = entry
                    break
        if log_entry is None:
            raise ValueError(f"No merge to reverse for entity {merged_entity_id!r}")

        winner_id = log_entry["entity_ids"][0]
        loser_id = merged_entity_id
        labels = log_entry.get("labels") or {}
        # Prefer the entity's CURRENT preferred label over the merge-time snapshot: if the winner
        # was renamed after the merge, the logged name is stale and the name-keyed restore queries
        # below would match nothing.
        winner_name = self.preferred_label_for_entity(winner_id) or labels.get(winner_id) or ""
        loser_name = self.preferred_label_for_entity(loser_id) or labels.get(loser_id) or ""
        meta = log_entry.get("meta") or {}
        sources_moved = list(meta.get("sources_moved") or [])
        labels_absorbed = list(meta.get("labels_absorbed") or [])
        articles_repointed = list(meta.get("articles_repointed") or [])

        # The loser must still be merged into this winner.
        row = self._conn.execute(
            "SELECT status, merged_into FROM concept_entities WHERE id=?", (loser_id,)
        ).fetchone()
        if row is None or row[0] != "merged" or row[1] != winner_id:
            raise ValueError(
                f"Entity {loser_id!r} is not merged into {winner_id!r}; cannot unmerge"
            )

        now = datetime.now().isoformat()
        winner_lk = _ck(winner_name)
        with self._tx():
            # Reactivate the loser.
            self._conn.execute(
                "UPDATE concept_entities SET status='active', merged_into=NULL, updated_at=?"
                " WHERE id=?",
                (now, loser_id),
            )
            # Move source edges and compile state back off the winner; reseed loser pending.
            for src in sources_moved:
                self._conn.execute(
                    "UPDATE OR IGNORE concepts SET entity_id=?, name=?"
                    " WHERE entity_id=? AND source_path=?",
                    (loser_id, loser_name, winner_id, src),
                )
                self._conn.execute(
                    "DELETE FROM concept_compile_state WHERE entity_id=? AND source_path=?",
                    (winner_id, src),
                )
                self._conn.execute(
                    """INSERT INTO concept_compile_state
                           (entity_id, source_path, concept_name, status, updated_at)
                       VALUES (?, ?, ?, 'pending', ?)
                       ON CONFLICT(entity_id, source_path) DO UPDATE
                           SET concept_name=excluded.concept_name,
                               status='pending', updated_at=excluded.updated_at""",
                    (loser_id, src, loser_name, now),
                )
                self._conn.execute(
                    "UPDATE OR IGNORE concept_occurrences SET concept_name=?"
                    " WHERE lower(concept_name)=lower(?) AND source_path=?",
                    (loser_name, winner_name, src),
                )
            # Repoint back exactly the wiki_articles the merge moved (recorded in meta), so a
            # loser-owned published article recovers its own live entity_id instead of staying
            # bound to the winner. Guard on the winner binding so we never steal an article that
            # was rebound elsewhere after the merge.
            for art_path in articles_repointed:
                self._conn.execute(
                    "UPDATE wiki_articles SET entity_id=? WHERE path=? AND entity_id=?",
                    (loser_id, art_path, winner_id),
                )
            # Drop the absorbed alias labels from the winner. The loser kept its own label
            # rows through the merge, so it needs no relabeling.
            for lbl in labels_absorbed:
                lk = _ck(lbl)
                if not lk or lk == winner_lk:
                    continue
                # Don't drop an alias that another still-merged loser also contributed to this
                # winner — it is still owed. (merge A→B and C→B both absorbing 'foo': unmerging A
                # must keep 'foo' because C is still merged into B and maps it.)
                if self._alias_owed_by_other_merge(winner_id, loser_id, lk):
                    continue
                self._conn.execute(
                    "DELETE FROM concept_labels WHERE entity_id=? AND role='alias' AND label_key=?",
                    (winner_id, lk),
                )

            # Drop candidates touching either entity — the graph changed; valid pairs re-derive
            # on the next ingest.
            if self._has_table("concept_merge_candidates"):
                self._conn.execute(
                    "DELETE FROM concept_merge_candidates"
                    " WHERE entity_a IN (?, ?) OR entity_b IN (?, ?)",
                    (winner_id, loser_id, winner_id, loser_id),
                )

            self._log_identity_op(
                "unmerge",
                [winner_id, loser_id],
                {winner_id: winner_name, loser_id: loser_name},
                meta={"sources_restored": sources_moved},
            )

        return {
            "winner": winner_name,
            "loser": loser_name,
            "sources_restored": sources_moved,
            "labels_absorbed": labels_absorbed,
        }

    def get_sources_for_entities(self, names: list[str]) -> dict[str, list[str]]:
        """Return {name: [source_paths]} for each named entity."""
        return {name: self.get_sources_for_concept(name) for name in names}

    def find_match_key_collisions(self) -> list[tuple[str, str, str]]:
        """Return (entity_id_a, entity_id_b, match_key) for active entities sharing a match_key.

        These are fold-collision pairs — merge candidates from plural/singular variants of the
        canonical name (issue #54: User/Users). Only *preferred* labels are compared: two entities
        that merely share an extracted *alias* match_key are not duplicates (e.g. "United States"
        and "Ultrasound" both aliased "US" are a legal shared alias, not a merge candidate), and
        folding aliases here produces false, confusing merge suggestions whose cited match_key
        belongs to neither preferred label.
        """
        if not self._has_table("concept_labels"):
            return []
        rows = self._conn.execute(
            """
            SELECT DISTINCT a.entity_id, b.entity_id, a.match_key
            FROM concept_labels a
            JOIN concept_labels b
              ON b.match_key = a.match_key AND b.entity_id > a.entity_id
            JOIN concept_entities ea ON ea.id = a.entity_id AND ea.status = 'active'
            JOIN concept_entities eb ON eb.id = b.entity_id AND eb.status = 'active'
            WHERE a.role = 'preferred' AND b.role = 'preferred'
            ORDER BY a.match_key
            """
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def list_identity_log(self, limit: int = 200) -> list[dict]:
        """Return recent identity operations for INDEX.json export."""
        if not self._has_table("concept_identity_log"):
            return []

        rows = self._conn.execute(
            "SELECT op, entity_ids, labels, meta, ts"
            " FROM concept_identity_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for row in rows:
            entry: dict = {"op": row[0], "ts": row[4]}
            try:
                entry["entity_ids"] = json.loads(row[1])
            except Exception:
                entry["entity_ids"] = []
            try:
                entry["labels"] = json.loads(row[2])
            except Exception:
                entry["labels"] = {}
            if row[3]:
                try:
                    entry["meta"] = json.loads(row[3])
                except Exception:
                    pass
            result.append(entry)
        return result

    def list_blessed_aliases(self) -> list[dict]:
        """Return human-blessed aliases (merge/rename) for the INDEX.json durability seed.

        Only ``source IN ('user','rename')`` aliases are exported: these are curation
        decisions that a re-ingest cannot reproduce, so they must survive a state.db rebuild
        (restoring them re-arms the re-mint guard). Extracted aliases are intentionally
        omitted — they regenerate on the next ingest, and blessing them on restore would
        block the order-independent promotion rule.
        """
        if not self._has_table("concept_labels"):
            return []
        rows = self._conn.execute(
            "SELECT entity_id, label, source FROM concept_labels"
            " WHERE role='alias' AND source IN ('user', 'rename')"
            " ORDER BY entity_id, label"
        ).fetchall()
        return [{"entity_id": r[0], "label": r[1], "source": r[2]} for r in rows]

    def restore_blessed_aliases(self, entries: list[dict]) -> int:
        """Re-attach human-blessed aliases to restored entities on a fresh-DB rebuild.

        Mirrors restore_entities_from_seed's precedence: only runs when the entity exists
        (it was just recreated from the seed) and never overwrites a live label. Returns the
        count restored. Label/match keys are recomputed (deterministic) rather than seeded.
        """
        if not self._has_table("concept_labels") or not entries:
            return 0
        now = datetime.now().isoformat()
        restored = 0
        with self._tx():
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                eid = entry.get("entity_id")
                label = entry.get("label")
                source = entry.get("source")
                if not eid or not label or source not in ("user", "rename"):
                    continue
                lk = _ck(label)
                if not lk:
                    continue
                if not self._conn.execute(
                    "SELECT 1 FROM concept_entities WHERE id=?", (eid,)
                ).fetchone():
                    continue
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO concept_labels"
                    " (entity_id, label, label_key, match_key, role, source, created_at)"
                    " VALUES (?, ?, ?, ?, 'alias', ?, ?)",
                    (eid, label, lk, _mk(label), source, now),
                )
                restored += cur.rowcount
        return restored

    def restore_entities_from_seed(self, entries: list[tuple[str, str]]) -> int:
        """Recreate entities with their ORIGINAL ids + preferred labels from a rebuild seed.

        Durability (decision 13): a vault whose state.db was deleted but whose
        .synto/INDEX.json survives rebuilds losslessly — entity ids are restored, not
        re-minted, so the identity log and merge/split history still line up. Precedence:
        if any entity already exists, state.db wins and this is a no-op (the caller, not
        this method, decides drift reporting). Returns the count restored.
        """
        if not self._has_table("concept_entities") or not self._has_table("concept_labels"):
            return 0
        if self._conn.execute("SELECT COUNT(*) FROM concept_entities").fetchone()[0]:
            return 0  # state.db wins — never overwrite a live DB from a seed
        now = datetime.now().isoformat()
        restored = 0
        seen_lk: set[str] = set()
        with self._tx():
            for name, entity_id in entries:
                if not name or not entity_id:
                    continue
                lk = _ck(name)
                if not lk or lk in seen_lk:
                    continue
                seen_lk.add(lk)
                if self._conn.execute(
                    "SELECT 1 FROM concept_entities WHERE id=?", (entity_id,)
                ).fetchone():
                    continue
                self._conn.execute(
                    "INSERT INTO concept_entities (id, kind, status, created_at, updated_at)"
                    " VALUES (?, 'concept', 'active', ?, ?)",
                    (entity_id, now, now),
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO concept_labels"
                    " (entity_id, label, label_key, match_key, role, source, created_at)"
                    " VALUES (?, ?, ?, ?, 'preferred', 'extracted', ?)",
                    (entity_id, name, lk, _mk(name), now),
                )
                restored += 1
        return restored

    def restore_identity_log(self, rows: list[dict]) -> None:
        """Restore the merge/split/rename log from a rebuild seed (chronological order).

        No-op if the log already has rows (state.db wins). `rows` is in the DESC shape
        returned by list_identity_log / written to INDEX.json, so it is reversed here.
        """
        if not self._has_table("concept_identity_log"):
            return
        if self._conn.execute("SELECT COUNT(*) FROM concept_identity_log").fetchone()[0]:
            return

        with self._tx():
            for r in reversed(rows):
                if not isinstance(r, dict) or not r.get("op"):
                    continue
                self._conn.execute(
                    "INSERT INTO concept_identity_log (op, entity_ids, labels, meta, ts)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        r.get("op"),
                        json.dumps(r.get("entity_ids", [])),
                        json.dumps(r.get("labels", {})),
                        json.dumps(r["meta"]) if r.get("meta") else None,
                        r.get("ts") or datetime.now().isoformat(),
                    ),
                )

    def list_active_entities_without_articles(self) -> list[tuple[str, str]]:
        """Return (entity_id, preferred_label) for active entities with no published/draft article.

        A draft awaiting approval counts as "has an article": the page is already compiled,
        so flagging it as an orphan (and advising "run compile") would be wrong. Only entities
        with no article of any kind — a failed compile or a dangling merge/split — are orphans.
        """
        if not self._has_table("concept_labels"):
            return []
        rows = self._conn.execute(
            """
            SELECT ce.id, cl.label
            FROM concept_entities ce
            JOIN concept_labels cl ON cl.entity_id = ce.id AND cl.role = 'preferred'
            WHERE ce.status = 'active'
              AND NOT EXISTS (
                  SELECT 1 FROM wiki_articles wa
                  WHERE wa.status IN ('published', 'draft')
                    AND (
                      lower(wa.title) = lower(cl.label)
                      OR lower(replace(wa.path, '\\', '/'))
                         LIKE '%/' || lower(cl.label_key) || '.md'
                    )
              )
            ORDER BY cl.label
            """
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def find_ambiguous_active_labels(self) -> list[tuple[str, int]]:
        """Return (label, entity_count) for labels claimed by >=2 active entities.

        A bare homonym label ("Mercury", an alias on both senses after a split) resolves
        ambiguously and should route to a disambiguation stub. Used by lint.
        """
        if not self._has_table("concept_labels"):
            return []
        rows = self._conn.execute(
            """
            SELECT MIN(cl.label) AS label, COUNT(DISTINCT cl.entity_id) AS n
            FROM concept_labels cl
            JOIN concept_entities ce ON ce.id = cl.entity_id AND ce.status = 'active'
            GROUP BY cl.label_key
            HAVING n >= 2
            ORDER BY label
            """
        ).fetchall()
        return [(r[0], int(r[1])) for r in rows]

    def count_legacy_backfill_aliases(self) -> int:
        """Number of aliases still sourced from the v18 legacy backfill (untrusted)."""
        if not self._has_table("concept_labels"):
            return 0
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM concept_labels WHERE source='legacy_backfill'"
            ).fetchone()[0]
        )

    def list_active_preferred_labels(self) -> list[str]:
        """Preferred labels of all active concept entities."""
        if not self._has_table("concept_labels"):
            return []
        rows = self._conn.execute(
            """
            SELECT cl.label FROM concept_labels cl
            JOIN concept_entities ce ON ce.id = cl.entity_id AND ce.status = 'active'
            WHERE cl.role = 'preferred'
            ORDER BY cl.label
            """
        ).fetchall()
        return [r[0] for r in rows]

    # ── Alias facade (backed by concept_labels) ──

    def upsert_aliases(
        self, concept_name: str, aliases: list[str], source: str = "extracted"
    ) -> None:
        """Merge aliases for a concept. Skips self-matches (alias == canonical).

        ``source`` defaults to 'extracted' (a weak LLM-guessed alias). Curation paths that record a
        human decision (e.g. rename keeping the old name) pass a blessed source ('rename'/'user')
        so the order-independent ingest rule links the surface instead of re-minting it.
        """
        canonical_lower = concept_name.lower()
        canonical_key = _ck(concept_name)
        now = datetime.now().isoformat()
        with self._tx():
            # Ensure the entity exists (creates if missing — upsert_aliases asserts ownership).
            entity_id = self._ensure_entity_for_name(concept_name, now)
            for alias in aliases:
                alias = alias.strip()
                if not alias or alias.lower() == canonical_lower:
                    continue
                # Primary write: concept_labels.
                if entity_id is not None:
                    alias_lk = _ck(alias)
                    if alias_lk and alias_lk != canonical_key:
                        self._conn.execute(
                            "INSERT INTO concept_labels"
                            " (entity_id, label, label_key, match_key, role, source, created_at)"
                            " VALUES (?, ?, ?, ?, 'alias', ?, ?)"
                            " ON CONFLICT(entity_id, label_key) DO NOTHING",
                            (entity_id, alias, alias_lk, _mk(alias), source, now),
                        )

    def get_aliases(self, concept_name: str) -> list[str]:
        """All aliases stored for a concept (case-insensitive match on concept_name)."""
        entity_id = self.entity_id_for_name(concept_name)
        if entity_id is None:
            return []
        rows = self._conn.execute(
            "SELECT label FROM concept_labels"
            " WHERE entity_id = ? AND role = 'alias' ORDER BY label",
            (entity_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def aliases_for_concept(self, concept_name: str) -> list[str]:
        return self.get_aliases(concept_name)

    def resolve_alias(self, surface: str) -> str | None:
        """Return canonical concept name if surface unambiguously matches exactly one concept."""
        result = self.resolve_label(surface)
        if result.ambiguous or not result.ids:
            return None
        return self.preferred_label_for_entity(result.ids[0])

    def list_alias_map(self) -> dict[str, str]:
        """Return {label_key: canonical_name} for all unambiguous alias labels.

        Labels claimed by more than one active entity are excluded — they are unsafe to rewrite.
        """
        rows = self._conn.execute(
            """
            SELECT cl.label_key, cl_p.label AS preferred
            FROM concept_labels cl
            JOIN concept_entities ce ON ce.id = cl.entity_id AND ce.status = 'active'
            JOIN concept_labels cl_p ON cl_p.entity_id = ce.id AND cl_p.role = 'preferred'
            WHERE cl.role = 'alias'
            """
        ).fetchall()
        counts2: dict[str, int] = {}
        mapping2: dict[str, str] = {}
        for alias_key, preferred in rows:
            counts2[alias_key] = counts2.get(alias_key, 0) + 1
            mapping2[alias_key] = preferred
        return {k: v for k, v in mapping2.items() if counts2[k] == 1}

    def list_blessed_alias_map(self) -> dict[str, str]:
        """Like list_alias_map but only HUMAN-BLESSED aliases (source user/rename).

        The pre-normalization candidate rewrite uses this so a WEAK (LLM-guessed) alias surface,
        when later extracted as a concept, is NOT silently folded into its host — it must reach
        the ingest classifier and mint its own entity (order-independent identity, v26). Blessed
        aliases are durable human decisions and remain safe to fold.
        """
        rows = self._conn.execute(
            """
            SELECT cl.label_key, cl_p.label AS preferred
            FROM concept_labels cl
            JOIN concept_entities ce ON ce.id = cl.entity_id AND ce.status = 'active'
            JOIN concept_labels cl_p ON cl_p.entity_id = ce.id AND cl_p.role = 'preferred'
            WHERE cl.role = 'alias' AND cl.source IN ('user', 'rename')
            """
        ).fetchall()
        counts: dict[str, int] = {}
        mapping: dict[str, str] = {}
        for alias_key, preferred in rows:
            counts[alias_key] = counts.get(alias_key, 0) + 1
            mapping[alias_key] = preferred
        return {k: v for k, v in mapping.items() if counts[k] == 1}

    def load_concept_alias_map(self) -> dict[str, list[str]]:
        """Return {concept_name: [aliases]} for all concepts with aliases.

        Used by query routing to bridge task-vocabulary questions to source-vocabulary
        page titles. Excludes ambiguous aliases (claimed by >= 2 distinct concepts).
        """
        rows = self._conn.execute(
            """
            SELECT cl_p.label AS concept_name, cl.label AS alias
            FROM concept_labels cl
            JOIN concept_entities ce ON ce.id = cl.entity_id AND ce.status = 'active'
            JOIN concept_labels cl_p ON cl_p.entity_id = ce.id AND cl_p.role = 'preferred'
            WHERE cl.role = 'alias'
              AND cl.label_key NOT IN (
                  SELECT label_key FROM concept_labels WHERE role = 'alias'
                  GROUP BY label_key
                  HAVING count(DISTINCT entity_id) >= 2
              )
            ORDER BY cl_p.label, cl.label
            """
        ).fetchall()
        result2: dict[str, list[str]] = {}
        for concept_name, alias in rows:
            result2.setdefault(concept_name, []).append(alias)
        return result2

    def list_frequent_aliases(self, threshold: int = 2) -> list[str]:
        """label_keys claimed by >= threshold distinct active entities.

        Used at export time to filter ambiguous aliases. Language-agnostic.
        """
        rows = self._conn.execute(
            """
            SELECT cl.label_key
            FROM concept_labels cl
            JOIN concept_entities ce ON ce.id = cl.entity_id AND ce.status = 'active'
            WHERE cl.role = 'alias'
            GROUP BY cl.label_key
            HAVING count(DISTINCT cl.entity_id) >= ?
            """,
            (threshold,),
        ).fetchall()
        return [r[0] for r in rows]

    def delete_aliases_for_concept(self, concept_name: str) -> None:
        """Remove all aliases for a concept."""
        entity_id = self.entity_id_for_name(concept_name)
        with self._tx():
            if entity_id is not None:
                self._conn.execute(
                    "DELETE FROM concept_labels WHERE entity_id = ? AND role = 'alias'",
                    (entity_id,),
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

    def list_source_concept_seeds(self) -> list[tuple[str, str, list[tuple[str, str]]]]:
        """Return content-hash-guarded source-to-concept links for rebuild seeds.

        Each concept entry is (name, entity_id) using the eid bound on the concepts row
        at extraction/upsert time (repointed on merge to follow identity, like other edges).
        Callers emit the stored eid directly instead of re-resolving the name at export time.
        """
        if not self._has_table("raw_notes") or not self._has_table("concepts"):
            return []
        rows = self._conn.execute(
            """
            SELECT c.source_path, r.content_hash, c.name, c.entity_id
            FROM concepts c
            JOIN raw_notes r ON r.path = c.source_path
            ORDER BY lower(c.source_path), c.source_path, lower(c.name), c.name
            """
        ).fetchall()

        grouped: list[tuple[str, str, list[tuple[str, str]]]] = []
        current_path: str | None = None
        current_hash = ""
        current_concepts: list[tuple[str, str]] = []
        for row in rows:
            source_path = str(row["source_path"])
            if current_path is not None and source_path != current_path:
                grouped.append((current_path, current_hash, current_concepts))
                current_concepts = []
            current_path = source_path
            current_hash = str(row["content_hash"])
            current_concepts.append((str(row["name"]), str(row["entity_id"] or "")))
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

        # Exact match via entity layer first.
        result = self.resolve_label(q)
        if result.ids and not result.ambiguous:
            preferred = self.preferred_label_for_entity(result.ids[0])
            if preferred:
                return preferred, self.aliases_for_concept(preferred)

        # Substring fallback on canonical concept names (covers prefix/infix queries).
        row = self._conn.execute(
            "SELECT DISTINCT name FROM concepts WHERE lower(name) LIKE ? ESCAPE '\\' LIMIT 1",
            (f"%{q_escaped}%",),
        ).fetchone()
        if row:
            name = row["name"]
            return name, self.aliases_for_concept(name)

        return None

    def get_compile_state(self, concept_name: str, source_path: str) -> sqlite3.Row | None:
        entity_id = self.entity_id_for_name(concept_name)
        if entity_id is None:
            return None
        return self.get_compile_state_for_entity(entity_id, source_path)

    def get_compile_state_for_entity(self, entity_id: str, source_path: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM concept_compile_state WHERE entity_id = ? AND source_path = ?",
            (entity_id, source_path),
        ).fetchone()

    def mark_concept_compile_state(
        self,
        concept_name: str,
        source_paths: list[str],
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        """Record compile status for a concept's source edges.

        Name-accepting wrapper: resolves to entity identity (minting if absent, as the
        ingest seam does) and keys the row on entity_id, carrying concept_name as a cache.
        Callers that already hold an entity_id should use mark_compile_state_for_entity.
        """
        now = datetime.now().isoformat()
        with self._tx():
            entity_id = self._ensure_entity_for_name(concept_name, now)
            if entity_id is None:
                return
            self._mark_compile_state_for_entity(
                entity_id, concept_name, source_paths, status, error, now
            )
        for source_path in source_paths:
            self.refresh_raw_compile_status(source_path)

    def mark_compile_state_for_entity(
        self,
        entity_id: str,
        source_paths: list[str],
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        """Record compile status keyed directly on entity_id (homonym-safe).

        The concept_name cache is refreshed from the entity's current preferred label.
        """
        now = datetime.now().isoformat()
        concept_name = self.preferred_label_for_entity(entity_id) or ""
        with self._tx():
            self._mark_compile_state_for_entity(
                entity_id, concept_name, source_paths, status, error, now
            )
        for source_path in source_paths:
            self.refresh_raw_compile_status(source_path)

    def _mark_compile_state_for_entity(
        self,
        entity_id: str,
        concept_name: str,
        source_paths: list[str],
        status: str,
        error: str | None,
        now: str,
    ) -> None:
        """Write compile-state rows (must be called inside a _tx)."""
        for source_path in source_paths:
            self._conn.execute(
                """INSERT INTO concept_compile_state
                       (entity_id, source_path, concept_name, status, error,
                        compiled_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(entity_id, source_path) DO UPDATE SET
                       concept_name=excluded.concept_name,
                       status=excluded.status,
                       error=excluded.error,
                       compiled_at=excluded.compiled_at,
                       updated_at=excluded.updated_at""",
                (
                    entity_id,
                    source_path,
                    concept_name,
                    status,
                    error,
                    now if status == "compiled" else None,
                    now,
                ),
            )

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
        """Preferred labels of concepts with pending/failed compile state, plus stub concepts.

        Scheduling keys on entity identity: the scheduler joins compile-state to
        concept_entities (status='active') by entity_id, so a retired/merged entity's
        stale rows and synthesis/disambiguation pages (no entity_id) never leak in.
        Two homonyms surface as their two distinct preferred labels.
        Excludes blocked concepts from normal scheduling.
        """
        rows = self._conn.execute(
            """
            SELECT DISTINCT cl.label AS name
            FROM concept_compile_state ccs
            JOIN concept_entities ce ON ce.id = ccs.entity_id AND ce.status = 'active'
            JOIN concept_labels cl ON cl.entity_id = ce.id AND cl.role = 'preferred'
            WHERE ccs.status IN ('pending', 'failed')
              AND lower(cl.label) NOT IN (SELECT lower(concept) FROM blocked_concepts)

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
        aliases = {a.casefold() for a in self.get_aliases(concept_name)}

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

    def concept_name_exists_exact(self, name: str, *, exclude_concept: str | None = None) -> bool:
        """True if ``name`` is already an exact concept name, alias, or knowledge item.

        Exact (case-insensitive) only — deliberately NOT the fuzzy
        ``find_concept_by_name_or_alias`` (which falls back to substring LIKE and would
        report false collisions, e.g. renaming to "Net" matching "Network").

        ``exclude_concept`` drops one concept's own identity from the check so a rename
        can promote that concept's existing alias to canonical (e.g. "Program Counter"
        with alias "PC" → "PC") without a false self-collision.
        """
        n = name.casefold()
        ex = exclude_concept.casefold() if exclude_concept else None
        # A concept/knowledge-item literally named `name` (other than the excluded one).
        if n != ex:
            for sql in (
                "SELECT 1 FROM concepts WHERE lower(name) = ? LIMIT 1",
                "SELECT 1 FROM knowledge_items WHERE lower(name) = ? LIMIT 1",
            ):
                if self._conn.execute(sql, (n,)).fetchone():
                    return True
        # An alias label_key claimed by a *different* entity.
        key = _ck(name)
        ex_key = _ck(exclude_concept) if exclude_concept else None
        row = self._conn.execute(
            """
            SELECT 1 FROM concept_labels cl
            JOIN concept_entities ce ON ce.id = cl.entity_id AND ce.status = 'active'
            JOIN concept_labels cl_p ON cl_p.entity_id = ce.id AND cl_p.role = 'preferred'
            WHERE cl.label_key = ?
              AND (? IS NULL OR cl_p.label_key != ?)
            LIMIT 1
            """,
            (key, ex_key, ex_key),
        ).fetchone()
        return row is not None

    def find_concept_exact(self, query: str) -> tuple[str, list[str]] | None:
        """Resolve a concept by EXACT (case-insensitive) name or alias only.

        No substring fallback — use this for destructive operations (rename) where a
        fuzzy neighbour (e.g. "Net" -> "Network") would silently target the wrong
        concept. Fuzzy callers (ingest, query, MCP) want find_concept_by_name_or_alias.
        """
        result = self.resolve_label(query)
        if result.ids and not result.ambiguous:
            preferred = self.preferred_label_for_entity(result.ids[0])
            if preferred:
                return preferred, self.aliases_for_concept(preferred)
        return None

    def rename_concept(self, old_name: str, new_name: str) -> None:
        """Migrate a concept's identity and behavioral state from ``old`` to ``new``.

        Atomically re-keys every table that binds a concept by name: identity
        (concepts, concept_compile_state, concept_occurrences, knowledge_items) and
        behavioral state whose consumers look up by current title (rejections — exact-match
        guidance; blocked_concepts — skipping silently unblocks; stubs). Caller must
        guarantee ``new`` is collision-free first.

        item_mentions is intentionally left untouched: those rows are generic
        source-evidence mentions, not concept canonical binding.
        """
        entity_id = self.entity_id_for_name(old_name)
        new_lk = _ck(new_name)
        new_mk = _mk(new_name)
        with self._tx():
            self._conn.execute(
                "UPDATE concepts SET name = ? WHERE lower(name) = lower(?)",
                (new_name, old_name),
            )
            # Update concept_labels: change preferred label.
            if entity_id is not None and new_lk:
                # Drop any alias that would collide with the new preferred label_key first.
                self._conn.execute(
                    "DELETE FROM concept_labels WHERE entity_id=? AND role='alias' AND label_key=?",
                    (entity_id, new_lk),
                )
                self._conn.execute(
                    "UPDATE concept_labels SET label=?, label_key=?, match_key=?"
                    " WHERE entity_id=? AND role='preferred'",
                    (new_name, new_lk, new_mk, entity_id),
                )
            # Refresh the compile-state name cache for this entity (identity, the
            # entity_id PK, does not move on a rename).
            if entity_id is not None:
                self._conn.execute(
                    "UPDATE concept_compile_state SET concept_name = ? WHERE entity_id = ?",
                    (new_name, entity_id),
                )
            self._conn.execute(
                "UPDATE concept_occurrences SET concept_name = ? "
                "WHERE lower(concept_name) = lower(?)",
                (new_name, old_name),
            )
            self._conn.execute(
                "UPDATE knowledge_items SET name = ? WHERE lower(name) = lower(?)",
                (new_name, old_name),
            )
            self._conn.execute(
                "UPDATE rejections SET concept = ? WHERE lower(concept) = lower(?)",
                (new_name, old_name),
            )
            self._conn.execute(
                "UPDATE blocked_concepts SET concept = ? WHERE lower(concept) = lower(?)",
                (new_name, old_name),
            )
            self._conn.execute(
                "UPDATE stubs SET concept = ? WHERE lower(concept) = lower(?)",
                (new_name, old_name),
            )
            # A candidate's stored surface/labels reference the old label — drop those touching the
            # renamed entity; still-valid pairs re-derive on the next ingest.
            if entity_id is not None and self._has_table("concept_merge_candidates"):
                self._conn.execute(
                    "DELETE FROM concept_merge_candidates WHERE entity_a = ? OR entity_b = ?",
                    (entity_id, entity_id),
                )

    def update_article_identity(self, old_path: str, new_path: str, new_title: str) -> None:
        """Repoint a tracked article row to a new path/title.

        An UPDATE (not delete+insert) so article_id and lineage survive. No-op if the
        old row is absent (DB-only rename of a never-compiled concept).

        content_hash is deliberately preserved, not recomputed: a concept rename changes
        only the frontmatter title, and content_hash is body-only. Recomputing it would
        sync the DB hash to the on-disk body and so erase manual-edit protection
        (compile skips a published article only while its body hash differs from the DB);
        that would expose a hand-fixed article to being clobbered on the next compile.
        """
        with self._tx():
            if not self._conn.execute(
                "SELECT 1 FROM wiki_articles WHERE path = ?", (old_path,)
            ).fetchone():
                return
            if old_path != new_path:
                self._conn.execute("DELETE FROM wiki_articles WHERE path = ?", (new_path,))
            self._conn.execute(
                "UPDATE wiki_articles SET path=?, title=?, updated_at=? WHERE path=?",
                (new_path, new_title, datetime.now().isoformat(), old_path),
            )

    def _upsert_article_row(self, record: WikiArticleRecord) -> None:
        """Write one wiki_articles row (must be called inside a _tx)."""
        article_id = self._resolve_article_id(record.path, record.article_id)
        self._conn.execute(
            """INSERT INTO wiki_articles
                   (
                        path, title, sources, content_hash, created_at, updated_at, status,
                        approved_at, approval_notes, kind, question_hash,
                        synthesis_sources, synthesis_source_hashes, article_id,
                        last_compile_pipeline, entity_id
                    )
                VALUES (:path, :title, :sources, :content_hash,
                        :created_at, :updated_at, :status,
                        :approved_at, :approval_notes, :kind, :question_hash,
                        :synthesis_sources, :synthesis_source_hashes, :article_id,
                        :last_compile_pipeline, :entity_id)
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
                    ),
                    entity_id=COALESCE(excluded.entity_id, wiki_articles.entity_id)""",
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
                "entity_id": record.entity_id,
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

    def published_path_for_entity(self, entity_id: str) -> str | None:
        """Vault-relative path of the published concept article for an entity, or None."""
        if not self._has_table("wiki_articles") or not entity_id:
            return None
        row = self._conn.execute(
            "SELECT path FROM wiki_articles WHERE entity_id = ? AND status = 'published'"
            " AND kind = 'concept' LIMIT 1",
            (entity_id,),
        ).fetchone()
        return row[0] if row else None

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
            self._mark_article_compiled(art)

    def _mark_article_compiled(self, art: WikiArticleRecord) -> None:
        """Mark an article's concept compiled, keyed on its bound entity when present.

        Recovers identity from the article's entity_id (homonym-safe) rather than the
        title; synthesis/disambiguation and legacy un-bound articles fall back to title.
        """
        if art.entity_id is not None:
            self.mark_compile_state_for_entity(art.entity_id, art.sources, "compiled")
        else:
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
            self._mark_article_compiled(art)

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
                    and rel_posix(path, vault) not in tracked_paths
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
                    rel_path = rel_posix(path, vault)
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
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM concept_labels WHERE role = 'alias'"
            ).fetchone()[0]
        )

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
        with self._tx() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO compile_runs
                   (run_ulid, pipeline_json, fast_model, heavy_model, started_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_ulid, pipeline_json, fast_model, heavy_model, now),
            )

    def finish_compile_run(
        self,
        run_ulid: str,
        article_count: int = 0,
        total_tokens: int = 0,
        total_cost_usd: float = 0.0,
    ) -> None:
        now = datetime.now().isoformat()
        with self._tx() as conn:
            conn.execute(
                """UPDATE compile_runs
                   SET finished_at = ?, article_count = ?, total_tokens = ?, total_cost_usd = ?
                   WHERE run_ulid = ?""",
                (now, article_count, total_tokens, total_cost_usd, run_ulid),
            )

    def update_article_compile_run(self, article_path: str, run_ulid: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE wiki_articles SET last_compile_pipeline = ? WHERE path = ?",
                (run_ulid, article_path),
            )

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

    def get_segments_for_source(self, source_id: str) -> list[sqlite3.Row]:
        """Return a source's segments (id, ordinal, structural_locator, text) in reading order.

        Used by ingest to build structure-aware analysis chunks aligned to segments, so the
        concepts a chunk yields can be attributed to known segment ids (concept_occurrences).
        """
        if not self._has_table("source_segments"):
            return []
        return self._conn.execute(
            """SELECT id, ordinal, structural_locator, text
               FROM source_segments WHERE source_id = ? ORDER BY ordinal""",
            (source_id,),
        ).fetchall()

    def clear_concept_occurrences_for_source(self, source_id: str) -> None:
        """Delete all concept→segment links for a source's segments (re-ingest = replace)."""
        if not self._has_table("concept_occurrences") or not self._has_table("source_segments"):
            return
        with self._tx():
            self._conn.execute(
                """DELETE FROM concept_occurrences WHERE source_segment_id IN
                   (SELECT id FROM source_segments WHERE source_id = ?)""",
                (source_id,),
            )

    def concept_occurrence_count(self) -> int:
        """Total concept→segment links — used by `synto doctor` for coverage reporting."""
        if not self._has_table("concept_occurrences"):
            return 0
        return int(self._conn.execute("SELECT count(*) FROM concept_occurrences").fetchone()[0])

    def select_passages_for_concept(
        self, canonical_name: str, max_passages: int
    ) -> list[sqlite3.Row]:
        """Return source segments linked to a concept, ordered by confidence then ordinal.

        Confidence defaults to 1.0 for extractors that don't set it, so the
        secondary order (ordinal ASC) acts as the effective ordering in that case.
        """
        return self._conn.execute(
            """SELECT s.id, s.source_id, s.ordinal, s.text, co.confidence,
                      d.origin_uri, d.license, d.id AS doc_id
               FROM concept_occurrences co
               JOIN source_segments s ON s.id = co.source_segment_id
               LEFT JOIN source_documents d ON d.id = s.source_id
               WHERE co.concept_name = ?
               ORDER BY co.confidence DESC, s.ordinal ASC
               LIMIT ?""",
            (canonical_name, max_passages),
        ).fetchall()

    def fts5_available(self) -> bool:
        """True if this SQLite build supports FTS5 (verbatim search index)."""
        return _fts5_available(self._conn)

    def source_segments_fts_exists(self) -> bool:
        """Cheap existence check for the FTS index (no row count)."""
        return (
            self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_segments_fts'"
            ).fetchone()
            is not None
        )

    def any_source_license_declared(self) -> bool:
        """True if at least one source_documents row has a non-null license."""
        if not self._has_table("source_documents"):
            return False
        return (
            self._conn.execute(
                "SELECT 1 FROM source_documents WHERE license IS NOT NULL LIMIT 1"
            ).fetchone()
            is not None
        )

    def source_segments_fts_status(self) -> tuple[bool, int, int]:
        """Return (fts_table_exists, fts_row_count, segment_row_count).

        Used by `synto doctor` to surface FTS index drift. When the FTS
        table is absent (vault below v16), fts_row_count is 0.
        """
        seg_count = self._conn.execute("SELECT count(*) FROM source_segments").fetchone()[0]
        fts_exists = (
            self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_segments_fts'"
            ).fetchone()
            is not None
        )
        if not fts_exists:
            return (False, 0, seg_count)
        fts_count = self._conn.execute("SELECT count(*) FROM source_segments_fts").fetchone()[0]
        return (True, fts_count, seg_count)

    def fetch_segment_by_id(self, segment_id: str) -> sqlite3.Row | None:
        """Return source_segments row joined with source_documents.origin_uri, or None."""
        return self._conn.execute(
            """SELECT s.source_id, s.identity, s.ordinal, s.content_hash, s.text,
                      d.origin_uri
               FROM source_segments s
               LEFT JOIN source_documents d ON d.id = s.source_id
               WHERE s.id = ?""",
            (segment_id,),
        ).fetchone()

    def search_segments_fts(self, match_arg: str, limit: int) -> list[sqlite3.Row]:
        """BM25 search across source_segments_fts; rows joined back to source_segments."""
        return self._conn.execute(
            """SELECT s.id AS segment_id, s.source_id, s.ordinal,
                      snippet(source_segments_fts, 0, '', '', '…', 32) AS snippet,
                      bm25(source_segments_fts) AS rank,
                      length(s.text) AS body_length
               FROM source_segments_fts
               JOIN source_segments s ON s.rowid = source_segments_fts.rowid
               WHERE source_segments_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (match_arg, limit),
        ).fetchall()

    def fetch_source_meta(self, source_ids: list[str]) -> dict[str, tuple[str | None, str | None]]:
        """Return {source_id: (license, origin_uri)} for the given source_ids."""
        if not source_ids:
            return {}
        placeholders = ",".join("?" * len(source_ids))
        rows = self._conn.execute(
            f"SELECT id, license, origin_uri FROM source_documents WHERE id IN ({placeholders})",
            source_ids,
        ).fetchall()
        return {r["id"]: (r["license"], r["origin_uri"]) for r in rows}

    def source_document_exists(self, source_id: str) -> bool:
        """True if source_documents has a row with this id."""
        row = self._conn.execute(
            "SELECT 1 FROM source_documents WHERE id = ?", (source_id,)
        ).fetchone()
        return row is not None

    def fetch_source_license(self, source_id: str) -> str | None:
        """Return the license string for a source, or None for unknown/null."""
        row = self._conn.execute(
            "SELECT license FROM source_documents WHERE id = ?", (source_id,)
        ).fetchone()
        return row["license"] if row is not None else None

    def count_segments_for_source(self, source_id: str) -> int:
        return self._conn.execute(
            "SELECT count(*) FROM source_segments WHERE source_id = ?", (source_id,)
        ).fetchone()[0]

    def list_segments_for_source(
        self, source_id: str, limit: int, offset: int
    ) -> list[sqlite3.Row]:
        """Return id/ordinal/length tuples for segments of a source, ordered by ordinal."""
        return self._conn.execute(
            """SELECT id, ordinal, length(text) AS length
               FROM source_segments
               WHERE source_id = ?
               ORDER BY ordinal
               LIMIT ? OFFSET ?""",
            (source_id, limit, offset),
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

        with self._tx() as conn:
            conn.execute(
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
        result_count: int | None = None,
        resolved_label: str | None = None,
    ) -> None:
        metadata: dict[str, object] = {"tool": tool, "args": args_summary}
        # `result_count` must be embedded even when 0 (a zero-result call is the
        # whole point of the backlog report); use `is not None`, not truthiness.
        if result_count is not None:
            metadata["result_count"] = result_count
        if resolved_label is not None:
            metadata["resolved_label"] = resolved_label
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
            metadata_json=json.dumps(metadata, sort_keys=True),
        )

    def count_mcp_audit_rows_since(self, since_ts: str) -> int:
        if not self._has_table("metric_events"):
            return 0
        row = self._conn.execute(
            "SELECT COUNT(*) FROM metric_events WHERE event_type='mcp_call' AND ts >= ?",
            (since_ts,),
        ).fetchone()
        return int(row[0])

    def zero_result_query_counts(
        self, since_ts: str, top_n: int = 20
    ) -> list[tuple[str, str | None, int]]:
        if not self._has_table("metric_events"):
            return []
        rows = self._conn.execute(
            """
            SELECT
                json_extract(metadata_json,'$.tool') AS tool,
                COALESCE(
                    json_extract(metadata_json,'$.args.query'),
                    json_extract(metadata_json,'$.args.concept_name')
                ) AS label,
                COUNT(*) AS cnt
            FROM metric_events
            WHERE event_type='mcp_call'
              AND ts >= ?
              AND success = 1
              AND json_extract(metadata_json,'$.result_count') = 0
              AND json_extract(metadata_json,'$.tool') IN (
                    'find_concept','search_articles',
                    'search_source_segments','get_source_passages'
              )
            GROUP BY tool, label
            ORDER BY cnt DESC, label ASC, MIN(id) ASC
            LIMIT ?
            """,
            (since_ts, top_n),
        ).fetchall()
        return [(r[0], r[1], int(r[2])) for r in rows]

    def single_source_concepts_in_demand(self, since_ts: str) -> list[tuple[str, int, int]]:
        # resolved_label is stored hashed when mcp.audit_detailed is off (the default)
        # and plaintext when it is on. Concept names in concept_occurrences are always
        # plaintext, so we cannot JOIN them in SQL — we match in Python against both the
        # plaintext name and its _hash8, which makes the report work under either mode
        # (and across mixed history) without weakening the privacy default.
        if not self._has_table("metric_events"):
            return []
        if not self._has_table("concept_occurrences"):
            return []
        # "Single-source" means backed by exactly one source DOCUMENT. concept_occurrences
        # keys on segment id, and one source has many segments, so COUNT(*) would count
        # segments, not sources — JOIN source_segments and count DISTINCT source_id.
        if self._has_table("source_segments"):
            single_sql = (
                "SELECT co.concept_name FROM concept_occurrences co "
                "JOIN source_segments s ON s.id = co.source_segment_id "
                "GROUP BY co.concept_name HAVING COUNT(DISTINCT s.source_id) = 1"
            )
        else:
            single_sql = (
                "SELECT concept_name FROM concept_occurrences "
                "GROUP BY concept_name HAVING COUNT(*) = 1"
            )
        single_names = [r[0] for r in self._conn.execute(single_sql).fetchall()]
        if not single_names:
            return []
        label_to_name = self._build_label_lookup(single_names)
        demand = self._conn.execute(
            """
            SELECT json_extract(metadata_json,'$.resolved_label') AS label, COUNT(*) AS hits
            FROM metric_events
            WHERE event_type='mcp_call' AND ts >= ?
              AND json_extract(metadata_json,'$.tool') IN ('find_concept','get_source_passages')
              AND json_extract(metadata_json,'$.resolved_label') IS NOT NULL
            GROUP BY label
            """,
            (since_ts,),
        ).fetchall()
        hits_by_name: dict[str, int] = {}
        for label, hits in demand:
            name = label_to_name.get(label)
            if name is not None:
                hits_by_name[name] = hits_by_name.get(name, 0) + int(hits)
        # Match the prior SQL ordering: hits DESC, then concept name ASC.
        ranked = sorted(hits_by_name.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(name, 1, hits) for name, hits in ranked]

    def repeat_weak_queries(
        self,
        since_ts: str,
        single_source_names: set[str],
        min_hits: int = 2,
    ) -> list[tuple[str | None, int, str]]:
        if not self._has_table("metric_events"):
            return []
        # Avoid building an empty IN () which is invalid SQL.
        if not single_source_names:
            return []
        # Match stored resolved_label in either form (plaintext or _hash8) — see
        # single_source_concepts_in_demand for why. The IN set spans both forms; the
        # returned target_concept is mapped back to the plaintext concept name.
        label_to_name = self._build_label_lookup(single_source_names)
        match_labels = sorted(label_to_name.keys())
        placeholders = ",".join("?" * len(match_labels))
        params: list[object] = [since_ts, *match_labels, min_hits]
        rows = self._conn.execute(
            f"""
            SELECT
                COALESCE(
                    json_extract(metadata_json,'$.args.query'),
                    json_extract(metadata_json,'$.args.concept_name')
                ) AS label,
                COUNT(*) AS hits,
                json_extract(metadata_json,'$.resolved_label') AS target_concept
            FROM metric_events
            WHERE event_type='mcp_call' AND ts >= ?
              AND json_extract(metadata_json,'$.tool') IN ('find_concept','get_source_passages')
              AND json_extract(metadata_json,'$.resolved_label') IN ({placeholders})
            GROUP BY label, target_concept
            HAVING COUNT(*) >= ?
            ORDER BY hits DESC, label ASC, MIN(id) ASC
            """,
            params,
        ).fetchall()
        return [(r[0], int(r[1]), label_to_name.get(r[2], r[2])) for r in rows]

    @staticmethod
    def _build_label_lookup(names) -> dict[str, str]:
        """Map every stored resolved_label form back to its plaintext concept name.

        Each concept name maps from both itself (audit_detailed=true) and its _hash8
        (audit_detailed=false). 8-char-prefix collisions are theoretically possible but
        negligible for an informational report.
        """
        lookup: dict[str, str] = {}
        for name in names:
            lookup[name] = name
            lookup[_hash8(name)] = name
        return lookup

    def tool_mix_sessions(
        self,
        since_ts: str,
        idle_gap_seconds: int = _SESSION_IDLE_GAP_SECONDS,
        min_calls: int = 5,
    ) -> list[tuple[int, int, int, int, int]]:
        if not self._has_table("metric_events"):
            return []
        rows = self._conn.execute(
            """
            WITH base AS (
                SELECT id, ts, COALESCE(vault_id, '_unknown_') AS part,
                       json_extract(metadata_json,'$.tool') AS tool
                FROM metric_events
                WHERE event_type='mcp_call' AND ts >= ?
            ),
            lagged AS (
                SELECT *,
                    LAG(ts) OVER (PARTITION BY part ORDER BY ts, id) AS prev_ts
                FROM base
            ),
            flagged AS (
                SELECT *,
                    CASE
                        WHEN prev_ts IS NULL THEN 1
                        WHEN (julianday(ts) - julianday(prev_ts)) * 86400.0 > ? THEN 1
                        ELSE 0
                    END AS is_new
                FROM lagged
            ),
            sessioned AS (
                SELECT *,
                    SUM(is_new) OVER (PARTITION BY part ORDER BY ts, id
                                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS sess
                FROM flagged
            )
            SELECT part, sess,
                COUNT(*) AS total,
                SUM(CASE WHEN tool IN (
                    'read_source_segment','search_source_segments',
                    'get_source_passages','list_segments'
                ) THEN 1 ELSE 0 END) AS verbatim,
                SUM(CASE WHEN tool = 'answer_question' THEN 1 ELSE 0 END) AS answer_question,
                SUM(CASE WHEN tool NOT IN (
                    'read_source_segment','search_source_segments',
                    'get_source_passages','list_segments','answer_question'
                ) THEN 1 ELSE 0 END) AS other
            FROM sessioned
            GROUP BY part, sess
            HAVING COUNT(*) >= ?
            ORDER BY part, sess
            """,
            (since_ts, idle_gap_seconds, min_calls),
        ).fetchall()
        # NULL vault rows are bucketed under '_unknown_', not dropped.
        return [(sid, int(r[2]), int(r[3]), int(r[4]), int(r[5])) for sid, r in enumerate(rows)]

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
        entity_id=row["entity_id"] if "entity_id" in keys else None,
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
