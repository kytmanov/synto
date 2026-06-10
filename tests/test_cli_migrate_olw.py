"""Tests for the `synto migrate-olw` CLI command."""

from __future__ import annotations

import sqlite3

from click.testing import CliRunner

from synto.cli import cli


def _make_legacy_vault(tmp_path):
    """Create a minimal legacy (.olw / wiki.toml) vault layout."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki.toml").write_text('[models]\nfast = "gemma4:e4b"\nheavy = "qwen2.5:14b"\n')
    olw = tmp_path / ".olw"
    olw.mkdir()
    sqlite3.connect(olw / "state.db").close()
    return tmp_path


def test_migrate_olw_copies_config_and_appdir(tmp_path):
    vault = _make_legacy_vault(tmp_path)

    result = CliRunner().invoke(cli, ["migrate-olw", "--vault", str(vault)])

    assert result.exit_code == 0, result.output
    assert (vault / "synto.toml").exists()
    assert (vault / ".synto").is_dir()
    assert (vault / ".synto" / "state.db").exists()


def test_migrate_olw_preserves_originals(tmp_path):
    vault = _make_legacy_vault(tmp_path)

    CliRunner().invoke(cli, ["migrate-olw", "--vault", str(vault)])

    assert (vault / "wiki.toml").exists(), "wiki.toml must be preserved (copy, not move)"
    assert (vault / ".olw").is_dir(), ".olw/ must be preserved (copy, not move)"
    assert (vault / ".olw" / "state.db").exists()


def test_migrate_olw_refuses_if_synto_toml_already_exists(tmp_path):
    vault = _make_legacy_vault(tmp_path)
    (vault / "synto.toml").write_text("[models]\n")

    result = CliRunner().invoke(cli, ["migrate-olw", "--vault", str(vault)])

    assert result.exit_code != 0
    assert "Refusing" in result.output or "Error" in result.output


def test_migrate_olw_refuses_if_no_legacy_layout(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "raw").mkdir()

    result = CliRunner().invoke(cli, ["migrate-olw", "--vault", str(tmp_path)])

    assert result.exit_code != 0


def test_migrate_olw_config_content_preserved(tmp_path):
    vault = _make_legacy_vault(tmp_path)

    CliRunner().invoke(cli, ["migrate-olw", "--vault", str(vault)])

    content = (vault / "synto.toml").read_text()
    assert "gemma4:e4b" in content
    assert "qwen2.5:14b" in content


def test_migrate_olw_updates_gitignore(tmp_path):
    vault = _make_legacy_vault(tmp_path)
    (vault / ".gitignore").write_text(".olw/\n")

    CliRunner().invoke(cli, ["migrate-olw", "--vault", str(vault)])

    gitignore = (vault / ".gitignore").read_text()
    assert ".synto/" in gitignore


def test_migrate_olw_gitignore_includes_lock_and_exports(tmp_path):
    vault = _make_legacy_vault(tmp_path)

    CliRunner().invoke(cli, ["migrate-olw", "--vault", str(vault)])

    gitignore = (vault / ".gitignore").read_text()
    assert ".synto/pipeline.lock" in gitignore
    assert ".synto/exports/" in gitignore


def test_migrate_olw_gitignore_idempotent(tmp_path):
    vault = _make_legacy_vault(tmp_path)

    CliRunner().invoke(cli, ["migrate-olw", "--vault", str(vault)])
    first = (vault / ".gitignore").read_text()

    # Restore legacy layout stub so second migration isn't refused
    (vault / "wiki.toml").write_text('[models]\nfast = "gemma4:e4b"\nheavy = "qwen2.5:14b"\n')
    # Count occurrences before and after a hypothetical re-run of the append loop
    assert first.count(".synto/pipeline.lock") == 1
    assert first.count(".synto/exports/") == 1


def test_migrate_olw_renames_telemetry_section_to_metrics(tmp_path):
    vault = _make_legacy_vault(tmp_path)
    (vault / "wiki.toml").write_text(
        '[models]\nfast = "gemma4:e4b"\nheavy = "qwen2.5:14b"\n\n[telemetry]\npersist = true\n'
    )

    result = CliRunner().invoke(cli, ["migrate-olw", "--vault", str(vault)])

    assert result.exit_code == 0, result.output
    content = (vault / "synto.toml").read_text()
    assert "[metrics]" in content
    assert "[telemetry]" not in content


def test_migrate_olw_upgrades_v9_db_and_drops_legacy_telemetry_tables(tmp_path):
    vault = _make_legacy_vault(tmp_path)
    db_path = vault / ".olw" / "state.db"
    db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_version (id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL);
        INSERT INTO schema_version (id, version) VALUES (1, 9);
        CREATE TABLE raw_notes (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'new',
            summary TEXT,
            quality TEXT,
            language TEXT,
            ingested_at TEXT,
            compiled_at TEXT,
            error TEXT,
            source_type TEXT NOT NULL DEFAULT 'notes',
            origin_uri TEXT,
            imported_at TEXT,
            normalized_hash TEXT,
            extractor_version TEXT,
            prompt_version TEXT
        );
        CREATE TABLE concepts (
            name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            PRIMARY KEY (name, source_path)
        );
        CREATE TABLE wiki_articles (
            path TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            sources TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            is_draft INTEGER NOT NULL DEFAULT 1,
            approved_at TEXT,
            approval_notes TEXT,
            kind TEXT NOT NULL DEFAULT 'concept',
            question_hash TEXT,
            synthesis_sources TEXT,
            synthesis_source_hashes TEXT,
            article_id TEXT,
            last_compile_pipeline TEXT
        );
        CREATE TABLE rejections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT NOT NULL,
            feedback TEXT NOT NULL,
            rejected_body TEXT,
            rejected_at TEXT NOT NULL
        );
        CREATE TABLE stubs (
            concept TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'auto'
        );
        CREATE TABLE blocked_concepts (concept TEXT PRIMARY KEY, blocked_at TEXT NOT NULL);
        CREATE TABLE concept_aliases (
            concept_name TEXT NOT NULL,
            alias TEXT NOT NULL,
            PRIMARY KEY (concept_name, alias)
        );
        CREATE TABLE knowledge_items (
            name TEXT PRIMARY KEY,
            kind TEXT NOT NULL DEFAULT 'ambiguous',
            subtype TEXT,
            status TEXT NOT NULL DEFAULT 'candidate',
            confidence REAL NOT NULL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE item_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            mention_text TEXT NOT NULL,
            context TEXT,
            evidence_level TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            UNIQUE(item_name, source_path, mention_text, evidence_level)
        );
        CREATE TABLE ingest_chunks (
            source_path TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            chunk_size INTEGER NOT NULL,
            checkpoint_schema INTEGER NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
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
            concept_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            compiled_at TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (concept_name, source_path)
        );
        CREATE TABLE source_documents (
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
        );
        CREATE TABLE source_segments (
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
            metadata_json TEXT
        );
        CREATE TABLE source_warnings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            severity TEXT NOT NULL,
            category TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE generated_assets (
            path TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            master_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_referenced_at TEXT,
            referenced_by_json TEXT NOT NULL DEFAULT '[]'
        );
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
        INSERT INTO telemetry_daily_rollups (day, vault_id, event_type, tier, calls) VALUES (
            '2026-05-15', 'vault', 'llm_call', 'fast', 2
        );
        """
    )
    conn.commit()
    conn.close()

    result = CliRunner().invoke(cli, ["migrate-olw", "--vault", str(vault)])

    assert result.exit_code == 0, result.output
    migrated = sqlite3.connect(vault / ".synto" / "state.db")
    version = migrated.execute("SELECT version FROM schema_version").fetchone()[0]
    tables = {
        row[0]
        for row in migrated.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert version == 20
    assert "metric_events" in tables
    assert "metric_daily_rollups" in tables
    assert "telemetry_events" not in tables
    assert "telemetry_daily_rollups" not in tables
    migrated.close()
