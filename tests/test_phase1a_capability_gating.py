from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from synto.config import Config
from synto.indexer import generate_index_json
from synto.pack_export import export_pack
from synto.readers import PackReader, VaultReader
from synto.state import StateDB


def _build_empty_v8_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_version (
            id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL
        );
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
            name TEXT NOT NULL, source_path TEXT NOT NULL,
            PRIMARY KEY (name, source_path)
        );
        CREATE TABLE wiki_articles (
            path TEXT PRIMARY KEY, title TEXT NOT NULL, sources TEXT NOT NULL,
            content_hash TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, is_draft INTEGER NOT NULL DEFAULT 1,
            approved_at TEXT, approval_notes TEXT,
            kind TEXT NOT NULL DEFAULT 'concept', question_hash TEXT,
            synthesis_sources TEXT, synthesis_source_hashes TEXT,
            article_id TEXT
        );
        INSERT INTO schema_version (id, version) VALUES (1, 8);
        """
    )
    conn.commit()
    conn.close()


def test_capability_gating_empty_v8_vault_round_trip(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / ".synto").mkdir()
    db_path = tmp_path / ".synto" / "state.db"
    _build_empty_v8_db(db_path)

    config = Config(vault=tmp_path)
    db = StateDB.open_readonly(db_path)
    try:
        reader = VaultReader(tmp_path)
        index_payload = json.loads(generate_index_json(config, db).read_text(encoding="utf-8"))
        out_dir = tmp_path / ".knowledge" / "agents"
        export_pack(config, target="agents", out=out_dir)
        manifest_payload = json.loads(
            (out_dir / "agent" / "manifest.json").read_text(encoding="utf-8")
        )
        pack_reader = PackReader(out_dir)

        assert reader.capabilities == frozenset({"articles", "concepts"})
        assert reader.has_capability("lifecycle") is False
        assert index_payload["pack"]["capabilities"] == ["articles", "concepts"]
        assert manifest_payload["pack"]["capabilities"] == ["articles", "concepts"]
        assert pack_reader.capabilities == reader.capabilities
        assert pack_reader.list_segments() == []
    finally:
        db.close()


def test_capability_gating_segments_round_trip(vault: Path, config: Config, db: StateDB) -> None:
    conn = sqlite3.connect(config.state_db_path)
    conn.execute(
        "INSERT INTO source_documents (id, title, source_type) VALUES (?, ?, ?)",
        ("src-1", "Source", "unknown_text"),
    )
    conn.execute(
        (
            "INSERT INTO source_segments "
            "(id, identity, ordinal, source_id, structural_locator, content_hash, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        ),
        ("seg-1", "src-1:0", 0, "src-1", "0", "hash", "text"),
    )
    conn.commit()
    conn.close()

    reader = VaultReader(vault)
    index_payload = json.loads(generate_index_json(config, db).read_text(encoding="utf-8"))
    out_dir = vault / ".knowledge" / "agents"
    export_pack(config, target="agents", out=out_dir)
    manifest_payload = json.loads((out_dir / "agent" / "manifest.json").read_text(encoding="utf-8"))
    pack_reader = PackReader(out_dir)
    segments = pack_reader.list_segments()

    expected = frozenset({"articles", "concepts", "lifecycle", "segments"})
    assert reader.capabilities == expected
    assert index_payload["pack"]["capabilities"] == [
        "articles",
        "concepts",
        "lifecycle",
        "segments",
    ]
    assert manifest_payload["pack"]["capabilities"] == [
        "articles",
        "concepts",
        "lifecycle",
        "segments",
    ]
    assert pack_reader.capabilities == expected
    assert [segment.id for segment in segments] == ["seg-1"]
    assert segments[0].source_id == "src-1"


def test_capability_gating_round_trip_matches_vault_and_pack_readers(
    vault: Path, config: Config, db: StateDB
) -> None:
    conn = sqlite3.connect(config.state_db_path)
    conn.execute(
        "INSERT INTO source_documents (id, title, source_type) VALUES (?, ?, ?)",
        ("src-1", "Source", "unknown_text"),
    )
    conn.execute(
        (
            "INSERT INTO source_segments "
            "(id, identity, ordinal, source_id, structural_locator, content_hash, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        ),
        ("seg-1", "src-1:0", 0, "src-1", "0", "hash", "text"),
    )
    conn.commit()
    conn.close()

    vault_reader = VaultReader(vault)
    out_dir = vault / ".knowledge" / "agents"
    export_pack(config, target="agents", out=out_dir)
    pack_reader = PackReader(out_dir)

    assert pack_reader.capabilities == vault_reader.capabilities
    assert pack_reader.has_capability("segments") is True
    assert pack_reader.has_capability("lifecycle") is True
    assert len(pack_reader.list_segments()) == len(vault_reader.list_segments()) == 1
