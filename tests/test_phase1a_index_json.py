from __future__ import annotations

import json
import sqlite3

from jsonschema import validate

from synto.indexer import generate_index_json, index_schema_path
from synto.models import RawNoteRecord, WikiArticleRecord
from synto.vault import write_note


def test_generate_index_json_is_schema_valid_and_deterministic(vault, config, db) -> None:
    write_note(config.wiki_dir / "Topic.md", {"title": "Topic"}, "Body")
    write_note(config.wiki_dir / "nested" / "Second.md", {"title": "Second"}, "Body")

    db.upsert_raw(
        RawNoteRecord(
            path="raw/a.md",
            content_hash="raw-h1",
            status="ingested",
            language="fr",
        )
    )
    db.upsert_concepts("raw/a.md", ["Topic", "Second"])
    db.upsert_aliases("Topic", ["Alias Topic"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/nested/Second.md",
            title="Second",
            sources=["raw/a.md"],
            content_hash="h2",
            status="published",
        )
    )

    first_path = generate_index_json(config, db)
    first_bytes = first_path.read_bytes()
    second_path = generate_index_json(config, db)
    second_bytes = second_path.read_bytes()

    payload = json.loads(first_bytes)
    schema = json.loads(index_schema_path().read_text(encoding="utf-8"))
    validate(payload, schema)

    assert first_bytes == second_bytes
    assert [article["path"] for article in payload["articles"]] == [
        "wiki/Topic.md",
        "wiki/nested/Second.md",
    ]
    assert payload["pack"]["capabilities"] == ["articles", "concepts"]
    assert payload["pack"]["language"] == ["fr"]
    assert payload["articles"][0]["aliases"] == ["Alias Topic"]
    assert payload["articles"][0]["entity_id"] == db.entity_id_for_name("Topic")
    assert payload["identity_log"] == []
    assert payload["papers"] == []
    sc = payload["source_concepts"]
    assert len(sc) == 1
    assert sc[0]["source_path"] == "raw/a.md"
    assert sc[0]["content_hash"] == "raw-h1"
    # concepts are now {name, entity_id} dicts; names must match (sorted)
    assert [c["name"] for c in sc[0]["concepts"]] == ["Second", "Topic"]
    assert all(isinstance(c["entity_id"], str) for c in sc[0]["concepts"])
    # entity_ids must round-trip back to their preferred labels
    assert all(
        db.preferred_label_for_entity(c["entity_id"]) == c["name"] for c in sc[0]["concepts"]
    )


def test_generate_index_json_expands_capabilities_when_segments_exist(vault, config, db) -> None:
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

    payload = json.loads(generate_index_json(config, db).read_text(encoding="utf-8"))

    assert payload["pack"]["capabilities"] == ["articles", "concepts", "lifecycle", "segments"]
    assert payload["sources"] == [{"id": "src-1", "title": "Source", "source_type": "unknown_text"}]
    assert payload["stats"]["source_segment_count"] == 1


def test_generate_index_json_includes_synthesis_and_drafts_in_stats(vault, config, db) -> None:
    write_note(config.drafts_dir / "Draft.md", {"title": "Draft"}, "Body")
    write_note(config.synthesis_dir / "Synth.md", {"title": "Synth"}, "Body")

    db.upsert_article(
        WikiArticleRecord(
            path="wiki/.drafts/Draft.md",
            title="Draft",
            sources=["raw/a.md"],
            content_hash="h1",
            status="draft",
        )
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/synthesis/Synth.md",
            title="Synth",
            sources=[],
            content_hash="h2",
            status="published",
            kind="synthesis",
            question_hash="qh-1",
        )
    )

    payload = json.loads(generate_index_json(config, db).read_text(encoding="utf-8"))

    assert payload["articles"] == []
    assert payload["stats"]["article_count"] == 1
    assert payload["stats"]["draft_count"] == 1
    assert payload["synthesis"] == [{"path": "wiki/synthesis/Synth.md", "title": "Synth"}]
