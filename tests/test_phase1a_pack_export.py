from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner
from jsonschema import validate

from synto.cli import cli
from synto.indexer import index_schema_path
from synto.models import RawNoteRecord, WikiArticleRecord
from synto.pack_export import _build_source_lookup, _export_source_refs, export_pack
from synto.readers import ArticleRef, PackReader
from synto.vault import write_note


def _build_v8_db(db_path: Path) -> None:
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
        INSERT INTO wiki_articles
            (path, title, sources, content_hash, created_at, updated_at, is_draft, kind, article_id)
            VALUES (
                'wiki/Test.md', 'Test Concept', '["raw/old.md"]', 'wh1',
                '2024-01-01T00:00:00', '2024-01-01T00:00:00', 0, 'concept',
                '01TESTULID0000000000000001'
            );
        """
    )
    conn.commit()
    conn.close()


def _write_wiki_toml(vault: Path) -> None:
    (vault / "synto.toml").write_text(
        """
[models]
fast = "test-fast"
heavy = "test-heavy"

[provider]
name = "ollama"
url = "http://localhost:11434"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_export_pack_round_trips_nested_articles_and_duplicate_basenames(vault, config, db) -> None:
    _write_wiki_toml(vault)
    first_path = config.wiki_dir / "nested" / "Topic.md"
    second_path = config.wiki_dir / "other" / "Topic.md"
    queries_path = config.queries_dir / "Saved.md"
    sources_path = config.sources_dir / "Source.md"
    synthesis_path = config.synthesis_dir / "Synth.md"
    draft_path = config.drafts_dir / "Draft.md"

    write_note(
        first_path,
        {"title": "Nested Topic", "sources": ["raw/a.md"]},
        "Nested body [[sources/source|Source]]",
    )
    write_note(second_path, {"title": "Other Topic"}, "Other body")
    write_note(queries_path, {"title": "Saved"}, "Saved body")
    write_note(
        sources_path,
        {"title": "Source", "source_file": "raw/a.md", "quality": "high"},
        "Source body",
    )
    write_note(synthesis_path, {"title": "Synth"}, "Synth body")
    write_note(draft_path, {"title": "Draft"}, "Draft body")
    write_note(
        config.sources_dir / "Uncited.md",
        {"title": "Uncited", "source_file": "raw/b.md", "quality": "low"},
        "Uncited body",
    )
    (config.raw_dir / "a.md").write_text("alpha", encoding="utf-8")
    (config.raw_dir / "b.md").write_text("beta", encoding="utf-8")
    (config.raw_dir / ".DS_Store").write_text("ignored", encoding="utf-8")
    (config.raw_dir / "processed").mkdir()
    (config.raw_dir / "processed" / "skip.md").write_text("skip", encoding="utf-8")
    (config.vault / "_resources").mkdir()
    ((config.vault / "_resources") / "img.png").write_text("img", encoding="utf-8")
    write_note(
        config.sources_dir / "Media.md",
        {"title": "Media", "source_file": "raw/media.md", "quality": "medium"},
        "Media note\n\n## Media\n- ![[./_resources/img.png]]",
    )
    (config.raw_dir / "media.md").write_text("gamma\n\n![[./_resources/img.png]]", encoding="utf-8")

    db.upsert_concepts("raw/a.md", ["Nested Topic", "Other Topic"])
    db.upsert_concepts("raw/b.md", ["Uncited Topic"])
    db.upsert_concepts("raw/media.md", ["Media Topic"])
    db.upsert_aliases("Nested Topic", ["NT"])
    db.upsert_raw(
        RawNoteRecord(path="raw/a.md", content_hash="raw-a", status="ingested", language="en")
    )
    db.upsert_raw(
        RawNoteRecord(path="raw/b.md", content_hash="raw-b", status="ingested", language="en")
    )
    db.upsert_raw(
        RawNoteRecord(
            path="raw/media.md",
            content_hash="raw-media",
            status="ingested",
            language="en",
        )
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/nested/Topic.md",
            title="Nested Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/other/Topic.md",
            title="Other Topic",
            sources=["raw/a.md"],
            content_hash="h2",
            status="published",
        )
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/synthesis/Synth.md",
            title="Synth",
            sources=[],
            content_hash="h3",
            status="published",
            kind="synthesis",
            question_hash="qh-1",
        )
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/.drafts/Draft.md",
            title="Draft",
            sources=["raw/a.md"],
            content_hash="h4",
            status="draft",
        )
    )

    out_dir = vault / ".knowledge" / "sample"
    result = export_pack(config, target="agents", out=out_dir)

    assert result.n_articles == 2
    assert (out_dir / "pack.toml").exists()
    assert (out_dir / "agent" / "manifest.json").exists()
    assert (out_dir / "agent" / "concepts.json").exists()
    assert (out_dir / "agent" / "sources.json").exists()
    assert (out_dir / "agent" / "routes.json").exists()
    assert (out_dir / "index" / "INDEX.json").exists()
    assert (out_dir / "articles" / "nested" / "Topic.md").exists()
    assert (out_dir / "articles" / "other" / "Topic.md").exists()
    assert (out_dir / "drafts" / "Draft.md").exists()
    assert (out_dir / "queries" / "Saved.md").exists()
    assert (out_dir / "sources" / "Source.md").exists()
    assert (out_dir / "sources" / "Uncited.md").exists()
    assert (out_dir / "sources" / "Media.md").exists()
    assert (out_dir / "synthesis" / "Synth.md").exists()
    assert (out_dir / "raw" / "a.md").exists()
    assert (out_dir / "raw" / "b.md").exists()
    assert (out_dir / "raw" / "media.md").exists()
    assert not (out_dir / "raw" / ".DS_Store").exists()
    assert not (out_dir / "raw" / "processed" / "skip.md").exists()
    assert (out_dir / "_resources" / "img.png").exists()

    reader = PackReader(out_dir)
    assert [ref.path for ref in reader.list_articles()] == [
        "articles/nested/Topic.md",
        "articles/other/Topic.md",
    ]
    assert "Nested body" in reader.read_article("Nested Topic").body
    assert reader.read_article("Other Topic").body == "Other body"
    assert reader.find_concept("NT") is not None

    schema = json.loads(index_schema_path().read_text(encoding="utf-8"))
    payload = json.loads((out_dir / "index" / "INDEX.json").read_text(encoding="utf-8"))
    validate(payload, schema)
    assert payload["articles"][0]["aliases"] == ["NT"]
    assert payload["papers"] == []
    assert sorted(payload["sources"], key=lambda item: item["id"]) == [
        {"id": "raw/a.md", "title": "Source", "source_type": "source_summary"},
        {"id": "raw/b.md", "title": "Uncited", "source_type": "source_summary"},
        {"id": "raw/media.md", "title": "Media", "source_type": "source_summary"},
    ]
    routes = json.loads((out_dir / "agent" / "routes.json").read_text(encoding="utf-8"))
    assert routes["schema_version"] == 1
    assert any(
        route["kind"] == "article" and route["surface"] == "Nested Topic"
        for route in routes["routes"]
    )
    assert any(
        route["kind"] == "source" and route["surface"] == "Source" for route in routes["routes"]
    )
    assert any(
        route["kind"] == "raw" and route["surface"] == "raw/a.md" for route in routes["routes"]
    )
    sources_payload = json.loads((out_dir / "agent" / "sources.json").read_text(encoding="utf-8"))
    source_entry = next(
        source for source in sources_payload["sources"] if source["title"] == "Source"
    )
    assert source_entry["path"] == "sources/Source.md"
    assert source_entry["raw_path"] == "raw/a.md"
    assert source_entry["referenced_by_articles"] == ["articles/nested/Topic.md"]
    assert source_entry["raw_included"] is True
    exported_article = (out_dir / "articles" / "nested" / "Topic.md").read_text(encoding="utf-8")
    assert "[[sources/Source|Source]]" in exported_article
    exported_source = (out_dir / "sources" / "Media.md").read_text(encoding="utf-8")
    assert "![[../_resources/img.png]]" in exported_source
    exported_raw = (out_dir / "raw" / "media.md").read_text(encoding="utf-8")
    assert "![[../_resources/img.png]]" in exported_raw
    manifest = json.loads((out_dir / "agent" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_included"] is True
    assert manifest["source_count"] == 3
    assert result.n_assets == 0


def test_export_pack_is_deterministic_and_preserves_user_agents_text(vault, config, db) -> None:
    _write_wiki_toml(vault)
    write_note(config.wiki_dir / "Topic.md", {"title": "Topic"}, "Body")
    db.upsert_concepts("raw/a.md", ["Topic"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )

    out_dir = vault / ".knowledge" / "sample"
    export_pack(config, target="agents", out=out_dir)

    agents_path = out_dir / "AGENTS.md"
    original_agents = agents_path.read_text(encoding="utf-8")
    original_index = (out_dir / "index" / "INDEX.json").read_bytes()
    original_manifest = (out_dir / "agent" / "manifest.json").read_bytes()
    agents_path.write_text(
        "User intro\n\n" + original_agents + "\nUser outro\n",
        encoding="utf-8",
    )

    export_pack(config, target="agents", out=out_dir)

    assert (out_dir / "index" / "INDEX.json").read_bytes() == original_index
    assert (out_dir / "agent" / "manifest.json").read_bytes() == original_manifest
    updated_agents = agents_path.read_text(encoding="utf-8")
    assert "User intro" in updated_agents
    assert "User outro" in updated_agents
    assert "synto:generated:start" in updated_agents
    assert "olw:generated:start" not in updated_agents
    assert (out_dir / "CLAUDE.md").read_text(encoding="utf-8") == updated_agents

    write_note(config.wiki_dir / "Topic.md", {"title": "Topic"}, "Updated body")
    export_pack(config, target="agents", out=out_dir)
    # manifest is stable (no structural changes)
    assert (out_dir / "agent" / "manifest.json").read_bytes() == original_manifest
    # INDEX.json now reflects the updated article summary
    updated_index = json.loads((out_dir / "index" / "INDEX.json").read_text(encoding="utf-8"))
    assert updated_index["articles"][0]["summary"] == "Updated body"


def test_pack_export_cli_invokes_agents_export(vault, config, db) -> None:
    _write_wiki_toml(vault)
    write_note(config.wiki_dir / "Topic.md", {"title": "Topic"}, "Body")
    db.upsert_concepts("raw/a.md", ["Topic"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )

    out_dir = vault / ".knowledge" / "cli-pack"
    result = CliRunner().invoke(
        cli,
        [
            "pack",
            "export",
            "--vault",
            str(vault),
            "--target",
            "agents",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Exported 1 articles" in result.output
    payload = json.loads((out_dir / "agent" / "manifest.json").read_text(encoding="utf-8"))
    assert payload["pack"]["capabilities"] == ["articles", "concepts"]


def test_export_pack_rejects_symlinked_optional_tree(vault, config, db) -> None:
    _write_wiki_toml(vault)
    write_note(config.wiki_dir / "Topic.md", {"title": "Topic"}, "Body")
    db.upsert_concepts("raw/a.md", ["Topic"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )
    outside = vault.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    config.queries_dir.mkdir(parents=True, exist_ok=True)
    (config.queries_dir / "link.md").symlink_to(outside)

    out_dir = vault / ".knowledge" / "sample"
    try:
        export_pack(config, target="agents", out=out_dir)
    except ValueError as exc:
        assert "symlinked" in str(exc)
    else:
        raise AssertionError("expected symlink export rejection")


def test_export_pack_works_with_readonly_v8_db(tmp_path: Path) -> None:
    from synto.config import Config

    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".synto").mkdir()
    _write_wiki_toml(tmp_path)
    write_note(tmp_path / "wiki" / "Test.md", {"title": "Test Concept"}, "Body")
    db_path = tmp_path / ".synto" / "state.db"
    _build_v8_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE wiki_articles SET article_id = ? WHERE path = ?",
        ("01TESTULID0000000000000001", "wiki/Test.md"),
    )
    conn.commit()
    conn.close()

    out_dir = tmp_path / ".knowledge" / "v8"
    result = export_pack(Config(vault=tmp_path), target="agents", out=out_dir)

    assert result.n_articles == 1
    assert (result.out_dir / "index" / "INDEX.json").exists()


# ── new pack export features ──────────────────────────────────────────────────


def _setup_two_article_vault(vault, config, db):
    """Create two cross-language articles for testing."""
    _write_wiki_toml(vault)
    write_note(
        config.wiki_dir / "Astrology.md",
        {"title": "Astrology", "confidence": 0.4, "tags": ["astrology"]},
        "## Overview\n\nAstrology is a pseudoscientific belief system.",
    )
    write_note(
        config.wiki_dir / "Астрология.md",
        {"title": "Астрология", "confidence": 0.35, "tags": []},
        "Астрология — псевдонаучная система предсказаний.",
    )
    db.upsert_concepts("raw/a.md", ["Astrology", "Астрология"])
    db.upsert_aliases("Astrology", ["астрология", "система/planets"])  # includes noise alias
    for title, path in [("Astrology", "wiki/Astrology.md"), ("Астрология", "wiki/Астрология.md")]:
        db.upsert_article(
            WikiArticleRecord(
                path=path,
                title=title,
                sources=["raw/a.md"],
                content_hash=f"h-{title}",
                status="published",
            )
        )


def test_pack_export_summaries_populated(vault, config, db):
    _setup_two_article_vault(vault, config, db)
    out_dir = vault / ".knowledge" / "test"
    export_pack(config, target="agents", out=out_dir)
    payload = json.loads((out_dir / "index" / "INDEX.json").read_text(encoding="utf-8"))
    summaries = {a["name"]: a["summary"] for a in payload["articles"]}
    assert summaries["Astrology"] == "Astrology is a pseudoscientific belief system."
    assert summaries["Астрология"] == "Астрология — псевдонаучная система предсказаний."


def test_pack_export_numeric_confidence(vault, config, db):
    _setup_two_article_vault(vault, config, db)
    out_dir = vault / ".knowledge" / "test"
    export_pack(config, target="agents", out=out_dir)
    payload = json.loads((out_dir / "index" / "INDEX.json").read_text(encoding="utf-8"))
    conf_map = {a["name"]: a["confidence"] for a in payload["articles"]}
    assert isinstance(conf_map["Astrology"], float)
    assert conf_map["Astrology"] == pytest.approx(0.4)
    assert isinstance(conf_map["Астрология"], float)
    assert conf_map["Астрология"] == pytest.approx(0.35)


def test_pack_export_noise_alias_filtered(vault, config, db):
    _setup_two_article_vault(vault, config, db)
    out_dir = vault / ".knowledge" / "test"
    export_pack(config, target="agents", out=out_dir)
    concepts = json.loads((out_dir / "agent" / "concepts.json").read_text(encoding="utf-8"))
    astrology_entry = next(c for c in concepts["concepts"] if c["name"] == "Astrology")
    # "система/planets" has "/" → must be filtered out
    assert all("/" not in a for a in astrology_entry["aliases"])
    # "астрология" (1 word) should be kept
    assert "астрология" in astrology_entry["aliases"]


def test_pack_export_cross_language_related_names(vault, config, db):
    _setup_two_article_vault(vault, config, db)
    out_dir = vault / ".knowledge" / "test"
    export_pack(config, target="agents", out=out_dir)
    concepts = json.loads((out_dir / "agent" / "concepts.json").read_text(encoding="utf-8"))
    astrology = next(c for c in concepts["concepts"] if c["name"] == "Astrology")
    astrologia_ru = next(c for c in concepts["concepts"] if c["name"] == "Астрология")
    # "астрология" alias of Astrology matches canonical name "Астрология"
    assert "Астрология" in astrology["related_names"]
    assert "Astrology" in astrologia_ru["related_names"]


def test_pack_export_frequent_alias_filtered(vault, config, db):
    """Aliases claimed by 2+ concepts are removed from export regardless of language."""
    _write_wiki_toml(vault)
    write_note(config.wiki_dir / "Alpha.md", {"title": "Alpha"}, "Alpha body.")
    write_note(config.wiki_dir / "Beta.md", {"title": "Beta"}, "Beta body.")
    for title, path in [("Alpha", "wiki/Alpha.md"), ("Beta", "wiki/Beta.md")]:
        db.upsert_article(
            WikiArticleRecord(
                path=path,
                title=title,
                sources=["raw/a.md"],
                content_hash=f"h-{title}",
                status="published",
            )
        )
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    # "shared-term" appears in both → ambiguous → must be filtered
    db.upsert_aliases("Alpha", ["shared-term", "unique-alpha"])
    db.upsert_aliases("Beta", ["shared-term", "unique-beta"])

    out_dir = vault / ".knowledge" / "test"
    export_pack(config, target="agents", out=out_dir)
    concepts = json.loads((out_dir / "agent" / "concepts.json").read_text(encoding="utf-8"))
    alpha = next(c for c in concepts["concepts"] if c["name"] == "Alpha")
    beta = next(c for c in concepts["concepts"] if c["name"] == "Beta")

    assert "shared-term" not in alpha["aliases"]
    assert "shared-term" not in beta["aliases"]
    assert "unique-alpha" in alpha["aliases"]
    assert "unique-beta" in beta["aliases"]


def test_pack_export_first_word_cross_language_related_names(vault, config, db):
    """First word of a multi-word alias matching a canonical name creates a cross-link."""
    _write_wiki_toml(vault)
    write_note(config.wiki_dir / "Precession.md", {"title": "Precession"}, "Precession body.")
    write_note(
        config.wiki_dir / "Prec-en.md",
        {"title": "Precession of the equinoxes"},
        "The equinoxes precess.",
    )
    for title, path in [
        ("Precession", "wiki/Precession.md"),
        ("Precession of the equinoxes", "wiki/Prec-en.md"),
    ]:
        db.upsert_article(
            WikiArticleRecord(
                path=path,
                title=title,
                sources=["raw/a.md"],
                content_hash=f"h-{title}",
                status="published",
            )
        )
    db.upsert_concepts("raw/a.md", ["Precession"])
    db.upsert_concepts("raw/b.md", ["Precession of the equinoxes"])
    # "precession of equinoxes method" — first word "precession" matches canonical "Precession"
    db.upsert_aliases("Precession of the equinoxes", ["precession of equinoxes method"])

    out_dir = vault / ".knowledge" / "test"
    export_pack(config, target="agents", out=out_dir)
    concepts = json.loads((out_dir / "agent" / "concepts.json").read_text(encoding="utf-8"))
    prec = next(c for c in concepts["concepts"] if c["name"] == "Precession")
    prec_en = next(c for c in concepts["concepts"] if c["name"] == "Precession of the equinoxes")

    assert "Precession of the equinoxes" in prec["related_names"]
    assert "Precession" in prec_en["related_names"]


def test_pack_export_article_path_in_concepts(vault, config, db):
    _setup_two_article_vault(vault, config, db)
    out_dir = vault / ".knowledge" / "test"
    export_pack(config, target="agents", out=out_dir)
    concepts = json.loads((out_dir / "agent" / "concepts.json").read_text(encoding="utf-8"))
    for c in concepts["concepts"]:
        assert "article_path" in c
        assert c["article_path"] is not None
        assert c["article_path"].startswith("articles/")


def test_pack_export_agents_md_has_workflow_and_languages(vault, config, db):
    _write_wiki_toml(vault)
    # Add notes with two different languages so languages get detected
    from synto.models import RawNoteRecord

    write_note(config.wiki_dir / "Topic.md", {"title": "Topic"}, "First paragraph.")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )
    db.upsert_raw(
        RawNoteRecord(path="raw/a.md", content_hash="h", status="ingested", language="en")
    )
    db.upsert_raw(
        RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested", language="ru")
    )

    out_dir = vault / ".knowledge" / "test"
    export_pack(config, target="agents", out=out_dir)
    agents_text = (out_dir / "AGENTS.md").read_text(encoding="utf-8")
    claude_text = (out_dir / "CLAUDE.md").read_text(encoding="utf-8")
    assert agents_text == claude_text
    assert "Concept lookup:" in agents_text
    assert "article_path" in agents_text
    assert "related_names" in agents_text
    assert "Languages:" in agents_text
    assert "Raw notes:" in agents_text
    assert "Sources:" in agents_text
    assert "Raw notes are included in `raw/` by default." in agents_text
    assert "## Contents" in agents_text
    assert "| Article | Confidence | Summary |" in agents_text
    assert "| high |" in agents_text or "| medium |" in agents_text or "| low |" in agents_text


# ── _export_source_refs edge cases ────────────────────────────────────────────


def test_export_source_refs_missing_source_file(vault, config, db):
    """Source pages without source_file metadata are silently skipped."""
    config.sources_dir.mkdir(parents=True, exist_ok=True)
    write_note(config.sources_dir / "Orphan.md", {"title": "Orphan"}, "No source_file here.")
    refs = _export_source_refs(config, db, [])
    assert not any(r["title"] == "Orphan" for r in refs)


def test_export_source_refs_sources_as_string(vault, config, db):
    """Article sources: given as a string (not list) is coerced to a list for cross-ref."""
    config.sources_dir.mkdir(parents=True, exist_ok=True)
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    write_note(
        config.sources_dir / "Note Source.md",
        {"title": "Note Source", "source_file": "raw/note.md"},
        "body",
    )
    # Article references the source as a bare string, not a list
    write_note(
        config.wiki_dir / "Article.md",
        {"title": "Article", "sources": "raw/note.md"},
        "body",
    )
    article_refs = [ArticleRef(id="1", name="Article", path="wiki/Article.md")]
    refs = _export_source_refs(config, db, article_refs)
    match = next((r for r in refs if r["title"] == "Note Source"), None)
    assert match is not None
    assert match["referenced_by_articles"]  # cross-reference was built


def test_export_source_refs_normalizes_legacy_windows_source_file(vault, config, db):
    """Legacy source_file backslashes still match DB and article metadata."""
    config.sources_dir.mkdir(parents=True, exist_ok=True)
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    write_note(
        config.sources_dir / "Legacy Source.md",
        {"title": "Legacy Source", "source_file": r"raw\note.md"},
        "## Concepts\n- [[Fallback Concept]]",
    )
    write_note(
        config.wiki_dir / "Article.md",
        {"title": "Article", "sources": ["raw/note.md"]},
        "body",
    )
    db.upsert_concepts("raw/note.md", ["Portable Concept"])
    db.upsert_raw(RawNoteRecord(path="raw/note.md", content_hash="hash-note", status="ingested"))

    refs = _export_source_refs(
        config,
        db,
        [ArticleRef(id="1", name="Article", path="wiki/Article.md")],
    )

    assert len(refs) == 1
    assert refs[0]["id"] == "raw/note.md"
    assert refs[0]["raw_path"] == "raw/note.md"
    assert refs[0]["concepts"] == ["Portable Concept"]
    assert refs[0]["referenced_by_articles"] == ["articles/Article.md"]
    assert refs[0]["raw_content_hash"] == "hash-note"


def test_build_source_lookup_normalizes_legacy_windows_raw_path():
    source = {
        "title": "Legacy Source",
        "path": "sources/Legacy Source.md",
        "raw_path": r"raw\note.md",
    }

    lookup = _build_source_lookup([source])

    assert lookup["note"] is source
    assert lookup["legacy source"] is source
    assert lookup["sources/legacy source.md"] is source


def test_export_source_refs_malformed_yaml(vault, config, db):
    """Source page with unparseable YAML frontmatter is skipped without crashing."""
    config.sources_dir.mkdir(parents=True, exist_ok=True)
    (config.sources_dir / "Broken.md").write_text("---\ntitle: [\n---\nbody", encoding="utf-8")
    refs = _export_source_refs(config, db, [])
    assert not any(r.get("title") == "Broken" for r in refs)


def test_export_source_refs_invalid_quality_type(vault, config, db):
    """quality metadata given as an int is passed through as-is in the result dict."""
    config.sources_dir.mkdir(parents=True, exist_ok=True)
    write_note(
        config.sources_dir / "Typed.md",
        {"title": "Typed", "source_file": "raw/x.md", "quality": 42},
        "body",
    )
    refs = _export_source_refs(config, db, [])
    match = next((r for r in refs if r["title"] == "Typed"), None)
    assert match is not None
    assert match["quality"] == 42


def test_routes_payload_emits_per_entity_routes_for_homonyms(db):
    """Two homonyms must export two distinct article routes, not collapse to one (feature 45).

    The route dedup keys on (kind, surface, path), so each entity's qualified label routes to
    its own article path — homonyms do not first-wins-overwrite each other.
    """
    from synto.pack_export import _routes_payload

    db.upsert_concepts("raw/a.md", ["Mercury (planet)"])
    db.upsert_concepts("raw/b.md", ["Mercury (element)"])
    for title, src in [("Mercury (planet)", "raw/a.md"), ("Mercury (element)", "raw/b.md")]:
        db.upsert_article(
            WikiArticleRecord(
                path=f"wiki/{title}.md",
                title=title,
                sources=[src],
                content_hash="h",
                status="published",
                entity_id=db.entity_id_for_name(title),
            )
        )

    routes = _routes_payload(db, [])["routes"]
    article_routes = {
        r["canonical"]: r["path"]
        for r in routes
        if r["kind"] == "article" and r["surface"] == r["canonical"]
    }
    assert article_routes["Mercury (planet)"] == "articles/Mercury (planet).md"
    assert article_routes["Mercury (element)"] == "articles/Mercury (element).md"
    assert article_routes["Mercury (planet)"] != article_routes["Mercury (element)"]
