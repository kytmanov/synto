"""Tests for the MCP server expansion (feature 23, single-vault scope).

Drives the tool *handlers* directly, not the FastMCP STDIO loop — the
handlers are closures inside `run_server`, so we re-use the same
construction path by extracting them via a small helper that mirrors
`run_server` minus the `server.run()` call.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from synto.config import Config, McpConfig
from synto.models import WikiArticleRecord
from synto.readers import ArticleFilter, VaultReader
from synto.state import StateDB
from synto.vault import atomic_write


def _write_article(
    path: Path,
    body: str,
    *,
    status: str | None = "published",
    confidence: float | None = None,
    single_source: bool | None = None,
    source_count: int | None = None,
    visibility: str = "public",
    tags: list[str] | None = None,
    lineage: list[dict] | None = None,
    aliases: list[str] | None = None,
) -> None:
    lines = ["---", f"title: {path.stem}", f"visibility: {visibility}"]
    if status is not None:
        lines.append(f"status: {status}")
    if confidence is not None:
        lines.append(f"confidence: {confidence}")
    if single_source is not None:
        lines.append(f"single_source: {'true' if single_source else 'false'}")
    if source_count is not None:
        lines.append(f"source_count: {source_count}")
    if tags is not None:
        lines.append(f"tags: [{', '.join(tags)}]")
    if lineage is not None:
        import json as _json

        lines.append(f"lineage: {_json.dumps(lineage)}")
    if aliases is not None:
        import json as _json

        lines.append(f"aliases: {_json.dumps(aliases)}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    atomic_write(path, "\n".join(lines) + "\n")


def _seed_article(
    db: StateDB,
    rel_path: str,
    title: str,
    *,
    kind: str = "concept",
) -> None:
    db.upsert_article(
        WikiArticleRecord(
            path=rel_path,
            title=title,
            sources=["raw/source.md"],
            content_hash=f"hash-{title}",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="published",
            kind=kind,
        )
    )


# ── handler harness ────────────────────────────────────────────────────────


def _build_tools(vault: Path, mcp_config: McpConfig | None = None):
    """Return (handlers-dict, config) using the exposed build_tool_handlers helper."""
    from synto.serve import build_tool_handlers

    config = Config(vault=vault, mcp=mcp_config or McpConfig(audit=False))
    reader = VaultReader(vault)
    audit_db = StateDB(config.state_db_path) if config.mcp.audit else None
    handlers = build_tool_handlers(reader, config, audit_db, vault_key="test-vault")
    return handlers, config


def _tools(vault: Path, mcp_config: McpConfig | None = None):
    handlers, _ = _build_tools(vault, mcp_config)
    return handlers


# ── tests ──────────────────────────────────────────────────────────────────


def test_list_articles_hides_drafts_by_default(vault, db):
    wiki = vault / "wiki"
    _write_article(wiki / "Pub.md", "published", status="published")
    _write_article(wiki / "Draft.md", "wip", status="draft")
    _seed_article(db, "wiki/Pub.md", "Pub")
    _seed_article(db, "wiki/Draft.md", "Draft")

    handlers = _tools(vault)
    result = handlers["list_articles"]()
    names = sorted(r["name"] for r in result)
    assert names == ["Pub"]


def test_list_articles_min_status_opt_in_to_drafts(vault, db):
    wiki = vault / "wiki"
    _write_article(wiki / "Pub.md", "published", status="published")
    _write_article(wiki / "Draft.md", "wip", status="draft")
    _seed_article(db, "wiki/Pub.md", "Pub")
    _seed_article(db, "wiki/Draft.md", "Draft")

    handlers = _tools(vault)
    result = handlers["list_articles"](min_status="draft")
    names = sorted(r["name"] for r in result)
    assert names == ["Draft", "Pub"]


def test_list_articles_legacy_no_status_still_visible(vault, db):
    wiki = vault / "wiki"
    _write_article(wiki / "Legacy.md", "older", status=None)
    _seed_article(db, "wiki/Legacy.md", "Legacy")

    handlers = _tools(vault)
    result = handlers["list_articles"]()
    assert [r["name"] for r in result] == ["Legacy"]
    assert result[0]["status"] is None


def test_list_articles_exclude_single_source(vault, db):
    wiki = vault / "wiki"
    _write_article(wiki / "Multi.md", "ok", single_source=False, source_count=3)
    _write_article(wiki / "Solo.md", "lonely", single_source=True, source_count=1)
    _seed_article(db, "wiki/Multi.md", "Multi")
    _seed_article(db, "wiki/Solo.md", "Solo")

    handlers = _tools(vault)
    result = handlers["list_articles"](exclude_single_source=True)
    assert [r["name"] for r in result] == ["Multi"]


def test_list_articles_surfaces_synthesis(vault, db):
    wiki = vault / "wiki"
    synthesis_dir = wiki / "synthesis"
    synthesis_dir.mkdir(parents=True, exist_ok=True)
    _write_article(wiki / "Concept.md", "c body")
    _write_article(synthesis_dir / "Synth.md", "s body")
    _seed_article(db, "wiki/Concept.md", "Concept", kind="concept")
    _seed_article(db, "wiki/synthesis/Synth.md", "Synth", kind="synthesis")

    handlers = _tools(vault)
    result = handlers["list_articles"]()
    by_name = {r["name"]: r for r in result}
    assert by_name["Concept"]["kind"] == "concept"
    assert by_name["Synth"]["kind"] == "synthesis"

    only_synth = handlers["list_articles"](kind="synthesis")
    assert [r["name"] for r in only_synth] == ["Synth"]


def test_search_articles_returns_matches_with_score(vault, db):
    wiki = vault / "wiki"
    _write_article(wiki / "Persistent Communication.md", "messaging concept")
    _write_article(wiki / "Other.md", "unrelated topic")
    _seed_article(db, "wiki/Persistent Communication.md", "Persistent Communication")
    _seed_article(db, "wiki/Other.md", "Other")

    handlers = _tools(vault)
    result = handlers["search_articles"]("persistent")
    assert len(result) == 1
    assert result[0]["name"] == "Persistent Communication"
    assert result[0]["score"] >= 1


def test_search_articles_empty_query_returns_empty(vault, db):
    handlers = _tools(vault)
    assert handlers["search_articles"]("") == []


def test_search_articles_min_status_default_hides_drafts(vault, db):
    wiki = vault / "wiki"
    _write_article(wiki / "Draft Topic.md", "draft body about topic", status="draft")
    _seed_article(db, "wiki/Draft Topic.md", "Draft Topic")

    handlers = _tools(vault)
    assert handlers["search_articles"]("topic") == []
    opted_in = handlers["search_articles"]("topic", min_status="draft")
    assert [r["name"] for r in opted_in] == ["Draft Topic"]


def test_get_concept_returns_definition_and_body(vault, db):
    wiki = vault / "wiki"
    _write_article(wiki / "MOM.md", "Message-oriented middleware decouples senders and receivers.")
    _seed_article(db, "wiki/MOM.md", "MOM")
    db.upsert_concepts("raw/source.md", ["MOM"])
    db.upsert_aliases("MOM", ["message-oriented middleware"])

    handlers = _tools(vault)
    result = handlers["get_concept"]("message-oriented middleware")
    assert result["name"] == "MOM"
    assert "Message-oriented middleware" in result["definition"]
    assert "Message-oriented middleware" in result["body"]
    assert "message-oriented middleware" in result["aliases"]


def test_get_concept_unknown_returns_empty_shape(vault, db):
    handlers = _tools(vault)
    result = handlers["get_concept"]("nope")
    assert result == {
        "name": "nope",
        "aliases": [],
        "canonical_article_id": None,
        "definition": "",
        "body": "",
        "frontmatter": {},
    }


def test_list_sources_returns_registered(vault, db):
    from types import SimpleNamespace

    db.upsert_source_document(
        SimpleNamespace(
            id="raw/source-a.md",
            title="Source A",
            source_type="markdown",
            origin_uri="file://a",
            imported_at=datetime.now().isoformat(),
            raw_hash="raw-hash",
            normalized_hash="hash-a",
            extractor_version="v1",
            license=None,
            redistribution="unknown",
            metadata={},
            bibliographic_metadata=None,
        )
    )

    handlers = _tools(vault)
    result = handlers["list_sources"]()
    assert any(s["id"] == "raw/source-a.md" for s in result)


def test_trace_lineage_returns_frontmatter_lineage(vault, db):
    wiki = vault / "wiki"
    lineage = [{"source": "raw/book.pdf", "section": "§6.2"}]
    _write_article(wiki / "Fallacies.md", "do not assume the network is reliable", lineage=lineage)
    _seed_article(db, "wiki/Fallacies.md", "Fallacies")

    handlers = _tools(vault)
    result = handlers["trace_lineage"]("Fallacies")
    assert result["article"] == "Fallacies"
    assert result["lineage"] == lineage


def test_answer_question_returns_index_missing_for_empty_vault(vault, db, monkeypatch):
    handlers = _tools(vault)

    class _Stub:
        pass

    from synto import client_factory

    monkeypatch.setattr(client_factory, "build_client", lambda *_a, **_k: _Stub())

    result = handlers["answer_question"]("anything")
    assert result["index_found"] is False
    assert result["selected_pages"] == []


def test_apply_filter_min_status_unknown_falls_through(vault, db):
    wiki = vault / "wiki"
    _write_article(wiki / "Pub.md", "published")
    _seed_article(db, "wiki/Pub.md", "Pub")

    reader = VaultReader(vault)
    refs = reader.list_articles(filter=ArticleFilter(min_status="bogus"))
    assert [r.name for r in refs] == ["Pub"]


def test_search_articles_matches_alias(vault, db):
    """Alias-only query resolves the article (dobryakov ROI-equation case)."""
    wiki = vault / "wiki"
    _write_article(
        wiki / "Agentic ROI framework.md",
        "Framework body without the alias surface.",
        aliases=["ROI equation", "cost dynamics"],
    )
    _seed_article(db, "wiki/Agentic ROI framework.md", "Agentic ROI framework")

    handlers = _tools(vault)
    result = handlers["search_articles"]("ROI equation")
    assert len(result) == 1
    assert result[0]["name"] == "Agentic ROI framework"
    assert result[0]["score"] >= 1


def test_search_articles_alias_misses_when_aliases_absent(vault, db):
    """No frontmatter aliases → alias-only query yields no result."""
    wiki = vault / "wiki"
    _write_article(wiki / "Plain.md", "totally unrelated body about cooking")
    _seed_article(db, "wiki/Plain.md", "Plain")

    handlers = _tools(vault)
    assert handlers["search_articles"]("ROI equation") == []


def test_search_articles_dedupes_when_term_in_name_and_alias(vault, db):
    """A query that hits both the name AND an alias returns the article once."""
    wiki = vault / "wiki"
    _write_article(
        wiki / "Tokens.md",
        "summary text",
        aliases=["tokens-as-capital", "tokens economy"],
    )
    _seed_article(db, "wiki/Tokens.md", "Tokens")

    handlers = _tools(vault)
    result = handlers["search_articles"]("tokens")
    assert len(result) == 1
    # "Tokens" name = 1 hit; aliases contain "tokens" twice → score >= 3 total.
    assert result[0]["score"] >= 3


def test_hash_args_passes_through_scalars():
    """Bools/ints/floats keep their value; strings still hash."""
    from synto.serve import _hash_args

    out = _hash_args({"flag": False, "n": 10, "ratio": 0.5, "name": "secret"})
    assert out["flag"] is False
    assert out["n"] == 10
    assert out["ratio"] == 0.5
    assert isinstance(out["name"], str)
    assert len(out["name"]) == 8 and out["name"] != "secret"
