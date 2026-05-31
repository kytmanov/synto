from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from synto.cli import cli
from synto.models import WikiArticleRecord
from synto.readers import ArticleNotFound
from synto.serve import _audit, _filter_visible_refs, _read_visible_article
from synto.state import StateDB
from synto.vault import atomic_write


def _write_article(
    path: Path,
    body: str,
    *,
    visibility: str | None = None,
    tags: list[str] | None = None,
):
    tags_line = f"tags: {json.dumps(tags or [])}\n"
    visibility_line = f"visibility: {visibility}\n" if visibility is not None else ""
    atomic_write(
        path,
        f"---\ntitle: {path.stem}\n{tags_line}{visibility_line}---\n\n{body}\n",
    )


def _insert_article(db: StateDB, rel_path: str, title: str):
    db.upsert_article(
        WikiArticleRecord(
            path=rel_path,
            title=title,
            sources=["raw/source.md"],
            content_hash=f"hash-{title}",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="published",
        )
    )


def test_serve_help_works():
    result = CliRunner().invoke(cli, ["serve", "--help"])

    assert result.exit_code == 0
    assert "stdio" in result.output
    assert "list_articles" in result.output


def test_run_server_prints_startup_banner_to_stderr_and_keeps_stdout_clean(
    vault, monkeypatch, capsys
) -> None:
    """run_server must announce itself on stderr (issue #30: silent hang) while
    keeping stdout pristine for JSON-RPC. A stray log/print on stdout corrupts the
    protocol stream and breaks a connected MCP client."""
    pytest.importorskip("mcp.server.fastmcp")
    import logging
    from unittest.mock import MagicMock

    from synto.serve import run_server

    # Stub FastMCP so server.run() returns instead of blocking on stdin.
    monkeypatch.setattr("mcp.server.fastmcp.FastMCP", MagicMock())

    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    # Simulate the CLI group callback that routes logging at stdout.
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(logging.StreamHandler(sys.stdout))
    try:
        run_server(vault)

        # The redirect must leave no logging handler pointed at stdout.
        assert root.handlers
        assert all(getattr(h, "stream", None) is not sys.stdout for h in root.handlers)
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)

    captured = capsys.readouterr()
    assert "Waiting for an MCP client" in captured.err
    assert captured.out == ""


def test_mcp_sdk_fastmcp_api_is_compatible_when_installed():
    pytest.importorskip("mcp.server.fastmcp")
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("test")

    assert hasattr(server, "tool")
    assert hasattr(server, "run")


def test_visibility_filter_excludes_private_articles(vault, db) -> None:
    wiki_path = vault / "wiki"
    public_path = wiki_path / "Public.md"
    private_path = wiki_path / "Private.md"
    _write_article(public_path, "public body", visibility="public")
    _write_article(private_path, "private body", visibility="private")
    _insert_article(db, "wiki/Public.md", "Public")
    _insert_article(db, "wiki/Private.md", "Private")

    from synto.config import Config
    from synto.readers import VaultReader

    config = Config(vault=vault)
    reader = VaultReader(vault)
    visible = _filter_visible_refs(reader, reader.list_articles(), config.mcp)

    assert [ref.name for ref in visible] == ["Public"]


def test_visibility_filter_default_private_mode(vault, db) -> None:
    wiki_path = vault / "wiki"
    explicit_public = wiki_path / "Explicit Public.md"
    implicit_private = wiki_path / "Implicit Private.md"
    _write_article(explicit_public, "public body", visibility="public")
    _write_article(implicit_private, "hidden body")
    _insert_article(db, "wiki/Explicit Public.md", "Explicit Public")
    _insert_article(db, "wiki/Implicit Private.md", "Implicit Private")

    from synto.config import Config, McpConfig
    from synto.readers import VaultReader

    config = Config(vault=vault, mcp=McpConfig(default_visibility="private"))
    reader = VaultReader(vault)
    visible = _filter_visible_refs(reader, reader.list_articles(), config.mcp)

    assert [ref.name for ref in visible] == ["Explicit Public"]


def test_excluded_tags_filter(vault, db) -> None:
    article_path = vault / "wiki" / "Secret.md"
    _write_article(article_path, "secret body", visibility="public", tags=["secret"])
    _insert_article(db, "wiki/Secret.md", "Secret")

    from synto.config import Config, McpConfig
    from synto.readers import VaultReader

    config = Config(vault=vault, mcp=McpConfig(exclude_tags=["secret"]))
    reader = VaultReader(vault)
    visible = _filter_visible_refs(reader, reader.list_articles(), config.mcp)

    assert visible == []


def test_read_article_hides_existence_for_hidden_article(vault, db) -> None:
    article_path = vault / "wiki" / "Hidden.md"
    _write_article(article_path, "hidden body", visibility="private")
    _insert_article(db, "wiki/Hidden.md", "Hidden")

    from synto.config import Config
    from synto.readers import VaultReader

    config = Config(vault=vault)
    reader = VaultReader(vault)

    try:
        _read_visible_article(reader, "Hidden", config.mcp)
    except ArticleNotFound:
        pass
    else:
        raise AssertionError("expected hidden article to look missing")


def test_find_concept_returns_none_when_canonical_article_hidden(vault, db) -> None:
    article_path = vault / "wiki" / "Canonical.md"
    _write_article(article_path, "hidden body", visibility="private")
    _insert_article(db, "wiki/Canonical.md", "Canonical")
    db.upsert_concepts("raw/source.md", ["Canonical"])
    db.upsert_aliases("Canonical", ["AliasName"])

    from synto.config import Config
    from synto.readers import VaultReader

    config = Config(vault=vault)
    reader = VaultReader(vault)
    concept = reader.find_concept("AliasName")

    assert concept is not None
    try:
        _read_visible_article(reader, concept.canonical_article_id or "", config.mcp)
    except ArticleNotFound:
        hidden = None
    else:
        hidden = concept

    assert hidden is None


def test_audit_log_off_by_default(vault, db) -> None:
    from synto.config import Config

    config = Config(vault=vault)
    _audit(
        db,
        vault_id="vault",
        tool="list_articles",
        arguments={"tag": "systems"},
        success=True,
        latency_ms=5,
        mcp_config=config.mcp,
    )

    assert db.list_metric_events() == []


def test_audit_log_on_records_calls(vault, db) -> None:
    from synto.config import Config, McpConfig

    config = Config(vault=vault, mcp=McpConfig(audit=True))
    _audit(
        db,
        vault_id="vault",
        tool="read_article",
        arguments={"name_or_id": "Some Title", "body": "very secret body"},
        success=True,
        latency_ms=7,
        mcp_config=config.mcp,
    )

    rows = db.list_metric_events()
    assert len(rows) == 1
    assert rows[0]["event_type"] == "mcp_call"
    payload = json.loads(rows[0]["metadata_json"])
    assert payload["tool"] == "read_article"
    assert payload["args"]["name_or_id"] is not None
    assert payload["args"]["body"] is not None
    assert payload["args"]["body"] != "very secret body"


def test_audit_failure_does_not_break_tool_response(vault, db, monkeypatch) -> None:
    from synto.config import Config, McpConfig

    config = Config(vault=vault, mcp=McpConfig(audit=True))

    def boom(**kwargs):
        raise RuntimeError("db broken")

    monkeypatch.setattr(db, "insert_mcp_audit_event", boom)

    _audit(
        db,
        vault_id="vault",
        tool="list_articles",
        arguments={},
        success=True,
        latency_ms=1,
        mcp_config=config.mcp,
    )
