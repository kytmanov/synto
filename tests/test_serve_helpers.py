"""Additional tests for serve.py helper functions and edge cases."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synto.config import McpConfig
from synto.readers import Article
from synto.serve import (
    _audit,
    _is_visible,
    _vault_id,
)


def test_vault_id_uses_resolved_path():
    """_vault_id resolves symlinks before hashing."""
    vid1 = _vault_id(Path("/tmp/../tmp/test-vault"))
    vid2 = _vault_id(Path("/tmp/test-vault"))
    assert vid1 == vid2


def test_is_visible_public_no_exclude_tags():
    """Public article with no exclude tags is visible."""
    article = Article(
        id="test",
        name="Test",
        path="wiki/Test.md",
        body="body",
        frontmatter={"visibility": "public", "tags": []},
    )
    mcp = McpConfig(default_visibility="public", exclude_tags=[])
    assert _is_visible(article, mcp) is True


def test_is_visible_private():
    """Private article is not visible."""
    article = Article(
        id="test",
        name="Test",
        path="wiki/Test.md",
        body="body",
        frontmatter={"visibility": "private", "tags": []},
    )
    mcp = McpConfig(default_visibility="public")
    assert _is_visible(article, mcp) is False


def test_is_visible_default_private_missing_visibility():
    """Article without visibility field uses default_visibility."""
    article = Article(
        id="test",
        name="Test",
        path="wiki/Test.md",
        body="body",
        frontmatter={"tags": []},
    )
    mcp = McpConfig(default_visibility="private")
    assert _is_visible(article, mcp) is False


def test_is_visible_default_public_missing_visibility():
    """Article without visibility field uses default_visibility=public."""
    article = Article(
        id="test",
        name="Test",
        path="wiki/Test.md",
        body="body",
        frontmatter={"tags": []},
    )
    mcp = McpConfig(default_visibility="public")
    assert _is_visible(article, mcp) is True


def test_is_visible_exclude_tags_match():
    """Public article with excluded tag is not visible."""
    article = Article(
        id="test",
        name="Test",
        path="wiki/Test.md",
        body="body",
        frontmatter={"visibility": "public", "tags": ["secret", "internal"]},
    )
    mcp = McpConfig(default_visibility="public", exclude_tags=["secret"])
    assert _is_visible(article, mcp) is False


def test_is_visible_exclude_tags_no_match():
    """Public article with non-excluded tag is visible."""
    article = Article(
        id="test",
        name="Test",
        path="wiki/Test.md",
        body="body",
        frontmatter={"visibility": "public", "tags": ["public"]},
    )
    mcp = McpConfig(default_visibility="public", exclude_tags=["secret"])
    assert _is_visible(article, mcp) is True


def test_audit_with_none_db():
    """_audit with db=None does not crash."""
    _audit(
        None,
        vault_id="v1",
        tool="test",
        arguments={"k": "v"},
        success=True,
        latency_ms=10,
        mcp_config=McpConfig(audit=True),
    )


def test_audit_with_audit_disabled():
    """_audit with audit=False does not record."""
    mock_db = MagicMock()
    _audit(
        mock_db,
        vault_id="v1",
        tool="test",
        arguments={"k": "v"},
        success=True,
        latency_ms=10,
        mcp_config=McpConfig(audit=False),
    )
    mock_db.insert_mcp_audit_event.assert_not_called()


def test_audit_db_error_does_not_raise():
    """_audit catches exceptions and does not propagate."""
    mock_db = MagicMock()
    mock_db.insert_mcp_audit_event.side_effect = RuntimeError("db broken")
    _audit(
        mock_db,
        vault_id="v1",
        tool="test",
        arguments={},
        success=True,
        latency_ms=5,
        mcp_config=McpConfig(audit=True),
    )
    # No exception raised


def test_run_server_rejects_non_stdio_transport():
    """run_server raises RuntimeError for non-stdio transport."""
    from synto.serve import run_server

    with patch("synto.serve._check_mcp_available"):
        with pytest.raises(RuntimeError, match="only stdio"):
            run_server(Path("/tmp/vault"), transport="http")
