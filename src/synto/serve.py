"""Minimal read-only MCP server wrapping VaultReader."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Config, McpConfig
from .readers import Article, ArticleFilter, ArticleNotFound, ArticleRef, VaultReader
from .state import StateDB

log = logging.getLogger(__name__)


def _check_mcp_available() -> None:
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "mcp library not installed. Install with: pip install synto[mcp]"
        ) from exc


def _vault_id(vault: Path) -> str:
    return hashlib.sha256(str(vault.resolve()).encode("utf-8")).hexdigest()[:16]


def _hash_args(arguments: dict[str, Any]) -> dict[str, str | None]:
    safe: dict[str, str | None] = {}
    for key, value in arguments.items():
        if value is None:
            safe[key] = None
        else:
            safe[key] = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8]
    return safe


def _tags_from_frontmatter(frontmatter: dict[str, object]) -> set[str]:
    tags = frontmatter.get("tags", [])
    if not isinstance(tags, list):
        return set()
    return {str(tag) for tag in tags}


def _is_visible(article: Article, mcp_config: McpConfig) -> bool:
    visibility = article.frontmatter.get("visibility", mcp_config.default_visibility)
    if visibility != "public":
        return False
    tags = _tags_from_frontmatter(article.frontmatter)
    return not bool(tags & set(mcp_config.exclude_tags))


def _filter_visible_refs(
    reader: VaultReader, refs: list[ArticleRef], mcp_config: McpConfig
) -> list[ArticleRef]:
    visible: list[ArticleRef] = []
    for ref in refs:
        try:
            article = reader.read_article(ref.id)
        except ArticleNotFound:
            continue
        if _is_visible(article, mcp_config):
            visible.append(ref)
    return visible


def _read_visible_article(reader: VaultReader, name_or_id: str, mcp_config: McpConfig) -> Article:
    article = reader.read_article(name_or_id)
    if not _is_visible(article, mcp_config):
        raise ArticleNotFound(name_or_id)
    return article


def _audit(
    db: StateDB | None,
    *,
    vault_id: str,
    tool: str,
    arguments: dict[str, Any],
    success: bool,
    latency_ms: int,
    mcp_config: McpConfig,
) -> None:
    if db is None or not mcp_config.audit:
        return
    try:
        db.insert_mcp_audit_event(
            ts=datetime.now(UTC).isoformat(),
            vault_id=vault_id,
            tool=tool,
            args_summary=_hash_args(arguments),
            latency_ms=latency_ms,
            success=success,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("mcp audit failed: %s", exc)


def _server_name(vault: Path) -> str:
    return f"synto:{hashlib.sha256(str(vault.resolve()).encode()).hexdigest()[:8]}"


def run_server(vault: Path, transport: str = "stdio") -> None:
    if transport != "stdio":
        raise RuntimeError("Phase 1A supports only stdio transport")

    _check_mcp_available()

    # The CLI group callback installs a RichHandler pointing at sys.stdout for
    # interactive use.  In stdio mode stdout is the JSON-RPC channel, so any
    # log line written there corrupts the protocol.  Suppress the mcp library's
    # INFO request tracing — it is noise to end users and not safe on this fd.
    import logging as _logging

    _logging.getLogger("mcp").setLevel(_logging.WARNING)

    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.exceptions import ToolError

    config = Config.from_vault(vault)
    reader = VaultReader(vault)
    db = StateDB(config.state_db_path) if config.mcp.audit else None
    server = FastMCP(
        _server_name(vault),
        instructions=(
            "Read-only access to published wiki articles from one Synto vault. "
            "Visibility filtering and optional audit logging are applied server-side."
        ),
        json_response=True,
    )
    vault_key = _vault_id(vault)

    @server.tool()
    def list_articles(
        tag: str | None = None, contains: str | None = None
    ) -> list[dict[str, object]]:
        started = time.monotonic()
        success = False
        arguments = {"tag": tag, "contains": contains}
        try:
            refs = reader.list_articles(filter=ArticleFilter(tag=tag, contains=contains))
            visible = _filter_visible_refs(reader, refs, config.mcp)
            success = True
            return [
                {
                    "id": ref.id,
                    "name": ref.name,
                    "path": ref.path,
                    "summary": ref.summary,
                    "tags": list(ref.tags),
                    "confidence": ref.confidence_score,
                    "source_count": ref.source_count,
                    "single_source": ref.single_source,
                    "source_quality": ref.source_quality,
                }
                for ref in visible
            ]
        finally:
            _audit(
                db,
                vault_id=vault_key,
                tool="list_articles",
                arguments=arguments,
                success=success,
                latency_ms=int((time.monotonic() - started) * 1000),
                mcp_config=config.mcp,
            )

    @server.tool()
    def read_article(name_or_id: str) -> dict[str, object]:
        started = time.monotonic()
        success = False
        arguments = {"name_or_id": name_or_id}
        try:
            try:
                article = _read_visible_article(reader, name_or_id, config.mcp)
            except ArticleNotFound as exc:
                raise ToolError(f"No article: {name_or_id!r}") from exc
            success = True
            return {
                "id": article.id,
                "name": article.name,
                "path": article.path,
                "body": article.body,
                "frontmatter": article.frontmatter,
            }
        finally:
            _audit(
                db,
                vault_id=vault_key,
                tool="read_article",
                arguments=arguments,
                success=success,
                latency_ms=int((time.monotonic() - started) * 1000),
                mcp_config=config.mcp,
            )

    @server.tool()
    def find_concept(query: str) -> dict[str, object] | None:
        started = time.monotonic()
        success = False
        arguments = {"query": query}
        try:
            concept = reader.find_concept(query)
            if concept is None or not concept.canonical_article_id:
                success = True
                return None
            try:
                _read_visible_article(reader, concept.canonical_article_id, config.mcp)
            except ArticleNotFound:
                success = True
                return None
            success = True
            return {
                "name": concept.name,
                "canonical_article_id": concept.canonical_article_id,
                "aliases": list(concept.aliases),
            }
        finally:
            _audit(
                db,
                vault_id=vault_key,
                tool="find_concept",
                arguments=arguments,
                success=success,
                latency_ms=int((time.monotonic() - started) * 1000),
                mcp_config=config.mcp,
            )

    try:
        server.run(transport="stdio")
    finally:
        if db is not None:
            db.close()


async def run_server_async(vault: Path, transport: str = "stdio") -> None:
    await asyncio.to_thread(run_server, vault, transport)
