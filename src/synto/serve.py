"""Read-only MCP server wrapping VaultReader plus a query-through tool."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Config, McpConfig
from .readers import (
    Article,
    ArticleFilter,
    ArticleNotFound,
    ArticleRef,
    VaultReader,
    _extract_first_paragraph,
)
from .state import StateDB

log = logging.getLogger(__name__)

_DEFAULT_MIN_STATUS = "published"
_SEARCH_LIMIT_CAP = 50

_SERVER_INSTRUCTIONS = (
    "Read-only access to wiki articles from one Synto vault. "
    "Visibility filtering and optional audit logging are applied server-side. "
    "By default only status='published' articles are returned; pass "
    "min_status='verified' or 'draft' to opt in to lower-trust material. "
    "`answer_question` triggers fast+heavy LLM calls (may incur cost on "
    "paid providers); all other tools are filesystem-only."
)


def _vault_id(vault: Path) -> str:
    return hashlib.sha256(str(vault.resolve()).encode("utf-8")).hexdigest()[:16]


def _hash_args(arguments: dict[str, Any]) -> dict[str, Any]:
    # bool/int/float are low-cardinality (often only two or three useful
    # values, e.g. limit/exclude_single_source). Hashing them loses all
    # signal for audit/debug without adding privacy. Strings, lists, and
    # dicts still get the 8-char sha256 prefix so user-supplied text
    # never lands raw in metric_events.
    safe: dict[str, Any] = {}
    for key, value in arguments.items():
        if value is None or isinstance(value, (bool, int, float)):
            safe[key] = value
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


def _license_allows(source_id: str, db: StateDB, mcp_config: McpConfig) -> bool:
    """Return True if the source's license permits verbatim access under current policy.

    Stub implementation — always permits. Replaced with the real gate in Stage 6.
    """
    return True


def _ref_to_dict(ref: ArticleRef) -> dict[str, object]:
    return {
        "id": ref.id,
        "name": ref.name,
        "path": ref.path,
        "summary": ref.summary,
        "tags": list(ref.tags),
        "confidence": ref.confidence_score,
        "source_count": ref.source_count,
        "single_source": ref.single_source,
        "source_quality": ref.source_quality,
        "status": ref.status,
        "kind": ref.kind,
    }


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


def _resolve_min_status(value: str | None) -> str:
    # Tool handlers expose `min_status: str | None = None`. None means
    # "the caller did not set it" → apply the agent-safe default. Empty
    # string means "no filter" — agent must opt in explicitly.
    if value is None:
        return _DEFAULT_MIN_STATUS
    return value


class _DefaultToolError(RuntimeError):
    """Placeholder exception type used when handlers are built outside FastMCP."""


def build_tool_handlers(
    reader: VaultReader,
    config: Config,
    db: StateDB | None,
    vault_key: str,
    *,
    tool_error_cls: type[Exception] = _DefaultToolError,
) -> dict[str, Callable[..., Any]]:
    """Construct the MCP tool handlers and return them keyed by tool name.

    Used by `run_server` to register against a FastMCP instance, and by tests
    to drive each handler directly without spinning up STDIO.
    `tool_error_cls` lets callers map "not found" errors to FastMCP's
    `ToolError` when running for real.
    """

    def list_articles(
        tag: str | None = None,
        contains: str | None = None,
        min_status: str | None = None,
        kind: str | None = None,
        exclude_single_source: bool = False,
    ) -> list[dict[str, object]]:
        started = time.monotonic()
        success = False
        arguments = {
            "tag": tag,
            "contains": contains,
            "min_status": min_status,
            "kind": kind,
            "exclude_single_source": exclude_single_source,
        }
        try:
            refs = reader.list_articles(
                filter=ArticleFilter(
                    tag=tag,
                    contains=contains,
                    min_status=_resolve_min_status(min_status),
                    kind=kind,
                    exclude_single_source=exclude_single_source,
                )
            )
            visible = _filter_visible_refs(reader, refs, config.mcp)
            success = True
            return [_ref_to_dict(ref) for ref in visible]
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

    def read_article(name_or_id: str) -> dict[str, object]:
        started = time.monotonic()
        success = False
        arguments = {"name_or_id": name_or_id}
        try:
            try:
                article = _read_visible_article(reader, name_or_id, config.mcp)
            except ArticleNotFound as exc:
                raise tool_error_cls(f"No article: {name_or_id!r}") from exc
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

    def search_articles(
        query: str,
        limit: int = 10,
        min_status: str | None = None,
        kind: str | None = None,
        exclude_single_source: bool = False,
    ) -> list[dict[str, object]]:
        """Lexical search across article name, summary, and aliases.

        Use `get_concept` for concept-graph lookups, and `answer_question`
        for a routed/synthesized answer. Returns up to `limit` results
        (capped at 50) sorted by substring-hit score.
        """
        started = time.monotonic()
        success = False
        arguments = {
            "query": query,
            "limit": limit,
            "min_status": min_status,
            "kind": kind,
            "exclude_single_source": exclude_single_source,
        }
        try:
            if not query:
                success = True
                return []
            capped_limit = max(0, min(int(limit), _SEARCH_LIMIT_CAP))
            # No `contains=` here — that filter only inspects name+summary
            # and would drop alias-only matches before scoring sees them.
            refs = reader.list_articles(
                filter=ArticleFilter(
                    min_status=_resolve_min_status(min_status),
                    kind=kind,
                    exclude_single_source=exclude_single_source,
                )
            )
            needle = query.casefold()
            scored_candidates: list[tuple[int, ArticleRef]] = []
            for ref in refs:
                haystack = f"{ref.name}\n{ref.summary or ''}\n{' '.join(ref.aliases)}".casefold()
                score = haystack.count(needle)
                if score:
                    scored_candidates.append((score, ref))
            # Visibility check after scoring to avoid file-reading every
            # article in vaults where the query matches a small subset.
            visible_refs = _filter_visible_refs(
                reader, [r for _, r in scored_candidates], config.mcp
            )
            visible_ids = {r.id for r in visible_refs}
            scored_visible = [(s, r) for s, r in scored_candidates if r.id in visible_ids]
            scored_visible.sort(key=lambda item: item[0], reverse=True)
            success = True
            results: list[dict[str, object]] = []
            for score, ref in scored_visible[:capped_limit]:
                payload = _ref_to_dict(ref)
                payload["score"] = score
                results.append(payload)
            return results
        finally:
            _audit(
                db,
                vault_id=vault_key,
                tool="search_articles",
                arguments=arguments,
                success=success,
                latency_ms=int((time.monotonic() - started) * 1000),
                mcp_config=config.mcp,
            )

    def get_concept(name: str) -> dict[str, object]:
        started = time.monotonic()
        success = False
        arguments = {"name": name}
        empty: dict[str, object] = {
            "name": name,
            "aliases": [],
            "canonical_article_id": None,
            "definition": "",
            "body": "",
            "frontmatter": {},
        }
        try:
            concept = reader.find_concept(name)
            if concept is None:
                success = True
                return empty
            payload: dict[str, object] = {
                "name": concept.name,
                "aliases": list(concept.aliases),
                "canonical_article_id": concept.canonical_article_id,
                "definition": "",
                "body": "",
                "frontmatter": {},
            }
            if concept.canonical_article_id:
                try:
                    article = _read_visible_article(
                        reader, concept.canonical_article_id, config.mcp
                    )
                    payload["body"] = article.body
                    payload["frontmatter"] = article.frontmatter
                    payload["definition"] = _extract_first_paragraph(article.body) or ""
                except ArticleNotFound:
                    pass
            success = True
            return payload
        finally:
            _audit(
                db,
                vault_id=vault_key,
                tool="get_concept",
                arguments=arguments,
                success=success,
                latency_ms=int((time.monotonic() - started) * 1000),
                mcp_config=config.mcp,
            )

    def list_sources() -> list[dict[str, object]]:
        started = time.monotonic()
        success = False
        try:
            sources = reader.list_sources()
            success = True
            return [
                {
                    "id": src.id,
                    "title": src.title,
                    "source_type": src.source_type,
                }
                for src in sources
            ]
        finally:
            _audit(
                db,
                vault_id=vault_key,
                tool="list_sources",
                arguments={},
                success=success,
                latency_ms=int((time.monotonic() - started) * 1000),
                mcp_config=config.mcp,
            )

    def trace_lineage(name_or_id: str) -> dict[str, object]:
        started = time.monotonic()
        success = False
        arguments = {"name_or_id": name_or_id}
        try:
            try:
                article = _read_visible_article(reader, name_or_id, config.mcp)
            except ArticleNotFound as exc:
                raise tool_error_cls(f"No article: {name_or_id!r}") from exc
            raw = article.frontmatter.get("lineage", [])
            lineage = raw if isinstance(raw, list) else []
            success = True
            return {"article": article.name, "lineage": lineage}
        finally:
            _audit(
                db,
                vault_id=vault_key,
                tool="trace_lineage",
                arguments=arguments,
                success=success,
                latency_ms=int((time.monotonic() - started) * 1000),
                mcp_config=config.mcp,
            )

    def answer_question(question: str, max_pages: int = 5) -> dict[str, object]:
        started = time.monotonic()
        success = False
        arguments = {"question": question, "max_pages": max_pages}
        try:
            from .client_factory import build_client
            from .engines import QueryConfig, QueryEngine

            client = build_client(config)
            engine = QueryEngine(
                reader=reader,
                fast_client=client,
                heavy_client=client,
                config=config,
                db=db,
                query_config=QueryConfig(max_pages=max(1, int(max_pages))),
            )
            answer = engine.query(question)
            visible_pages: list[str] = []
            for page_name in engine.last_selected_pages:
                try:
                    _read_visible_article(reader, page_name, config.mcp)
                except ArticleNotFound:
                    continue
                visible_pages.append(page_name)
            success = True
            return {
                "answer": answer.text,
                "title": answer.title,
                "selected_pages": visible_pages,
                "index_found": engine.last_index_found,
            }
        finally:
            _audit(
                db,
                vault_id=vault_key,
                tool="answer_question",
                arguments=arguments,
                success=success,
                latency_ms=int((time.monotonic() - started) * 1000),
                mcp_config=config.mcp,
            )

    def read_source_segment(segment_id: str, max_chars: int | None = None) -> dict[str, object]:
        """Fetch one verbatim paragraph by segment id.

        Use when you already have a segment id from another tool and want the
        exact source text. Returns the raw paragraph body as ingested, not a
        synto-generated synthesis. Pass max_chars to cap the returned body size
        (capped at 16000; default is unbounded for single-segment fetches).
        """
        started = time.monotonic()
        success = False
        arguments: dict[str, Any] = {"segment_id": segment_id, "max_chars": max_chars}
        try:
            assert db is not None
            row = db._conn.execute(
                """SELECT s.source_id, s.identity, s.ordinal, s.content_hash, s.text,
                          d.origin_uri
                   FROM source_segments s
                   LEFT JOIN source_documents d ON d.id = s.source_id
                   WHERE s.id = ?""",
                (segment_id,),
            ).fetchone()
            if row is None:
                raise tool_error_cls(f"unknown segment_id: {segment_id!r}")
            if not _license_allows(row["source_id"], db, config.mcp):
                raise tool_error_cls(
                    f"source {row['source_id']!r} is restricted by license policy"
                )
            body: str = row["text"]
            truncated = False
            if max_chars is not None:
                cap = min(max_chars, 16000)
                if len(body) > cap:
                    body = body[:cap] + "…"
                    truncated = True
            success = True
            return {
                "segment_id": segment_id,
                "source_id": row["source_id"],
                "identity": row["identity"],
                "ordinal": row["ordinal"],
                "content_hash": row["content_hash"],
                "body": body,
                "source_path": row["origin_uri"],
                "truncated": truncated,
            }
        finally:
            _audit(
                db,
                vault_id=vault_key,
                tool="read_source_segment",
                arguments=arguments,
                success=success,
                latency_ms=int((time.monotonic() - started) * 1000),
                mcp_config=config.mcp,
            )

    def search_source_segments(query: str, limit: int = 10) -> dict[str, object]:
        """Full-text search (BM25) across raw source paragraphs.

        Use when the user wants the source's own words, not a synto-generated
        synthesis. Returns ranked snippets with segment ids; fetch the full body
        with read_source_segment. limit is capped at 50.
        """
        started = time.monotonic()
        success = False
        arguments: dict[str, Any] = {"query": query, "limit": limit}
        try:
            assert db is not None
            if not query or not query.strip():
                raise tool_error_cls("query must be non-empty")
            if limit < 1:
                raise tool_error_cls("limit must be at least 1")
            limit = min(limit, 50)
            # Wrap in double-quotes to treat user input as a literal phrase.
            # Internal double-quotes are doubled to escape them in FTS5 syntax.
            match_arg = '"' + query.replace('"', '""') + '"'
            rows = db._conn.execute(
                """SELECT s.id AS segment_id, s.source_id, s.ordinal,
                          snippet(source_segments_fts, 0, '', '', '…', 32) AS snippet,
                          bm25(source_segments_fts) AS rank
                   FROM source_segments_fts
                   JOIN source_segments s ON s.rowid = source_segments_fts.rowid
                   WHERE source_segments_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (match_arg, limit),
            ).fetchall()
            # Batch fetch licenses and origin_uris for all distinct sources
            source_ids = list({r["source_id"] for r in rows})
            source_meta: dict[str, tuple[str | None, str | None]] = {}
            if source_ids:
                placeholders = ",".join("?" * len(source_ids))
                for src in db._conn.execute(
                    f"SELECT id, license, origin_uri FROM source_documents WHERE id IN ({placeholders})",
                    source_ids,
                ).fetchall():
                    source_meta[src["id"]] = (src["license"], src["origin_uri"])
            results = []
            hidden = 0
            for r in rows:
                if not _license_allows(r["source_id"], db, config.mcp):
                    hidden += 1
                    continue
                _license, origin_uri = source_meta.get(r["source_id"], (None, None))
                results.append({
                    "segment_id": r["segment_id"],
                    "source_id": r["source_id"],
                    "ordinal": r["ordinal"],
                    "snippet": r["snippet"],
                    "score": -r["rank"],  # BM25 is negative; flip so higher = better
                    "source_path": origin_uri,
                })
            success = True
            return {"results": results, "hidden_by_policy": hidden}
        finally:
            _audit(
                db,
                vault_id=vault_key,
                tool="search_source_segments",
                arguments=arguments,
                success=success,
                latency_ms=int((time.monotonic() - started) * 1000),
                mcp_config=config.mcp,
            )

    handlers: dict[str, Callable[..., Any]] = {
        "list_articles": list_articles,
        "read_article": read_article,
        "find_concept": find_concept,
        "search_articles": search_articles,
        "get_concept": get_concept,
        "list_sources": list_sources,
        "trace_lineage": trace_lineage,
        "answer_question": answer_question,
    }
    if db is not None:
        handlers["read_source_segment"] = read_source_segment
        handlers["search_source_segments"] = search_source_segments
    return handlers


def run_server(vault: Path, transport: str = "stdio") -> None:
    if transport != "stdio":
        raise RuntimeError("Phase 1A supports only stdio transport")

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
    # Always open db — new verbatim-source tools query it regardless of audit setting.
    # _audit() still respects config.mcp.audit before writing any audit rows.
    db = StateDB(config.state_db_path)
    server = FastMCP(
        _server_name(vault),
        instructions=_SERVER_INSTRUCTIONS,
        json_response=True,
    )
    vault_key = _vault_id(vault)
    handlers = build_tool_handlers(reader, config, db, vault_key, tool_error_cls=ToolError)
    for handler in handlers.values():
        server.tool()(handler)

    try:
        server.run(transport="stdio")
    finally:
        db.close()


async def run_server_async(vault: Path, transport: str = "stdio") -> None:
    await asyncio.to_thread(run_server, vault, transport)
