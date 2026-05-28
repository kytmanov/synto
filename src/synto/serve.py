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

from .config import Config, McpConfig, McpSourceAccessConfig
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


# Per-process cache of effective source-access mode keyed by (vault_key, configured_mode).
# Populated once at run_server() startup so individual handlers don't re-query the DB.
_effective_mode_cache: dict[tuple[str, str], str] = {}


def _effective_source_access_mode(
    db: StateDB, configured: McpSourceAccessConfig, vault_key: str
) -> str:
    """Return effective source-access mode after grandfather check.

    Legacy vaults (zero declared licenses) get the configured "permissive_only" mode
    relaxed to "all" so upgrades from v0.3.0 are seamless. Once any source has a
    license set, the configured mode takes effect on the next process restart.

    "all" and "deny" modes pass through unchanged — grandfather only relaxes
    "permissive_only", it never overrides an explicit choice.

    Result is cached per (vault_key, configured.mode) for the process lifetime.
    """
    if configured.mode != "permissive_only":
        return configured.mode
    cache_key = (vault_key, configured.mode)
    cached = _effective_mode_cache.get(cache_key)
    if cached is not None:
        return cached
    declared = db._conn.execute(
        "SELECT 1 FROM source_documents WHERE license IS NOT NULL LIMIT 1"
    ).fetchone()
    effective = "permissive_only" if declared is not None else "all"
    _effective_mode_cache[cache_key] = effective
    return effective


def _license_allows_value(license_str: str | None, mode: str, permissive: set[str]) -> bool:
    """Pure-function gate. `mode` is the *effective* mode (post-grandfather).
    `permissive` is the casefolded set of allowed license strings.
    """
    if mode == "all":
        return True
    if mode == "deny":
        return False
    # permissive_only
    if license_str is None:
        return False
    return license_str.casefold() in permissive


def _license_allows(source_id: str, db: StateDB, mcp_config: McpConfig, vault_key: str) -> bool:
    """Single-source convenience wrapper for read_source_segment and list_segments.

    For multi-row tools (search_source_segments, get_source_passages), use the
    batched source-meta + _license_allows_value pattern instead to avoid N+1.
    """
    effective_mode = _effective_source_access_mode(db, mcp_config.source_access, vault_key)
    if effective_mode == "all":
        return True
    if effective_mode == "deny":
        return False
    # permissive_only — need the license value
    row = db._conn.execute(
        "SELECT license FROM source_documents WHERE id = ?", (source_id,)
    ).fetchone()
    if row is None:
        return False  # Orphan source. For single-source tools, this collapses to "policy denial"
        # which is acceptable because they already checked the source exists upstream.
    license_str: str | None = row["license"]
    permissive = {lic.casefold() for lic in mcp_config.source_access.permissive_licenses}
    return _license_allows_value(license_str, effective_mode, permissive)


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
            if not _license_allows(row["source_id"], db, config.mcp, vault_key):
                raise tool_error_cls(f"source {row['source_id']!r} is restricted by license policy")
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
        The response includes `hidden_by_policy` (count blocked by license gate) and
        `orphan_segments` (count whose source document is missing from the registry).
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
                _sql = (
                    "SELECT id, license, origin_uri FROM source_documents"
                    f" WHERE id IN ({placeholders})"
                )
                for src in db._conn.execute(_sql, source_ids).fetchall():
                    source_meta[src["id"]] = (src["license"], src["origin_uri"])
            # Effective mode + permissive set computed once for the whole result page.
            effective_mode = _effective_source_access_mode(db, config.mcp.source_access, vault_key)
            permissive = {lic.casefold() for lic in config.mcp.source_access.permissive_licenses}
            results: list[dict[str, Any]] = []
            hidden_by_policy = 0
            orphan_segments = 0
            for r in rows:
                meta = source_meta.get(r["source_id"])
                if meta is None:
                    # source_id present in source_segments but missing from source_documents
                    # → orphan. Distinct from policy denial.
                    orphan_segments += 1
                    continue
                license_str, origin_uri = meta
                if not _license_allows_value(license_str, effective_mode, permissive):
                    hidden_by_policy += 1
                    continue
                results.append(
                    {
                        "segment_id": r["segment_id"],
                        "source_id": r["source_id"],
                        "ordinal": r["ordinal"],
                        "snippet": r["snippet"],
                        "score": -r["rank"],  # BM25 is negative; flip so higher = better
                        "source_path": origin_uri,
                    }
                )
            success = True
            return {
                "results": results,
                "hidden_by_policy": hidden_by_policy,
                "orphan_segments": orphan_segments,
            }
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

    def get_source_passages(
        concept_name: str, max_passages: int = 5, max_chars: int | None = None
    ) -> dict[str, object]:
        """Return verbatim paragraphs explicitly linked to a concept.

        Use when you know which concept the user is asking about and want
        the source's own explanation, not a synto-generated synthesis.
        Results are ordered by extraction confidence then reading order.
        Pass max_chars to cap each paragraph (default 8000, max 16000).
        The response includes `hidden_by_policy` (count blocked by license gate) and
        `orphan_segments` (count whose source document is missing from the registry).
        """
        started = time.monotonic()
        success = False
        arguments: dict[str, Any] = {
            "concept_name": concept_name,
            "max_passages": max_passages,
            "max_chars": max_chars,
        }
        try:
            assert db is not None
            if max_passages > 20:
                raise tool_error_cls("max_passages must be at most 20")
            max_passages = max(1, max_passages)
            resolved = db.find_concept_by_name_or_alias(concept_name)
            if resolved is None:
                success = True
                return {"results": [], "hidden_by_policy": 0, "orphan_segments": 0}
            canonical, _aliases = resolved
            rows = db.select_passages_for_concept(canonical, max_passages)
            # select_passages_for_concept already JOINs source_documents and returns
            # origin_uri, license, and doc_id — no separate batch fetch needed.
            # doc_id is NULL when source_id has no matching source_documents row (orphan).
            char_cap = min(max_chars, 16000) if max_chars is not None else 8000
            # Effective mode + permissive set computed once for the whole result page.
            effective_mode = _effective_source_access_mode(db, config.mcp.source_access, vault_key)
            permissive = {lic.casefold() for lic in config.mcp.source_access.permissive_licenses}
            results: list[dict[str, Any]] = []
            hidden_by_policy = 0
            orphan_segments = 0
            for r in rows:
                if r["doc_id"] is None:
                    # LEFT JOIN miss: source_id in source_segments with no source_documents row.
                    orphan_segments += 1
                    continue
                license_str: str | None = r["license"]
                origin_uri: str | None = r["origin_uri"]
                if not _license_allows_value(license_str, effective_mode, permissive):
                    hidden_by_policy += 1
                    continue
                body: str = r["text"]
                truncated = False
                if len(body) > char_cap:
                    body = body[:char_cap] + "…"
                    truncated = True
                results.append(
                    {
                        "segment_id": r["id"],
                        "source_id": r["source_id"],
                        "ordinal": r["ordinal"],
                        "body": body,
                        "confidence": r["confidence"],
                        "source_path": origin_uri,
                        "truncated": truncated,
                    }
                )
            success = True
            return {
                "results": results,
                "hidden_by_policy": hidden_by_policy,
                "orphan_segments": orphan_segments,
            }
        finally:
            _audit(
                db,
                vault_id=vault_key,
                tool="get_source_passages",
                arguments=arguments,
                success=success,
                latency_ms=int((time.monotonic() - started) * 1000),
                mcp_config=config.mcp,
            )

    def list_segments(source_id: str, limit: int = 200, offset: int = 0) -> dict[str, object]:
        """Enumerate a single source document's paragraphs in reading order.

        Use for sequential exploration of a known document. Returns segment ids
        and character lengths (not full bodies) — fetch individual bodies with
        read_source_segment. limit is capped at 500.
        """
        started = time.monotonic()
        success = False
        arguments: dict[str, Any] = {"source_id": source_id, "limit": limit, "offset": offset}
        try:
            assert db is not None
            limit = min(max(1, limit), 500)
            offset = max(0, offset)
            src = db._conn.execute(
                "SELECT id, license FROM source_documents WHERE id = ?", (source_id,)
            ).fetchone()
            if src is None:
                raise tool_error_cls(f"unknown source_id: {source_id!r}")
            if not _license_allows(source_id, db, config.mcp, vault_key):
                raise tool_error_cls(f"source {source_id!r} is restricted by license policy")
            total = db._conn.execute(
                "SELECT count(*) FROM source_segments WHERE source_id = ?", (source_id,)
            ).fetchone()[0]
            rows = db._conn.execute(
                """SELECT id, ordinal, length(text) AS length
                   FROM source_segments
                   WHERE source_id = ?
                   ORDER BY ordinal
                   LIMIT ? OFFSET ?""",
                (source_id, limit, offset),
            ).fetchall()
            segments = [
                {"segment_id": r["id"], "ordinal": r["ordinal"], "length": r["length"]}
                for r in rows
            ]
            success = True
            return {
                "source_id": source_id,
                "total": total,
                "returned": len(segments),
                "segments": segments,
            }
        finally:
            _audit(
                db,
                vault_id=vault_key,
                tool="list_segments",
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
        handlers["get_source_passages"] = get_source_passages
        handlers["list_segments"] = list_segments
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
    vault_key = _vault_id(vault)
    # Surface grandfather behaviour exactly once at startup so legacy vault users
    # know why permissive_only is being treated as "all".
    effective_mode = _effective_source_access_mode(db, config.mcp.source_access, vault_key)
    if config.mcp.source_access.mode == "permissive_only" and effective_mode == "all":
        log.info(
            "synto: legacy vault detected (0 declared licenses); MCP source-access mode treated"
            ' as "all" for this session. Set [mcp.source_access] in synto.toml or declare'
            " licenses on sources to engage the privacy gate."
        )
    server = FastMCP(
        _server_name(vault),
        instructions=_SERVER_INSTRUCTIONS,
        json_response=True,
    )
    handlers = build_tool_handlers(reader, config, db, vault_key, tool_error_cls=ToolError)
    for handler in handlers.values():
        server.tool()(handler)

    try:
        server.run(transport="stdio")
    finally:
        db.close()


async def run_server_async(vault: Path, transport: str = "stdio") -> None:
    await asyncio.to_thread(run_server, vault, transport)
