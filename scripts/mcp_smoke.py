#!/usr/bin/env python
"""Standalone MCP protocol smoke test.

Creates isolated vaults, bootstraps state DBs, starts `synto serve` as a
subprocess per suite, and exercises every tool + edge case over stdio.

Exit 0 = all checks passed.
Exit 1 = at least one check failed.

NOTE: Use the direct synto binary (not `uv run synto`).  When launched via
`uv run`, uv routes child stderr through its own stdout, injecting log lines
into the JSON-RPC stream and causing parse errors on the client side.

Environment variables:
  REPORT_FILE  Path to write a JSON results file (written even on failure).
               stdout remains human-readable when this is set.

Flags:
  --json       Print a single JSON results object to stdout (suppresses
               streaming output).
  --suite NAME Run only the named suite (use --suite list to show available).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── Output helpers ────────────────────────────────────────────────────────────

_COLOR = sys.stdout.isatty()


def _c(code: str) -> str:
    return code if _COLOR else ""


_GREEN = "\033[32m"
_RED = "\033[31m"
_HEAD = "\033[1m"
_RST = "\033[0m"

PASS_SYM = f"{_c(_GREEN)}✓{_c(_RST)}" if _COLOR else "PASS"
FAIL_SYM = f"{_c(_RED)}✗{_c(_RST)}" if _COLOR else "FAIL"

_failures: list[str] = []
_results: list[dict] = []
_current_suite: str = ""
_JSON_MODE: bool = False


def check(label: str, ok: bool, detail: str = "") -> None:
    _results.append(
        {
            "suite": _current_suite,
            "name": label,
            "passed": ok,
            "detail": detail or None,
        }
    )
    if ok:
        if not _JSON_MODE:
            print(f"  {PASS_SYM} {label}")
    else:
        msg = label + (f": {detail}" if detail else "")
        _failures.append(msg)
        if not _JSON_MODE:
            print(f"  {FAIL_SYM} {msg}")


def header(title: str) -> None:
    if not _JSON_MODE:
        print(f"\n{_c(_HEAD)}{title}{_c(_RST)}")


# ── Vault bootstrap ───────────────────────────────────────────────────────────


def _base_toml() -> str:
    return (
        '[models]\nfast = "dummy"\nheavy = "dummy"\n\n'
        '[provider]\nname = "ollama"\nurl = "http://localhost:11434"\n'
    )


def _insert(db, path: str, title: str, status: str = "published") -> None:
    from synto.models import WikiArticleRecord

    db.upsert_article(
        WikiArticleRecord(
            path=path,
            title=title,
            sources=["raw/note.md"],
            content_hash=f"h-{title}",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=status,
        )
    )


def make_main_vault(tmp: Path) -> Path:
    """Primary test vault: public + private + default-visibility articles, concepts."""
    from synto.state import StateDB

    vault = tmp / "vault"
    (vault / "wiki").mkdir(parents=True)
    (vault / ".synto").mkdir()
    (vault / "raw").mkdir()
    (vault / "synto.toml").write_text(_base_toml())

    (vault / "wiki" / "Neural Networks.md").write_text(
        '---\ntitle: Neural Networks\ntags: ["ml"]\nvisibility: public\n---\n\n'
        "Neural networks learn by adjusting weights through backpropagation.\n"
    )
    (vault / "wiki" / "Internal Notes.md").write_text(
        "---\ntitle: Internal Notes\ntags: []\nvisibility: private\n---\n\nInternal only.\n"
    )
    # No visibility field → default_visibility="public" (server default) → visible
    (vault / "wiki" / "Default Visibility.md").write_text(
        '---\ntitle: Default Visibility\ntags: ["default"]\n---\n\n'
        "This article has no explicit visibility field.\n"
    )

    db = StateDB(vault / ".synto" / "state.db")
    _insert(db, "wiki/Neural Networks.md", "Neural Networks")
    _insert(db, "wiki/Internal Notes.md", "Internal Notes")
    _insert(db, "wiki/Default Visibility.md", "Default Visibility")

    db.upsert_concepts("raw/note.md", ["Neural Networks"])
    db.upsert_aliases("Neural Networks", ["NN", "neural net"])
    # Concept whose canonical article is private — find_concept must return null
    db.upsert_concepts("raw/note.md", ["Internal Notes"])
    db.close()
    return vault


def make_exclude_tags_vault(tmp: Path) -> Path:
    """Vault with exclude_tags = [\"secret\"] in mcp config."""
    from synto.state import StateDB

    vault = tmp / "exclude_vault"
    (vault / "wiki").mkdir(parents=True)
    (vault / ".synto").mkdir()
    (vault / "raw").mkdir()
    (vault / "synto.toml").write_text(_base_toml() + '\n[mcp]\nexclude_tags = ["secret"]\n')

    (vault / "wiki" / "Open Article.md").write_text(
        '---\ntitle: Open Article\ntags: ["general"]\nvisibility: public\n---\n\nVisible body.\n'
    )
    (vault / "wiki" / "Secret Article.md").write_text(
        '---\ntitle: Secret Article\ntags: ["secret", "general"]\nvisibility: public\n---\n\nHidden body.\n'
    )
    # An article that is both tagged secret AND private — double-hidden
    (vault / "wiki" / "Double Hidden.md").write_text(
        '---\ntitle: Double Hidden\ntags: ["secret"]\nvisibility: private\n---\n\nDouble hidden.\n'
    )

    db = StateDB(vault / ".synto" / "state.db")
    _insert(db, "wiki/Open Article.md", "Open Article")
    _insert(db, "wiki/Secret Article.md", "Secret Article")
    _insert(db, "wiki/Double Hidden.md", "Double Hidden")
    db.close()
    return vault


def make_private_default_vault(tmp: Path) -> Path:
    """Vault where default_visibility = private."""
    from synto.state import StateDB

    vault = tmp / "private_vault"
    (vault / "wiki").mkdir(parents=True)
    (vault / ".synto").mkdir()
    (vault / "raw").mkdir()
    (vault / "synto.toml").write_text(_base_toml() + '\n[mcp]\ndefault_visibility = "private"\n')
    (vault / "wiki" / "Explicit Public.md").write_text(
        "---\ntitle: Explicit Public\ntags: []\nvisibility: public\n---\n\nVisible.\n"
    )
    (vault / "wiki" / "No Visibility Field.md").write_text(
        "---\ntitle: No Visibility Field\ntags: []\n---\n\nShould be hidden.\n"
    )

    db = StateDB(vault / ".synto" / "state.db")
    _insert(db, "wiki/Explicit Public.md", "Explicit Public")
    _insert(db, "wiki/No Visibility Field.md", "No Visibility Field")
    db.close()
    return vault


def make_audit_vault(tmp: Path) -> Path:
    """Vault with audit = true."""
    from synto.state import StateDB

    vault = tmp / "audit_vault"
    (vault / "wiki").mkdir(parents=True)
    (vault / ".synto").mkdir()
    (vault / "raw").mkdir()
    (vault / "synto.toml").write_text(_base_toml() + "\n[mcp]\naudit = true\n")
    (vault / "wiki" / "Public.md").write_text(
        "---\ntitle: Public\ntags: []\nvisibility: public\n---\n\nBody.\n"
    )

    db = StateDB(vault / ".synto" / "state.db")
    _insert(db, "wiki/Public.md", "Public")
    db.close()
    return vault


def make_verbatim_vault(tmp: Path) -> Path:
    """Vault for verbatim source tool tests.

    Two source documents: book1 (CC-BY, permissive) and secret (proprietary, blocked).
    Three segments for book1; one for secret. One concept linked to two book1 segments.
    No explicit [mcp] config — default permissive_only mode is used.
    """
    from synto.state import StateDB

    vault = tmp / "verbatim_vault"
    (vault / "wiki").mkdir(parents=True)
    (vault / ".synto").mkdir()
    (vault / "raw").mkdir()
    (vault / "synto.toml").write_text(_base_toml())

    db = StateDB(vault / ".synto" / "state.db")

    # Open source — CC-BY license passes default permissive_only gate
    db._conn.execute(
        "INSERT INTO source_documents"
        " (id, source_type, origin_uri, title, imported_at, redistribution, license)"
        " VALUES ('book1', 'pdf', '/raw/book1.pdf', 'Open Book',"
        " '2024-01-01T00:00:00', 'unknown', 'CC-BY')"
    )
    segments = [
        (
            "book1:p:0:aa",
            "book1:p:0",
            0,
            "p:0",
            "h0",
            "Quantum mechanics describes the behaviour of matter at small scales.",
        ),
        (
            "book1:p:1:aa",
            "book1:p:1",
            1,
            "p:1",
            "h1",
            "Wave function collapse is the reduction of quantum superposition.",
        ),
        (
            "book1:p:2:aa",
            "book1:p:2",
            2,
            "p:2",
            "h2",
            "Entanglement is a correlation between quantum particles.",
        ),
    ]
    for seg_id, identity, ordinal, locator, chash, text in segments:
        db._conn.execute(
            "INSERT INTO source_segments"
            " (id, identity, ordinal, source_id, structural_locator, content_hash, text)"
            " VALUES (?, ?, ?, 'book1', ?, ?, ?)",
            (seg_id, identity, ordinal, locator, chash, text),
        )

    # Restricted source — proprietary license blocked under default mode
    db._conn.execute(
        "INSERT INTO source_documents"
        " (id, source_type, origin_uri, title, imported_at, redistribution, license)"
        " VALUES ('secret', 'pdf', '/raw/secret.pdf', 'Secret Book',"
        " '2024-01-01T00:00:00', 'unknown', 'proprietary')"
    )
    db._conn.execute(
        "INSERT INTO source_segments"
        " (id, identity, ordinal, source_id, structural_locator, content_hash, text)"
        " VALUES ('secret:p:0:aa', 'secret:p:0', 0, 'secret', 'p:0', 'hs0',"
        " 'Hidden content here')"
    )

    # Concept linked to two book1 segments at different confidences
    db._conn.execute("INSERT INTO concepts (name, source_path) VALUES ('Quantum', 'raw/book1.md')")
    db._conn.execute(
        "INSERT INTO concept_occurrences"
        " (concept_name, source_segment_id, ordinal, confidence)"
        " VALUES ('Quantum', 'book1:p:0:aa', 0, 0.9)"
    )
    db._conn.execute(
        "INSERT INTO concept_occurrences"
        " (concept_name, source_segment_id, ordinal, confidence)"
        " VALUES ('Quantum', 'book1:p:1:aa', 1, 0.6)"
    )
    db._conn.commit()
    db.close()
    return vault


# ── StdioServerParameters factory ────────────────────────────────────────────


def _params(vault: Path):
    from mcp.client.stdio import StdioServerParameters

    # Use the direct venv binary, NOT `uv run synto`.
    # `uv run` routes child stderr through its own stdout, which injects log
    # lines into the JSON-RPC stream and causes parse errors on the client.
    synto_bin = Path(sys.executable).parent / "synto"
    return StdioServerParameters(command=str(synto_bin), args=["serve", "--vault", str(vault)])


def _sc(res, key: str | None = None):
    """Extract from structuredContent, handling both wrapped and direct shapes."""
    sc = res.structuredContent
    if sc is None:
        return None
    if key:
        return sc.get(key)
    return sc


# ── Test suites ───────────────────────────────────────────────────────────────


async def suite_list_articles(vault: Path) -> None:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_params(vault)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            # ── tool list sanity ───────────────────────────────────────────
            tools = await s.list_tools()
            names = {t.name for t in tools.tools}
            check(
                "tool list has 12 expected names (v0.4.0)",
                names
                == {
                    "list_articles",
                    "read_article",
                    "find_concept",
                    "search_articles",
                    "get_concept",
                    "list_sources",
                    "trace_lineage",
                    "answer_question",
                    "read_source_segment",
                    "search_source_segments",
                    "get_source_passages",
                    "list_segments",
                },
                str(names),
            )

            # ── no filter: visibility ──────────────────────────────────────
            res = await s.call_tool("list_articles", {})
            articles = _sc(res, "result") or []
            check("list_articles no-filter: returns list", isinstance(articles, list))
            titles = [a["name"] for a in articles]
            check("list_articles: public article present", "Neural Networks" in titles, str(titles))
            check(
                "list_articles: private article absent", "Internal Notes" not in titles, str(titles)
            )
            check(
                "list_articles: no-visibility article present (default=public)",
                "Default Visibility" in titles,
                str(titles),
            )

            # ── response item structure ────────────────────────────────────
            if articles:
                item = next(a for a in articles if a["name"] == "Neural Networks")
                for field in ("id", "name", "path", "tags"):
                    check(
                        f"list_articles item has '{field}' field",
                        field in item,
                        str(list(item.keys())),
                    )
                # summary may be None for empty body, but key must exist
                check(
                    "list_articles item has 'summary' key",
                    "summary" in item,
                    str(list(item.keys())),
                )
                check("list_articles item tags is a list", isinstance(item.get("tags"), list))
                check(
                    "list_articles item id is a non-empty string",
                    isinstance(item.get("id"), str) and len(item["id"]) > 0,
                )

            # ── tag filter ─────────────────────────────────────────────────
            res = await s.call_tool("list_articles", {"tag": "ml"})
            tagged = _sc(res, "result") or []
            tagged_titles = [a["name"] for a in tagged]
            check(
                "tag filter: 'ml' article present",
                "Neural Networks" in tagged_titles,
                str(tagged_titles),
            )
            check(
                "tag filter: non-ml article absent",
                "Default Visibility" not in tagged_titles,
                str(tagged_titles),
            )
            check("tag filter: private article absent", "Internal Notes" not in tagged_titles)
            check(
                "tag filter: all results have the tag",
                all("ml" in a["tags"] for a in tagged),
                str(tagged),
            )

            # ── tag filter: no match → empty list, not error ───────────────
            res = await s.call_tool("list_articles", {"tag": "xyznonexistent_tag_12345"})
            empty = _sc(res, "result")
            check("tag filter: unknown tag → [] not error", empty == [], repr(empty))

            # ── contains filter ────────────────────────────────────────────
            res = await s.call_tool("list_articles", {"contains": "neural"})
            matched = _sc(res, "result") or []
            matched_titles = [a["name"] for a in matched]
            check(
                "contains filter: matching article present",
                "Neural Networks" in matched_titles,
                str(matched_titles),
            )
            check(
                "contains filter: non-matching article absent",
                "Default Visibility" not in matched_titles,
                str(matched_titles),
            )

            # ── contains filter: no match → empty list, not error ──────────
            res = await s.call_tool("list_articles", {"contains": "xyznothing_12345"})
            empty2 = _sc(res, "result")
            check("contains filter: unknown term → [] not error", empty2 == [], repr(empty2))


async def suite_read_article(vault: Path) -> None:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_params(vault)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            # ── happy path: full response structure ────────────────────────
            res = await s.call_tool("read_article", {"name_or_id": "Neural Networks"})
            check("read_article success: not isError", not res.isError)
            art = _sc(res)
            for field in ("id", "name", "path", "body", "frontmatter"):
                check(
                    f"read_article response has '{field}'",
                    art is not None and field in art,
                    str(list(art.keys()) if art else None),
                )
            if art:
                check(
                    "read_article name matches request",
                    art["name"] == "Neural Networks",
                    repr(art["name"]),
                )
                check("read_article body contains expected text", "weights" in art.get("body", ""))
                check(
                    "read_article frontmatter is a dict", isinstance(art.get("frontmatter"), dict)
                )
                check(
                    "read_article frontmatter has visibility",
                    "visibility" in art.get("frontmatter", {}),
                    str(art.get("frontmatter", {}).keys()),
                )
                check(
                    "read_article id is a non-empty string",
                    isinstance(art.get("id"), str) and len(art["id"]) > 0,
                )

            # ── case-insensitive lookup ────────────────────────────────────
            res_lower = await s.call_tool("read_article", {"name_or_id": "neural networks"})
            check(
                "read_article case-insensitive lookup works",
                not res_lower.isError,
                res_lower.content[0].text[:60] if res_lower.content else "empty",
            )

            # ── private article: indistinguishable from nonexistent ────────
            res_priv = await s.call_tool("read_article", {"name_or_id": "Internal Notes"})
            res_noex = await s.call_tool("read_article", {"name_or_id": "XYZ_does_not_exist_abc"})
            check("read_article private → isError", bool(res_priv.isError))
            check("read_article nonexistent → isError", bool(res_noex.isError))

            # Both errors should use the same message template — no hint of existence
            priv_text = res_priv.content[0].text if res_priv.content else ""
            noex_text = res_noex.content[0].text if res_noex.content else ""
            check(
                "read_article private error: does not say 'private' or 'hidden'",
                "private" not in priv_text.lower() and "hidden" not in priv_text.lower(),
                repr(priv_text[:100]),
            )
            check(
                "read_article private error format matches nonexistent format",
                "No article:" in priv_text and "No article:" in noex_text,
                f"priv={priv_text[:60]!r}  noex={noex_text[:60]!r}",
            )


async def suite_find_concept(vault: Path) -> None:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_params(vault)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            # ── alias lookup (first alias) ─────────────────────────────────
            res = await s.call_tool("find_concept", {"query": "NN"})
            data = _sc(res, "result")
            check("find_concept first alias: non-null", data is not None, repr(data))
            if data:
                check(
                    "find_concept first alias: correct name",
                    data.get("name") == "Neural Networks",
                    repr(data.get("name")),
                )
                check(
                    "find_concept first alias: canonical_article_id set",
                    bool(data.get("canonical_article_id")),
                    repr(data.get("canonical_article_id")),
                )
                aliases = data.get("aliases", [])
                check(
                    "find_concept: aliases list contains first alias", "NN" in aliases, str(aliases)
                )
                check(
                    "find_concept: aliases list contains second alias",
                    "neural net" in aliases,
                    str(aliases),
                )

            # ── alias lookup (second alias) ────────────────────────────────
            res2 = await s.call_tool("find_concept", {"query": "neural net"})
            data2 = _sc(res2, "result")
            check("find_concept second alias: non-null", data2 is not None, repr(data2))
            if data2:
                check(
                    "find_concept second alias: same canonical name",
                    data2.get("name") == "Neural Networks",
                    repr(data2.get("name")),
                )

            # ── canonical name lookup ──────────────────────────────────────
            res3 = await s.call_tool("find_concept", {"query": "Neural Networks"})
            data3 = _sc(res3, "result")
            check("find_concept canonical name: non-null", data3 is not None)
            if data3:
                check(
                    "find_concept canonical name: id matches alias lookup id",
                    data3.get("canonical_article_id") == (data or {}).get("canonical_article_id"),
                )

            # ── unknown query → null ───────────────────────────────────────
            res4 = await s.call_tool("find_concept", {"query": "XYZ_totally_unknown_55555"})
            data4 = _sc(res4, "result")
            check(
                "find_concept unknown: null, not error",
                data4 is None and not res4.isError,
                f"data={repr(data4)} isError={res4.isError}",
            )

            # ── concept with private canonical article → null ──────────────
            res5 = await s.call_tool("find_concept", {"query": "Internal Notes"})
            data5 = _sc(res5, "result")
            check("find_concept private canonical article → null", data5 is None, repr(data5))


async def suite_exclude_tags(vault: Path) -> None:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_params(vault)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            # ── list_articles: excluded article absent ─────────────────────
            res = await s.call_tool("list_articles", {})
            articles = _sc(res, "result") or []
            titles = [a["name"] for a in articles]
            check("exclude_tags: open article visible", "Open Article" in titles, str(titles))
            check(
                "exclude_tags: secret-tagged article absent",
                "Secret Article" not in titles,
                str(titles),
            )
            check("exclude_tags: double-hidden absent", "Double Hidden" not in titles, str(titles))

            # ── list_articles tag="general": excluded article still absent ──
            # "Secret Article" has tag "general" AND "secret"; even when filtering
            # by "general", the exclude_tags="secret" rule must still hide it.
            res_gen = await s.call_tool("list_articles", {"tag": "general"})
            gen_titles = [a["name"] for a in (_sc(res_gen, "result") or [])]
            check(
                "exclude_tags + tag filter: open article with 'general' tag present",
                "Open Article" in gen_titles,
                str(gen_titles),
            )
            check(
                "exclude_tags + tag filter: secret article absent even with matching tag",
                "Secret Article" not in gen_titles,
                str(gen_titles),
            )

            # ── read_article: excluded article looks nonexistent ───────────
            res_sec = await s.call_tool("read_article", {"name_or_id": "Secret Article"})
            check(
                "exclude_tags: read_article on excluded → isError",
                bool(res_sec.isError),
                repr(res_sec.content[0].text[:80] if res_sec.content else "empty"),
            )

            # ── read_article: non-excluded article works ───────────────────
            res_open = await s.call_tool("read_article", {"name_or_id": "Open Article"})
            check("exclude_tags: read_article on open article works", not res_open.isError)


async def suite_default_visibility_private(vault: Path) -> None:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_params(vault)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            res = await s.call_tool("list_articles", {})
            articles = _sc(res, "result") or []
            titles = [a["name"] for a in articles]
            check(
                "default=private: explicit public visible", "Explicit Public" in titles, str(titles)
            )
            check(
                "default=private: no-visibility article hidden",
                "No Visibility Field" not in titles,
                str(titles),
            )
            check("default=private: list contains exactly 1 article", len(titles) == 1, str(titles))

            # Verify read_article for the hidden article also returns an error
            res_hid = await s.call_tool("read_article", {"name_or_id": "No Visibility Field"})
            check(
                "default=private: read_article on no-visibility article → isError",
                bool(res_hid.isError),
            )

            # And the explicitly public one is readable
            res_pub = await s.call_tool("read_article", {"name_or_id": "Explicit Public"})
            check("default=private: read_article on explicit public works", not res_pub.isError)


async def suite_audit(vault: Path) -> None:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_params(vault)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            await s.call_tool("list_articles", {})
            await s.call_tool("read_article", {"name_or_id": "Public"})
            # A failed call — nonexistent article — should also be logged with success=false
            await s.call_tool("read_article", {"name_or_id": "XYZ_nonexistent_zzz"})

    # Let child process exit and flush DB
    time.sleep(0.5)

    db_path = vault / ".synto" / "state.db"
    check("audit DB exists", db_path.exists())
    if not db_path.exists():
        return

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT event_type, metadata_json, latency_ms, success "
        "FROM metric_events WHERE event_type='mcp_call' ORDER BY id"
    ).fetchall()
    con.close()

    check(
        "audit: 3 mcp_call rows (list + read_success + read_fail)",
        len(rows) == 3,
        f"got {len(rows)}",
    )

    if len(rows) >= 3:
        tools_logged = [json.loads(r["metadata_json"])["tool"] for r in rows]
        check("audit: list_articles logged", "list_articles" in tools_logged, str(tools_logged))
        check("audit: read_article logged", "read_article" in tools_logged, str(tools_logged))

        # First row: list_articles, success=1
        p0 = json.loads(rows[0]["metadata_json"])
        check("audit row 0: tool=list_articles", p0["tool"] == "list_articles", str(p0))
        check("audit row 0: success=1", rows[0]["success"] == 1, f"got {rows[0]['success']!r}")
        check(
            "audit row 0: latency_ms ≥ 0",
            isinstance(rows[0]["latency_ms"], int) and rows[0]["latency_ms"] >= 0,
            str(rows[0]["latency_ms"]),
        )

        # Second row: read_article success
        p1 = json.loads(rows[1]["metadata_json"])
        check("audit row 1: tool=read_article", p1["tool"] == "read_article", str(p1))
        check("audit row 1: success=1", rows[1]["success"] == 1, str(rows[1]["success"]))

        # Third row: read_article failure — success must be 0
        p2 = json.loads(rows[2]["metadata_json"])
        check("audit row 2: tool=read_article", p2["tool"] == "read_article", str(p2))
        check(
            "audit row 2: failed call logged with success=0",
            rows[2]["success"] == 0,
            f"got {rows[2]['success']!r}",
        )

        # All rows: string args are hashed; scalars (bool/int/float) pass through.
        # v0.3.0 change: _hash_args no longer hashes low-cardinality scalars.
        for i, row in enumerate(rows):
            payload = json.loads(row["metadata_json"])
            check(
                f"audit row {i}: args hashed or scalar",
                all(
                    v is None
                    or isinstance(v, (bool, int, float))
                    or (isinstance(v, str) and len(v) == 8)
                    for v in payload.get("args", {}).values()
                ),
                str(payload.get("args")),
            )


async def suite_audit_disabled(vault: Path) -> None:
    """Verify that audit=false (the default) writes zero metric rows."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_params(vault)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            await s.call_tool("list_articles", {})
            await s.call_tool("read_article", {"name_or_id": "Neural Networks"})

    time.sleep(0.3)

    db_path = vault / ".synto" / "state.db"
    # The vault was bootstrapped with a state DB (for VaultReader), so it exists
    if db_path.exists():
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT COUNT(*) AS n FROM metric_events WHERE event_type='mcp_call'"
        ).fetchone()
        con.close()
        check("audit=false: zero mcp_call rows written", rows["n"] == 0, f"got {rows['n']} rows")
    else:
        # No DB at all is also acceptable (no audit = no DB creation by server)
        check("audit=false: no state DB created by server (audit writes skipped)", True)


async def suite_verbatim_source_tools(vault: Path) -> None:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_params(vault)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            # ── read_source_segment: happy path ────────────────────────────
            res = await s.call_tool("read_source_segment", {"segment_id": "book1:p:0:aa"})
            check(
                "read_source_segment: not isError",
                not res.isError,
                res.content[0].text[:80] if res.content else "empty",
            )
            data = _sc(res)
            for field in ("segment_id", "source_id", "body", "source_path", "truncated"):
                check(
                    f"read_source_segment response has '{field}'",
                    data is not None and field in data,
                    str(list(data.keys()) if data else None),
                )
            if data:
                check(
                    "read_source_segment: body non-empty",
                    isinstance(data.get("body"), str) and len(data["body"]) > 0,
                )
                check(
                    "read_source_segment: source_path is origin_uri",
                    data.get("source_path") == "/raw/book1.pdf",
                    repr(data.get("source_path")),
                )
                check(
                    "read_source_segment: truncated=False for full read",
                    data.get("truncated") is False,
                    repr(data.get("truncated")),
                )

            # ── read_source_segment: truncation ────────────────────────────
            res_trunc = await s.call_tool(
                "read_source_segment", {"segment_id": "book1:p:0:aa", "max_chars": 5}
            )
            check("read_source_segment max_chars: not isError", not res_trunc.isError)
            td = _sc(res_trunc)
            if td:
                check(
                    "read_source_segment max_chars: body truncated",
                    len(td.get("body", "")) <= 6,  # 5 chars + ellipsis
                    repr(td.get("body")),
                )
                check("read_source_segment max_chars: truncated=True", td.get("truncated") is True)

            # ── read_source_segment: unknown id ────────────────────────────
            res_unk = await s.call_tool("read_source_segment", {"segment_id": "nonexistent:id"})
            check("read_source_segment unknown id: isError", bool(res_unk.isError))

            # ── read_source_segment: policy blocked ────────────────────────
            res_pol = await s.call_tool("read_source_segment", {"segment_id": "secret:p:0:aa"})
            check("read_source_segment proprietary source: isError (policy)", bool(res_pol.isError))

            # ── search_source_segments: basic BM25 ────────────────────────
            res_fts = await s.call_tool("search_source_segments", {"query": "quantum"})
            check("search_source_segments: not isError", not res_fts.isError)
            fts = _sc(res_fts)
            results = fts.get("results", []) if fts else []
            check(
                "search_source_segments: results non-empty",
                len(results) > 0,
                f"got {len(results)} results",
            )
            if results:
                for field in ("segment_id", "snippet", "score"):
                    check(
                        f"search_source_segments result has '{field}'",
                        field in results[0],
                        str(list(results[0].keys())),
                    )
                scores = [r["score"] for r in results]
                check(
                    "search_source_segments: scores descending",
                    scores == sorted(scores, reverse=True),
                    str(scores),
                )

            # ── search_source_segments: privacy gate ──────────────────────
            res_hid = await s.call_tool("search_source_segments", {"query": "hidden content"})
            check("search_source_segments hidden query: not isError", not res_hid.isError)
            hd = _sc(res_hid)
            if hd:
                check(
                    "search_source_segments: proprietary segment filtered",
                    hd.get("results") == [],
                    repr(hd.get("results")),
                )
                check(
                    "search_source_segments: hidden_by_policy == 1",
                    hd.get("hidden_by_policy") == 1,
                    repr(hd.get("hidden_by_policy")),
                )

            # ── search_source_segments: empty query → error ────────────────
            res_empty = await s.call_tool("search_source_segments", {"query": ""})
            check("search_source_segments empty query: isError", bool(res_empty.isError))

            # ── search_source_segments: FTS5 special chars ────────────────
            res_sp = await s.call_tool(
                "search_source_segments", {"query": '"quoted":value AND OR NOT'}
            )
            check(
                "search_source_segments: special chars do not raise",
                not res_sp.isError,
                res_sp.content[0].text[:80] if res_sp.isError and res_sp.content else "",
            )

            # ── get_source_passages: known concept ─────────────────────────
            res_gsp = await s.call_tool("get_source_passages", {"concept_name": "Quantum"})
            check("get_source_passages: not isError", not res_gsp.isError)
            gsp = _sc(res_gsp)
            passages = gsp.get("results", []) if gsp else []
            check(
                "get_source_passages: 2 passages returned",
                len(passages) == 2,
                f"got {len(passages)}",
            )
            if passages:
                for field in ("segment_id", "source_id", "body", "confidence"):
                    check(
                        f"get_source_passages result has '{field}'",
                        field in passages[0],
                        str(list(passages[0].keys())),
                    )
                check(
                    "get_source_passages: first confidence ≈ 0.9",
                    abs(passages[0].get("confidence", 0) - 0.9) < 0.01,
                    repr(passages[0].get("confidence")),
                )

            # ── get_source_passages: max_passages=1 ───────────────────────
            res_mp = await s.call_tool(
                "get_source_passages", {"concept_name": "Quantum", "max_passages": 1}
            )
            check("get_source_passages max_passages=1: not isError", not res_mp.isError)
            mp = _sc(res_mp)
            if mp:
                check(
                    "get_source_passages max_passages=1: exactly 1 result",
                    len(mp.get("results", [])) == 1,
                    str(len(mp.get("results", []))),
                )

            # ── get_source_passages: unknown concept → empty, not error ───
            res_unk2 = await s.call_tool(
                "get_source_passages", {"concept_name": "XYZ_no_concept_99999"}
            )
            check("get_source_passages unknown concept: not isError", not res_unk2.isError)
            uk2 = _sc(res_unk2)
            if uk2:
                check(
                    "get_source_passages unknown concept: empty results",
                    uk2.get("results") == [],
                    repr(uk2.get("results")),
                )

            # ── list_segments: basic enumeration ──────────────────────────
            res_ls = await s.call_tool("list_segments", {"source_id": "book1"})
            check("list_segments: not isError", not res_ls.isError)
            ls = _sc(res_ls)
            if ls:
                check("list_segments: total == 3", ls.get("total") == 3, repr(ls.get("total")))
                check(
                    "list_segments: returned == 3",
                    ls.get("returned") == 3,
                    repr(ls.get("returned")),
                )
                segs = ls.get("segments", [])
                check(
                    "list_segments: segments ordered by ordinal",
                    [s["ordinal"] for s in segs] == sorted(s["ordinal"] for s in segs),
                )
                if segs:
                    for field in ("segment_id", "ordinal", "length"):
                        check(
                            f"list_segments segment has '{field}'",
                            field in segs[0],
                            str(list(segs[0].keys())),
                        )

            # ── list_segments: pagination ──────────────────────────────────
            res_pg = await s.call_tool(
                "list_segments", {"source_id": "book1", "limit": 2, "offset": 1}
            )
            check("list_segments pagination: not isError", not res_pg.isError)
            pg = _sc(res_pg)
            if pg:
                check(
                    "list_segments pagination: total still 3",
                    pg.get("total") == 3,
                    repr(pg.get("total")),
                )
                check(
                    "list_segments pagination: returned == 2",
                    pg.get("returned") == 2,
                    repr(pg.get("returned")),
                )

            # ── list_segments: unknown source → error ─────────────────────
            res_nox = await s.call_tool("list_segments", {"source_id": "nope_src"})
            check("list_segments unknown source: isError", bool(res_nox.isError))

            # ── list_segments: proprietary source → policy error ──────────
            res_pol2 = await s.call_tool("list_segments", {"source_id": "secret"})
            check("list_segments proprietary source: isError (policy)", bool(res_pol2.isError))


# ── Suite registry ────────────────────────────────────────────────────────────

_VAULT_FACTORIES = {
    "main": make_main_vault,
    "exclude": make_exclude_tags_vault,
    "private": make_private_default_vault,
    "audit": make_audit_vault,
    "verbatim": make_verbatim_vault,
}

SUITES: dict[str, tuple] = {
    "list_articles": (suite_list_articles, "main"),
    "read_article": (suite_read_article, "main"),
    "find_concept": (suite_find_concept, "main"),
    "exclude_tags": (suite_exclude_tags, "exclude"),
    "default_visibility_private": (suite_default_visibility_private, "private"),
    "audit": (suite_audit, "audit"),
    "audit_disabled": (suite_audit_disabled, "main"),
    "verbatim_source_tools": (suite_verbatim_source_tools, "verbatim"),
}


# ── Runner ────────────────────────────────────────────────────────────────────


async def _run(name: str, coro) -> None:
    """Run one suite with timeout, converting exceptions to FAIL checks."""
    global _current_suite
    _current_suite = name
    header(name)
    try:
        await asyncio.wait_for(coro, timeout=30.0)
    except TimeoutError:
        check("suite timed out after 30s", False)
    except Exception as exc:
        check("suite raised exception", False, type(exc).__name__ + ": " + str(exc))


async def run_all(suite_filter: str | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="mcp-smoke-") as tmp:
        tmp_path = Path(tmp)

        if suite_filter is not None:
            fn, vault_key = SUITES[suite_filter]
            vault = _VAULT_FACTORIES[vault_key](tmp_path)
            await _run(suite_filter, fn(vault))
            return

        # Full run — build each vault once, reuse across suites that share it
        vaults = {key: factory(tmp_path) for key, factory in _VAULT_FACTORIES.items()}
        for name, (fn, vault_key) in SUITES.items():
            await _run(name, fn(vaults[vault_key]))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    global _JSON_MODE

    parser = argparse.ArgumentParser(
        description="MCP protocol smoke tests for synto serve.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        action="store_true",
        help="Print results as a single JSON object to stdout (suppresses streaming output).",
    )
    parser.add_argument(
        "--suite",
        metavar="NAME",
        help="Run a single named suite. Use 'list' to print available names.",
    )
    args = parser.parse_args()

    if args.suite == "list":
        print("\n".join(SUITES))
        return 0

    if args.suite and args.suite not in SUITES:
        print(
            f"Unknown suite {args.suite!r}.\nAvailable: {', '.join(SUITES)}",
            file=sys.stderr,
        )
        return 1

    _JSON_MODE = args.json_out
    t0 = time.monotonic()

    try:
        asyncio.run(run_all(args.suite))
    except Exception:
        if not _JSON_MODE:
            traceback.print_exc()
        return 1

    duration = int(time.monotonic() - t0)
    passed = sum(1 for r in _results if r["passed"])
    failed = len(_results) - passed

    report = {
        "passed": passed,
        "failed": failed,
        "duration_s": duration,
        "checks": _results,
    }

    report_file = os.environ.get("REPORT_FILE")
    if report_file:
        Path(report_file).write_text(json.dumps(report, indent=2))

    if _JSON_MODE:
        print(json.dumps(report))
        return 1 if failed else 0

    print(f"\n{'─' * 50}")
    if _failures:
        print(f"{_c(_RED)}{len(_failures)} FAILED:{_c(_RST)}")
        for f in _failures:
            print(f"  • {f}")
        return 1

    print(f"{_c(_GREEN)}All checks passed.{_c(_RST)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
