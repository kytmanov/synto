#!/usr/bin/env python
"""Standalone MCP protocol smoke test.

Creates isolated vaults, bootstraps state DBs, starts `synto serve` as a
subprocess per suite, and exercises every tool + edge case over stdio.

Exit 0 = all checks passed.
Exit 1 = at least one check failed.

NOTE: Use the direct synto binary (not `uv run synto`).  When launched via
`uv run`, uv routes child stderr through its own stdout, injecting log lines
into the JSON-RPC stream and causing parse errors on the client side.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── Output helpers ────────────────────────────────────────────────────────────

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
HEAD = "\033[1m"
RST = "\033[0m"

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  {PASS} {label}")
    else:
        msg = label + (f": {detail}" if detail else "")
        print(f"  {FAIL} {msg}")
        _failures.append(msg)


def header(title: str) -> None:
    print(f"\n{HEAD}{title}{RST}")


# ── Vault bootstrap ───────────────────────────────────────────────────────────

def _base_toml() -> str:
    return (
        "[models]\nfast = \"dummy\"\nheavy = \"dummy\"\n\n"
        "[provider]\nname = \"ollama\"\nurl = \"http://localhost:11434\"\n"
    )


def _insert(db, path: str, title: str, is_draft: bool = False) -> None:
    from synto.models import WikiArticleRecord
    db.upsert_article(WikiArticleRecord(
        path=path, title=title, sources=["raw/note.md"],
        content_hash=f"h-{title}", created_at=datetime.now(),
        updated_at=datetime.now(), is_draft=is_draft,
    ))


def make_main_vault(tmp: Path) -> Path:
    """Primary test vault: public + private + default-visibility articles, concepts."""
    from synto.state import StateDB

    vault = tmp / "vault"
    (vault / "wiki").mkdir(parents=True)
    (vault / ".synto").mkdir()
    (vault / "raw").mkdir()
    (vault / "synto.toml").write_text(_base_toml())

    (vault / "wiki" / "Neural Networks.md").write_text(
        "---\ntitle: Neural Networks\ntags: [\"ml\"]\nvisibility: public\n---\n\n"
        "Neural networks learn by adjusting weights through backpropagation.\n"
    )
    (vault / "wiki" / "Internal Notes.md").write_text(
        "---\ntitle: Internal Notes\ntags: []\nvisibility: private\n---\n\nInternal only.\n"
    )
    # No visibility field → default_visibility="public" (server default) → visible
    (vault / "wiki" / "Default Visibility.md").write_text(
        "---\ntitle: Default Visibility\ntags: [\"default\"]\n---\n\n"
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
    (vault / "synto.toml").write_text(
        _base_toml() + "\n[mcp]\nexclude_tags = [\"secret\"]\n"
    )

    (vault / "wiki" / "Open Article.md").write_text(
        "---\ntitle: Open Article\ntags: [\"general\"]\nvisibility: public\n---\n\nVisible body.\n"
    )
    (vault / "wiki" / "Secret Article.md").write_text(
        "---\ntitle: Secret Article\ntags: [\"secret\", \"general\"]\nvisibility: public\n---\n\nHidden body.\n"
    )
    # An article that is both tagged secret AND private — double-hidden
    (vault / "wiki" / "Double Hidden.md").write_text(
        "---\ntitle: Double Hidden\ntags: [\"secret\"]\nvisibility: private\n---\n\nDouble hidden.\n"
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
    (vault / "synto.toml").write_text(
        _base_toml() + "\n[mcp]\ndefault_visibility = \"private\"\n"
    )
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
    header("list_articles")

    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_params(vault)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            # ── tool list sanity ───────────────────────────────────────────
            tools = await s.list_tools()
            names = {t.name for t in tools.tools}
            check("tool list has exactly 3 expected names",
                  names == {"list_articles", "read_article", "find_concept"}, str(names))

            # ── no filter: visibility ──────────────────────────────────────
            res = await s.call_tool("list_articles", {})
            articles = _sc(res, "result") or []
            check("list_articles no-filter: returns list", isinstance(articles, list))
            titles = [a["name"] for a in articles]
            check("list_articles: public article present", "Neural Networks" in titles, str(titles))
            check("list_articles: private article absent", "Internal Notes" not in titles, str(titles))
            check("list_articles: no-visibility article present (default=public)",
                  "Default Visibility" in titles, str(titles))

            # ── response item structure ────────────────────────────────────
            if articles:
                item = next(a for a in articles if a["name"] == "Neural Networks")
                for field in ("id", "name", "path", "tags"):
                    check(f"list_articles item has '{field}' field", field in item, str(list(item.keys())))
                # summary may be None for empty body, but key must exist
                check("list_articles item has 'summary' key", "summary" in item, str(list(item.keys())))
                check("list_articles item tags is a list", isinstance(item.get("tags"), list))
                check("list_articles item id is a non-empty string",
                      isinstance(item.get("id"), str) and len(item["id"]) > 0)

            # ── tag filter ─────────────────────────────────────────────────
            res = await s.call_tool("list_articles", {"tag": "ml"})
            tagged = _sc(res, "result") or []
            tagged_titles = [a["name"] for a in tagged]
            check("tag filter: 'ml' article present", "Neural Networks" in tagged_titles, str(tagged_titles))
            check("tag filter: non-ml article absent",
                  "Default Visibility" not in tagged_titles, str(tagged_titles))
            check("tag filter: private article absent", "Internal Notes" not in tagged_titles)
            check("tag filter: all results have the tag",
                  all("ml" in a["tags"] for a in tagged), str(tagged))

            # ── tag filter: no match → empty list, not error ───────────────
            res = await s.call_tool("list_articles", {"tag": "xyznonexistent_tag_12345"})
            empty = _sc(res, "result")
            check("tag filter: unknown tag → [] not error", empty == [], repr(empty))

            # ── contains filter ────────────────────────────────────────────
            res = await s.call_tool("list_articles", {"contains": "neural"})
            matched = _sc(res, "result") or []
            matched_titles = [a["name"] for a in matched]
            check("contains filter: matching article present",
                  "Neural Networks" in matched_titles, str(matched_titles))
            check("contains filter: non-matching article absent",
                  "Default Visibility" not in matched_titles, str(matched_titles))

            # ── contains filter: no match → empty list, not error ──────────
            res = await s.call_tool("list_articles", {"contains": "xyznothing_12345"})
            empty2 = _sc(res, "result")
            check("contains filter: unknown term → [] not error", empty2 == [], repr(empty2))


async def suite_read_article(vault: Path) -> None:
    header("read_article")

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
                check(f"read_article response has '{field}'",
                      art is not None and field in art, str(list(art.keys()) if art else None))
            if art:
                check("read_article name matches request", art["name"] == "Neural Networks",
                      repr(art["name"]))
                check("read_article body contains expected text",
                      "weights" in art.get("body", ""))
                check("read_article frontmatter is a dict",
                      isinstance(art.get("frontmatter"), dict))
                check("read_article frontmatter has visibility",
                      "visibility" in art.get("frontmatter", {}),
                      str(art.get("frontmatter", {}).keys()))
                check("read_article id is a non-empty string",
                      isinstance(art.get("id"), str) and len(art["id"]) > 0)

            # ── case-insensitive lookup ────────────────────────────────────
            res_lower = await s.call_tool("read_article", {"name_or_id": "neural networks"})
            check("read_article case-insensitive lookup works", not res_lower.isError,
                  res_lower.content[0].text[:60] if res_lower.content else "empty")

            # ── private article: indistinguishable from nonexistent ────────
            res_priv = await s.call_tool("read_article", {"name_or_id": "Internal Notes"})
            res_noex = await s.call_tool("read_article", {"name_or_id": "XYZ_does_not_exist_abc"})
            check("read_article private → isError", bool(res_priv.isError))
            check("read_article nonexistent → isError", bool(res_noex.isError))

            # Both errors should use the same message template — no hint of existence
            priv_text = res_priv.content[0].text if res_priv.content else ""
            noex_text = res_noex.content[0].text if res_noex.content else ""
            check("read_article private error: does not say 'private' or 'hidden'",
                  "private" not in priv_text.lower() and "hidden" not in priv_text.lower(),
                  repr(priv_text[:100]))
            check("read_article private error format matches nonexistent format",
                  "No article:" in priv_text and "No article:" in noex_text,
                  f"priv={priv_text[:60]!r}  noex={noex_text[:60]!r}")


async def suite_find_concept(vault: Path) -> None:
    header("find_concept")

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
                check("find_concept first alias: correct name",
                      data.get("name") == "Neural Networks", repr(data.get("name")))
                check("find_concept first alias: canonical_article_id set",
                      bool(data.get("canonical_article_id")), repr(data.get("canonical_article_id")))
                aliases = data.get("aliases", [])
                check("find_concept: aliases list contains first alias",
                      "NN" in aliases, str(aliases))
                check("find_concept: aliases list contains second alias",
                      "neural net" in aliases, str(aliases))

            # ── alias lookup (second alias) ────────────────────────────────
            res2 = await s.call_tool("find_concept", {"query": "neural net"})
            data2 = _sc(res2, "result")
            check("find_concept second alias: non-null", data2 is not None, repr(data2))
            if data2:
                check("find_concept second alias: same canonical name",
                      data2.get("name") == "Neural Networks", repr(data2.get("name")))

            # ── canonical name lookup ──────────────────────────────────────
            res3 = await s.call_tool("find_concept", {"query": "Neural Networks"})
            data3 = _sc(res3, "result")
            check("find_concept canonical name: non-null", data3 is not None)
            if data3:
                check("find_concept canonical name: id matches alias lookup id",
                      data3.get("canonical_article_id") == (data or {}).get("canonical_article_id"))

            # ── unknown query → null ───────────────────────────────────────
            res4 = await s.call_tool("find_concept", {"query": "XYZ_totally_unknown_55555"})
            data4 = _sc(res4, "result")
            check("find_concept unknown: null, not error",
                  data4 is None and not res4.isError,
                  f"data={repr(data4)} isError={res4.isError}")

            # ── concept with private canonical article → null ──────────────
            res5 = await s.call_tool("find_concept", {"query": "Internal Notes"})
            data5 = _sc(res5, "result")
            check("find_concept private canonical article → null",
                  data5 is None, repr(data5))


async def suite_exclude_tags(vault: Path) -> None:
    header("exclude_tags filtering")

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
            check("exclude_tags: secret-tagged article absent",
                  "Secret Article" not in titles, str(titles))
            check("exclude_tags: double-hidden absent", "Double Hidden" not in titles, str(titles))

            # ── list_articles tag="general": excluded article still absent ──
            # "Secret Article" has tag "general" AND "secret"; even when filtering
            # by "general", the exclude_tags="secret" rule must still hide it.
            res_gen = await s.call_tool("list_articles", {"tag": "general"})
            gen_titles = [a["name"] for a in (_sc(res_gen, "result") or [])]
            check("exclude_tags + tag filter: open article with 'general' tag present",
                  "Open Article" in gen_titles, str(gen_titles))
            check("exclude_tags + tag filter: secret article absent even with matching tag",
                  "Secret Article" not in gen_titles, str(gen_titles))

            # ── read_article: excluded article looks nonexistent ───────────
            res_sec = await s.call_tool("read_article", {"name_or_id": "Secret Article"})
            check("exclude_tags: read_article on excluded → isError", bool(res_sec.isError),
                  repr(res_sec.content[0].text[:80] if res_sec.content else "empty"))

            # ── read_article: non-excluded article works ───────────────────
            res_open = await s.call_tool("read_article", {"name_or_id": "Open Article"})
            check("exclude_tags: read_article on open article works", not res_open.isError)


async def suite_default_visibility_private(vault: Path) -> None:
    header("default_visibility = private")

    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_params(vault)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            res = await s.call_tool("list_articles", {})
            articles = _sc(res, "result") or []
            titles = [a["name"] for a in articles]
            check("default=private: explicit public visible", "Explicit Public" in titles, str(titles))
            check("default=private: no-visibility article hidden",
                  "No Visibility Field" not in titles, str(titles))
            check("default=private: list contains exactly 1 article",
                  len(titles) == 1, str(titles))

            # Verify read_article for the hidden article also returns an error
            res_hid = await s.call_tool("read_article", {"name_or_id": "No Visibility Field"})
            check("default=private: read_article on no-visibility article → isError",
                  bool(res_hid.isError))

            # And the explicitly public one is readable
            res_pub = await s.call_tool("read_article", {"name_or_id": "Explicit Public"})
            check("default=private: read_article on explicit public works", not res_pub.isError)


async def suite_audit(vault: Path) -> None:
    header("Audit logging (enabled)")

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

    check("audit: 3 mcp_call rows (list + read_success + read_fail)",
          len(rows) == 3, f"got {len(rows)}")

    if len(rows) >= 3:
        tools_logged = [json.loads(r["metadata_json"])["tool"] for r in rows]
        check("audit: list_articles logged", "list_articles" in tools_logged, str(tools_logged))
        check("audit: read_article logged", "read_article" in tools_logged, str(tools_logged))

        # First row: list_articles, success=1
        p0 = json.loads(rows[0]["metadata_json"])
        check("audit row 0: tool=list_articles", p0["tool"] == "list_articles", str(p0))
        check("audit row 0: success=1", rows[0]["success"] == 1,
              f"got {rows[0]['success']!r}")
        check("audit row 0: latency_ms ≥ 0",
              isinstance(rows[0]["latency_ms"], int) and rows[0]["latency_ms"] >= 0,
              str(rows[0]["latency_ms"]))

        # Second row: read_article success
        p1 = json.loads(rows[1]["metadata_json"])
        check("audit row 1: tool=read_article", p1["tool"] == "read_article", str(p1))
        check("audit row 1: success=1", rows[1]["success"] == 1, str(rows[1]["success"]))

        # Third row: read_article failure — success must be 0
        p2 = json.loads(rows[2]["metadata_json"])
        check("audit row 2: tool=read_article", p2["tool"] == "read_article", str(p2))
        check("audit row 2: failed call logged with success=0",
              rows[2]["success"] == 0, f"got {rows[2]['success']!r}")

        # All rows: args are hashed not plaintext
        for i, row in enumerate(rows):
            payload = json.loads(row["metadata_json"])
            check(f"audit row {i}: args hashed (8-char hex or None)",
                  all(
                      v is None or (isinstance(v, str) and len(v) == 8)
                      for v in payload.get("args", {}).values()
                  ), str(payload.get("args")))


async def suite_audit_disabled(vault: Path) -> None:
    """Verify that audit=false (the default) writes zero metric rows."""
    header("Audit disabled by default")

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
        check("audit=false: zero mcp_call rows written", rows["n"] == 0,
              f"got {rows['n']} rows")
    else:
        # No DB at all is also acceptable (no audit = no DB creation by server)
        check("audit=false: no state DB created by server (audit writes skipped)", True)


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_all() -> None:
    with tempfile.TemporaryDirectory(prefix="mcp-smoke-") as tmp:
        tmp_path = Path(tmp)

        main_vault = make_main_vault(tmp_path)
        excl_vault = make_exclude_tags_vault(tmp_path)
        priv_vault = make_private_default_vault(tmp_path)
        audit_vault = make_audit_vault(tmp_path)

        await asyncio.wait_for(suite_list_articles(main_vault), timeout=30.0)
        await asyncio.wait_for(suite_read_article(main_vault), timeout=30.0)
        await asyncio.wait_for(suite_find_concept(main_vault), timeout=30.0)
        await asyncio.wait_for(suite_exclude_tags(excl_vault), timeout=30.0)
        await asyncio.wait_for(suite_default_visibility_private(priv_vault), timeout=30.0)
        await asyncio.wait_for(suite_audit(audit_vault), timeout=30.0)
        await asyncio.wait_for(suite_audit_disabled(main_vault), timeout=30.0)


def main() -> int:
    try:
        asyncio.run(run_all())
    except Exception:
        traceback.print_exc()
        return 1

    print(f"\n{'─' * 50}")
    if _failures:
        print(f"\033[31m{len(_failures)} FAILED:\033[0m")
        for f in _failures:
            print(f"  • {f}")
        return 1

    print(f"\033[32mAll checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
