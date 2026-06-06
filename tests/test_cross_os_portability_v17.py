"""Cross-OS vault portability (issue #55).

A user built a vault on Windows and moved it — including ``.synto/state.db`` — to a Linux
box. Windows stored vault-relative paths with backslash separators (``raw\\note.md``); on
Linux the same note resolves to ``raw/note.md``. The content-hash dedup then matched a row
at a "different" path and skipped every note as a duplicate, ignoring its source.

These tests encode *why* the fix matters: a vault must remain recognizable after a cross-OS
move so incremental updates keep working, rather than re-ingesting or silently skipping.

The root cause is path separators, NOT line endings (the user's guess): the content hash is
computed on a newline-normalized body, so CRLF↔LF is intentionally a non-issue — locked in
by ``test_line_endings_do_not_change_content_hash``.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synto.models import ItemMentionRecord, RawNoteRecord, WikiArticleRecord
from synto.paths import rel_posix
from synto.pipeline.ingest import _content_hash
from synto.state import _DB_PATH_LIST_PARAMS, _DB_PATH_PARAMS, StateDB
from synto.vault import parse_note, write_note


def _seed_windows_built_db(db_path: Path) -> str:
    """Create a v16-era DB whose rows use Windows separators, as a Windows build would.

    Rows are inserted with raw SQL (bypassing the Pydantic normalizers) and the schema
    version is pinned back to 16 so reopening exercises only the v17 separator migration.
    Returns the body hash stored on the raw note.
    """
    db = StateDB(db_path)
    body_hash = _content_hash("Qubit body\n")
    db._conn.execute(
        "INSERT INTO raw_notes (path, content_hash, status) VALUES (?, ?, ?)",
        ("raw\\qubit.md", body_hash, "ingested"),
    )
    db._conn.execute(
        "INSERT INTO concepts (name, source_path) VALUES (?, ?)",
        ("Qubit", "raw\\qubit.md"),
    )
    db._conn.execute(
        "INSERT INTO concept_compile_state "
        "(concept_name, source_path, status, updated_at) VALUES (?, ?, 'compiled', ?)",
        ("Qubit", "raw\\qubit.md", "2026-01-01"),
    )
    db._conn.execute(
        """INSERT INTO wiki_articles
               (path, title, sources, content_hash, created_at, updated_at, status,
                kind, synthesis_sources, synthesis_source_hashes, article_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "wiki\\Qubit.md",
            "Qubit",
            json.dumps(["raw\\qubit.md", "raw\\note two.md"]),
            "ch",
            "2026-01-01",
            "2026-01-01",
            "published",
            "concept",
            json.dumps([]),
            json.dumps([]),
            "aid-qubit",
        ),
    )
    db._conn.execute("UPDATE schema_version SET version = 16 WHERE id = 1")
    db._conn.commit()
    db._conn.close()
    return body_hash


def test_v17_migration_rewrites_separators_on_a_windows_built_db(tmp_path: Path) -> None:
    """Moving a Windows-built DB to Linux must repair stored paths on first open."""
    db_path = tmp_path / "state.db"
    _seed_windows_built_db(db_path)

    db = StateDB(db_path)  # reopen → runs the v17 migration

    assert db._conn.execute("SELECT version FROM schema_version").fetchone()[0] == 17
    assert db._conn.execute("SELECT path FROM raw_notes").fetchone()[0] == "raw/qubit.md"
    assert db._conn.execute("SELECT source_path FROM concepts").fetchone()[0] == "raw/qubit.md"
    assert (
        db._conn.execute("SELECT source_path FROM concept_compile_state").fetchone()[0]
        == "raw/qubit.md"
    )
    assert db._conn.execute("SELECT path FROM wiki_articles").fetchone()[0] == "wiki/Qubit.md"


def test_v17_migration_does_not_corrupt_json_source_lists(tmp_path: Path) -> None:
    """JSON-encoded source lists must be decoded/re-encoded, not REPLACE'd.

    A Windows path is JSON-escaped on disk as ``raw\\\\qubit.md``; a naive REPLACE of a
    single backslash would yield ``raw//qubit.md``. Guards against that regression.
    """
    db_path = tmp_path / "state.db"
    _seed_windows_built_db(db_path)

    db = StateDB(db_path)

    sources = json.loads(db._conn.execute("SELECT sources FROM wiki_articles").fetchone()[0])
    assert sources == ["raw/qubit.md", "raw/note two.md"]
    assert all("//" not in s and "\\" not in s for s in sources)


def test_moved_vault_recognizes_existing_note_instead_of_skipping_it(tmp_path: Path) -> None:
    """The reported symptom: after the move, the note is found, not flagged a duplicate.

    Reproduces the exact dedup decision from ``ingest_note``: a Linux re-ingest computes the
    POSIX relative path and compares it against the stored path returned by content hash.
    Before the fix that comparison was ``'raw\\qubit.md' != 'raw/qubit.md'`` → True → skip.
    """
    db_path = tmp_path / "state.db"
    body_hash = _seed_windows_built_db(db_path)

    db = StateDB(db_path)
    vault = tmp_path
    linux_note = vault / "raw" / "qubit.md"

    existing = db.get_raw_by_hash(body_hash)
    assert existing is not None
    # The dedup guard must NOT treat the moved note as a duplicate of itself.
    assert existing.path == rel_posix(linux_note, vault)
    # And a path-keyed lookup with the Linux path resolves the migrated row.
    assert db.get_raw("raw/qubit.md") is not None


def test_record_models_normalize_separators_on_construction() -> None:
    """Write-side guarantee: no caller can persist a backslash path, on any OS."""
    raw = RawNoteRecord(path="raw\\note.md", content_hash="x")
    assert raw.path == "raw/note.md"

    article = WikiArticleRecord(
        path="wiki\\A.md",
        title="A",
        sources=["raw\\n1.md", "raw\\n2.md"],
        content_hash="h",
        synthesis_sources=["raw\\s.md"],
    )
    assert article.path == "wiki/A.md"
    assert article.sources == ["raw/n1.md", "raw/n2.md"]
    assert article.synthesis_sources == ["raw/s.md"]


def test_state_transitions_match_windows_style_caller_paths(tmp_path: Path) -> None:
    """verify/publish/approve/delete must hit the POSIX-stored row even when the caller
    passes a Windows-style path.

    compile.py builds keys with ``str(path.relative_to(...))`` — backslash-separated on
    Windows. Articles are stored as POSIX (model validator), so without boundary
    normalization the mutating ``WHERE path = ?`` silently no-ops and publish/approve do
    nothing.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/.drafts/Foo.md", title="Foo", sources=[], content_hash="h", status="draft"
        )
    )

    db.verify_article("wiki\\.drafts\\Foo.md")
    assert db.get_article("wiki/.drafts/Foo.md").status == "verified"

    db.publish_article("wiki\\.drafts\\Foo.md", "wiki\\Foo.md")
    assert db.get_article("wiki/Foo.md").status == "published"

    db.approve_article("wiki\\Foo.md", notes="ok")
    assert db.get_article("wiki/Foo.md").approved_at is not None

    db.delete_article("wiki\\Foo.md")
    assert db.get_article("wiki/Foo.md") is None


def test_v17_migration_resolves_duplicate_backslash_and_posix_rows(tmp_path: Path) -> None:
    """A cross-OS-moved vault re-compiled on the new OS can hold both `wiki\\Q.md` and
    `wiki/Q.md` (pre-fix exact-path lookups kept them as distinct rows). Normalizing the
    primary key in place would collide; the migration must keep the POSIX row and drop the
    stale backslash duplicate instead of aborting.
    """
    db_path = tmp_path / "state.db"
    db = StateDB(db_path)
    for path, status, aid in [("wiki\\Q.md", "draft", "a1"), ("wiki/Q.md", "published", "a2")]:
        db._conn.execute(
            """INSERT INTO wiki_articles
                   (path, title, sources, content_hash, created_at, updated_at, status,
                    kind, synthesis_sources, synthesis_source_hashes, article_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                path,
                "Q",
                json.dumps([]),
                "h",
                "2026-01-01",
                "2026-01-01",
                status,
                "concept",
                json.dumps([]),
                json.dumps([]),
                aid,
            ),
        )
    # Same collision on a raw_notes primary key.
    db._conn.execute(
        "INSERT INTO raw_notes (path, content_hash, status) VALUES (?, ?, ?)",
        ("raw\\n.md", "h1", "ingested"),
    )
    db._conn.execute(
        "INSERT INTO raw_notes (path, content_hash, status) VALUES (?, ?, ?)",
        ("raw/n.md", "h2", "ingested"),
    )
    db._conn.execute("UPDATE schema_version SET version = 16 WHERE id = 1")
    db._conn.commit()
    db._conn.close()

    db = StateDB(db_path)  # must not raise on the PK collision

    arts = db._conn.execute("SELECT path, status FROM wiki_articles").fetchall()
    assert [(r[0], r[1]) for r in arts] == [("wiki/Q.md", "published")]
    raws = [r[0] for r in db._conn.execute("SELECT path FROM raw_notes").fetchall()]
    assert raws == ["raw/n.md"]


def test_line_endings_do_not_change_content_hash(tmp_path: Path) -> None:
    """Lock in that CRLF↔LF is a non-issue: a git line-ending rewrite must not change a
    note's dedup identity. This is why the user's line-ending hypothesis was wrong."""
    meta = {"title": "T"}
    lf_path = tmp_path / "lf.md"
    crlf_path = tmp_path / "crlf.md"
    write_note(lf_path, meta, "Hello world\nSecond line\n")
    crlf_path.write_bytes(lf_path.read_bytes().replace(b"\n", b"\r\n"))

    _, lf_body = parse_note(lf_path)
    _, crlf_body = parse_note(crlf_path)
    assert _content_hash(lf_body) == _content_hash(crlf_body)


# ── Abstraction guarantees (the normalization machinery itself) ────────────────


def test_vault_rel_path_type_rejects_non_str() -> None:
    """The `VaultRelPath` type wraps a *guarded* validator: a non-str field must raise a
    clean Pydantic `ValidationError`, not crash in `to_posix` (`int` has no `.replace`)."""
    with pytest.raises(ValidationError):
        RawNoteRecord(path=123, content_hash="h")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ItemMentionRecord(
            item_name="x",
            source_path=object(),  # type: ignore[arg-type]
            mention_text="m",
            evidence_level="source_supported",
        )


def test_db_path_decorator_does_not_mangle_path_typed_argument(tmp_path: Path) -> None:
    """The decorator's `isinstance(str)` guard is load-bearing: `_infer_orphan_draft_status`
    has a parameter named `path` but receives a `Path`. Normalizing it would call
    `Path.replace` (a file rename) — the guard must pass the `Path` through untouched."""
    db = StateDB(tmp_path / "state.db")
    note = tmp_path / "a b.md"  # space — exercises a real filesystem read
    write_note(note, {"status": "verified"}, "body\n")

    assert db._infer_orphan_draft_status(note) == "verified"
    assert note.exists()  # not accidentally renamed by a mis-applied str.replace


def test_db_path_param_naming_convention_is_complete() -> None:
    """Enforce the decorator's naming convention as an invariant: every `StateDB` parameter
    whose name contains `path` and is typed `str`/`list[str]` must be covered by the
    decorator's constants. A future method with an uncovered path param fails here instead of
    silently bypassing normalization (the convention is the contract)."""
    covered = _DB_PATH_PARAMS | _DB_PATH_LIST_PARAMS
    str_like = {"str", "str | None", "list[str]", "list[str] | None"}
    offenders: list[str] = []
    for name, fn in vars(StateDB).items():
        if not inspect.isfunction(fn):
            continue
        for p in inspect.signature(fn).parameters.values():
            if p.name == "self":
                continue
            ann = p.annotation if isinstance(p.annotation, str) else None
            if "path" in p.name.lower() and ann in str_like and p.name not in covered:
                offenders.append(f"{name}({p.name}: {ann})")
    assert not offenders, f"path-named str params not covered by the decorator: {offenders}"
