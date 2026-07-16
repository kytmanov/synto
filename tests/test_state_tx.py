"""Transaction semantics of StateDB._tx() — the depth-0 BEGIN contract.

These pin the fix for the early-commit bug: without an explicit BEGIN at
depth 0, sqlite3's legacy isolation mode only implicit-BEGINs before DML,
so a nested _tx()'s SAVEPOINT issued before any outer DML becomes the
outermost transaction and its RELEASE commits for real — the outer frame's
rollback then has nothing to undo. Direct db._tx() use here is intentional:
these are semantics tests for the primitive itself.
"""

from __future__ import annotations

import pytest

from synto.models import RawNoteRecord
from synto.state import StateDB


@pytest.fixture
def db(tmp_path):
    d = StateDB(tmp_path / ".synto" / "state.db")
    yield d
    d.close()


def _raw(path: str = "raw/a.md") -> RawNoteRecord:
    return RawNoteRecord(path=path, content_hash="h", status="new")


def test_nested_public_call_keeps_outer_transaction_open(db):
    with db._tx():
        db.upsert_raw(_raw())
        assert db._conn.in_transaction


def test_outer_raise_rolls_back_nested_write(db):
    with pytest.raises(RuntimeError):
        with db._tx():
            db.upsert_raw(_raw())
            raise RuntimeError("boom after nested write")
    assert db.get_raw("raw/a.md") is None


def test_depth0_ddl_frame_is_atomic(db):
    with pytest.raises(RuntimeError):
        with db._tx():
            db._conn.execute("CREATE TABLE tx_probe (id INTEGER)")
            raise RuntimeError("boom after DDL")
    assert not db._has_table("tx_probe")


def test_caught_nested_failure_preserves_outer_write(db):
    with db._tx():
        db.upsert_raw(_raw("raw/outer.md"))
        try:
            with db._tx():
                db.upsert_raw(_raw("raw/inner.md"))
                raise RuntimeError("inner boom")
        except RuntimeError:
            pass
    assert db.get_raw("raw/outer.md") is not None
    assert db.get_raw("raw/inner.md") is None


def test_three_level_nesting_innermost_rollback(db):
    with db._tx():
        db.upsert_raw(_raw("raw/l1.md"))
        with db._tx():
            db.upsert_raw(_raw("raw/l2.md"))
            try:
                with db._tx():
                    db.upsert_raw(_raw("raw/l3.md"))
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
    assert db.get_raw("raw/l1.md") is not None
    assert db.get_raw("raw/l2.md") is not None
    assert db.get_raw("raw/l3.md") is None
