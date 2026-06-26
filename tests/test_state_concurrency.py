"""Regression tests for issue #75: SQLite transaction races under parallel ingest.

`StateDB` shares one `sqlite3.Connection` across threads (`check_same_thread=False`), and
`LLMCache` commits on that same connection. During parallel ingest the worker threads commit
via the cache while the main thread is mid-`_tx()`, which used to corrupt the connection's
transaction state and fail with `cannot commit - no transaction is active` /
`cannot start a transaction within a transaction`.

These tests encode the contract that the connection tolerates concurrent transactions + cache
writes. They go red if the serialization (the StateDB lock + routing cache writes through
`_tx()`) is removed.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from synto.cache import LLMCache
from synto.state import StateDB


def _insert_chunk_row(conn, source_path: str, chunk_index: int) -> None:
    now = datetime.now().isoformat()
    conn.execute(
        """INSERT INTO ingest_chunks
               (source_path, content_hash, chunk_index, chunk_count, chunk_size,
                checkpoint_schema, result_json, created_at, updated_at)
           VALUES (?, 'h', ?, 1, 100, 2, '{}', ?, ?)""",
        (source_path, chunk_index, now, now),
    )


def test_cache_commit_does_not_break_main_transaction_atomicity(tmp_path: Path) -> None:
    """A worker's cache write must not commit the main thread's open transaction.

    The real harm of #75 is a broken transaction boundary: a worker's `cache.put()` commit
    landing inside the main thread's `_tx()` commits the main thread's not-yet-finished work.
    Here the main thread writes a row inside `_tx()`, lets a worker fire a `cache.put()`, then
    forces a rollback. If the worker's commit leaked into the main transaction, the row
    survives the rollback (atomicity broken). The handoff uses a timeout because, once
    serialized, the worker correctly blocks on the lock until the main `_tx()` exits — so the
    timeout path is the *fixed* behavior, not a flake.
    """
    db = StateDB(tmp_path / "state.db")
    cache = LLMCache(db)

    main_in_tx = threading.Event()
    worker_committed = threading.Event()
    worker_error: list[Exception] = []

    def worker() -> None:
        try:
            main_in_tx.wait(timeout=5.0)
            # On master this commits immediately (the shared connection has no lock), ending the
            # main thread's open transaction. After the fix this blocks until the main `_tx()`
            # releases the lock, so it can never commit mid-transaction.
            cache.put("model", [{"role": "user", "content": "race"}], "resp")
            worker_committed.set()
        except Exception as exc:  # noqa: BLE001 - surfaced via assertion below
            worker_error.append(exc)

    t = threading.Thread(target=worker)
    t.start()

    class _ForceRollback(RuntimeError):
        pass

    try:
        with db._tx() as conn:
            _insert_chunk_row(conn, "raw/main.md", 0)
            main_in_tx.set()
            # On master the worker commits within microseconds, so this returns fast and the
            # premature commit has already leaked the row out of our transaction. After the fix
            # the worker is blocked on the lock, so this waits out the timeout and the row stays
            # inside our (about-to-roll-back) transaction.
            worker_committed.wait(timeout=0.5)
            raise _ForceRollback
    except _ForceRollback:
        pass

    t.join(timeout=5.0)
    assert not worker_error, f"worker raised: {worker_error}"

    # The main thread rolled back, so its row must be gone. On master the worker's commit
    # leaked it past the rollback and this count is 1. Intentional bare read: the worker thread
    # has already joined, so this verification runs single-threaded (no concurrent writer).
    remaining = db._conn.execute("SELECT COUNT(*) FROM ingest_chunks").fetchone()[0]
    assert remaining == 0, "main transaction was committed by a concurrent cache write (#75)"

    # The worker's cache write must still have succeeded (no deadlock, no lost work).
    assert cache.get("model", [{"role": "user", "content": "race"}]) == "resp"


def test_parallel_cache_and_tx_writes_raise_no_transaction_errors(tmp_path: Path) -> None:
    """Hammer the real interleaving: worker cache put/get vs main-thread `_tx()` writes.

    Mirrors parallel ingest, where chunk-analysis workers write only via the cache while the
    main thread persists chunk checkpoints through `_tx()`. Pre-fix this reliably surfaces
    `sqlite3.OperationalError` transaction-state errors; post-fix every thread completes clean.
    """
    db = StateDB(tmp_path / "state.db")
    cache = LLMCache(db)

    n_workers = 8
    iters = 150
    errors: list[Exception] = []
    start = threading.Barrier(n_workers + 1)

    def worker(wid: int) -> None:
        try:
            start.wait()
            for i in range(iters):
                messages = [{"role": "user", "content": f"{wid}:{i}"}]
                cache.put("model", messages, f"resp-{wid}-{i}")
                cache.get("model", messages)
        except Exception as exc:  # noqa: BLE001 - surfaced via assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(n_workers)]
    for t in threads:
        t.start()

    start.wait()
    try:
        for i in range(iters):
            db.upsert_ingest_chunk(f"raw/note{i % 4}.md", "h", i, iters, 100, "{}")
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)

    for t in threads:
        t.join()

    assert errors == [], f"concurrent cache + _tx writes raised: {errors}"


def test_read_serializes_against_a_held_write_transaction(tmp_path: Path) -> None:
    """`_read()` is the supported primitive for a read that races a writer.

    It takes the same re-entrant lock as `_tx()`, so a read issued while another thread holds an
    open write transaction must *block* until that transaction commits and then observe the
    committed row — serialized, not racing. We deliberately do not assert that a *bare* `_conn`
    read interleaves safely: CPython's sqlite3 Connection does not guarantee that under true
    concurrency, so such a test would be flaky. This pins the guarantee `_read()` actually makes.
    """
    db = StateDB(tmp_path / "state.db")

    writer_in_tx = threading.Event()
    reader_done = threading.Event()
    reader_count: list[int] = []
    reader_error: list[Exception] = []

    def reader() -> None:
        try:
            writer_in_tx.wait(timeout=5.0)
            # Blocks here until the writer's _tx() releases the lock, then reads committed state.
            with db._read() as conn:
                reader_count.append(
                    conn.execute("SELECT COUNT(*) FROM ingest_chunks").fetchone()[0]
                )
            reader_done.set()
        except Exception as exc:  # noqa: BLE001 - surfaced via assertion below
            reader_error.append(exc)

    t = threading.Thread(target=reader)
    t.start()

    with db._tx() as conn:
        _insert_chunk_row(conn, "raw/writer.md", 0)
        writer_in_tx.set()
        # While we hold the transaction, the reader's _read() must stay blocked on the lock — it
        # cannot complete. If _read() did not serialize, it would read immediately and set this.
        assert not reader_done.wait(timeout=0.3), "_read() did not serialize against a held _tx()"

    # Transaction committed and lock released: the reader now unblocks and sees the committed row.
    assert reader_done.wait(timeout=5.0), "_read() deadlocked against a committed _tx()"
    t.join(timeout=5.0)
    assert not reader_error, f"reader raised: {reader_error}"
    assert reader_count == [1], f"reader saw uncommitted/stale state: {reader_count}"
