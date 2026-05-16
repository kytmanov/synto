"""
Tests for watcher.py — debounce logic only.
No filesystem watching, no Ollama, no threads started beyond Timer.
"""

from __future__ import annotations

import threading
import time

from synto.watcher import _DebounceHandler

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_handler(debounce: float = 0.05) -> tuple[_DebounceHandler, list[list[str]]]:
    """Return (handler, calls) where calls accumulates callback invocations."""
    calls: list[list[str]] = []
    handler = _DebounceHandler(callback=lambda paths: calls.append(paths), debounce_secs=debounce)
    return handler, calls


class _FakeEvent:
    """Minimal filesystem event stub."""

    def __init__(self, path: str, is_directory: bool = False):
        self.src_path = path
        self.is_directory = is_directory


class _FakeMoveEvent:
    def __init__(self, src: str, dest: str):
        self.src_path = src
        self.dest_path = dest
        self.is_directory = False


# ── Debounce behaviour ────────────────────────────────────────────────────────


def test_single_event_fires_after_debounce():
    handler, calls = _make_handler(debounce=0.05)
    handler.on_created(_FakeEvent("/vault/raw/note.md"))
    time.sleep(0.15)
    assert len(calls) == 1
    assert "/vault/raw/note.md" in calls[0]


def test_multiple_rapid_events_batched_into_one_call():
    handler, calls = _make_handler(debounce=0.1)
    handler.on_created(_FakeEvent("/vault/raw/a.md"))
    handler.on_modified(_FakeEvent("/vault/raw/b.md"))
    handler.on_created(_FakeEvent("/vault/raw/c.md"))
    time.sleep(0.3)
    # All three events → single callback with all paths
    assert len(calls) == 1
    assert len(calls[0]) == 3


def test_non_md_files_ignored():
    handler, calls = _make_handler(debounce=0.05)
    handler.on_created(_FakeEvent("/vault/raw/image.png"))
    handler.on_modified(_FakeEvent("/vault/raw/.DS_Store"))
    time.sleep(0.15)
    assert calls == []


def test_directory_events_ignored():
    handler, calls = _make_handler(debounce=0.05)
    handler.on_created(_FakeEvent("/vault/raw/subdir", is_directory=True))
    time.sleep(0.15)
    assert calls == []


def test_moved_event_with_md_dest_handled():
    handler, calls = _make_handler(debounce=0.05)
    handler.on_moved(_FakeMoveEvent("/tmp/obs_tmp123", "/vault/raw/note.md"))
    time.sleep(0.15)
    assert len(calls) == 1
    assert "/vault/raw/note.md" in calls[0]


def test_moved_event_non_md_dest_ignored():
    handler, calls = _make_handler(debounce=0.05)
    handler.on_moved(_FakeMoveEvent("/vault/raw/note.md", "/vault/raw/note.txt"))
    time.sleep(0.15)
    assert calls == []


def test_debounce_resets_on_new_event():
    handler, calls = _make_handler(debounce=0.15)
    handler.on_created(_FakeEvent("/vault/raw/a.md"))
    time.sleep(0.05)  # before debounce fires
    handler.on_modified(_FakeEvent("/vault/raw/b.md"))  # resets timer
    time.sleep(0.05)  # still before debounce fires
    assert calls == []  # not fired yet
    time.sleep(0.2)  # now it should fire
    assert len(calls) == 1
    assert len(calls[0]) == 2


def test_deduplication_of_same_path():
    handler, calls = _make_handler(debounce=0.05)
    handler.on_modified(_FakeEvent("/vault/raw/note.md"))
    handler.on_modified(_FakeEvent("/vault/raw/note.md"))
    handler.on_modified(_FakeEvent("/vault/raw/note.md"))
    time.sleep(0.15)
    assert len(calls) == 1
    assert calls[0].count("/vault/raw/note.md") == 1  # deduplicated via set


def test_flush_fires_immediately():
    handler, calls = _make_handler(debounce=60.0)  # very long debounce
    handler.on_created(_FakeEvent("/vault/raw/note.md"))
    assert calls == []  # not fired yet
    handler.flush()
    assert len(calls) == 1


def test_flush_with_no_pending_events_is_noop():
    handler, calls = _make_handler(debounce=0.05)
    handler.flush()  # nothing pending
    assert calls == []


def test_two_batches_separated_by_silence():
    handler, calls = _make_handler(debounce=0.08)
    # First batch
    handler.on_created(_FakeEvent("/vault/raw/a.md"))
    time.sleep(0.2)
    # Second batch
    handler.on_created(_FakeEvent("/vault/raw/b.md"))
    time.sleep(0.2)
    assert len(calls) == 2
    assert "/vault/raw/a.md" in calls[0]
    assert "/vault/raw/b.md" in calls[1]


def test_thread_safety_concurrent_events():
    """Fire many events from multiple threads simultaneously — no crash, single batch."""
    handler, calls = _make_handler(debounce=0.1)

    def _fire(name: str):
        handler.on_created(_FakeEvent(f"/vault/raw/{name}.md"))

    threads = [threading.Thread(target=_fire, args=(f"note{i}",)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    time.sleep(0.3)
    assert len(calls) == 1
    assert len(calls[0]) == 20
