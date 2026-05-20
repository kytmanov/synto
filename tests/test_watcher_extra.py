"""Additional tests for watcher.py watch() function and edge cases."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from synto.watcher import _DebounceHandler, watch


class _FakeEvent:
    def __init__(self, path: str, is_directory: bool = False):
        self.src_path = path
        self.is_directory = is_directory


def test_watch_uses_config_debounce(tmp_path):
    """watch() reads debounce_secs from config when not overridden."""
    (tmp_path / "raw").mkdir()
    config = MagicMock()
    config.raw_dir = tmp_path / "raw"
    config.pipeline.watch_debounce = 0.05

    client = MagicMock()
    db = MagicMock()
    calls = []

    # Patch observer to not actually block
    mock_observer = MagicMock()
    mock_observer.is_alive.side_effect = [True, False]  # runs once then stops

    with patch("synto.watcher.Observer", return_value=mock_observer):
        watch(config, client, db, on_event=lambda paths: calls.append(paths))

    assert mock_observer.schedule.called
    assert mock_observer.start.called
    assert mock_observer.stop.called
    assert mock_observer.join.called


def test_watch_uses_override_debounce(tmp_path):
    """watch() uses debounce_secs override when provided."""
    (tmp_path / "raw").mkdir()
    config = MagicMock()
    config.raw_dir = tmp_path / "raw"
    config.pipeline.watch_debounce = 999.0  # should be ignored

    client = MagicMock()
    db = MagicMock()
    calls = []

    mock_observer = MagicMock()
    mock_observer.is_alive.side_effect = [True, False]

    with patch("synto.watcher.Observer", return_value=mock_observer):
        watch(
            config,
            client,
            db,
            on_event=lambda paths: calls.append(paths),
            debounce_secs=0.05,
        )

    # Verify handler was created with the override value
    schedule_call = mock_observer.schedule.call_args
    handler = schedule_call[0][0]
    assert handler._debounce_secs == 0.05


def test_watch_flushes_on_exit(tmp_path):
    """watch() flushes pending events when exiting."""
    (tmp_path / "raw").mkdir()
    config = MagicMock()
    config.raw_dir = tmp_path / "raw"
    config.pipeline.watch_debounce = 60.0  # long debounce

    client = MagicMock()
    db = MagicMock()
    calls = []

    mock_observer = MagicMock()
    mock_observer.is_alive.side_effect = [True, False]

    with patch("synto.watcher.Observer", return_value=mock_observer):
        # Inject an event before watch exits
        handler = _DebounceHandler(callback=lambda paths: calls.append(paths), debounce_secs=60.0)
        handler.on_created(_FakeEvent(str(tmp_path / "raw" / "note.md")))

        with patch("synto.watcher._DebounceHandler", return_value=handler):
            watch(config, client, db, on_event=lambda paths: calls.append(paths))

    # flush() should have been called, firing the pending event
    assert len(calls) == 1


def test_watch_handles_keyboard_interrupt(tmp_path):
    """watch() catches KeyboardInterrupt and cleans up gracefully."""
    (tmp_path / "raw").mkdir()
    config = MagicMock()
    config.raw_dir = tmp_path / "raw"
    config.pipeline.watch_debounce = 0.05

    client = MagicMock()
    db = MagicMock()

    mock_observer = MagicMock()
    mock_observer.is_alive.side_effect = KeyboardInterrupt()

    with patch("synto.watcher.Observer", return_value=mock_observer):
        watch(config, client, db, on_event=lambda paths: None)

    # Should have stopped and joined despite the interrupt
    assert mock_observer.stop.called
    assert mock_observer.join.called


def test_watch_observer_scheduled_on_raw_dir(tmp_path):
    """Observer is scheduled on config.raw_dir."""
    (tmp_path / "raw").mkdir()
    config = MagicMock()
    config.raw_dir = tmp_path / "raw"
    config.pipeline.watch_debounce = 0.05

    mock_observer = MagicMock()
    mock_observer.is_alive.return_value = False

    with patch("synto.watcher.Observer", return_value=mock_observer):
        watch(config, MagicMock(), MagicMock(), on_event=lambda paths: None)

    schedule_call = mock_observer.schedule.call_args
    assert schedule_call[1]["recursive"] is True
    assert schedule_call[0][1] == str(tmp_path / "raw")


def test_debounce_handler_moved_event_no_dest_path():
    """on_moved with no dest_path attribute is handled gracefully."""
    calls = []
    handler = _DebounceHandler(callback=lambda paths: calls.append(paths), debounce_secs=0.05)

    class NoDestEvent:
        is_directory = False

    handler.on_moved(NoDestEvent())
    time.sleep(0.15)
    assert calls == []


def test_debounce_handler_moved_event_dest_not_string():
    """on_moved with dest_path that's not a string is handled."""
    calls = []
    handler = _DebounceHandler(callback=lambda paths: calls.append(paths), debounce_secs=0.05)

    class BadDestEvent:
        is_directory = False
        dest_path = None

    handler.on_moved(BadDestEvent())
    time.sleep(0.15)
    assert calls == []
