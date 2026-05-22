"""Additional tests for pipeline/lock.py uncovered paths."""

from __future__ import annotations

import platform
from pathlib import Path

import pytest

from synto.paths import effective_app_dir
from synto.pipeline.lock import (
    _warn_if_synced,
    has_invalid_lock_file,
    lock_holder_pid,
    pipeline_lock,
)


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / ".synto").mkdir()
    return tmp_path


# ── _warn_if_synced ──────────────────────────────────────────────────────────


def test_warn_if_synced_dropbox(caplog):
    """Warning emitted when vault is inside a synced directory."""
    vault = Path("/Users/me/Dropbox/my-vault")
    with caplog.at_level("WARNING"):
        _warn_if_synced(vault)
    assert "Dropbox" in caplog.text


def test_warn_if_synced_no_warning_for_local(caplog):
    """No warning when vault is on local filesystem."""
    vault = Path("/Users/me/projects/vault")
    with caplog.at_level("WARNING"):
        _warn_if_synced(vault)
    assert caplog.text == ""


# ── has_invalid_lock_file ────────────────────────────────────────────────────


def test_has_invalid_lock_file_no_file(vault):
    """Returns False when lock file doesn't exist."""
    assert has_invalid_lock_file(vault) is False


def test_has_invalid_lock_file_valid_pid(vault):
    """Returns False when lock file contains a valid PID."""
    lock_path = vault / ".synto" / "pipeline.lock"
    lock_path.write_text("12345")
    assert has_invalid_lock_file(vault) is False


def test_has_invalid_lock_file_empty(vault):
    """Returns True when lock file is empty."""
    lock_path = vault / ".synto" / "pipeline.lock"
    lock_path.write_text("")
    assert has_invalid_lock_file(vault) is True


# ── lock_holder_pid Windows path ─────────────────────────────────────────────


def test_lock_holder_pid_windows_live(vault, monkeypatch):
    """Windows: returns PID when process is alive."""
    monkeypatch.setattr("synto.pipeline.lock._IS_POSIX", False)
    lock_path = effective_app_dir(vault) / "pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("12345")
    monkeypatch.setattr("synto.pipeline.lock._windows_pid_alive", lambda pid: True)
    assert lock_holder_pid(vault) == 12345


def test_lock_holder_pid_windows_dead(vault, monkeypatch):
    """Windows: returns None when process is dead."""
    monkeypatch.setattr("synto.pipeline.lock._IS_POSIX", False)
    lock_path = effective_app_dir(vault) / "pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("12345")
    monkeypatch.setattr("synto.pipeline.lock._windows_pid_alive", lambda pid: False)
    assert lock_holder_pid(vault) is None


# ── Windows lock broken file cleanup ─────────────────────────────────────────


def test_windows_lock_broken_file_cleanup(vault, monkeypatch):
    """Windows: broken lock file is cleaned up and lock acquired."""
    monkeypatch.setattr("synto.pipeline.lock._IS_POSIX", False)
    lock_path = effective_app_dir(vault) / "pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("not-a-pid")
    monkeypatch.setattr("synto.pipeline.lock._windows_pid_alive", lambda pid: True)
    with pipeline_lock(vault) as acquired:
        assert acquired is True


# ── POSIX lock with block=True ───────────────────────────────────────────────


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_blocks_when_held(vault):
    """block=True waits until lock is released."""
    import threading
    import time

    results = []

    def hold_lock():
        with pipeline_lock(vault, block=False) as acquired:
            results.append(("holder", acquired))
            time.sleep(0.2)

    def wait_lock():
        with pipeline_lock(vault, block=True) as acquired:
            results.append(("waiter", acquired))

    t1 = threading.Thread(target=hold_lock)
    t2 = threading.Thread(target=wait_lock)
    t1.start()
    time.sleep(0.05)  # let holder acquire first
    t2.start()
    t1.join()
    t2.join()

    assert results[0] == ("holder", True)
    assert results[1] == ("waiter", True)
