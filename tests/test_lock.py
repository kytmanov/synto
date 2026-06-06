"""Tests for pipeline/lock.py."""

from __future__ import annotations

import os
import platform
import threading
from pathlib import Path

import pytest

from synto.paths import effective_app_dir
from synto.pipeline.lock import has_invalid_lock_file, lock_holder_pid, pipeline_lock


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / ".synto").mkdir()
    return tmp_path


# ── Basic lock acquisition ────────────────────────────────────────────────────


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_acquired_yields_true(vault):
    with pipeline_lock(vault) as acquired:
        assert acquired is True


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_held_yields_false(vault):
    with pipeline_lock(vault) as acquired:
        assert acquired is True
        # Second non-blocking attempt while first is held
        with pipeline_lock(vault, block=False) as acquired2:
            assert acquired2 is False


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_released_after_context(vault):
    with pipeline_lock(vault) as acquired:
        assert acquired is True
    # After context exits, lock should be acquirable again
    with pipeline_lock(vault) as acquired:
        assert acquired is True


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_released_on_exception(vault):
    try:
        with pipeline_lock(vault) as acquired:
            assert acquired is True
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    # Lock must be released even though exception was raised
    with pipeline_lock(vault) as acquired:
        assert acquired is True


# ── Lock file creation ────────────────────────────────────────────────────────


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_file_created(vault):
    with pipeline_lock(vault):
        assert (vault / ".synto" / "pipeline.lock").exists()


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_file_contains_pid(vault):
    import os

    with pipeline_lock(vault):
        pid = lock_holder_pid(vault)
        assert pid == os.getpid()


# ── lock_holder_pid ───────────────────────────────────────────────────────────


def test_lock_holder_pid_no_file(vault):
    # No lock file exists yet
    assert lock_holder_pid(vault) is None


def test_lock_holder_pid_unreadable(vault):
    # Write garbage
    lock_path = vault / ".synto" / "pipeline.lock"
    lock_path.write_text("not-a-pid")
    assert lock_holder_pid(vault) is None


def test_has_invalid_lock_file_detects_garbage(vault):
    lock_path = vault / ".synto" / "pipeline.lock"
    lock_path.write_text("not-a-pid")

    assert has_invalid_lock_file(vault) is True


# ── NFS flock emulation (issue #56) ───────────────────────────────────────────


def _install_nfs_flock(monkeypatch):
    """Make fcntl.flock behave like NFS: exclusive lock on a read-only fd → EBADF.

    On NFS the kernel emulates flock() as fcntl() POSIX byte-range locks, and a
    POSIX write lock requires a writable fd; a read-only fd returns EBADF.
    """
    import errno
    import fcntl

    real_flock = fcntl.flock

    def fake_flock(fd, operation):
        writable = getattr(fd, "writable", lambda: True)()
        if operation & fcntl.LOCK_EX and not writable:
            raise OSError(errno.EBADF, "Bad file descriptor")
        return real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", fake_flock)


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_holder_pid_on_nfs_does_not_crash(vault, monkeypatch):
    """Regression for #56: probing the lock on NFS must not raise Bad file descriptor.

    The probe must take a SHARED lock (which needs only a readable fd), so the
    NFS-emulated flock never sees an exclusive lock on a read-only fd. If anyone
    reverts to an exclusive probe, the fake raises EBADF, the narrowed errno guard
    re-raises it, and this test goes red.
    """
    lock_path = vault / ".synto" / "pipeline.lock"
    lock_path.write_text("4242")  # valid PID, but no live flock holder
    _install_nfs_flock(monkeypatch)

    assert lock_holder_pid(vault) is None


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores file permission bits",
)
def test_lock_holder_pid_stale_lock_on_readonly_file(vault):
    """Review regression: a readable-but-not-writable stale lock must read as free.

    A mode-0444 lock file (read-only mount / snapshot) used to make the old
    writable open() raise PermissionError, which the broad OSError guard reported
    as a phantom holder — suppressing stale-lock cleanup. The shared-lock probe
    opens read-only, so it correctly reports the dead lock as free.
    """
    lock_path = vault / ".synto" / "pipeline.lock"
    lock_path.write_text("4242")  # valid PID, no live holder
    os.chmod(lock_path, 0o444)

    assert lock_holder_pid(vault) is None


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_holder_pid_locking_unavailable_assumes_held(vault, monkeypatch):
    """nolock filesystem (ENOLCK): can't probe, so conservatively assume held."""
    import errno
    import fcntl

    lock_path = vault / ".synto" / "pipeline.lock"
    lock_path.write_text("4242")

    def enolck(fd, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", enolck)

    assert lock_holder_pid(vault) == 4242


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_holder_pid_unexpected_oserror_propagates(vault, monkeypatch):
    """An unexpected probe failure must not be swallowed (fail loud)."""
    import errno
    import fcntl

    lock_path = vault / ".synto" / "pipeline.lock"
    lock_path.write_text("4242")

    def eio(fd, operation):
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(fcntl, "flock", eio)

    with pytest.raises(OSError):
        lock_holder_pid(vault)


# ── Thread safety ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_only_one_thread_acquires_lock(vault):
    """Concurrent non-blocking attempts: exactly one succeeds."""
    results: list[bool] = []
    barrier = threading.Barrier(2)

    def try_lock():
        with pipeline_lock(vault, block=False) as acquired:
            barrier.wait()  # both threads enter before either exits
            results.append(acquired)

    t1 = threading.Thread(target=try_lock)
    t2 = threading.Thread(target=try_lock)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results.count(True) == 1
    assert results.count(False) == 1


# ── Stale PID detection ───────────────────────────────────────────────────────


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_holder_pid_returns_none_after_release(vault):
    """lock_holder_pid returns None once the flock is released (not just stale file)."""
    with pipeline_lock(vault) as acquired:
        assert acquired is True
        pid_during = lock_holder_pid(vault)
        assert pid_during is not None  # lock held → pid visible
    # After context exit flock is released; file still exists but lock is free
    pid_after = lock_holder_pid(vault)
    assert pid_after is None  # stale file should not report lock as held


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_holder_pid_returns_pid_while_held(vault):
    import os

    with pipeline_lock(vault) as acquired:
        assert acquired is True
        pid = lock_holder_pid(vault)
        assert pid == os.getpid()


# ── PID written after flock acquired ─────────────────────────────────────────


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_pid_written_after_lock_acquired(vault):
    """Lock file should contain our PID only after the flock is held."""
    import os

    with pipeline_lock(vault) as acquired:
        assert acquired is True
        lock_path = vault / ".synto" / "pipeline.lock"
        pid_in_file = int(lock_path.read_text().strip())
        assert pid_in_file == os.getpid()


# ── First-caller smoke (renamed from misleading "windows_yields_true_without_lock") ──


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_first_caller_acquires_lock(vault):
    with pipeline_lock(vault) as acquired:
        assert acquired is True


# ── Windows lock (mocked) ─────────────────────────────────────────────────────


def test_windows_lock_acquired(vault, monkeypatch):
    """Windows path: no existing lock file → lock acquired."""
    monkeypatch.setattr("synto.pipeline.lock._IS_POSIX", False)
    monkeypatch.setattr("synto.pipeline.lock._windows_pid_alive", lambda pid: True)
    with pipeline_lock(vault) as acquired:
        assert acquired is True


def test_windows_lock_refused_when_live(vault, monkeypatch):
    """Windows path: existing lock held by live PID → not acquired."""
    monkeypatch.setattr("synto.pipeline.lock._IS_POSIX", False)
    lock_path = effective_app_dir(vault) / "pipeline.lock"
    lock_path.parent.mkdir(exist_ok=True)
    lock_path.write_text("12345")
    monkeypatch.setattr("synto.pipeline.lock._windows_pid_alive", lambda pid: True)
    with pipeline_lock(vault) as acquired:
        assert acquired is False


def test_windows_lock_cleans_stale_and_acquires(vault, monkeypatch):
    """Windows path: existing lock with dead PID → stale cleaned, lock acquired."""
    monkeypatch.setattr("synto.pipeline.lock._IS_POSIX", False)
    lock_path = effective_app_dir(vault) / "pipeline.lock"
    lock_path.parent.mkdir(exist_ok=True)
    lock_path.write_text("99999999")
    monkeypatch.setattr("synto.pipeline.lock._windows_pid_alive", lambda pid: False)
    with pipeline_lock(vault) as acquired:
        assert acquired is True
    assert not lock_path.exists()  # cleaned up on context exit
