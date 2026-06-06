"""
Pipeline concurrency lock.

Prevents concurrent pipeline runs (synto watch + synto compile, etc.) from
racing on the same StateDB.

- POSIX (Linux/macOS): uses fcntl.flock() — advisory, auto-released on process
  death (no stale-lock problem).
- Windows: uses O_CREAT|O_EXCL atomic file creation + PID liveness check via
  ctypes (stdlib only). Stale locks from crashed processes are cleaned up
  automatically on the next acquire attempt.

Vault must be on a local filesystem — flock() is unreliable on NFS/Dropbox.
"""

from __future__ import annotations

import contextlib
import logging
import platform
from pathlib import Path

from ..paths import effective_app_dir

log = logging.getLogger(__name__)

_IS_POSIX = platform.system() != "Windows"

# Known sync directories that indicate a remote/synced vault
_SYNC_DIRS = {"Dropbox", "OneDrive", "iCloud Drive", "Google Drive"}


def _windows_pid_alive(pid: int) -> bool:
    """Return True if the given PID is a running process on Windows (ctypes, stdlib)."""
    import ctypes

    SYNCHRONIZE = 0x00100000
    handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, 0, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def _acquire_windows_lock(lock_path: Path) -> bool:
    """Atomically create the lock file (O_CREAT|O_EXCL). Cleans stale PIDs.

    Returns True if the lock was acquired, False if held by a live process.
    """
    import os

    max_retries = 3
    for _ in range(max_retries):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(lock_path.read_text().strip())
            except Exception:
                lock_path.unlink(missing_ok=True)
                continue  # broken file cleaned up, retry
            if _windows_pid_alive(pid):
                return False  # live lock held by another process
            lock_path.unlink(missing_ok=True)
            # stale lock cleaned up, retry
    return False  # exhausted retries


def _warn_if_synced(vault: Path) -> None:
    parts = set(vault.parts)
    for sync_dir in _SYNC_DIRS:
        if sync_dir in parts:
            log.warning(
                "Vault is inside '%s' — pipeline lock (flock) may be unreliable on synced "
                "filesystems. Ensure the Synto app dir is on a local path.",
                sync_dir,
            )
            break


@contextlib.contextmanager
def pipeline_lock(vault: Path, block: bool = False):
    """
    Acquire an exclusive pipeline lock for the vault.

    Yields True if the lock was acquired, False if it was already held.
    The lock is released on context exit, including on exceptions.

    Usage::

        with pipeline_lock(config.vault) as acquired:
            if not acquired:
                console.print("⚠ pipeline already running")
                return
            # ... do pipeline work ...
    """
    if not _IS_POSIX:
        lock_path = effective_app_dir(vault) / "pipeline.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if not _acquire_windows_lock(lock_path):
            yield False
            return
        try:
            yield True
        finally:
            lock_path.unlink(missing_ok=True)
        return

    import fcntl

    _warn_if_synced(vault)

    lock_path = effective_app_dir(vault) / "pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Open with "a+" (create if absent, no truncation) so a competing process
    # that fails to acquire the lock does not clear the incumbent's PID.
    # We truncate and write the PID ourselves only after the lock is held.
    with open(lock_path, "a+") as f:
        import os

        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if block else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        # We now hold the lock — overwrite with our PID.
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()))
        f.flush()
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def lock_holder_pid(vault: Path) -> int | None:
    """Return PID if the pipeline lock is actively held, None otherwise.

    Verifies the lock is actually held (not just a stale lock file) by
    attempting a non-blocking shared acquire. If that succeeds the lock is
    free; if it raises BlockingIOError the lock is live.
    """
    lock_path = effective_app_dir(vault) / "pipeline.lock"
    if not lock_path.exists():
        return None
    try:
        pid = int(lock_path.read_text().strip())
    except Exception:
        return None
    if not _IS_POSIX:
        return pid if _windows_pid_alive(pid) else None
    import errno
    import fcntl

    try:
        # Detect the holder with a SHARED lock: pipeline_lock() always takes an
        # exclusive lock, so a non-blocking shared acquire fails iff it is live.
        # A read lock needs only a readable fd (the file is already proven
        # readable above), so this works on NFS — where flock() is emulated as
        # fcntl() locks and an exclusive lock on a read-only fd returns EBADF —
        # and on read-only mounts / mode-0444 stale lock files.
        with open(lock_path) as f:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
        return None  # acquired → nobody holding it
    except BlockingIOError:
        return pid  # lock is live
    except OSError as e:
        # Locking subsystem unavailable (e.g. nolock NFS mount → ENOLCK). Can't
        # probe liveness; assume held so we never advise deleting a possibly-live
        # lock. Re-raise anything unexpected rather than masking real bugs.
        if e.errno in (errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOTSUP):
            return pid
        raise


def has_invalid_lock_file(vault: Path) -> bool:
    """Return True when pipeline.lock exists but does not contain a valid PID."""
    lock_path = effective_app_dir(vault) / "pipeline.lock"
    if not lock_path.exists():
        return False
    try:
        int(lock_path.read_text().strip())
    except Exception:
        return True
    return False
