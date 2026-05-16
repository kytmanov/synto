"""Git safety net: auto-commit and safe undo via git revert."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .paths import AUTO_COMMIT_PREFIX, LEGACY_AUTO_COMMIT_PREFIX

log = logging.getLogger(__name__)


def _run(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=check)


def _is_auto_commit_subject(subject: str) -> bool:
    """Match only direct auto-commit subjects, not git-generated revert subjects."""

    return subject.startswith(f"{AUTO_COMMIT_PREFIX} ") or subject.startswith(
        f"{LEGACY_AUTO_COMMIT_PREFIX} "
    )


def _has_pre_staged_changes(vault: Path) -> bool:
    """Return True if the index has staged changes before synto touches anything."""
    try:
        result = _run(["git", "diff", "--cached", "--name-only"], cwd=vault, check=False)
        return bool(result.stdout.strip())
    except Exception:
        return False  # fail open: if the check itself errors, don't block


def git_commit(
    vault: Path,
    message: str,
    paths: list[str] | None = None,
) -> str:
    """Stage paths and commit. Returns 'committed', 'nothing', 'blocked', or 'failed'.

    Returns 'blocked' when the user already has staged changes — synto will not
    touch the index in that case to avoid accidentally bundling unrelated changes.

    paths defaults to wiki/, raw/, vault-schema.md, .synto/ (full snapshot).
    Pass a subset to create targeted commits (e.g. ingest vs approve).
    """
    if paths is None:
        paths = ["wiki/", "raw/", "vault-schema.md", ".synto/"]
    try:
        if _has_pre_staged_changes(vault):
            log.warning("git_commit: pre-staged changes detected — skipping auto-commit")
            return "blocked"
        _run(["git", "add"] + paths, cwd=vault)
        # Check if there's anything staged
        result = _run(["git", "status", "--porcelain"], cwd=vault)
        if not result.stdout.strip():
            log.debug("git_commit: nothing to commit")
            return "nothing"
        _run(["git", "commit", "-m", f"{AUTO_COMMIT_PREFIX} {message}"], cwd=vault)
        log.info("git commit: %s %s", AUTO_COMMIT_PREFIX, message)
        return "committed"
    except subprocess.CalledProcessError as e:
        log.warning("git commit failed: %s", e.stderr)
        return "failed"


def git_log_auto(vault: Path, n: int = 10) -> list[dict]:
    """Return last N synto/legacy auto-commits as list of {hash, message}."""
    try:
        result = _run(
            ["git", "log", f"--max-count={n * 3}", "--oneline", "--format=%H %s"],
            cwd=vault,
        )
        commits = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and _is_auto_commit_subject(parts[1]):
                commits.append({"hash": parts[0], "message": parts[1]})
                if len(commits) >= n:
                    break
        return commits
    except subprocess.CalledProcessError:
        return []


def git_undo(vault: Path, steps: int = 1) -> list[str]:
    """
    Revert last N synto/legacy auto-commits using git revert (safe — creates new commits).
    Returns list of reverted commit messages.
    """
    commits = git_log_auto(vault, n=steps)
    if not commits:
        return []
    reverted = []
    for c in commits:
        try:
            _run(
                ["git", "-c", "merge.conflictstyle=merge", "revert", "--no-edit", c["hash"]],
                cwd=vault,
            )
            reverted.append(c["message"])
        except subprocess.CalledProcessError as e:
            log.warning("git revert failed for %s: %s", c["hash"], e.stderr)
            break
    return reverted


def git_init(vault: Path) -> None:
    """Init git repo if not already initialised."""
    if not (vault / ".git").exists():
        _run(["git", "init"], cwd=vault)
        log.info("Initialised git repo at %s", vault)
