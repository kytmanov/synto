"""Additional tests for git_ops.py uncovered paths."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from synto import git_ops
from synto.git_ops import git_commit, git_init, git_log_auto, git_undo


def test_git_log_auto_stops_at_n_commits(monkeypatch):
    """git_log_auto returns exactly n commits even when more exist."""
    log_output = "\n".join(
        [
            "aaa111 [synto] commit 1",
            "bbb222 [synto] commit 2",
            "ccc333 [synto] commit 3",
            "ddd444 [synto] commit 4",
            "eee555 [synto] commit 5",
        ]
    )

    class Result:
        stdout = log_output

    monkeypatch.setattr(git_ops, "_run", lambda args, cwd, check=True: Result())
    commits = git_log_auto(Path("/tmp/fake"), n=2)
    assert len(commits) == 2
    assert commits[0]["message"] == "[synto] commit 1"
    assert commits[1]["message"] == "[synto] commit 2"


def test_git_log_auto_skips_empty_lines(monkeypatch):
    """Empty lines in git log output are skipped."""
    log_output = "\n\naaa111 [synto] commit 1\n\nbbb222 [synto] commit 2\n\n"

    class Result:
        stdout = log_output

    monkeypatch.setattr(git_ops, "_run", lambda args, cwd, check=True: Result())
    commits = git_log_auto(Path("/tmp/fake"), n=5)
    assert len(commits) == 2


def test_git_log_auto_returns_empty_on_process_error(monkeypatch):
    """git_log_auto returns [] when git log fails."""

    def fail_run(args, cwd, check=True):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(git_ops, "_run", fail_run)
    commits = git_log_auto(Path("/tmp/fake"))
    assert commits == []


def test_git_log_auto_empty_output(monkeypatch):
    """git_log_auto returns [] when repo has no commits."""

    class Result:
        stdout = ""

    monkeypatch.setattr(git_ops, "_run", lambda args, cwd, check=True: Result())
    commits = git_log_auto(Path("/tmp/fake"))
    assert commits == []


def test_git_log_auto_malformed_line(monkeypatch):
    """Lines without space separator are skipped."""
    log_output = "no-space-here\naaa111 [synto] valid commit"

    class Result:
        stdout = log_output

    monkeypatch.setattr(git_ops, "_run", lambda args, cwd, check=True: Result())
    commits = git_log_auto(Path("/tmp/fake"), n=5)
    assert len(commits) == 1
    assert commits[0]["hash"] == "aaa111"


# ── git_undo ──────────────────────────────────────────────────────────────────


def test_git_undo_stops_on_revert_failure(monkeypatch):
    """git_undo stops reverting when one revert fails."""
    monkeypatch.setattr(
        git_ops,
        "git_log_auto",
        lambda vault, n: [
            {"hash": "aaa", "message": "[synto] first"},
            {"hash": "bbb", "message": "[synto] second"},
        ],
    )

    call_count = 0

    def mock_run(args, cwd, check=True):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise subprocess.CalledProcessError(1, "git", stderr="conflict")
        r = MagicMock()
        r.stdout = ""
        return r

    monkeypatch.setattr(git_ops, "_run", mock_run)
    result = git_undo(Path("/tmp/fake"), steps=2)
    assert result == ["[synto] first"]


# ── git_init ──────────────────────────────────────────────────────────────────


def test_git_init_skips_when_git_exists(tmp_path, monkeypatch):
    """git_init does nothing when .git already exists."""
    (tmp_path / ".git").mkdir()
    calls = []

    def mock_run(args, cwd, check=True):
        calls.append(args)
        r = MagicMock()
        r.stdout = ""
        return r

    monkeypatch.setattr(git_ops, "_run", mock_run)
    git_init(tmp_path)
    assert calls == []


# ── _has_pre_staged_changes ───────────────────────────────────────────────────


def test_has_pre_staged_changes_fail_open(monkeypatch):
    """Returns False when the check itself errors (fail open)."""

    def mock_run(args, cwd, check=True):
        raise RuntimeError("git broken")

    monkeypatch.setattr(git_ops, "_run", mock_run)
    assert git_ops._has_pre_staged_changes(Path("/tmp/fake")) is False


# ── git_commit with custom paths ──────────────────────────────────────────────


def test_git_commit_with_custom_paths(tmp_path, monkeypatch):
    """git_commit stages only the specified paths."""
    staged_paths = []

    def mock_run(args, cwd, check=True):
        r = MagicMock()
        if args[0] == "git" and args[1] == "add":
            staged_paths.extend(args[2:])
        elif "status" in args:
            r.stdout = "M wiki/Article.md\n"
        else:
            r.stdout = ""
        return r

    monkeypatch.setattr(git_ops, "_run", mock_run)
    git_commit(tmp_path, "test", paths=["wiki/Article.md"])
    assert staged_paths == ["wiki/Article.md"]
