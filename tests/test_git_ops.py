from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from synto import git_ops
from synto.git_ops import git_commit


def test_is_auto_commit_subject_matches_only_direct_prefix():
    assert git_ops._is_auto_commit_subject("[synto] approve: 3 articles published") is True
    assert (
        git_ops._is_auto_commit_subject('Revert "[synto] approve: 3 articles published"') is False
    )
    assert git_ops._is_auto_commit_subject("baseline") is False


def test_git_log_auto_ignores_revert_subjects(monkeypatch):
    log_output = "\n".join(
        [
            'aaa111 Revert "[synto] approve: 2 articles published"',
            "bbb222 [synto] approve: 2 articles published",
            "ccc333 [synto] ingest: 2 notes",
            "ddd444 baseline",
        ]
    )

    class Result:
        stdout = log_output

    monkeypatch.setattr(git_ops, "_run", lambda args, cwd, check=True: Result())

    commits = git_ops.git_log_auto(Path("/tmp/fake"), n=3)

    assert commits == [
        {"hash": "bbb222", "message": "[synto] approve: 2 articles published"},
        {"hash": "ccc333", "message": "[synto] ingest: 2 notes"},
    ]


# ── git_commit return values ──────────────────────────────────────────────────


def test_git_commit_returns_committed(tmp_path, monkeypatch):
    """Returns 'committed' when git stages and commits changes."""

    def mock_run(args, cwd, check=True):
        r = MagicMock()
        r.stdout = "M wiki/Article.md\n" if "status" in args else ""
        return r

    monkeypatch.setattr(git_ops, "_run", mock_run)
    assert git_commit(tmp_path, "test message") == "committed"


def test_git_commit_returns_nothing_when_no_changes(tmp_path, monkeypatch):
    """Returns 'nothing' when nothing is staged after git add."""

    def mock_run(args, cwd, check=True):
        r = MagicMock()
        r.stdout = ""
        return r

    monkeypatch.setattr(git_ops, "_run", mock_run)
    assert git_commit(tmp_path, "test message") == "nothing"


def test_git_commit_returns_failed_on_error(tmp_path, monkeypatch):
    """Returns 'failed' when git raises CalledProcessError."""

    def mock_run(args, cwd, check=True):
        raise subprocess.CalledProcessError(1, "git", stderr="not a git repo")

    monkeypatch.setattr(git_ops, "_run", mock_run)
    assert git_commit(tmp_path, "test message") == "failed"


def test_git_commit_blocked_when_pre_staged(tmp_path, monkeypatch):
    """Returns 'blocked' when the index already has staged changes."""

    def mock_run(args, cwd, check=True):
        r = MagicMock()
        r.stdout = "wiki/secret.md\n" if "diff" in args else ""
        return r

    monkeypatch.setattr(git_ops, "_run", mock_run)
    assert git_commit(tmp_path, "msg") == "blocked"


def test_git_commit_not_blocked_when_nothing_pre_staged(tmp_path, monkeypatch):
    """Proceeds normally when no pre-staged changes exist."""

    def mock_run(args, cwd, check=True):
        r = MagicMock()
        if "diff" in args:
            r.stdout = ""  # nothing pre-staged
        elif "status" in args:
            r.stdout = "M wiki/Article.md\n"
        else:
            r.stdout = ""
        return r

    monkeypatch.setattr(git_ops, "_run", mock_run)
    assert git_commit(tmp_path, "msg") == "committed"


# ── failed commit must not leave the index staged (real git) ──────────────────


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def _real_vault(tmp_path: Path) -> Path:
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.email", "t@t"], tmp_path)
    _git(["config", "user.name", "t"], tmp_path)
    _git(["config", "commit.gpgsign", "false"], tmp_path)
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "a.md").write_text("a\n", encoding="utf-8")
    return tmp_path


def _staged(vault: Path) -> list[str]:
    return _git(["diff", "--cached", "--name-only"], vault).stdout.split()


def test_git_commit_failure_unstages_so_next_commit_is_not_blocked(tmp_path):
    """A rejected commit (here: a pre-commit hook) must not leave synto's paths staged.

    Otherwise the next call mistakes them for the user's staged work and returns
    'blocked' forever. Applies to unborn branches too (first commit in a new vault).
    """
    vault = _real_vault(tmp_path)
    hook = vault / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)

    assert git_commit(vault, "ingest", ["wiki/"]) == "failed"
    assert _staged(vault) == []

    hook.unlink()
    (vault / "wiki" / "b.md").write_text("b\n", encoding="utf-8")
    assert git_commit(vault, "approve", ["wiki/"]) == "committed"


def test_git_commit_failed_add_on_ignored_path_unstages(tmp_path):
    """`git add` exits non-zero on an ignored path but still stages the others."""
    vault = _real_vault(tmp_path)
    (vault / ".gitignore").write_text(".synto/\n", encoding="utf-8")
    (vault / ".synto").mkdir()
    (vault / ".synto" / "state.db").write_text("x", encoding="utf-8")

    assert git_commit(vault, "ingest", ["wiki/", ".synto/"]) == "failed"
    assert _staged(vault) == []
    assert git_commit(vault, "ingest", ["wiki/"]) == "committed"


def test_git_commit_failure_after_history_keeps_prior_commits(tmp_path):
    """Unstaging on a branch with history resets the index to HEAD, not to empty."""
    vault = _real_vault(tmp_path)
    assert git_commit(vault, "first", ["wiki/"]) == "committed"
    hook = vault / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    (vault / "wiki" / "a.md").write_text("changed\n", encoding="utf-8")

    assert git_commit(vault, "second", ["wiki/"]) == "failed"
    assert _staged(vault) == []
    tracked = _git(["ls-files"], vault).stdout.split()
    assert tracked == ["wiki/a.md"]


# ── git_undo dirty-tree guard ─────────────────────────────────────────────────


def test_git_undo_raises_when_working_tree_dirty(tmp_path, monkeypatch):
    """Raises RuntimeError when tracked files have uncommitted changes."""

    def mock_run(args, cwd, check=True):
        r = MagicMock()
        r.stdout = " M wiki/Article.md\n"  # modified tracked file
        return r

    monkeypatch.setattr(git_ops, "_run", mock_run)
    import pytest

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        git_ops.git_undo(tmp_path)


def test_git_undo_ignores_untracked_files(tmp_path, monkeypatch):
    """Untracked files (e.g. .gitignore) don't block undo — git revert works fine with them."""
    call_log: list[list[str]] = []

    def mock_run(args, cwd, check=True):
        call_log.append(list(args))
        r = MagicMock()
        if "porcelain" in args:
            r.stdout = "?? .gitignore\n"  # untracked only — should not block
        elif args[:2] == ["git", "log"]:
            r.stdout = "abc123 [synto] ingest: 1 note\n"
        else:
            r.stdout = ""
        return r

    monkeypatch.setattr(git_ops, "_run", mock_run)
    reverted = git_ops.git_undo(tmp_path, steps=1)
    assert reverted == ["[synto] ingest: 1 note"]


def test_git_undo_proceeds_when_working_tree_clean(tmp_path, monkeypatch):
    """Reverts the target commit when the working tree is clean."""
    call_log: list[list[str]] = []

    def mock_run(args, cwd, check=True):
        call_log.append(list(args))
        r = MagicMock()
        if "porcelain" in args:
            r.stdout = ""  # clean
        elif args[:2] == ["git", "log"]:
            r.stdout = "abc123 [synto] ingest: 1 note\n"
        else:
            r.stdout = ""
        return r

    monkeypatch.setattr(git_ops, "_run", mock_run)
    reverted = git_ops.git_undo(tmp_path, steps=1)
    assert reverted == ["[synto] ingest: 1 note"]
    assert any("revert" in " ".join(c) for c in call_log)
