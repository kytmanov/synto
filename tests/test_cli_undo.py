"""CLI wiring for `synto undo` — the identity-op divergence guard (QA finding F1).

`git revert` restores the working tree but never the gitignored state.db, so reverting a
concept identity-op commit (merge/split/unmerge/rename) silently diverges the DB from disk.
These pin that the command refuses such a batch, names the DB-aware inverse, still honours
--force, and leaves ordinary data ops (ingest/compile) revertible. The reversal-hint helper
is unit-tested directly so each op's inverse is encoded precisely.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from synto.cli import _identity_op_reversal_hint, _is_identity_op_subject, cli


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _commit(vault: Path, subject: str) -> None:
    _git(["add", "-A"], vault)
    _git(["commit", "-m", subject, "--allow-empty"], vault)


def _count(vault: Path) -> int:
    out = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=vault,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    for sub in ("raw", "wiki", "wiki/.drafts", ".synto"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    _git(["init"], tmp_path)
    # Persist identity in the repo config: `synto undo` runs `git revert` in its own
    # subprocess (git_ops.git_undo), which does not pass the per-call `-c user.*` flags
    # that the `_git` helper uses. Without repo-local identity the revert cannot commit on
    # a machine that has no global git identity (e.g. CI), so it silently adds no commit.
    _git(["config", "user.email", "t@t"], tmp_path)
    _git(["config", "user.name", "t"], tmp_path)
    _git(["config", "commit.gpgsign", "false"], tmp_path)
    (tmp_path / "wiki" / "seed.md").write_text("seed")
    _commit(tmp_path, "initial")
    return tmp_path


# ── command-level guard ─────────────────────────────────────────────────────────


def test_undo_refuses_identity_op_and_names_inverse(vault: Path) -> None:
    (vault / "wiki" / "seed.md").write_text("after merge")
    _commit(vault, "[synto] concept merge: Alpha → Beta")
    before = _count(vault)

    result = CliRunner().invoke(cli, ["undo", "--vault", str(vault)])

    assert result.exit_code == 1, result.output
    assert 'concept unmerge "Alpha"' in result.output
    # Refusal must not run git revert — no new commit, file untouched.
    assert _count(vault) == before
    assert (vault / "wiki" / "seed.md").read_text() == "after merge"


def test_undo_force_reverts_identity_op_with_db_caveat(vault: Path) -> None:
    (vault / "wiki" / "seed.md").write_text("after merge")
    _commit(vault, "[synto] concept merge: Alpha → Beta")
    before = _count(vault)

    result = CliRunner().invoke(cli, ["undo", "--force", "--vault", str(vault)])

    assert result.exit_code == 0, result.output
    assert _count(vault) == before + 1, "revert should add a commit"
    assert (vault / "wiki" / "seed.md").read_text() == "seed", "file reverted"
    # The caveat must steer to the concept inverse, NOT to `synto compile` (which cannot
    # reconcile a merge), and flag the state.db divergence.
    assert "state.db" in result.output
    assert "concept" in result.output
    assert "compile` will NOT reconcile" in result.output


def test_undo_allows_non_identity_data_op(vault: Path) -> None:
    (vault / "wiki" / "seed.md").write_text("after ingest")
    _commit(vault, "[synto] ingest: 1 note")
    before = _count(vault)

    result = CliRunner().invoke(cli, ["undo", "--vault", str(vault)])

    assert result.exit_code == 0, result.output
    assert _count(vault) == before + 1
    assert (vault / "wiki" / "seed.md").read_text() == "seed"
    # Non-identity ops keep the original soft caveat pointing at compile.
    assert "synto compile" in result.output


# ── reversal-hint helper (per-op inverse) ────────────────────────────────────────


@pytest.mark.parametrize(
    "subject,expected",
    [
        ("[synto] concept merge: Alpha → Beta", 'synto concept unmerge "Alpha"'),
        ("[synto] concept unmerge: Alpha ← Beta", 'synto concept merge "Alpha" "Beta"'),
        (
            "[synto] concept rename: Old Name → New Name",
            'synto concept rename "New Name" "Old Name"',
        ),
    ],
)
def test_reversal_hint_names_correct_inverse(subject: str, expected: str) -> None:
    assert _is_identity_op_subject(subject)
    assert expected in _identity_op_reversal_hint(subject)


def test_reversal_hint_split_has_no_single_inverse(vault: Path) -> None:
    hint = _identity_op_reversal_hint("[synto] concept split: Mercury → senses")
    assert "merge" in hint and "no single inverse" in hint


def test_non_identity_subject_is_not_flagged() -> None:
    assert not _is_identity_op_subject("[synto] ingest: 1 note")
    assert not _is_identity_op_subject("[synto] compile: 3 drafts")
