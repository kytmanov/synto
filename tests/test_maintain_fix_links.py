"""Additional tests for maintain.py fix_broken_links and edge cases."""

from __future__ import annotations

from pathlib import Path

import frontmatter as fm_lib
import pytest

from synto.config import Config
from synto.models import LintIssue
from synto.pipeline.maintain import fix_broken_links
from synto.state import StateDB
from synto.vault import atomic_write, parse_note


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".synto").mkdir()
    return tmp_path


@pytest.fixture
def config(vault: Path) -> Config:
    return Config(vault=vault)


@pytest.fixture
def db(config: Config) -> StateDB:
    return StateDB(config.state_db_path)


def _write_article(config: Config, title: str, body: str) -> Path:
    from synto.vault import sanitize_filename

    path = config.wiki_dir / f"{sanitize_filename(title)}.md"
    post = fm_lib.Post(body, title=title, status="published", tags=[], sources=[])
    atomic_write(path, fm_lib.dumps(post))
    return path


def test_fix_broken_links_no_alias_map(config, db):
    """Broken links but no alias map → all still broken."""
    issues = [
        LintIssue(
            path="wiki/Article.md",
            issue_type="broken_link",
            description="[[Unknown Topic]] not found",
            suggestion="Fix link",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 0
    assert len(report.still_broken) == 1


def test_fix_broken_links_rewrites_alias(config, db):
    """Alias maps to known canonical → link is rewritten."""
    canonical = "Machine Learning"
    _write_article(config, canonical, "## Body\n\nContent.")
    db.upsert_aliases(canonical, ["ML"])

    article = config.wiki_dir / "Test Article.md"
    post = fm_lib.Post(
        "See [[ML]] for details.",
        title="Test Article",
        status="published",
        tags=[],
        sources=[],
    )
    atomic_write(article, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Test Article.md",
            issue_type="broken_link",
            description="[[ML]] not found",
            suggestion="Fix link",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 1
    assert len(report.repaired_links) == 1
    old_link, new_link = report.repaired_links[0][1], report.repaired_links[0][2]
    assert old_link == "[[ML]]"
    assert new_link == "[[Machine Learning|ML]]"

    _, body = parse_note(article)
    assert "[[Machine Learning|ML]]" in body


def test_fix_broken_links_unknown_alias_stays_broken(config, db):
    """Alias not in map → stays in still_broken."""
    _write_article(config, "Known Topic", "## Body\n\nContent.")

    article = config.wiki_dir / "Test.md"
    post = fm_lib.Post(
        "See [[Unknown]] for details.",
        title="Test",
        status="published",
        tags=[],
        sources=[],
    )
    atomic_write(article, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Test.md",
            issue_type="broken_link",
            description="[[Unknown]] not found",
            suggestion="Fix link",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 0
    assert len(report.still_broken) == 1


def test_fix_broken_links_alias_maps_to_nonexistent_canonical(config, db):
    """Alias maps to canonical but canonical article doesn't exist → still broken."""
    db.upsert_aliases("Ghost Topic", ["GT"])
    # No article for "Ghost Topic" exists

    article = config.wiki_dir / "Test.md"
    post = fm_lib.Post(
        "See [[GT]] for details.",
        title="Test",
        status="published",
        tags=[],
        sources=[],
    )
    atomic_write(article, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Test.md",
            issue_type="broken_link",
            description="[[GT]] not found",
            suggestion="Fix link",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 0
    assert len(report.still_broken) == 1


def test_fix_broken_links_dry_run(config, db):
    """Dry run does not modify files."""
    canonical = "Neural Networks"
    _write_article(config, canonical, "## Body\n\nContent.")
    db.upsert_aliases(canonical, ["NN"])

    article = config.wiki_dir / "Ref.md"
    post = fm_lib.Post(
        "See [[NN]] for details.",
        title="Ref",
        status="published",
        tags=[],
        sources=[],
    )
    atomic_write(article, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description="[[NN]] not found",
            suggestion="Fix link",
        )
    ]
    report = fix_broken_links(config, db, issues, dry_run=True)
    assert report.repaired == 1

    _, body = parse_note(article)
    assert "[[NN]]" in body  # not rewritten in dry run


def test_fix_broken_links_multiple_issues_same_file(config, db):
    """Multiple broken links in same file → one write, multiple repairs."""
    canonical_a = "Concept A"
    canonical_b = "Concept B"
    _write_article(config, canonical_a, "Body A")
    _write_article(config, canonical_b, "Body B")
    db.upsert_aliases(canonical_a, ["CA"])
    db.upsert_aliases(canonical_b, ["CB"])

    article = config.wiki_dir / "Multi.md"
    post = fm_lib.Post(
        "See [[CA]] and [[CB]] for details.",
        title="Multi",
        status="published",
        tags=[],
        sources=[],
    )
    atomic_write(article, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Multi.md",
            issue_type="broken_link",
            description="[[CA]] not found",
            suggestion="Fix",
        ),
        LintIssue(
            path="wiki/Multi.md",
            issue_type="broken_link",
            description="[[CB]] not found",
            suggestion="Fix",
        ),
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 2


def test_fix_broken_links_skips_parse_error(config, db):
    """File that can't be parsed → skipped, issues stay broken."""
    canonical = "Topic"
    _write_article(config, canonical, "Body")
    db.upsert_aliases(canonical, ["T"])

    # Write an invalid file that will fail parsing
    bad_file = config.vault / "wiki" / "Bad.md"
    bad_file.write_bytes(b"\x80\x81\x82")  # invalid bytes

    issues = [
        LintIssue(
            path="wiki/Bad.md",
            issue_type="broken_link",
            description="[[T]] not found",
            suggestion="Fix",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert len(report.skipped_files) == 1
    assert len(report.still_broken) == 1


def test_fix_broken_links_preserves_code_blocks(config, db):
    """Links inside code fences are not rewritten."""
    canonical = "Python"
    _write_article(config, canonical, "## Body")
    db.upsert_aliases(canonical, ["PY"])

    article = config.wiki_dir / "Code.md"
    body = """## Article

Normal text references [[PY]].

```
# This [[PY]] should not be touched
code = "[[PY]]"
```

After code block, [[PY]] again.
"""
    post = fm_lib.Post(
        body,
        title="Code",
        status="published",
        tags=[],
        sources=[],
    )
    atomic_write(article, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Code.md",
            issue_type="broken_link",
            description="[[PY]] not found",
            suggestion="Fix",
        )
    ]
    fix_broken_links(config, db, issues)

    _, body_after = parse_note(article)
    # Code block content should be unchanged
    assert 'code = "[[PY]]"' in body_after
    # Outside code should be rewritten
    assert "[[Python|PY]]" in body_after


def test_fix_broken_links_repair_report_records_path(config, db):
    """Repair report records the source file path for each repair."""
    canonical = "Topic"
    _write_article(config, canonical, "Body")
    db.upsert_aliases(canonical, ["T"])

    article = config.wiki_dir / "Report.md"
    post = fm_lib.Post(
        "See [[T]].",
        title="Report",
        status="published",
        tags=[],
        sources=[],
    )
    atomic_write(article, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Report.md",
            issue_type="broken_link",
            description="[[T]] not found",
            suggestion="Fix",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired_links[0][0] == "wiki/Report.md"


def test_fix_broken_links_with_fragment(config, db):
    """Alias link with heading fragment — target extraction strips fragment."""
    canonical = "Machine Learning"
    _write_article(config, canonical, "## Body\n\n## Subsection\n\nContent.")
    db.upsert_aliases(canonical, ["ML"])

    article = config.wiki_dir / "Frag.md"
    post = fm_lib.Post(
        "See [[ML#Subsection]] for details.",
        title="Frag",
        status="published",
        tags=[],
        sources=[],
    )
    atomic_write(article, fm_lib.dumps(post))

    # The lint issue description extracts the target without the fragment
    issues = [
        LintIssue(
            path="wiki/Frag.md",
            issue_type="broken_link",
            description="[[ML]] not found",
            suggestion="Fix link",
        )
    ]
    fix_broken_links(config, db, issues)

    _, body = parse_note(article)
    # The fragment form [[ML#Subsection]] gets rewritten to [[Machine Learning#Subsection|ML]]
    assert "[[Machine Learning#Subsection|ML]]" in body
