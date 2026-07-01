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


def test_fix_broken_links_heals_dangling_link_to_existing_page(config, db):
    """[[Phase II)]] beside an existing 'Phase II' page is rewritten to [[Phase II]] (issue #53).

    The malformed link is genuinely broken in Obsidian (it targets 'Phase II)'); healing it both
    fixes the link and removes it from still_broken so no duplicate stub is created.
    """
    _write_article(config, "Phase II", "## Body\n\nContent.")

    article = config.wiki_dir / "Ref.md"
    post = fm_lib.Post(
        "See [[Phase II)]] for details.", title="Ref", status="published", tags=[], sources=[]
    )
    atomic_write(article, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description="[[Phase II)]] has no matching wiki page",
            suggestion="Fix link",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 1
    assert report.still_broken == []

    _, body = parse_note(article)
    assert "[[Phase II]]" in body
    assert "[[Phase II)]]" not in body


def test_fix_broken_links_preserves_punctuation_before_dangling_bracket(config, db):
    """[[Yahoo!)]] beside an existing 'Yahoo!' page must heal to [[Yahoo!]], not [[Yahoo]]."""
    _write_article(config, "Yahoo!", "## Body\n\nContent.")

    article = config.wiki_dir / "Ref.md"
    post = fm_lib.Post(
        "See [[Yahoo!)]] for details.", title="Ref", status="published", tags=[], sources=[]
    )
    atomic_write(article, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description="[[Yahoo!)]] has no matching wiki page",
            suggestion="Fix link",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 1
    assert report.still_broken == []

    _, body = parse_note(article)
    assert "[[Yahoo!]]" in body
    assert "[[Yahoo)]]" not in body
    assert "[[Yahoo]]" not in body


def test_fix_broken_links_resyncs_content_hash(config, db):
    """A machine link-rewrite must not make compile treat the article as manually edited.

    Regression for #83: fix_broken_links rewrote the published body but left the DB
    content_hash reflecting the pre-rewrite body, so the next `compile` saw
    on-disk-hash != DB-hash and skipped the concept as 'manually edited'. The stored
    hash must track the body the writer actually left on disk.
    """
    from synto.models import WikiArticleRecord
    from synto.pipeline.compile import _content_hash

    canonical = "Machine Learning"
    _write_article(config, canonical, "## Body\n\nContent.")
    db.upsert_aliases(canonical, ["ML"])

    article = config.wiki_dir / "Test Article.md"
    post = fm_lib.Post(
        "See [[ML]] for details.", title="Test Article", status="published", tags=[], sources=[]
    )
    atomic_write(article, fm_lib.dumps(post))
    rel = str(article.relative_to(config.vault))
    # Register the article as if freshly published: DB hash == on-disk body hash.
    _, body_before = parse_note(article)
    db.upsert_article(
        WikiArticleRecord(
            path=rel,
            title="Test Article",
            sources=[],
            content_hash=_content_hash(body_before),
            status="published",
        )
    )

    issues = [
        LintIssue(
            path=rel,
            issue_type="broken_link",
            description="[[ML]] not found",
            suggestion="Fix link",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 1

    _, body_after = parse_note(article)
    art = db.get_article(rel)
    assert art is not None
    # The stored hash tracks the rewritten body …
    assert art.content_hash == _content_hash(body_after)
    # … so compile's manual-edit guard (blank-tolerant) does not fire on this machine rewrite.
    assert not (art.content_hash and art.content_hash != _content_hash(body_after))


def test_fix_broken_links_preserves_synthesis_kind(config, db):
    """Repairing a link inside a synthesis page must not demote it to kind='concept'.

    Regression for review Issue 3: the shared _update_article_hash rebuilt a partial
    WikiArticleRecord, and the upsert clobbered kind/question_hash from the record's defaults —
    so a synthesis page with a repairable link lost its synthesis identity and provenance.
    """
    from synto.models import WikiArticleRecord
    from synto.pipeline.compile import _content_hash

    canonical = "Machine Learning"
    _write_article(config, canonical, "## Body\n\nContent.")
    db.upsert_aliases(canonical, ["ML"])

    synth_dir = config.wiki_dir / "synthesis"
    synth_dir.mkdir(parents=True, exist_ok=True)
    synth = synth_dir / "Topic.md"
    post = fm_lib.Post(
        "Discusses [[ML]] at length.",
        title="Topic",
        status="published",
        kind="synthesis",
        question_hash="qh123",
        tags=[],
        sources=[],
    )
    atomic_write(synth, fm_lib.dumps(post))
    rel = str(synth.relative_to(config.vault))
    _, body_before = parse_note(synth)
    db.upsert_article(
        WikiArticleRecord(
            path=rel,
            title="Topic",
            sources=[],
            content_hash=_content_hash(body_before),
            status="published",
            kind="synthesis",
            question_hash="qh123",
        )
    )

    issues = [
        LintIssue(
            path=rel,
            issue_type="broken_link",
            description="[[ML]] not found",
            suggestion="Fix link",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 1

    art = db.get_article(rel)
    assert art is not None
    assert art.kind == "synthesis"  # not demoted to "concept"
    assert art.question_hash == "qh123"  # provenance intact
    _, body_after = parse_note(synth)
    assert art.content_hash == _content_hash(body_after)


def test_fix_broken_links_targets_sanitized_stem_not_title(config, db):
    """A title with filename-forbidden chars must be linked by its stem, not the raw title.

    'TCP/IP' is stored as TCPIP.md; a wikilink only resolves to the sanitized stem. The repair must
    emit [[TCPIP|TCP/IP]], not [[TCP/IP]] (which stays broken) — and not falsely report success.
    """
    from synto.vault import sanitize_filename

    tcp = _write_article(config, "TCP/IP", "## Body\n\nNetworking.")
    assert tcp.name == "TCPIP.md"  # title sanitized to stem on disk

    article = config.wiki_dir / "Ref.md"
    post = fm_lib.Post(
        "See [[TCP/IP)]] for details.", title="Ref", status="published", tags=[], sources=[]
    )
    atomic_write(article, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description="[[TCP/IP)]] has no matching wiki page",
            suggestion="Fix link",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 1
    assert report.still_broken == []

    _, body = parse_note(article)
    assert "[[TCPIP|TCP/IP]]" in body
    assert sanitize_filename("TCP/IP") == "TCPIP"


def test_fix_broken_links_alias_to_forbidden_char_canonical(config, db):
    """An alias whose canonical title has forbidden chars resolves to the canonical's stem."""
    _write_article(config, "TCP/IP", "## Body")
    db.upsert_aliases("TCP/IP", ["IP stack"])

    article = config.wiki_dir / "Ref.md"
    post = fm_lib.Post(
        "See [[IP stack]] for details.", title="Ref", status="published", tags=[], sources=[]
    )
    atomic_write(article, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description="[[IP stack]] has no matching wiki page",
            suggestion="Fix link",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 1
    assert report.still_broken == []

    _, body = parse_note(article)
    assert "[[TCPIP|IP stack]]" in body


def test_fix_broken_links_preserves_path_style_link(config, db):
    """A dangling path link to a real source page heals to the path, not a sanitized stem.

    [[sources/Paper)]] must become [[sources/Paper]] (the '/' is meaningful and resolves by path) —
    never [[sourcesPaper|sources/Paper]], which stays broken.
    """
    sources = config.wiki_dir / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    paper = sources / "Paper.md"
    post = fm_lib.Post("Source summary.", title="Paper", status="published", tags=[], sources=[])
    atomic_write(paper, fm_lib.dumps(post))

    article = config.wiki_dir / "Ref.md"
    ref = fm_lib.Post(
        "See [[sources/Paper)]] for details.", title="Ref", status="published", tags=[], sources=[]
    )
    atomic_write(article, fm_lib.dumps(ref))

    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description="[[sources/Paper)]] has no matching wiki page",
            suggestion="Run `synto run` ...",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 1
    assert report.still_broken == []

    _, body = parse_note(article)
    assert "[[sources/Paper]]" in body
    assert "sourcesPaper" not in body


def test_fix_broken_links_unknown_path_style_not_corrupted(config, db):
    """A path link with no matching page is left untouched — not sanitized, not stubbed."""
    from synto.pipeline.maintain import create_stubs

    article = config.wiki_dir / "Ref.md"
    ref = fm_lib.Post(
        "See [[sources/Ghost)]] for details.", title="Ref", status="published", tags=[], sources=[]
    )
    atomic_write(article, fm_lib.dumps(ref))

    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description="[[sources/Ghost)]] has no matching wiki page",
            suggestion="Run `synto run` ...",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 0
    assert len(report.still_broken) == 1

    _, body = parse_note(article)
    assert "[[sources/Ghost)]]" in body  # untouched
    assert "sourcesGhost" not in body  # not corrupted into a stem

    # And no bogus stub is created for a path-style target.
    created = create_stubs(config, db, broken_link_issues=report.still_broken)
    assert created == []


def test_fix_broken_links_dangling_unresolved_rewrites_and_stays_broken(config, db):
    """A malformed link to an unknown page is rewritten to the clean target AND stays broken.

    The body is normalized to [[Foo]] so the stub create_stubs makes for it ('Foo.md') matches —
    but it remains in still_broken because no page resolves it yet.
    """
    article = config.wiki_dir / "Ref.md"
    post = fm_lib.Post(
        "See [[Foo)]] for details.", title="Ref", status="published", tags=[], sources=[]
    )
    atomic_write(article, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description="[[Foo)]] has no matching wiki page",
            suggestion="Fix link",
        )
    ]
    report = fix_broken_links(config, db, issues)
    assert report.repaired == 1
    assert len(report.still_broken) == 1

    _, body = parse_note(article)
    assert "[[Foo]]" in body
    assert "[[Foo)]]" not in body


def test_fix_broken_links_dangling_dry_run_does_not_write(config, db):
    _write_article(config, "Phase II", "## Body")

    article = config.wiki_dir / "Ref.md"
    post = fm_lib.Post("See [[Phase II)]].", title="Ref", status="published", tags=[], sources=[])
    atomic_write(article, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description="[[Phase II)]] has no matching wiki page",
            suggestion="Fix link",
        )
    ]
    report = fix_broken_links(config, db, issues, dry_run=True)
    assert report.repaired == 1

    _, body = parse_note(article)
    assert "[[Phase II)]]" in body  # untouched in dry run


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
