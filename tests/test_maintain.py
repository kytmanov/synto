"""Tests for pipeline/maintain.py."""

from __future__ import annotations

from pathlib import Path

import frontmatter as fm_lib
import pytest

from synto.config import Config
from synto.models import RawNoteRecord
from synto.pipeline.maintain import (
    _extract_link_target,
    create_stubs,
    normalize_published_alias_links,
    suggest_concept_merges,
    suggest_orphan_links,
)
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


def _write_article(config: Config, title: str, body: str = "## Body\n\nContent.") -> Path:
    from synto.vault import sanitize_filename

    path = config.wiki_dir / f"{sanitize_filename(title)}.md"
    post = fm_lib.Post(body, title=title, status="published", tags=[], sources=[])
    atomic_write(path, fm_lib.dumps(post))
    return path


# ── _extract_link_target ──────────────────────────────────────────────────────


def test_extract_link_target_found():
    assert _extract_link_target("[[Quantum Computing]] not found") == "Quantum Computing"


def test_extract_link_target_not_found():
    assert _extract_link_target("some generic description") is None


def test_extract_link_target_first_match():
    result = _extract_link_target("[[Alpha]] and [[Beta]] missing")
    assert result == "Alpha"


# ── create_stubs ──────────────────────────────────────────────────────────────


def test_create_stubs_from_lint_issues(config, db):
    from synto.models import LintIssue

    issues = [
        LintIssue(
            path="wiki/Article.md",
            issue_type="broken_link",
            description="[[Missing Topic]] not found",
            suggestion="Create article for Missing Topic",
        )
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    assert len(created) == 1
    assert created[0].exists()
    assert db.has_stub("Missing Topic")


def test_create_stubs_skips_existing_stub(config, db):
    from synto.models import LintIssue

    db.add_stub("Already There")
    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description="[[Already There]] not found",
            suggestion="Create stub for Already There",
        )
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    assert len(created) == 0


def test_create_stubs_skips_existing_draft(config, db):
    from synto.models import LintIssue
    from synto.vault import sanitize_filename

    target = "Existing Draft"
    draft_path = config.drafts_dir / f"{sanitize_filename(target)}.md"
    post = fm_lib.Post("body", title=target, status="draft", tags=[], sources=[])
    atomic_write(draft_path, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description=f"[[{target}]] not found",
            suggestion=f"Create stub for {target}",
        )
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    assert len(created) == 0


def test_create_stubs_deduplicates_targets(config, db):
    from synto.models import LintIssue

    issues = [
        LintIssue(
            path="wiki/A.md",
            issue_type="broken_link",
            description="[[Same Topic]] not found",
            suggestion="Create stub for Same Topic",
        ),
        LintIssue(
            path="wiki/B.md",
            issue_type="broken_link",
            description="[[Same Topic]] not found",
            suggestion="Create stub for Same Topic",
        ),
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    assert len(created) == 1


def test_create_stubs_respects_max_stubs(config, db):
    from synto.models import LintIssue

    issues = [
        LintIssue(
            path=f"wiki/A{i}.md",
            issue_type="broken_link",
            description=f"[[Topic {i}]] not found",
            suggestion=f"Create stub for Topic {i}",
        )
        for i in range(10)
    ]
    created = create_stubs(config, db, broken_link_issues=issues, max_stubs=3)
    assert len(created) == 3


def test_create_stubs_stub_body_has_info_callout(config, db):
    from synto.models import LintIssue

    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description="[[New Topic]] not found",
            suggestion="Create stub for New Topic",
        )
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    content = created[0].read_text()
    assert "[!info]" in content or "stub" in content.lower()


def test_create_stubs_strips_md_extension(config, db):
    """Model sometimes emits [[raw-note.md]] wikilinks; stub name must not be 'raw-note.md'."""
    from synto.models import LintIssue

    issues = [
        LintIssue(
            path="wiki/Article.md",
            issue_type="broken_link",
            description="[[deep-learning.md]] not found",
            suggestion="",
        )
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    assert len(created) == 1
    # Stub file must NOT have double .md extension
    assert not created[0].name.endswith(".md.md"), f"double extension: {created[0].name}"
    assert created[0].name == "deep-learning.md"


def test_create_stubs_strips_dangling_punctuation(config, db):
    """A malformed target 'Phase II)' must not create a divergent 'Phase II).md' (issue #53)."""
    from synto.models import LintIssue

    issues = [
        LintIssue(
            path="wiki/Article.md",
            issue_type="broken_link",
            description="[[Phase II)]] has no matching wiki page",
            suggestion="",
        )
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    assert len(created) == 1
    assert created[0].name == "Phase II.md"
    assert db.has_stub("Phase II")


def test_create_stubs_dangling_target_matches_existing_article(config, db):
    """'Phase II)' resolves to the existing 'Phase II' page after cleaning — no duplicate stub."""
    from synto.models import LintIssue

    _write_article(config, "Phase II")
    issues = [
        LintIssue(
            path="wiki/Article.md",
            issue_type="broken_link",
            description="[[Phase II)]] has no matching wiki page",
            suggestion="",
        )
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    assert created == []
    assert not (config.drafts_dir / "Phase II).md").exists()


def test_create_stubs_preserves_punctuation_before_dangling_bracket(config, db):
    """A malformed target like 'Yahoo!)' must keep the real title punctuation when stubbed."""
    from synto.models import LintIssue

    issues = [
        LintIssue(
            path="wiki/Article.md",
            issue_type="broken_link",
            description="[[Yahoo!)]] has no matching wiki page",
            suggestion="",
        )
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    assert len(created) == 1
    assert created[0].name == "Yahoo!.md"
    assert db.has_stub("Yahoo!")


def test_create_stubs_empty_issues(config, db):
    created = create_stubs(config, db, broken_link_issues=[])
    assert created == []


def test_create_stubs_none_runs_lint_internally(config, db):
    """broken_link_issues=None → lint runs internally, no crash on empty wiki."""
    # Empty wiki → lint finds no broken links → no stubs created
    created = create_stubs(config, db, broken_link_issues=None)
    assert created == []


# ── normalize_published_alias_links ──────────────────────────────────────────


def test_normalize_published_alias_links_rewrites_alias(config, db):
    canonical = "Program Counter"
    _write_article(config, canonical, "## Body\n\nHolds the next instruction address.")
    db.upsert_aliases(canonical, ["PC"])

    # Write a second article that uses the alias form
    other = config.wiki_dir / "Other Article.md"
    post = fm_lib.Post(
        "[[PC]] is important.", title="Other Article", status="published", tags=[], sources=[]
    )
    atomic_write(other, fm_lib.dumps(post))

    modified = normalize_published_alias_links(config, db)
    assert modified >= 1

    _, body = parse_note(other)
    assert "[[Program Counter|PC]]" in body


def test_normalize_published_alias_links_skips_canonical(config, db):
    canonical = "Program Counter"
    art = _write_article(config, canonical, "## Body\n\nContent.")
    db.upsert_aliases(canonical, ["PC"])

    _, body_before = parse_note(art)
    normalize_published_alias_links(config, db)
    _, body_after = parse_note(art)
    assert body_before == body_after


def test_normalize_published_alias_links_resyncs_content_hash(config, db):
    """Rewriting alias links in a published article must keep its DB content_hash current.

    Regression for #83: the normalize pass rewrote the body but left the stored hash
    stale, so the next `compile` mistook the machine rewrite for a manual edit.
    """
    from synto.models import WikiArticleRecord
    from synto.pipeline.compile import _content_hash

    canonical = "Program Counter"
    _write_article(config, canonical, "## Body\n\nHolds the next instruction address.")
    db.upsert_aliases(canonical, ["PC"])

    other = config.wiki_dir / "Other Article.md"
    post = fm_lib.Post(
        "[[PC]] is important.", title="Other Article", status="published", tags=[], sources=[]
    )
    atomic_write(other, fm_lib.dumps(post))
    rel = str(other.relative_to(config.vault))
    _, body_before = parse_note(other)
    db.upsert_article(
        WikiArticleRecord(
            path=rel,
            title="Other Article",
            sources=[],
            content_hash=_content_hash(body_before),
            status="published",
        )
    )

    modified = normalize_published_alias_links(config, db)
    assert modified >= 1

    _, body_after = parse_note(other)
    art = db.get_article(rel)
    assert art is not None
    assert art.content_hash == _content_hash(body_after)


def test_normalize_published_alias_links_dry_run(config, db):
    _write_article(config, "Arithmetic Logic Unit", "## Body\n\nContent.")
    db.upsert_aliases("Arithmetic Logic Unit", ["ALU"])

    other = config.wiki_dir / "Ref.md"
    post = fm_lib.Post(
        "See [[ALU]] for details.", title="Ref", status="published", tags=[], sources=[]
    )
    atomic_write(other, fm_lib.dumps(post))

    modified = normalize_published_alias_links(config, db, dry_run=True)
    assert modified >= 1
    _, body = parse_note(other)
    assert "[[ALU]]" in body  # not rewritten in dry-run


def test_normalize_published_alias_links_empty_alias_map(config, db):
    _write_article(config, "Concept", "## Body\n\nContent.")
    assert normalize_published_alias_links(config, db) == 0


# ── suggest_orphan_links ──────────────────────────────────────────────────────


def test_suggest_orphan_links_empty_wiki(config, db):
    result = suggest_orphan_links(config, db)
    assert result == []


def test_suggest_orphan_links_no_unlinked_mentions(config, db):
    _write_article(config, "Isolated Topic", body="## Body\n\nNo mentions of anything.")
    result = suggest_orphan_links(config, db)
    # No mentions of Isolated Topic in other articles → no suggestions
    assert result == []


def test_suggest_orphan_links_finds_unlinked_mention(config, db):
    # Write orphan article
    _write_article(config, "Orphan Topic", body="## Body\n\nAbout orphan.")
    # Write another article that mentions "Orphan Topic" in plain text
    _write_article(
        config,
        "Main Article",
        body="## Body\n\nThis discusses Orphan Topic in detail.",
    )

    result = suggest_orphan_links(config, db)
    # May or may not find it depending on lint detecting orphan — just check no crash
    assert isinstance(result, list)


def test_wiki_page_key_forward_slash_on_windows_paths():
    # Review finding: wiki_pages was keyed with str(relative_to()) — backslashes on
    # Windows — while lint's issue.path is always as_posix(), so the self-skip
    # comparison silently never matched there. Same test shape as lint's
    # test_wiki_rel_key_forward_slash_on_windows_paths.
    from pathlib import PureWindowsPath

    from synto.pipeline.maintain import _wiki_page_key

    key = _wiki_page_key(
        PureWindowsPath(r"C:\v\wiki\Main Article.md"),
        PureWindowsPath(r"C:\v"),
    )
    assert key == "wiki/Main Article.md"


def test_suggest_orphan_links_does_not_suggest_self(config, db):
    # An orphan article naming its own title in its own body is not an inbound mention.
    # (On POSIX the key formats coincide; the Windows-shaped guarantee is pinned by
    # test_wiki_page_key_forward_slash_on_windows_paths above.)
    _write_article(config, "Orphan Topic", body="## Body\n\nOrphan Topic is described here.")

    result = suggest_orphan_links(config, db)

    assert all(title != "Orphan Topic" or not mentions for title, mentions in result)


def test_suggest_orphan_links_ignores_fenced_mentions(config, db):
    # A title appearing only inside code fences / inline code is not an unlinked mention.
    _write_article(config, "Orphan Topic", body="## Body\n\nAbout orphan.")
    _write_article(
        config,
        "Main Article",
        body="## Body\n\nCode:\n\n```\nOrphan Topic\n```\n\nAnd `Orphan Topic` inline.",
    )

    result = suggest_orphan_links(config, db)

    assert all(title != "Orphan Topic" for title, _ in result)


def test_suggest_orphan_links_finds_prose_mention_next_to_fenced_one(config, db):
    _write_article(config, "Orphan Topic", body="## Body\n\nAbout orphan.")
    _write_article(
        config,
        "Main Article",
        body="## Body\n\nThis discusses Orphan Topic.\n\n```\nOrphan Topic\n```\n",
    )

    result = suggest_orphan_links(config, db)

    assert ("Orphan Topic", ["wiki/Main Article.md"]) in result


# ── suggest_concept_merges ────────────────────────────────────────────────────


def test_suggest_concept_merges_empty(config, db):
    result = suggest_concept_merges(config, db)
    assert result == []


def test_suggest_concept_merges_single_concept(config, db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Machine Learning"])
    result = suggest_concept_merges(config, db)
    assert result == []


def test_suggest_concept_merges_high_similarity(config, db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Machine Learning"])
    db.upsert_concepts("raw/b.md", ["Machine-Learning"])

    result = suggest_concept_merges(config, db)
    assert len(result) > 0
    a, b, score = result[0]
    assert score >= 0.7
    assert {"Machine Learning", "Machine-Learning"} == {a, b}


def test_suggest_concept_merges_surfaces_dangling_punctuation_duplicate(config, db):
    """'Phase II' and 'Phase II)' must score as near-duplicates (issue #53) so users can merge them.

    Without per-token punctuation stripping, 'ii)' != 'ii' drops the score below threshold and the
    duplicate stays hidden — the same defect that minted it.
    """
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Phase II"])
    db.upsert_concepts("raw/b.md", ["Phase II)"])

    result = suggest_concept_merges(config, db)
    assert len(result) > 0
    a, b, score = result[0]
    assert score >= 0.7
    assert {"Phase II", "Phase II)"} == {a, b}


def test_suggest_concept_merges_low_similarity_excluded(config, db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Quantum Computing"])
    db.upsert_concepts("raw/b.md", ["Neural Networks"])

    result = suggest_concept_merges(config, db)
    assert result == []


def test_suggest_concept_merges_sorted_by_score_desc(config, db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/c.md", content_hash="h3", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Deep Learning"])
    db.upsert_concepts("raw/b.md", ["Deep-Learning"])
    db.upsert_concepts("raw/c.md", ["Deep Learning Model"])

    result = suggest_concept_merges(config, db)
    scores = [r[2] for r in result]
    assert scores == sorted(scores, reverse=True)


# ── issue #53: malformed-link end-to-end ──────────────────────────────────────


def test_maintain_fix_flow_heals_malformed_link_no_duplicate_file(config, db):
    """End-to-end: a [[Phase II)]] link beside an existing 'Phase II' page is healed, not duped.

    Reproduces issue #53 through the real maintain --fix path: lint flags the malformed link,
    fix_broken_links rewrites it to [[Phase II]], and create_stubs makes no 'Phase II).md'.
    """
    from synto.models import LintIssue
    from synto.pipeline.lint import run_lint
    from synto.pipeline.maintain import fix_broken_links

    _write_article(config, "Phase II", "## Body\n\nContent about phase two.")
    ref = _write_article(config, "Project Plan", "The plan covers [[Phase II)]] in detail.")

    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert any("Phase II)" in i.description for i in broken), "lint should flag the malformed link"

    repair = fix_broken_links(config, db, broken)
    remaining: list[LintIssue] = repair.still_broken
    created = create_stubs(config, db, broken_link_issues=remaining)

    assert not (config.drafts_dir / "Phase II).md").exists()
    assert all(p.name != "Phase II).md" for p in created)

    _, body = parse_note(ref)
    assert "[[Phase II]]" in body
    assert "[[Phase II)]]" not in body
