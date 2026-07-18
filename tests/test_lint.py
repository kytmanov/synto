"""Tests for pipeline/lint.py — no LLM, no Ollama required."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from synto.config import Config
from synto.models import WikiArticleRecord
from synto.pipeline.lint import run_lint
from synto.state import StateDB
from synto.vault import parse_note, write_note


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".synto").mkdir()
    return tmp_path


@pytest.fixture
def config(vault):
    return Config(vault=vault)


@pytest.fixture
def db(config):
    return StateDB(config.state_db_path)


def _write_page(
    config: Config, title: str, body: str = "", meta_override: dict | None = None
) -> Path:
    meta = {"title": title, "tags": ["test"], "status": "published"}
    if meta_override:
        meta.update(meta_override)
    path = config.wiki_dir / f"{title}.md"
    write_note(path, meta, body or f"Content about {title}.")
    return path


# ── Health score ──────────────────────────────────────────────────────────────


def test_no_pages_returns_healthy(vault, config, db):
    result = run_lint(config, db)
    assert result.health_score == 100.0
    assert result.issues == []


def test_clean_wiki_scores_100(vault, config, db):
    _write_page(config, "Quantum Computing", "See also [[Machine Learning]].")
    _write_page(config, "Machine Learning", "Related to [[Quantum Computing]].")
    result = run_lint(config, db)
    # Both pages link to each other — no orphans; no broken links; all fields present
    orphan_issues = [i for i in result.issues if i.issue_type == "orphan"]
    broken_issues = [i for i in result.issues if i.issue_type == "broken_link"]
    assert not orphan_issues
    assert not broken_issues


# ── Missing frontmatter ───────────────────────────────────────────────────────


def test_missing_frontmatter_detected(vault, config, db):
    # Write a page without frontmatter
    path = config.wiki_dir / "Bare.md"
    path.write_text("Just a body, no frontmatter.", encoding="utf-8")

    result = run_lint(config, db)
    types = [i.issue_type for i in result.issues]
    assert "missing_frontmatter" in types


def test_missing_fields_reported(vault, config, db):
    # Write page missing 'tags' and 'status'
    path = config.wiki_dir / "NoTags.md"
    write_note(path, {"title": "NoTags"}, "Content.")

    result = run_lint(config, db)
    missing_issues = [i for i in result.issues if i.issue_type == "missing_frontmatter"]
    assert missing_issues
    assert any("tags" in i.description or "status" in i.description for i in missing_issues)


def test_fix_mode_adds_missing_fields(vault, config, db):
    path = config.wiki_dir / "NoStatus.md"
    write_note(path, {"title": "NoStatus", "tags": []}, "Body.")

    run_lint(config, db, fix=True)

    import frontmatter

    post = frontmatter.load(str(path))
    assert "status" in post.metadata


# ── Orphan detection ──────────────────────────────────────────────────────────


def test_orphan_detected(vault, config, db):
    _write_page(config, "Isolated Page", "No links to or from anywhere.")
    result = run_lint(config, db)
    orphans = [i for i in result.issues if i.issue_type == "orphan"]
    assert orphans
    assert "Isolated Page" in orphans[0].path


def test_orphan_not_flagged_when_linked(vault, config, db):
    _write_page(config, "Alpha", "See [[Beta]].")
    _write_page(config, "Beta", "See [[Alpha]].")
    result = run_lint(config, db)
    orphans = [i for i in result.issues if i.issue_type == "orphan"]
    assert not orphans


def test_index_md_not_checked(vault, config, db):
    """index.md and log.md are system files — skip them."""
    (config.wiki_dir / "index.md").write_text("# Index\n", encoding="utf-8")
    (config.wiki_dir / "log.md").write_text("# Log\n", encoding="utf-8")
    result = run_lint(config, db)
    paths = [i.path for i in result.issues]
    assert not any("index.md" in p or "log.md" in p for p in paths)


# ── Broken links ──────────────────────────────────────────────────────────────


def test_broken_wikilink_detected(vault, config, db):
    _write_page(config, "Alpha", "See [[Ghost Page]] for details.")
    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert broken
    assert "Ghost Page" in broken[0].description


def test_valid_wikilink_not_broken(vault, config, db):
    _write_page(config, "Alpha", "See [[Beta]] for details.")
    _write_page(config, "Beta", "Linked from Alpha.")
    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert not broken


def test_parenthesized_title_base_resolves(vault, config, db):
    _write_page(config, "Alpha", "See [[Workflow]].")
    _write_page(
        config,
        "Workflow (Process Pattern)",
        "Linked from Alpha.",
        meta_override={"title": "Workflow (Process Pattern)"},
    )

    result = run_lint(config, db)

    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert not broken


def test_url_wikilinks_not_broken(vault, config, db):
    """[[https://example.com]] and domain/path links must not trigger broken_link."""
    body = "See [[https://example.com/page]] and [[scrummasters.com.ua/book]]."
    _write_page(config, "Alpha", body)
    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert not broken


def test_vault_path_fragments_not_broken(vault, config, db):
    """LLM sometimes writes [[wiki/]], [[raw/]], [[source]] as links — not real pages."""
    body = "See [[wiki/]] and [[raw/]] and [[source]] and [[sources]] and [[wiki/.drafts/]]."
    _write_page(config, "Alpha", body)
    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert not broken


def test_source_path_wikilink_valid_when_source_page_exists(vault, config, db):
    (config.sources_dir).mkdir(parents=True, exist_ok=True)
    _write_page(config, "Alpha", "See [[sources/Source Note|S1]].")
    write_note(
        config.sources_dir / "Source Note.md",
        {"title": "Source Note", "tags": ["source"], "status": "published"},
        "Source summary.",
    )

    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert not broken


def test_source_path_wikilink_missing_is_broken(vault, config, db):
    _write_page(config, "Alpha", "See [[sources/Missing Source|S1]].")

    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert broken
    assert "sources/Missing Source" in broken[0].description


def test_wiki_rel_key_forward_slash_on_windows_paths():
    # #26: [[sources/X]] links (always forward-slash) were falsely reported broken
    # on Windows because the title-index key used backslashes. The key must be
    # forward-slash regardless of how wiki pages are enumerated by the OS.
    from pathlib import PureWindowsPath

    from synto.pipeline.lint import _wiki_rel_key

    key = _wiki_rel_key(
        PureWindowsPath(r"C:\v\wiki\sources\Source Note.md"),
        PureWindowsPath(r"C:\v\wiki"),
    )
    assert key == "sources/source note"


def test_duplicate_broken_links_deduplicated(vault, config, db):
    """Same broken target appearing multiple times in one page → only one issue."""
    body = "See [[Ghost]] here. Also [[Ghost]] there. And [[Ghost]] again."
    _write_page(config, "Alpha", body)
    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert len(broken) == 1
    assert "Ghost" in broken[0].description


def test_malformed_bracket_link_detected(vault, config, db):
    _write_page(config, "Alpha", "This mentions [astronomy] without a URL.")

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_link"]
    assert malformed
    assert "[astronomy]" in malformed[0].description


def test_citation_markers_not_malformed_links(vault, config, db):
    _write_page(config, "Alpha", "Claim [S1]. Joint [S1,S2].")

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_link"]
    assert not malformed


def test_obsidian_callout_marker_not_malformed_link(vault, config, db):
    _write_page(config, "Alpha", "> [!info] This is a callout.")

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_link"]
    assert not malformed


def test_inline_math_interval_not_flagged(vault, config, db):
    _write_page(config, "Alpha", "The rate $H_i \\in [0, 1]$ defines coverage.")

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_link"]
    assert not malformed


def test_display_math_bracket_not_flagged(vault, config, db):
    _write_page(config, "Alpha", "$$x \\in [0, \\infty)$$")

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_link"]
    assert not malformed


def test_malformed_bracket_link_detected_in_draft(vault, config, db):
    write_note(
        config.drafts_dir / "Draft.md",
        {"title": "Draft", "tags": [], "status": "draft"},
        "Draft mentions [astronomy] without a URL.",
    )

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_link"]
    assert malformed
    assert malformed[0].path == "wiki/.drafts/Draft.md"


def test_dangling_bracket_detected_in_draft(vault, config, db):
    write_note(
        config.drafts_dir / "Draft.md",
        {"title": "Draft", "tags": [], "status": "draft"},
        "The article ends with a broken link fragment [",
    )

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_link"]
    assert malformed
    assert "Dangling '['" in malformed[0].description


def test_broken_wikilink_detected_in_draft(vault, config, db):
    write_note(
        config.drafts_dir / "Draft.md",
        {"title": "Draft", "tags": [], "status": "draft"},
        "Draft links to [[Invented Page]].",
    )

    result = run_lint(config, db)

    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert broken
    assert broken[0].path == "wiki/.drafts/Draft.md"


def test_malformed_embed_detected_in_draft(vault, config, db):
    write_note(
        config.drafts_dir / "Draft.md",
        {"title": "Draft", "tags": [], "status": "draft"},
        "Draft has bad media !./_resources/file.pdf.",
    )

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_embed"]
    assert malformed
    assert "file.pdf" in malformed[0].description


def test_malformed_embed_fix_repairs_draft(vault, config, db):
    draft = config.drafts_dir / "Draft.md"
    write_note(
        draft,
        {"title": "Draft", "tags": [], "status": "draft"},
        "Draft has bad media !./_resources/file.pdf.",
    )

    result = run_lint(config, db, fix=True)

    malformed = [i for i in result.issues if i.issue_type == "malformed_embed"]
    assert malformed
    assert "![[./_resources/file.pdf]]" in draft.read_text()


def test_malformed_latex_detected(vault, config, db):
    _write_page(config, "Alpha", "Equation:\n\\[\na=b\n\\]")

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_latex"]
    assert malformed


def test_malformed_latex_not_reported_as_malformed_link(vault, config, db):
    _write_page(config, "Alpha", "Equation:\n\\[\na=b\n\\]")

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_link"]
    assert not malformed


def test_lint_fix_repairs_malformed_latex(vault, config, db):
    page = _write_page(config, "Alpha", "Equation:\n\\[\na=b\n\\]")

    run_lint(config, db, fix=True)

    assert "$$\na=b\n$$" in page.read_text()


def test_lint_fix_updates_synthesis_frontmatter_hash_for_malformed_latex(vault, config, db):
    from synto.pipeline.lint import _body_hash

    path = config.synthesis_dir / "Topic.md"
    write_note(
        path,
        {
            "title": "Topic",
            "tags": ["synthesis"],
            "kind": "synthesis",
            "status": "published",
            "question_hash": "qh",
            "content_hash": "wrong",
        },
        "Equation:\n\\[\na=b\n\\]",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/synthesis/Topic.md",
            title="Topic",
            sources=[],
            content_hash="wrong",
            status="published",
            kind="synthesis",
            question_hash="qh",
        )
    )

    run_lint(config, db, fix=True)

    import frontmatter

    post = frontmatter.load(path)
    assert post.content == "Equation:\n$$\na=b\n$$"
    assert post.metadata["content_hash"] == _body_hash(post.content)


def test_write_fixed_note_hashes_ondisk_body_not_prewrite(vault, config, db):
    """_write_fixed_note must store the hash of the body as it round-trips on disk.

    Regression for review Issue 4: it previously hashed the pre-write body. A body ending in a
    newline hashes differently once the frontmatter round-trip strips it, which would
    re-introduce the #83 stale/manual-edit false positive.
    """
    from synto.pipeline.lint import _body_hash, _write_fixed_note

    page = config.wiki_dir / "Alpha.md"
    body = "Line one.\n\nLine two.\n"  # trailing newline is stripped on read-back
    write_note(page, {"title": "Alpha", "status": "published"}, body)
    rel = "wiki/Alpha.md"
    db.upsert_article(
        WikiArticleRecord(
            path=rel, title="Alpha", sources=[], content_hash="old", status="published"
        )
    )

    _write_fixed_note(page, rel, {"title": "Alpha", "status": "published"}, body, db)

    _, ondisk = parse_note(page)
    assert ondisk != body  # the round-trip stripped the trailing newline
    art = db.get_article(rel)
    assert art is not None
    assert art.content_hash == _body_hash(ondisk)
    assert art.content_hash != _body_hash(body)  # not the pre-write hash


def test_write_fixed_note_survives_post_write_parse_failure(vault, config, db, monkeypatch, caplog):
    """A reparse fault right after writing must update the row best-effort, not leave a stale hash.

    Regression for review Issue 1: _write_fixed_note reparses the file it just wrote to hash the
    round-tripped body. If that parse raises (a transient FS fault), the old code skipped the DB
    update and let the exception abort the caller's whole fix pass — reintroducing the #83
    stale/manual-edit false positive. The failure must now be logged loudly and the row updated to
    the intended body's hash instead of the stale one.
    """
    import logging

    from synto.pipeline.lint import _body_hash, _write_fixed_note

    page = config.wiki_dir / "Alpha.md"
    body = "Fixed machine body."
    write_note(page, {"title": "Alpha", "status": "published"}, body)
    rel = "wiki/Alpha.md"
    db.upsert_article(
        WikiArticleRecord(
            path=rel, title="Alpha", sources=[], content_hash="old", status="published"
        )
    )

    def _boom(_path):
        raise OSError("transient read failure")

    # A non-synthesis page reparses exactly once (post-write); write_note does not parse.
    monkeypatch.setattr("synto.pipeline.lint.parse_note", _boom)

    with caplog.at_level(logging.WARNING):
        _write_fixed_note(page, rel, {"title": "Alpha", "status": "published"}, body, db)

    art = db.get_article(rel)
    assert art is not None
    assert art.content_hash == _body_hash(body)  # best-effort fallback, not the stale "old"
    assert art.content_hash != "old"
    assert "post-write reparse" in caplog.text


def test_blank_content_hash_not_flagged_stale(vault, config, db):
    """A machine-written stub carrying the placeholder content_hash="" is not a manual edit.

    Regression for review Issue 2: the stale detector compared hashes with no guard, so any row
    with the documented "" placeholder (unmerge stubs, crash windows) was always reported stale.
    """
    _write_page(config, "Placeholder", "Fresh machine body.")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Placeholder.md",
            title="Placeholder",
            sources=[],
            content_hash="",
            status="published",
        )
    )

    result = run_lint(config, db)
    stale = [i for i in result.issues if i.issue_type == "stale" and "Placeholder" in i.path]
    assert not stale


def test_lint_fix_repairs_plain_source_citations(vault, config, db):
    page = _write_page(
        config,
        "Alpha",
        "Claim [S1].\n\n## Sources\n- [S1] [[sources/Alpha Source|Alpha Source]]",
    )

    run_lint(config, db, fix=True)

    assert "Claim [S1](#Sources)." in page.read_text()
    assert "- [S1] [[sources/Alpha Source|Alpha Source]]" in page.read_text()


def test_lint_fix_repairs_linked_source_legend_labels(vault, config, db):
    page = _write_page(
        config,
        "Alpha",
        "Claim [S1](#Sources).\n\n"
        "## Sources\n- [S1](#Sources) [[sources/Alpha Source|Alpha Source]]",
    )

    run_lint(config, db, fix=True)

    assert "Claim [S1](#Sources)." in page.read_text()
    assert "- [S1] [[sources/Alpha Source|Alpha Source]]" in page.read_text()


def test_markdown_anchor_links_not_inline_tags(vault, config, db):
    _write_page(config, "Alpha", "Claim [S1](#Sources).")

    result = run_lint(config, db)

    inline = [i for i in result.issues if i.issue_type == "inline_tag"]
    assert not inline


def test_lint_fix_updates_article_hash(vault, config, db):
    body = "Claim [S1].\n\n## Sources\n- [S1] [[sources/Alpha Source|Alpha Source]]"
    page = _write_page(config, "Alpha", body)
    from synto.pipeline.lint import _body_hash

    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Alpha.md",
            title="Alpha",
            sources=[],
            content_hash=_body_hash(body),
            status="published",
        )
    )

    run_lint(config, db, fix=True)
    result = run_lint(config, db)

    assert "Claim [S1](#Sources)." in page.read_text()
    assert not [i for i in result.issues if i.issue_type == "stale"]


def test_graph_quality_flags_welcome(vault, config, db):
    (config.vault / "Welcome.md").write_text("Welcome. [[create a link]]")

    result = run_lint(config, db)

    graph = [i for i in result.issues if i.issue_type == "graph_noise"]
    assert graph
    assert graph[0].path == "Welcome.md"


def test_graph_quality_flags_media_embeds_in_drafts(vault, config, db):
    write_note(
        config.drafts_dir / "Draft.md",
        {"title": "Draft", "tags": [], "status": "draft"},
        "Draft embeds ![[./_resources/file.pdf]].",
    )

    result = run_lint(config, db)

    graph = [i for i in result.issues if i.issue_type == "graph_noise"]
    assert graph
    assert "media embeds" in graph[0].description


def test_graph_quality_flags_duplicate_raw_source_titles(vault, config, db):
    raw = config.raw_dir / "Reference note.md"
    raw.write_text("Raw body.")
    write_note(
        config.sources_dir / "Reference Note.md",
        {"title": "Reference Note", "tags": ["source"], "status": "published"},
        "Source body.",
    )

    result = run_lint(config, db)

    graph = [i for i in result.issues if i.issue_type == "graph_noise"]
    assert any("duplicate raw note titles" in i.description for i in graph)


def test_graph_quality_flags_low_concept_connectivity(vault, config, db):
    write_note(
        config.drafts_dir / "Alpha.md",
        {"title": "Alpha", "tags": [], "status": "draft"},
        "No links.",
    )
    write_note(
        config.drafts_dir / "Beta.md",
        {"title": "Beta", "tags": [], "status": "draft"},
        "No links.",
    )

    result = run_lint(config, db)

    connectivity = [i for i in result.issues if i.issue_type == "graph_connectivity"]
    assert len(connectivity) == 1
    assert "and 1 more" in connectivity[0].description


def test_graph_quality_ignores_draft_aliases_for_connectivity(vault, config, db):
    write_note(
        config.drafts_dir / "Alpha.md",
        {"title": "Alpha", "aliases": ["Alias Alpha"], "tags": [], "status": "draft"},
        "No links.",
    )

    result = run_lint(config, db)

    connectivity = [i for i in result.issues if i.issue_type == "graph_connectivity"]
    assert connectivity == []


def test_graph_quality_checks_can_be_disabled(vault, config, db):
    config.pipeline.graph_quality_checks = False
    (config.vault / "Welcome.md").write_text("Welcome. [[create a link]]")

    result = run_lint(config, db)

    assert not [i for i in result.issues if i.issue_type == "graph_noise"]


def test_graph_quality_issues_do_not_reduce_health_score(vault, config, db):
    (config.vault / "Welcome.md").write_text("Welcome. [[create a link]]")
    raw = config.raw_dir / "Reference note.md"
    raw.write_text("Raw body.")
    write_note(
        config.sources_dir / "Reference Note.md",
        {"title": "Reference Note", "tags": ["source"], "status": "published"},
        "Source body.",
    )

    result = run_lint(config, db)

    assert [i for i in result.issues if i.issue_type == "graph_noise"]
    assert result.health_score == 100.0
    assert result.advisory_issue_count == len(result.issues)


# ── Low confidence ────────────────────────────────────────────────────────────


def test_low_confidence_detected(vault, config, db):
    _write_page(
        config,
        "Weak",
        meta_override={"confidence": 0.1, "title": "Weak", "tags": [], "status": "published"},
    )
    result = run_lint(config, db)
    low = [i for i in result.issues if i.issue_type == "low_confidence"]
    assert low


def test_high_confidence_not_flagged(vault, config, db):
    _write_page(
        config,
        "Strong",
        meta_override={"confidence": 0.8, "title": "Strong", "tags": [], "status": "published"},
    )
    result = run_lint(config, db)
    low = [i for i in result.issues if i.issue_type == "low_confidence"]
    assert not low


# ── Stale (manually edited) ───────────────────────────────────────────────────


def test_stale_detected_on_hash_mismatch(vault, config, db):
    path = _write_page(config, "Edited")
    rel = str(path.relative_to(vault))
    # Register with a WRONG hash
    db.upsert_article(
        WikiArticleRecord(
            path=rel,
            title="Edited",
            sources=[],
            content_hash="wrong_hash",
            status="published",
        )
    )

    result = run_lint(config, db)
    stale = [i for i in result.issues if i.issue_type == "stale"]
    assert stale


def test_not_stale_when_hash_matches(vault, config, db):
    import hashlib

    path = _write_page(config, "Fresh")
    rel = str(path.relative_to(vault))
    from synto.vault import parse_note

    _, body = parse_note(path)
    correct_hash = hashlib.sha256(body.encode()).hexdigest()
    db.upsert_article(
        WikiArticleRecord(
            path=rel,
            title="Fresh",
            sources=[],
            content_hash=correct_hash,
            status="published",
        )
    )

    result = run_lint(config, db)
    stale = [i for i in result.issues if i.issue_type == "stale"]
    assert not stale


def test_write_fixed_note_does_not_mutate_caller_meta(vault, config, db):
    """_write_fixed_note must not leak its synthesis content_hash write into the caller's dict.

    Regression for review Issue 6: the function mutated the passed `meta` in place, an
    observable side-effect on the caller's dict. Callers pass a freshly parsed dict today, but
    the write is unnecessary and future-fragile.
    """
    from synto.pipeline.lint import _write_fixed_note

    path = config.synthesis_dir / "Synth.md"
    config.synthesis_dir.mkdir(parents=True, exist_ok=True)
    meta = {"title": "Synth", "kind": "synthesis", "question_hash": "abc"}
    caller_meta = dict(meta)

    _write_fixed_note(path, str(path.relative_to(vault)), caller_meta, "Body.", db)

    assert caller_meta == meta, "caller's meta dict must be unchanged"
    assert "content_hash" not in caller_meta


# ── Filename drift ────────────────────────────────────────────────────────────


def test_filename_drift_detected_for_legacy_stem(vault, config, db):
    # A file created by the old sanitizer ("Foo." kept its trailing dot) no longer
    # matches what new wikilinks will point at — lint must surface it.
    path = config.wiki_dir / "Foo..md"
    write_note(path, {"title": "Foo.", "tags": [], "status": "published"}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Foo..md", title="Foo.", sources=[], content_hash="h", status="published"
        )
    )

    result = run_lint(config, db)

    drift = [i for i in result.issues if i.issue_type == "filename_drift"]
    assert drift
    assert "wiki/Foo..md" in drift[0].path


def test_filename_drift_collision_suggests_concept_rename(vault, config, db):
    # Two titles collapsing to one canonical stem ("Foo" + "Foo.") can never be fixed by
    # maintain --fix (it correctly skips); the issue must say so and point at the real
    # remedy instead of sending the user in a --fix loop.
    for stem, title in (("Foo", "Foo"), ("Foo.", "Foo.")):
        path = config.wiki_dir / f"{stem}.md"
        write_note(path, {"title": title, "tags": [], "status": "published"}, "Body.")
        db.upsert_article(
            WikiArticleRecord(
                path=f"wiki/{stem}.md",
                title=title,
                sources=[],
                content_hash="h",
                status="published",
            )
        )

    result = run_lint(config, db)

    drift = [i for i in result.issues if i.issue_type == "filename_drift"]
    assert drift
    assert "concept rename" in drift[0].suggestion


def test_no_filename_drift_for_canonical_or_retitled(vault, config, db):
    _write_page(config, "Alpha", "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Alpha.md", title="Alpha", sources=[], content_hash="h", status="published"
        )
    )
    # Manual retitle in frontmatter/DB (stem doesn't re-sanitize to the title's form) is
    # not drift — that's a deliberate user edit, not a sanitizer-rule change.
    _write_page(config, "Beta", "Body.", meta_override={"title": "Renamed Concept"})
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Beta.md",
            title="Renamed Concept",
            sources=[],
            content_hash="h2",
            status="published",
        )
    )

    result = run_lint(config, db)

    assert not [i for i in result.issues if i.issue_type == "filename_drift"]


# ── Invalid tags ─────────────────────────────────────────────────────────────


def test_invalid_tag_detected(vault, config, db):
    _write_page(
        config,
        "BadTags",
        meta_override={"tags": ["bad tag", "C++"], "status": "published"},
    )
    result = run_lint(config, db)
    tag_issues = [i for i in result.issues if i.issue_type == "invalid_tag"]
    assert tag_issues
    assert "bad tag" in tag_issues[0].description


def test_valid_tags_no_issue(vault, config, db):
    _write_page(
        config,
        "GoodTags",
        meta_override={"tags": ["physics", "machine-learning"], "status": "published"},
    )
    result = run_lint(config, db)
    tag_issues = [i for i in result.issues if i.issue_type == "invalid_tag"]
    assert not tag_issues


def test_fix_mode_sanitizes_tags(vault, config, db):
    import frontmatter as fm

    path = _write_page(
        config,
        "FixTags",
        meta_override={"tags": ["bad tag", "physics"], "status": "published"},
    )
    run_lint(config, db, fix=True)
    post = fm.load(str(path))
    assert "bad-tag" in post.metadata["tags"]
    assert "bad tag" not in post.metadata["tags"]


def test_lint_checks_source_pages(vault, config, db):
    """Tags in wiki/sources/ pages are also checked."""
    sources_dir = config.wiki_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    src_path = sources_dir / "MySource.md"
    write_note(src_path, {"title": "MySource", "tags": ["bad tag"], "status": "published"}, "Body.")
    result = run_lint(config, db)
    tag_issues = [i for i in result.issues if i.issue_type == "invalid_tag"]
    assert any("sources" in i.path for i in tag_issues)


def test_lint_flags_orphan_synthesis_file(vault, config, db):
    write_note(
        config.synthesis_dir / "Orphan.md",
        {"title": "Orphan", "tags": ["synthesis"], "kind": "synthesis", "status": "published"},
        "Body.",
    )

    result = run_lint(config, db)

    orphan = [i for i in result.issues if i.issue_type == "orphan"]
    assert any(i.path == "wiki/synthesis/Orphan.md" for i in orphan)


def test_lint_flags_missing_synthesis_source_page(vault, config, db):
    path = config.synthesis_dir / "Topic.md"
    write_note(
        path,
        {
            "title": "Topic",
            "tags": ["synthesis"],
            "kind": "synthesis",
            "status": "published",
            "source_pages": ["Missing Topic"],
            "source_page_hashes": [{"path": "wiki/Missing Topic.md", "hash": "abc"}],
        },
        "Body.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/synthesis/Topic.md",
            title="Topic",
            sources=[],
            content_hash="hash",
            status="published",
            kind="synthesis",
            question_hash="qh",
        )
    )

    result = run_lint(config, db)

    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert any("Missing Topic" in i.description for i in broken)


def test_lint_flags_synthesis_source_hash_drift(vault, config, db):
    _write_page(config, "Alpha", "Original body.")
    path = config.synthesis_dir / "Topic.md"
    write_note(
        path,
        {
            "title": "Topic",
            "tags": ["synthesis"],
            "kind": "synthesis",
            "status": "published",
            "source_pages": ["Alpha"],
            "source_page_hashes": [{"path": "wiki/Alpha.md", "hash": "wrong"}],
        },
        "Body.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/synthesis/Topic.md",
            title="Topic",
            sources=[],
            content_hash="hash",
            status="published",
            kind="synthesis",
            question_hash="qh",
        )
    )

    result = run_lint(config, db)

    stale = [i for i in result.issues if i.issue_type == "stale"]
    assert any("wiki/Alpha.md" in i.description for i in stale)


def test_lint_flags_synthesis_chain(vault, config, db):
    write_note(
        config.synthesis_dir / "Parent.md",
        {"title": "Parent", "tags": ["synthesis"], "kind": "synthesis", "status": "published"},
        "Body.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/synthesis/Parent.md",
            title="Parent",
            sources=[],
            content_hash="hash-parent",
            status="published",
            kind="synthesis",
            question_hash="qh-parent",
        )
    )
    write_note(
        config.synthesis_dir / "Child.md",
        {
            "title": "Child",
            "tags": ["synthesis"],
            "kind": "synthesis",
            "status": "published",
            "source_pages": ["Parent"],
            "source_page_hashes": [{"path": "wiki/synthesis/Parent.md", "hash": "hash-parent"}],
        },
        "Body.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/synthesis/Child.md",
            title="Child",
            sources=[],
            content_hash="hash-child",
            status="published",
            kind="synthesis",
            question_hash="qh-child",
        )
    )

    result = run_lint(config, db)

    chains = [i for i in result.issues if i.issue_type == "synthesis_chain"]
    assert chains


def test_lint_flags_synthesis_title_shadowing_concept(vault, config, db):
    _write_page(config, "Topic", "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Topic.md",
            title="Topic",
            sources=[],
            content_hash="hash-concept",
            status="published",
        )
    )
    write_note(
        config.synthesis_dir / "Topic Summary.md",
        {
            "title": "Topic",
            "tags": ["synthesis"],
            "kind": "synthesis",
            "status": "published",
            "source_pages": ["Topic"],
            "source_page_hashes": [{"path": "wiki/Topic.md", "hash": "hash-concept"}],
        },
        "Body.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/synthesis/Topic Summary.md",
            title="Topic",
            sources=[],
            content_hash="hash-synthesis",
            status="published",
            kind="synthesis",
            question_hash="qh",
        )
    )

    result = run_lint(config, db)

    graph = [i for i in result.issues if i.issue_type == "graph_noise"]
    assert any("shadows an existing concept title" in i.description for i in graph)


# ── Summary string ────────────────────────────────────────────────────────────


def test_summary_mentions_issue_counts(vault, config, db):
    _write_page(config, "Solo", "No links.")  # orphan
    result = run_lint(config, db)
    assert "orphan" in result.summary


def test_summary_healthy_when_no_issues(vault, config, db):
    result = run_lint(config, db)
    assert "healthy" in result.summary.lower()


def test_missing_media_counts_as_advisory_issue(vault, config, db):
    _write_page(
        config,
        "Media Note",
        "Diagram ![[./_resources/missing.pdf]].",
        meta_override={"title": "Media Note", "tags": [], "status": "published"},
    )
    result = run_lint(config, db)
    assert [i for i in result.issues if i.issue_type == "missing_media"]
    assert result.advisory_issue_count == 1


# ── Config sanity ─────────────────────────────────────────────────────────────


def test_lint_warns_on_stale_article_max_tokens(vault, config, db):
    """Existing synto.toml files written by older `synto setup` runs pin
    article_max_tokens=4096 (the old default). Surface a config_outdated issue
    so users discover the stale value via `synto maintain --dry-run`."""
    config.pipeline.article_max_tokens = 4096
    result = run_lint(config, db)
    config_issues = [i for i in result.issues if i.issue_type == "config_outdated"]
    assert config_issues, "expected a config_outdated issue when article_max_tokens==4096"
    assert "article_max_tokens" in config_issues[0].description
    assert "16384" in config_issues[0].suggestion


def test_lint_silent_when_article_max_tokens_below_legacy_default(vault, config, db):
    config.pipeline.article_max_tokens = 2048
    result = run_lint(config, db)
    assert not [i for i in result.issues if i.issue_type == "config_outdated"]


def test_lint_silent_when_article_max_tokens_at_new_default(vault, config, db):
    config.pipeline.article_max_tokens = 16384
    result = run_lint(config, db)
    assert not [i for i in result.issues if i.issue_type == "config_outdated"]


# ── Stale lock ────────────────────────────────────────────────────────────────


def test_dead_windows_lock_detected(config, db, monkeypatch):
    monkeypatch.setattr("synto.pipeline.lock._IS_POSIX", False)
    monkeypatch.setattr("synto.pipeline.lock._windows_pid_alive", lambda pid: False)

    lock_path = config.vault / ".synto" / "pipeline.lock"
    lock_path.parent.mkdir(exist_ok=True)
    lock_path.write_text("99999999")  # PID far above OS max → guaranteed dead
    result = run_lint(config, db)
    stale = [i for i in result.issues if i.issue_type == "stale_lock"]
    assert stale
    assert "99999999" in stale[0].description


def test_no_stale_lock_when_absent(config, db):
    result = run_lint(config, db)
    assert not any(i.issue_type == "stale_lock" for i in result.issues)


def test_own_lock_not_flagged_as_stale(config, db):
    """Lint runs inside the held pipeline lock during run/maintain; our own lock
    must not be reported as stale. On NFS the liveness probe can't self-detect, so
    this is guarded by a current-PID short-circuit (also avoids the probe's fd
    close dropping the held lock under NFS POSIX-lock emulation)."""
    import os

    lock_path = config.vault / ".synto" / "pipeline.lock"
    lock_path.parent.mkdir(exist_ok=True)
    lock_path.write_text(str(os.getpid()))
    result = run_lint(config, db)
    assert not any(i.issue_type == "stale_lock" for i in result.issues)


def test_released_posix_lock_file_not_flagged_as_stale(config, db, monkeypatch):
    monkeypatch.setattr("synto.pipeline.lock._IS_POSIX", True)

    lock_path = config.vault / ".synto" / "pipeline.lock"
    lock_path.parent.mkdir(exist_ok=True)
    lock_path.write_text("99999999")

    result = run_lint(config, db)

    assert not any(i.issue_type == "stale_lock" for i in result.issues)


def test_invalid_lock_file_detected(config, db):
    lock_path = config.vault / ".synto" / "pipeline.lock"
    lock_path.parent.mkdir(exist_ok=True)
    lock_path.write_text("not-a-pid")
    result = run_lint(config, db)
    stale = [i for i in result.issues if i.issue_type == "stale_lock"]
    assert stale
    assert "invalid" in stale[0].description.lower()


# ── Missing media ─────────────────────────────────────────────────────────────


def test_missing_media_detected(config, db):
    _write_page(config, "Article", "See ![[_resources/photo.png]] for details.")
    result = run_lint(config, db)
    missing = [i for i in result.issues if i.issue_type == "missing_media"]
    assert missing
    assert "photo.png" in missing[0].description


def test_present_media_not_flagged(config, db):
    resources = config.vault / "_resources"
    resources.mkdir()
    (resources / "photo.png").write_bytes(b"fake")
    _write_page(config, "Article", "See ![[_resources/photo.png]] for details.")
    result = run_lint(config, db)
    assert not any(i.issue_type == "missing_media" for i in result.issues)


def test_missing_media_bare_name_flagged(config, db):
    _write_page(config, "Article", "See ![[photo.png]] for details.")
    result = run_lint(config, db)
    assert any(i.issue_type == "missing_media" for i in result.issues)


def test_note_transclusion_not_flagged_as_missing_media(config, db):
    _write_page(config, "Other Note", "other content")
    _write_page(config, "Article", "See ![[Other Note]] for details.")
    result = run_lint(config, db)
    assert not any(i.issue_type == "missing_media" for i in result.issues)


def test_missing_media_detected_in_raw_note(config, db):
    raw_note = config.vault / "raw" / "My Note.md"
    write_note(raw_note, {"title": "My Note"}, "See ![[_resources/missing.png]]")
    result = run_lint(config, db)
    missing = [i for i in result.issues if i.issue_type == "missing_media"]
    assert any("missing.png" in i.description for i in missing)


# ── Identity checks (feature 45) ───────────────────────────────────────────────


def test_ambiguous_label_flagged_when_no_disambiguation_page(config, db):
    # A split leaves the bare label "Mercury" as a shared alias on two senses.
    # Without a disambiguation page on disk, the bare label is unresolvable — flag it.
    db.upsert_concepts("raw/planets.md", ["Mercury"])
    db.upsert_concepts("raw/chemistry.md", ["Mercury"])
    db.split_entity(
        "Mercury",
        [
            {"name": "Mercury (planet)", "sources": ["raw/planets.md"]},
            {"name": "Mercury (element)", "sources": ["raw/chemistry.md"]},
        ],
    )

    result = run_lint(config, db)
    ambiguous = [i for i in result.issues if i.issue_type == "ambiguous_label_needs_disambiguation"]
    assert any("Mercury" in i.description for i in ambiguous)


def test_ambiguous_label_suppressed_by_disambiguation_page(config, db):
    # The same split, but a disambiguation page now exists on disk — the check must
    # consult db.list_articles() (the line that previously crashed on `all_articles`)
    # and suppress the warning.
    db.upsert_concepts("raw/planets.md", ["Mercury"])
    db.upsert_concepts("raw/chemistry.md", ["Mercury"])
    db.split_entity(
        "Mercury",
        [
            {"name": "Mercury (planet)", "sources": ["raw/planets.md"]},
            {"name": "Mercury (element)", "sources": ["raw/chemistry.md"]},
        ],
    )
    dis_path = config.wiki_dir / "Mercury.md"
    write_note(dis_path, {"title": "Mercury", "kind": "disambiguation"}, "may refer to:")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Mercury.md",
            title="Mercury",
            sources=[],
            content_hash="",
            status="published",
            kind="disambiguation",
        )
    )

    result = run_lint(config, db)
    ambiguous = [i for i in result.issues if i.issue_type == "ambiguous_label_needs_disambiguation"]
    assert not ambiguous


# ── Manual-rename-as-relabel (Decision 10) ─────────────────────────────────────


def _publish_concept(config: Config, db: StateDB, name: str, body: str) -> str:
    """Create an entity + a published concept article on disk, tracked with its hash."""
    db.upsert_concepts("raw/source.md", [name])
    eid = db.entity_id_for_name(name)
    path = config.wiki_dir / f"{name}.md"
    write_note(path, {"title": name, "status": "published", "tags": ["t"]}, body)
    _, on_disk_body = parse_note(path)
    db.upsert_article(
        WikiArticleRecord(
            path=f"wiki/{name}.md",
            title=name,
            sources=["raw/source.md"],
            content_hash=hashlib.sha256(on_disk_body.encode()).hexdigest(),
            status="published",
            kind="concept",
            entity_id=eid,
        )
    )
    return eid


def test_manual_relabel_flagged_without_fix(config, db):
    eid = _publish_concept(config, db, "Foo", "Body about Foo.")
    # User renames the file on disk; body is unchanged.
    (config.wiki_dir / "Foo.md").rename(config.wiki_dir / "Bar.md")

    result = run_lint(config, db)  # no fix
    relabel = [i for i in result.issues if i.issue_type == "manual_relabel_adopted"]
    assert relabel and "Bar" in relabel[0].description
    # Nothing adopted: preferred label is still "Foo".
    assert db.preferred_label_for_entity(eid) == "Foo"


def test_manual_relabel_adopted_under_fix(config, db):
    eid = _publish_concept(config, db, "Foo", "Body about Foo.")
    # A page linking to the old name, to prove inbound links get rewritten.
    write_note(
        config.wiki_dir / "Linker.md",
        {"title": "Linker", "status": "published", "tags": ["t"]},
        "See [[Foo]] for details.",
    )
    (config.wiki_dir / "Foo.md").rename(config.wiki_dir / "Bar.md")

    run_lint(config, db, fix=True)

    # Preferred label adopted the new filename; old name demoted to an alias.
    assert db.preferred_label_for_entity(eid) == "Bar"
    assert "Foo" in {a for a in db.get_aliases("Bar")}
    # Inbound link rewritten to the new stem.
    assert "[[Bar]]" in (config.wiki_dir / "Linker.md").read_text()


def test_manual_relabel_adopted_blesses_old_label(config, db):
    """The demoted old label must be a BLESSED alias (source='rename'), not a weak 'extracted'
    one — parity with `concept rename`. A weak alias would re-mint the old name on the next
    ingest, re-fragmenting the concept the adoption just unified.
    """
    eid = _publish_concept(config, db, "Foo", "Body about Foo.")
    (config.wiki_dir / "Foo.md").rename(config.wiki_dir / "Bar.md")

    run_lint(config, db, fix=True)

    row = db._conn.execute(
        "SELECT source FROM concept_labels WHERE entity_id=? AND lower(label)=lower('Foo')",
        (eid,),
    ).fetchone()
    assert row is not None, "old label should survive as an alias"
    assert row[0] == "rename", f"expected blessed 'rename' source, got {row[0]!r}"


def test_manual_relabel_collision_not_adopted(config, db):
    foo_id = _publish_concept(config, db, "Foo", "Body about Foo.")
    # A second active entity already owns the label "Bar".
    db.upsert_concepts("raw/bar.md", ["Bar"])
    assert db.preferred_label_for_entity(db.entity_id_for_name("Bar")) == "Bar"
    # User renames Foo's file to "Bar.md" — colliding with the Bar entity's label.
    (config.wiki_dir / "Foo.md").rename(config.wiki_dir / "Bar.md")

    result = run_lint(config, db, fix=True)
    collisions = [i for i in result.issues if i.issue_type == "homonym_filename_collision"]
    assert collisions
    # Foo must not have been silently relabeled to the colliding name.
    assert db.preferred_label_for_entity(foo_id) == "Foo"


# ── Disambiguation stubs (F4) ──────────────────────────────────────────────────


def test_disambiguation_stub_not_flagged_missing_or_stale(vault, config, db):
    """Disambiguation stubs are auto-generated system pages. Legacy ones (written before the
    creation-site fix) carry bare frontmatter and an empty DB content_hash, so the generic
    missing_frontmatter/stale checks fire on every run — pure noise, and --fix would inject
    bogus fields into a generated page. A real concept page must still be flagged on both axes.
    """
    dis = config.wiki_dir / "Mercury.md"
    write_note(
        dis,
        {"title": "Mercury", "kind": "disambiguation"},  # legacy bare frontmatter
        "**Mercury** may refer to:\n\n- [[Mercury (planet)]]\n",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Mercury.md",
            title="Mercury",
            sources=[],
            content_hash="",  # legacy empty hash vs a real body → would read as stale
            status="published",
            kind="disambiguation",
        )
    )
    # Control: an ordinary concept page that genuinely is missing fields and stale.
    normal = config.wiki_dir / "Normal.md"
    write_note(normal, {"title": "Normal"}, "Body about normal.")
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Normal.md",
            title="Normal",
            sources=[],
            content_hash="deadbeef",
            status="published",
        )
    )

    result = run_lint(config, db)
    flagged = {(i.path, i.issue_type) for i in result.issues}
    assert ("wiki/Mercury.md", "missing_frontmatter") not in flagged
    assert ("wiki/Mercury.md", "stale") not in flagged
    assert ("wiki/Normal.md", "missing_frontmatter") in flagged
    assert ("wiki/Normal.md", "stale") in flagged


# ── Code fences and inline code must not trigger content scanners (#93) ───────


def test_json_in_fenced_code_block_not_malformed_link(vault, config, db):
    # The #93 repro: a valid package.json example inside a ```json fence. Following
    # the [[...]] suggestion would corrupt the code sample.
    _write_page(
        config,
        "Alpha",
        'Example:\n\n```json\n{\n  "workspaces": ["apps/*", "packages/*"]\n}\n```\n',
    )

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_link"]
    assert not malformed


def test_brackets_in_inline_code_not_malformed_link(vault, config, db):
    _write_page(config, "Alpha", "Set the `[config]` section header first.")

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_link"]
    assert not malformed


def test_malformed_link_outside_fence_still_detected(vault, config, db):
    # Positive control against over-masking: the fence is skipped, prose is not.
    _write_page(
        config,
        "Alpha",
        'Mentions [astronomy] in prose.\n\n```json\n["apps/*", "packages/*"]\n```\n',
    )

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_link"]
    assert len(malformed) == 1
    assert "[astronomy]" in malformed[0].description


def test_embed_like_token_in_fence_not_flagged_or_rewritten(vault, config, db):
    page = _write_page(config, "Alpha", "Shell example:\n\n```\n!diagram.png\n```\n")

    result = run_lint(config, db, fix=True)

    embeds = [i for i in result.issues if i.issue_type == "malformed_embed"]
    assert not embeds
    _, body = parse_note(page)
    assert "!diagram.png" in body
    assert "![[" not in body


def test_citation_in_fence_not_rewritten_by_fix(vault, config, db):
    page = _write_page(
        config,
        "Alpha",
        "Claim [S9].\n\n```\ncite [S1] here\n```\n\n## Sources\n- S9: somewhere\n",
    )

    run_lint(config, db, fix=True)

    _, body = parse_note(page)
    assert "cite [S1] here" in body  # fence byte-identical
    assert "[S9](#Sources)" in body  # prose citation still repaired


def test_latex_like_line_in_fence_not_malformed_latex(vault, config, db):
    _write_page(config, "Alpha", "LaTeX source example:\n\n```latex\n\\[\na=b\n\\]\n```\n")

    result = run_lint(config, db)

    malformed = [i for i in result.issues if i.issue_type == "malformed_latex"]
    assert not malformed
