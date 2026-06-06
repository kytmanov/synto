"""Tests for pipeline/lint.py — no LLM, no Ollama required."""

from __future__ import annotations

from pathlib import Path

import pytest

from synto.config import Config
from synto.models import WikiArticleRecord
from synto.pipeline.lint import run_lint
from synto.state import StateDB
from synto.vault import write_note


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


def test_stale_lock_detected(config, db):
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
