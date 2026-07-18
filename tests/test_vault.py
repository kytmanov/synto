"""Tests for vault.py — pure functions, no LLM required."""

from __future__ import annotations

import pytest

import synto.vault as vault
from synto.vault import (
    atomic_write,
    build_wiki_frontmatter,
    chunk_text,
    ensure_wikilinks,
    extract_wikilinks,
    generate_aliases,
    list_wiki_articles,
    next_available_path,
    normalize_wikilinks,
    parse_note,
    rename_wikilink_targets,
    sanitize_filename,
    sanitize_wikilink_target,
    strip_image_text_blocks,
    write_note,
)

__all__ = ["vault"]

# ── parse_note ────────────────────────────────────────────────────────────────


def test_parse_note_with_frontmatter(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("---\ntitle: Test\ntags: [a, b]\n---\n\nBody text here.")
    meta, body = parse_note(p)
    assert meta["title"] == "Test"
    assert meta["tags"] == ["a", "b"]
    assert "Body text here" in body


def test_parse_note_no_frontmatter(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("Just body text, no frontmatter.")
    meta, body = parse_note(p)
    assert meta == {}
    assert "Just body text" in body


def test_parse_note_dashes_in_body(tmp_path):
    """python-frontmatter must not get confused by --- in body."""
    p = tmp_path / "note.md"
    p.write_text("---\ntitle: Test\n---\n\nHeader\n---\nSeparator above.")
    meta, body = parse_note(p)
    assert meta["title"] == "Test"
    assert "Separator above" in body


def test_write_note_roundtrip(tmp_path):
    p = tmp_path / "out.md"
    write_note(p, {"title": "Hello", "tags": ["x"]}, "Body content.")
    meta, body = parse_note(p)
    assert meta["title"] == "Hello"
    assert "Body content" in body


# ── wikilinks ─────────────────────────────────────────────────────────────────


def test_extract_wikilinks():
    content = "See [[Quantum Entanglement]] and [[Bell States|Bell's theorem]]."
    links = extract_wikilinks(content)
    assert "Quantum Entanglement" in links
    assert "Bell States" in links


def test_extract_wikilinks_excludes_image_embeds():
    content = "![[photo.png]] and [[Real Link]]"
    assert extract_wikilinks(content) == ["Real Link"]


def test_extract_wikilinks_excludes_pdf():
    assert extract_wikilinks("![[doc.pdf]]") == []


def test_extract_wikilinks_keeps_note_transclusion():
    """![[other-note]] (no media extension) = note transclusion, keep it."""
    assert extract_wikilinks("![[other-note]]") == ["other-note"]


def test_extract_wikilinks_ignores_inline_code():
    assert extract_wikilinks("Use `[[Not A Link]]` and [[Real Link]].") == ["Real Link"]


def test_extract_wikilinks_excludes_jpg():
    assert extract_wikilinks("![[image.jpg]]") == []


def test_ensure_wikilinks_no_mangle_image_alt():
    """![Quantum Computing](img.png) must not become ![[[ Quantum Computing]]](img.png)."""
    content = "See ![Quantum Computing](img.png) for details."
    result = ensure_wikilinks(content, ["Quantum Computing"])
    assert "![Quantum Computing](img.png)" in result
    assert "![[[" not in result


def test_ensure_wikilinks_no_mangle_obsidian_embed():
    """![[Quantum Computing]] must not become ![[[[Quantum Computing]]]]."""
    content = "See ![[Quantum Computing]] for details."
    result = ensure_wikilinks(content, ["Quantum Computing"])
    assert "![[Quantum Computing]]" in result
    assert "![[[[" not in result


def test_ensure_wikilinks_basic():
    content = "Quantum Entanglement is a physical phenomenon."
    result = ensure_wikilinks(content, ["Quantum Entanglement"])
    assert "[[Quantum Entanglement]]" in result


def test_ensure_wikilinks_no_double_wrap():
    content = "See [[Quantum Entanglement]] already."
    result = ensure_wikilinks(content, ["Quantum Entanglement"])
    assert result.count("[[Quantum Entanglement]]") == 1


def test_ensure_wikilinks_word_boundary():
    """Should not wrap partial matches."""
    content = "Python scripting is used here."
    result = ensure_wikilinks(content, ["Python"])
    # "Python" is a standalone word here — should link
    assert "[[Python]]" in result


def test_ensure_wikilinks_no_substring_in_word():
    """Should NOT wrap 'Python' inside 'CPython'."""
    content = "CPython is the reference implementation."
    result = ensure_wikilinks(content, ["Python"])
    assert "[[Python]]" not in result
    assert "CPython" in result


def test_ensure_wikilinks_skip_code_blocks():
    content = "Use `Python` in code. Python is great."
    result = ensure_wikilinks(content, ["Python"])
    # Should only link the second "Python", not the one in backticks
    assert "`Python`" in result or "`[[Python]]`" not in result


def test_ensure_wikilinks_restores_inline_code_after_length_change():
    content = "Machine learning then `code`"
    result = ensure_wikilinks(content, ["Machine learning"])
    assert result == "[[Machine learning]] then `code`"


def test_ensure_wikilinks_restores_fenced_code_after_length_change():
    content = "Python before\n```\nPython in code\n```"
    result = ensure_wikilinks(content, ["Python"])
    assert result == "[[Python]] before\n```\nPython in code\n```"


def test_ensure_wikilinks_restores_embed_after_length_change():
    content = "Python before ![[Python.png]]"
    result = ensure_wikilinks(content, ["Python"])
    assert result == "[[Python]] before ![[Python.png]]"


def test_ensure_wikilinks_empty_targets():
    content = "Some text here."
    assert ensure_wikilinks(content, []) == content


def test_ensure_wikilinks_backslash_target_emits_filename_target():
    # A LaTeX title \int is written to int.md (sanitize_filename strips "\"), so the link
    # target must be int to resolve; the raw title is kept as display: [[int|\int]].
    assert ensure_wikilinks(r"x\int y", [r"\int"]) == r"x[[int|\int]] y"


def test_ensure_wikilinks_target_matches_filename_stem():
    # The link target must equal the file sanitize_filename() would create, for any title
    # carrying filename-forbidden chars (here "/"); the raw title is preserved as display.
    title = "TCP/IP"
    result = ensure_wikilinks("see TCP/IP here", [title])
    assert result == f"see [[{sanitize_filename(title)}|{title}]] here"
    assert result == "see [[TCPIP|TCP/IP]] here"


def test_ensure_wikilinks_idempotent_for_normalized_target():
    # Re-running over an already-normalized link must not double-wrap it.
    once = ensure_wikilinks(r"x\int y", [r"\int"])
    assert ensure_wikilinks(once, [r"\int"]) == once


def test_ensure_wikilinks_backslash_target_no_match_does_not_raise():
    # re.sub parses the replacement template eagerly, so a backslash title raised
    # re.PatternError even when the body never matched the pattern.
    assert ensure_wikilinks("no latex here", [r"\int"]) == "no latex here"


def test_ensure_wikilinks_multi_occurrence_idempotent():
    # The guard skips a title once its emitted (normalized) form exists anywhere — without
    # that, each run would link one more plain occurrence (run1→1st, run2→2nd, …). Only the
    # first mention is linked, and re-running is a no-op.
    once = ensure_wikilinks("TCP/IP is great. TCP/IP rules.", ["TCP/IP"])
    assert once == "[[TCPIP|TCP/IP]] is great. TCP/IP rules."
    assert ensure_wikilinks(once, ["TCP/IP"]) == once


def test_ensure_wikilinks_skips_already_linked_normalized():
    # Reviewer scenario: with the normalized link already present, the remaining plain
    # mention stays plain — identical to how a normal title behaves (link-once), e.g.
    # ensure_wikilinks("[[Python]] and Python", ["Python"]) is also a no-op.
    body = "[[TCPIP]] and TCP/IP"
    assert ensure_wikilinks(body, ["TCP/IP"]) == body


# ── chunk_text ────────────────────────────────────────────────────────────────

# ── sanitize_filename ─────────────────────────────────────────────────────────


def test_sanitize_wikilink_target_matches_backslash_stripped_filename():
    assert sanitize_filename(r"\int") == "int"
    assert sanitize_wikilink_target(r"\int") == "int"


def test_sanitize_filename_strips_forbidden():
    assert sanitize_filename('A*B"C/D') == "ABCD"


def test_sanitize_filename_max_len():
    long_title = "word " * 30  # 150 chars
    result = sanitize_filename(long_title.strip(), max_len=20)
    assert len(result) <= 20


def test_sanitize_filename_empty_becomes_untitled():
    assert sanitize_filename("***///") == "untitled"


def test_sanitize_filename_normal():
    assert sanitize_filename("Quantum Computing") == "Quantum Computing"


# ── sanitize_filename: Windows-safe output ────────────────────────────────────
# Vaults are cross-platform; a name that cannot be created on Windows is a bug on
# every OS, because the vault stops syncing/cloning there.


@pytest.mark.parametrize("title", ["CON", "nul", "Nul", "COM7", "lpt3", "AUX", "PRN"])
def test_sanitize_filename_deserves_reserved_device_names(title):
    result = sanitize_filename(title)
    assert result.casefold() != title.casefold()
    assert result  # never empty


def test_sanitize_filename_reserved_name_stays_link_consistent():
    # Same derivation for filename and link target — must hold for the de-reserved form.
    assert sanitize_wikilink_target("NUL") == sanitize_filename("NUL")


def test_sanitize_filename_strips_trailing_dots():
    # Windows silently drops trailing dots, so "Foo..md" can't round-trip.
    assert sanitize_filename("Foo.") == "Foo"
    assert sanitize_filename("etc..") == "etc"


def test_sanitize_filename_keeps_interior_dots():
    assert sanitize_filename("Node.js") == "Node.js"
    assert sanitize_filename(".NET") == ".NET"


def test_sanitize_filename_strips_control_chars():
    assert sanitize_filename("Foo\x07Bar\x1f") == "FooBar"


def test_parse_note_body_identical_for_crlf_and_lf_bytes(tmp_path):
    # Content hashes are computed on parse_note bodies, so CRLF (Windows-authored) and
    # LF bytes of the same note must parse to the identical body — otherwise a git
    # line-ending rewrite flips every note to "manually edited" on the next compile.
    lf = tmp_path / "lf.md"
    crlf = tmp_path / "crlf.md"
    lf.write_bytes(b"---\ntitle: T\n---\nHello\nWorld\n")
    crlf.write_bytes(b"---\r\ntitle: T\r\n---\r\nHello\r\nWorld\r\n")

    _, lf_body = parse_note(lf)
    _, crlf_body = parse_note(crlf)

    assert lf_body == crlf_body


def test_sanitize_filename_nonlatin_unchanged():
    assert sanitize_filename("Каталог шаблонов") == "Каталог шаблонов"
    assert sanitize_filename("日本語ノート") == "日本語ノート"


@pytest.mark.parametrize(
    "title",
    [
        "NUL",
        "Foo.",
        "CON.",
        "Foo\x07Bar",
        "word " * 30,
        "Каталог шаблонов",
        "日本語ノート",
        "***///",
    ],
)
def test_sanitize_filename_idempotent(title):
    # filename_drift detection treats sanitize_filename(stem) == sanitize_filename(title)
    # as "converged" — that only holds if the function is a fixpoint of itself.
    once = sanitize_filename(title.strip())
    assert sanitize_filename(once) == once


# ── atomic_write ──────────────────────────────────────────────────────────────


def test_atomic_write_creates_file(tmp_path):
    p = tmp_path / "out.md"
    atomic_write(p, "hello world")
    assert p.read_text() == "hello world"


def test_atomic_write_overwrites(tmp_path):
    p = tmp_path / "out.md"
    p.write_text("old")
    atomic_write(p, "new")
    assert p.read_text() == "new"


def test_atomic_write_no_tmp_left(tmp_path):
    p = tmp_path / "out.md"
    atomic_write(p, "content")
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == []


def test_next_available_path_returns_same_path_when_free(tmp_path):
    path = tmp_path / "Topic.md"
    assert next_available_path(path) == path


def test_next_available_path_suffixes_on_collision(tmp_path):
    (tmp_path / "Topic.md").write_text("x")
    assert next_available_path(tmp_path / "Topic.md") == tmp_path / "Topic-2.md"


def test_next_available_path_suffixes_case_insensitive_collision(tmp_path):
    (tmp_path / "Foo.md").write_text("x")
    assert next_available_path(tmp_path / "foo.md") == tmp_path / "foo-2.md"


def test_next_available_path_skips_taken_suffixes(tmp_path):
    (tmp_path / "Topic.md").write_text("x")
    (tmp_path / "Topic-2.md").write_text("x")
    assert next_available_path(tmp_path / "Topic.md") == tmp_path / "Topic-3.md"


def test_next_available_path_honors_reserved_names(tmp_path):
    path = tmp_path / "Topic.md"

    assert next_available_path(path, reserved_names=["Topic.md"]) == tmp_path / "Topic-2.md"


def test_next_available_path_honors_reserved_names_case_insensitively(tmp_path):
    path = tmp_path / "foo.md"

    assert next_available_path(path, reserved_names=["Foo.md"]) == tmp_path / "foo-2.md"


def test_next_available_path_skips_reserved_suffixes(tmp_path):
    path = tmp_path / "Topic.md"

    assert (
        next_available_path(path, reserved_names=["Topic.md", "Topic-2.md"])
        == tmp_path / "Topic-3.md"
    )


# ── generate_aliases ──────────────────────────────────────────────────────────


def test_generate_aliases_lowercase():
    aliases = generate_aliases("Quantum Computing", "some text")
    assert "quantum computing" in aliases


def test_generate_aliases_same_case_no_duplicate():
    aliases = generate_aliases("quantum computing", "some text")
    assert "quantum computing" not in aliases  # title == lower, skip


def test_generate_aliases_abbreviation():
    text = "Quantum Computing (QC) is fascinating."
    aliases = generate_aliases("Quantum Computing", text)
    assert "QC" in aliases


def test_generate_aliases_multiple_abbreviations():
    text = "Machine Learning (ML) and Deep Learning (DL) are related."
    aliases = generate_aliases("Machine Learning", text)
    assert "ML" in aliases
    assert "DL" not in aliases  # only matches "Machine Learning (..."


# ── sanitize_wikilink_target ──────────────────────────────────────────────────


def test_sanitize_wikilink_target_strips_closing_bracket():
    assert sanitize_wikilink_target("foo]bar") == "foobar"


def test_sanitize_wikilink_target_strips_opening_bracket():
    assert sanitize_wikilink_target("foo[bar") == "foobar"


def test_sanitize_wikilink_target_strips_pipe():
    assert sanitize_wikilink_target("A|B") == "AB"


def test_sanitize_wikilink_target_strips_hash():
    assert sanitize_wikilink_target("title#section") == "titlesection"


def test_sanitize_wikilink_target_passthrough():
    assert sanitize_wikilink_target("Normal Title") == "Normal Title"


def test_sanitize_wikilink_target_strips_filename_forbidden_chars():
    # A link target must equal the filename stem to resolve. Chars that sanitize_filename
    # strips (here ":", "/", "*") must be stripped from the target too — otherwise the
    # link points at a name no file has.
    for title in ["Python: Guide", "TCP/IP", "C*", r"\int", 'A*B"C/D']:
        assert sanitize_wikilink_target(title) == sanitize_filename(title)


def test_sanitize_wikilink_target_strips_colon():
    # Regression for the old "preserves colon" behavior: "Python: Guide" is written to
    # "Python Guide.md", so the link target must be "Python Guide", not "Python: Guide".
    assert sanitize_wikilink_target("Python: Guide") == "Python Guide"


# ── build_wiki_frontmatter ────────────────────────────────────────────────────


def test_build_wiki_frontmatter_sanitizes_tags():
    meta = build_wiki_frontmatter(
        title="Test",
        tags=["quantum computing", "C++ stuff"],
        sources=[],
        confidence=0.8,
    )
    assert meta["tags"] == ["quantum-computing", "c-stuff"]


def test_build_wiki_frontmatter_preserves_valid_tags():
    meta = build_wiki_frontmatter(
        title="Test",
        tags=["physics", "ai"],
        sources=[],
        confidence=0.8,
    )
    assert meta["tags"] == ["physics", "ai"]


def test_build_wiki_frontmatter_deduplicates_tags():
    meta = build_wiki_frontmatter(
        title="Test",
        tags=["AI", "ai", "machine learning", "machine-learning"],
        sources=[],
        confidence=0.5,
    )
    assert meta["tags"] == ["ai", "machine-learning"]


def test_build_wiki_frontmatter_preserves_existing_tags_when_new_tags_are_empty():
    meta = build_wiki_frontmatter(
        title="Test",
        tags=[],
        sources=[],
        confidence=0.5,
        existing_meta={"tags": ["astrology", "zodiac"]},
    )
    assert meta["tags"] == ["astrology", "zodiac"]


def test_chunk_text_heading_split():
    text = (
        "# Title\n\nIntro paragraph.\n\n## Section 1\n\n"
        "Content one.\n\n## Section 2\n\nContent two."
    )
    chunks = chunk_text(text, chunk_size=500)
    assert len(chunks) >= 1


def test_chunk_text_sliding_window():
    # Generate text longer than chunk_size words
    words = ["word"] * 1000
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # All chunks should be non-empty
    assert all(c.strip() for c in chunks)


def test_chunk_text_short_note():
    text = "Short note."
    chunks = chunk_text(text, chunk_size=512)
    assert chunks == ["Short note."]


def test_list_wiki_articles_excludes_source_and_meta_pages(tmp_path):
    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "sources").mkdir(parents=True)
    (wiki_dir / ".drafts").mkdir(parents=True)
    (wiki_dir / "queries").mkdir(parents=True)
    (wiki_dir / "synthesis").mkdir(parents=True)

    write_note(wiki_dir / "Concept.md", {"title": "Concept"}, "body")
    write_note(wiki_dir / "sources" / "Raw.md", {"title": "Raw Source"}, "body")
    write_note(wiki_dir / "queries" / "Saved.md", {"title": "Saved"}, "body")
    write_note(wiki_dir / "synthesis" / "Synth.md", {"title": "Synth"}, "body")
    write_note(wiki_dir / "index.md", {"title": "Index"}, "body")
    write_note(wiki_dir / "log.md", {"title": "Log"}, "body")
    write_note(wiki_dir / ".drafts" / "Draft.md", {"title": "Draft"}, "body")

    result = list_wiki_articles(wiki_dir)

    assert [(title, path.name) for title, path in result] == [("Concept", "Concept.md")]


# ── normalize_wikilinks ───────────────────────────────────────────────────────


def test_normalize_wikilinks_alias_rewrite():
    alias_map = {"ml": "Machine Learning"}
    known = {"Machine Learning"}
    result = normalize_wikilinks("See [[ml]] for details.", alias_map, known)
    assert result == "See [[Machine Learning|ml]] for details."


def test_normalize_wikilinks_canonical_passthrough():
    alias_map = {"machine learning": "Machine Learning"}
    known = {"Machine Learning"}
    result = normalize_wikilinks("See [[Machine Learning]] for details.", alias_map, known)
    assert result == "See [[Machine Learning]] for details."


def test_normalize_wikilinks_unknown_passthrough():
    result = normalize_wikilinks("See [[UnknownTopic]] for details.", {}, {"Machine Learning"})
    assert result == "See [[UnknownTopic]] for details."


def test_normalize_wikilinks_case_insensitive_alias():
    """Alias lookup is case-insensitive; display text preserves original case."""
    alias_map = {"ml": "Machine Learning"}
    known = {"Machine Learning"}
    result = normalize_wikilinks("See [[ML]] for details.", alias_map, known)
    assert result == "See [[Machine Learning|ML]] for details."


def test_normalize_wikilinks_preserves_fragment():
    alias_map = {"ml": "Machine Learning"}
    known = {"Machine Learning"}
    result = normalize_wikilinks("See [[ml#Overview]] for more.", alias_map, known)
    assert result == "See [[Machine Learning#Overview|ml]] for more."


def test_normalize_wikilinks_preserves_display_text():
    alias_map = {"ml": "Machine Learning"}
    known = {"Machine Learning"}
    result = normalize_wikilinks("See [[ml|custom display]] for more.", alias_map, known)
    assert result == "See [[Machine Learning|custom display]] for more."


def test_normalize_wikilinks_skips_fenced_code():
    alias_map = {"ml": "Machine Learning"}
    known = {"Machine Learning"}
    body = "```\n[[ml]] stays as-is\n```"
    assert normalize_wikilinks(body, alias_map, known) == body


def test_normalize_wikilinks_skips_inline_code():
    alias_map = {"ml": "Machine Learning"}
    known = {"Machine Learning"}
    body = "Use `[[ml]]` in code."
    assert normalize_wikilinks(body, alias_map, known) == body


# ── rename_wikilink_targets ───────────────────────────────────────────────────


# Filename stems keep spaces (only Obsidian-forbidden chars are stripped), so a
# concept "Quantm Computing" lives at "Quantm Computing.md" and links read
# [[Quantm Computing]]. Stem and name diverge only when the name has forbidden chars.


def test_rename_repoints_bare_link():
    body = "See [[Quantm Computing]] for more."
    out = rename_wikilink_targets(
        body, "Quantm Computing", "Quantum Computing", "Quantum Computing"
    )
    assert out == "See [[Quantum Computing]] for more."


def test_rename_repoints_target_when_display_echoes_old_name():
    body = "See [[Quantm Computing|Quantm Computing]]."
    out = rename_wikilink_targets(
        body, "Quantm Computing", "Quantum Computing", "Quantum Computing"
    )
    assert out == "See [[Quantum Computing]]."


def test_rename_preserves_deliberate_display_text():
    # An author's intentional, different display must survive — only the target moves.
    body = "Read [[Quantm Computing|the quantum chapter]]."
    out = rename_wikilink_targets(
        body, "Quantm Computing", "Quantum Computing", "Quantum Computing"
    )
    assert out == "Read [[Quantum Computing|the quantum chapter]]."


def test_rename_preserves_fragment():
    body = "Jump to [[Quantm Computing#History]]."
    out = rename_wikilink_targets(
        body, "Quantm Computing", "Quantum Computing", "Quantum Computing"
    )
    assert out == "Jump to [[Quantum Computing#History]]."


def test_rename_leaves_other_links_untouched():
    body = "Links: [[Other]] and [[Quantm Computing]]."
    out = rename_wikilink_targets(
        body, "Quantm Computing", "Quantum Computing", "Quantum Computing"
    )
    assert "[[Other]]" in out
    assert "[[Quantum Computing]]" in out


def test_rename_skips_code_blocks():
    body = "`[[Quantm Computing]]` stays literal."
    out = rename_wikilink_targets(
        body, "Quantm Computing", "Quantum Computing", "Quantum Computing"
    )
    assert out == body


def test_rename_emits_display_link_when_new_name_has_forbidden_chars():
    # Renaming "TCP" → "TCP/IP": stem "TCPIP" ≠ name, so emit [[TCPIP|TCP/IP]].
    body = "See [[TCP]]."
    out = rename_wikilink_targets(body, "TCP", "TCPIP", "TCP/IP")
    assert out == "See [[TCPIP|TCP/IP]]."


def test_rename_repoints_case_variant_links():
    # Targets match the stem case-insensitively (old_key = old_stem.casefold()), so links
    # an author wrote in any case still repoint — otherwise [[quantm computing]] would be
    # left dangling after the rename while [[Quantm Computing]] was fixed.
    body = "See [[quantm computing]] and [[QUANTM COMPUTING]]."
    out = rename_wikilink_targets(
        body, "Quantm Computing", "Quantum Computing", "Quantum Computing"
    )
    assert out == "See [[Quantum Computing]] and [[Quantum Computing]]."


# ── strip_image_text_blocks ─────────────────────────────────────────────────────

# Real extractor output: an "omitted" marker line, then a Start/End block whose body
# is OCR gibberish. Feeding this to an LLM wastes tokens and risks verbatim copying,
# so it must be gone before the body reaches a model.
_OCR_BLOCK = (
    "**==> picture [409 x 22] intentionally omitted <==**\n\n"
    "**----- Start of picture text -----**<br>\n"
    "es OVI Ta we -yo OVI CIV L y α<br>**----- End of picture text -----**<br>\n"
)


def test_strip_removes_block_and_marker_but_keeps_prose():
    body = f"Real intro paragraph.\n\n{_OCR_BLOCK}\n**Figure 2.** Caption that is real."
    out = strip_image_text_blocks(body)
    assert "Start of picture text" not in out
    assert "End of picture text" not in out
    assert "intentionally omitted" not in out
    assert "es OVI Ta we" not in out  # the gibberish itself is gone
    assert "Real intro paragraph." in out
    assert "**Figure 2.** Caption that is real." in out


def test_strip_handles_multiple_blocks():
    body = f"A\n{_OCR_BLOCK}\nB\n{_OCR_BLOCK}\nC"
    out = strip_image_text_blocks(body)
    assert "picture text" not in out
    # Non-greedy match must not swallow the prose between the two blocks.
    assert "A" in out and "B" in out and "C" in out


def test_strip_is_noop_on_clean_text():
    body = "# Title\n\nA normal paragraph with no extractor markers at all."
    assert strip_image_text_blocks(body) == body


def test_strip_tolerates_dash_count_and_whitespace():
    body = "**---  Start of picture text  ---**<br>junk<br>**--- End of picture text ---**<br>"
    assert "picture text" not in strip_image_text_blocks(body)
