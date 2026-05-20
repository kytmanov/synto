"""Tests for markdown_math.py."""

from __future__ import annotations

from synto.markdown_math import (
    has_malformed_obsidian_math,
    mask_markdown_regions,
    restore_markdown_regions,
    sanitize_obsidian_math,
)

# ── mask_markdown_regions ─────────────────────────────────────────────────────


def test_mask_code_fences():
    """Code fences are masked."""
    text = "Before\n```python\nprint('hello')\n```\nAfter"
    masked, replacements = mask_markdown_regions(text)
    assert "__SYNTO_MARKDOWN_MASK_0__" in masked
    assert "```" not in masked
    restored = restore_markdown_regions(masked, replacements)
    assert restored == text


def test_mask_inline_code():
    """Inline code is masked."""
    text = "Use `print()` for output."
    masked, replacements = mask_markdown_regions(text)
    assert "__SYNTO_MARKDOWN_MASK_0__" in masked
    restored = restore_markdown_regions(masked, replacements)
    assert restored == text


def test_mask_display_math():
    """Display math $$...$$ is masked."""
    text = "Formula: $$E = mc^2$$ done."
    masked, replacements = mask_markdown_regions(text)
    assert "$$" not in masked
    restored = restore_markdown_regions(masked, replacements)
    assert restored == text


def test_mask_inline_math():
    """Inline math $...$ is masked."""
    text = "The value is $x + y$ here."
    masked, replacements = mask_markdown_regions(text)
    assert "$x" not in masked
    restored = restore_markdown_regions(masked, replacements)
    assert restored == text


def test_mask_paren_math():
    """\\(...\\) math is masked."""
    text = r"Use \(a^2 + b^2 = c^2\) for pythagorean."
    masked, replacements = mask_markdown_regions(text)
    assert r"\(" not in masked
    restored = restore_markdown_regions(masked, replacements)
    assert restored == text


def test_mask_images():
    """Image markdown is masked."""
    text = "See ![alt](image.png) here."
    masked, replacements = mask_markdown_regions(text)
    assert "![" not in masked
    restored = restore_markdown_regions(masked, replacements)
    assert restored == text


def test_mask_links():
    """Markdown links are masked."""
    text = "Visit [Google](https://google.com)."
    masked, replacements = mask_markdown_regions(text)
    assert "[Google]" not in masked
    restored = restore_markdown_regions(masked, replacements)
    assert restored == text


def test_mask_wikilinks_enabled():
    """Wikilinks are masked by default."""
    text = "See [[Machine Learning]] for details."
    masked, replacements = mask_markdown_regions(text)
    assert "[[" not in masked
    restored = restore_markdown_regions(masked, replacements)
    assert restored == text


def test_mask_wikilinks_disabled():
    """Wikilinks not masked when mask_wikilinks=False."""
    text = "See [[Machine Learning]] for details."
    masked, _ = mask_markdown_regions(text, mask_wikilinks=False)
    assert "[[Machine Learning]]" in masked


def test_mask_embeds_enabled():
    """Obsidian embeds are masked by default."""
    text = "Embed: ![[some-note]]"
    masked, replacements = mask_markdown_regions(text)
    assert "![" not in masked
    restored = restore_markdown_regions(masked, replacements)
    assert restored == text


def test_mask_embeds_disabled():
    """Embeds not masked when mask_embeds=False — but wikilink part still masked."""
    text = "Embed: ![[some-note]]"
    masked, _ = mask_markdown_regions(text, mask_embeds=False)
    # The ! prefix remains but [[some-note]] is masked by wikilink pattern
    assert "!" in masked
    assert "[[some-note]]" not in masked


def test_mask_multiple_regions():
    """Multiple different regions are masked with unique tokens."""
    text = "Code: `x`\nMath: $y$\nLink: [a](b)"
    masked, replacements = mask_markdown_regions(text)
    assert len(replacements) == 3
    restored = restore_markdown_regions(masked, replacements)
    assert restored == text


# ── sanitize_obsidian_math ────────────────────────────────────────────────────


def test_sanitize_display_math_double_backslash():
    r"""\\[...\\] is converted to $$...$$."""
    text = r"\\[E = mc^2\\]"
    result = sanitize_obsidian_math(text)
    assert "$$" in result
    assert r"\[" not in result


def test_sanitize_display_math_single_backslash():
    r"\[...\] is converted to $$...$$." ""
    text = r"\[E = mc^2\]"
    result = sanitize_obsidian_math(text)
    assert "$$" in result


def test_sanitize_bare_latex_line():
    r"""Bare \frac line is wrapped in $$."""
    text = r"\frac{1}{2}"
    result = sanitize_obsidian_math(text)
    assert "$$" in result


def test_sanitize_preserves_code_blocks():
    """Math-like content in code blocks is not modified."""
    text = "```\n\\frac{1}{2}\n```"
    result = sanitize_obsidian_math(text)
    assert result == text


def test_sanitize_preserves_heading_lines():
    """Lines starting with # are not treated as latex."""
    text = "# Heading with \\alpha"
    result = sanitize_obsidian_math(text)
    assert "$$" not in result


def test_sanitize_preserves_list_lines():
    """Lines starting with - or * are not treated as latex."""
    text = r"- \frac{1}{2}"
    result = sanitize_obsidian_math(text)
    assert "$$" not in result


def test_sanitize_preserves_numbered_list():
    """Numbered list items are not treated as latex."""
    text = r"1. \frac{1}{2}"
    result = sanitize_obsidian_math(text)
    assert "$$" not in result


def test_sanitize_preserves_blockquote():
    """Blockquote lines are not treated as latex."""
    text = r"> \frac{1}{2}"
    result = sanitize_obsidian_math(text)
    assert "$$" not in result


def test_sanitize_double_backslash_command():
    r"""\\frac (escaped backslash) is handled correctly."""
    text = r"\\frac{a}{b}"
    result = sanitize_obsidian_math(text)
    # Should be wrapped in $$
    assert "$$" in result


def test_sanitize_empty_display_math():
    """Empty display math is left unchanged."""
    text = r"\[\]"
    result = sanitize_obsidian_math(text)
    assert result == text


# ── has_malformed_obsidian_math ───────────────────────────────────────────────


def test_has_malformed_no_changes():
    """Content that doesn't need sanitizing returns False."""
    text = "Normal text with no math."
    assert has_malformed_obsidian_math(text) is False


def test_has_malformed_needs_sanitizing():
    """Content with \\[...\\] returns True."""
    text = r"\\[E = mc^2\\]"
    assert has_malformed_obsidian_math(text) is True


def test_has_malformed_bare_latex():
    """Bare latex lines return True."""
    text = r"\frac{1}{2}"
    assert has_malformed_obsidian_math(text) is True
