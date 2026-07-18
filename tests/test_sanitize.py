"""Tests for sanitize.py — pure functions, no I/O."""

from __future__ import annotations

import unicodedata

import pytest

from synto.sanitize import clean_display_name, sanitize_tag, sanitize_tags

# ── sanitize_tag ──────────────────────────────────────────────────────────────


def test_spaces_to_hyphens():
    assert sanitize_tag("quantum computing") == "quantum-computing"


def test_multiple_spaces():
    assert sanitize_tag("machine learning basics") == "machine-learning-basics"


def test_cpp_special_chars_removed():
    assert sanitize_tag("C++ programming") == "c-programming"


def test_leading_hash_stripped():
    assert sanitize_tag("#my-tag") == "my-tag"


def test_valid_tag_passthrough():
    assert sanitize_tag("physics") == "physics"


def test_garbage_returns_empty():
    assert sanitize_tag("!!!") == ""


def test_slashes_preserved():
    assert sanitize_tag("science/physics") == "science/physics"


def test_pure_numbers_valid():
    assert sanitize_tag("2024") == "2024"


def test_underscores_preserved():
    assert sanitize_tag("my_tag") == "my_tag"


def test_hyphens_preserved():
    assert sanitize_tag("already-hyphenated") == "already-hyphenated"


def test_mixed_case_lowercased():
    assert sanitize_tag("MachineLearning") == "machinelearning"


def test_uppercase_tag_lowercased():
    assert sanitize_tag("AI") == "ai"


def test_leading_hyphen_stripped():
    assert sanitize_tag("-bad-start") == "bad-start"


def test_leading_underscore_stripped():
    assert sanitize_tag("_bad-start") == "bad-start"


def test_whitespace_only_returns_empty():
    assert sanitize_tag("   ") == ""


def test_empty_string_returns_empty():
    assert sanitize_tag("") == ""


def test_at_symbol_removed():
    assert sanitize_tag("tag@user") == "taguser"


# ── sanitize_tag: language-agnostic (any script survives, Obsidian allows Unicode tags) ──


def test_accented_latin_preserved():
    assert sanitize_tag("café") == "café"


def test_cyrillic_preserved():
    assert sanitize_tag("каталог") == "каталог"


def test_cyrillic_lowercased():
    assert sanitize_tag("Каталог") == "каталог"


def test_cyrillic_nested_path_preserved():
    assert sanitize_tag("код/архитектура") == "код/архитектура"


def test_cjk_preserved():
    # Caseless script — lowercasing must be a no-op, not a mangle.
    assert sanitize_tag("日本語") == "日本語"


def test_cyrillic_spaces_to_hyphens():
    assert sanitize_tag("машинное обучение") == "машинное-обучение"


def test_leading_punctuation_stripped_before_cyrillic():
    assert sanitize_tag("#каталог") == "каталог"


def test_punctuation_removed_from_unicode_tag():
    assert sanitize_tag("каталог!") == "каталог"


def test_garbage_still_empty_with_unicode_rules():
    assert sanitize_tag("!!!@@@###") == ""


# ── sanitize_tag: idempotence and normalization ───────────────────────────────
# lint's invalid_tag check is `t != sanitize_tag(t)`, so any input where a second pass
# differs from the first makes lint re-flag the tag --fix just wrote (review finding on
# Turkish İ: filter-before-lower let the combining dot of lower("İ") survive one pass).


@pytest.mark.parametrize(
    "raw",
    [
        "İstanbul",  # lower() emits U+0307 combining dot — the reported case
        unicodedata.normalize("NFD", "café"),  # decomposed input (macOS-style)
        unicodedata.normalize("NFD", "Ёлка"),
        "ẞ",  # capital sharp s → ß
        "ǅungla",  # titlecase digraph
        "каталог",
        "日本語",
        "quantum computing",
        "C++ programming",
        "#my-tag",
        "2024",
        "!!!",
    ],
)
def test_sanitize_tag_idempotent(raw):
    once = sanitize_tag(raw)
    assert sanitize_tag(once) == once


def test_turkish_dotted_capital_folds_cleanly():
    assert sanitize_tag("İstanbul") == "istanbul"


def test_nfd_input_normalized_to_nfc():
    # Decomposed "café" (e + combining acute) must keep its accent, as NFC.
    nfd = unicodedata.normalize("NFD", "café")
    assert sanitize_tag(nfd) == "café"
    assert unicodedata.is_normalized("NFC", sanitize_tag(nfd))


# ── sanitize_tags ─────────────────────────────────────────────────────────────


def test_dedup_case_insensitive():
    result = sanitize_tags(["AI", "ai"])
    assert result == ["ai"]


def test_filters_empty_results():
    result = sanitize_tags(["!!!", "valid"])
    assert result == ["valid"]


def test_preserves_order():
    result = sanitize_tags(["beta", "alpha", "gamma"])
    assert result == ["beta", "alpha", "gamma"]


def test_filters_all_garbage():
    result = sanitize_tags(["!!!", "@@@", "###"])
    assert result == []


def test_empty_list():
    assert sanitize_tags([]) == []


def test_mixed_valid_and_invalid():
    result = sanitize_tags(["quantum computing", "!!!", "physics"])
    assert result == ["quantum-computing", "physics"]


def test_no_duplicate_after_sanitize():
    # Both sanitize to "machine-learning"
    result = sanitize_tags(["machine learning", "machine-learning"])
    assert result == ["machine-learning"]


# ── clean_display_name ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        # The #53 case: dangling closer with no opener → drop it so it stops diverging.
        ("Phase II)", "Phase II"),
        ("Phase II))", "Phase II"),
        ("(draft", "draft"),
        ("[draft", "draft"),
        ("Phase II.)", "Phase II."),
        # Balanced punctuation is part of the title → kept verbatim.
        ("Extreme Programming (XP)", "Extreme Programming (XP)"),
        ("f(x)", "f(x)"),
        ("(see note)", "(see note)"),
        # A matched leading "(" must survive even when an interior bracket is unbalanced — the edge
        # check matches the specific bracket, not whole-string counts.
        ("(a) (b", "(a) (b"),
        ("(RFC 1234) (draft", "(RFC 1234) (draft"),
        # Only the genuinely dangling edge bracket is trimmed; the matched pair is kept.
        ("(a))", "(a)"),
        ("(see note) extra)", "(see note) extra"),
        # Ordinary trailing punctuation is part of the name and kept — sanitize_filename keeps
        # these chars, so there is no filename divergence to fix (cf. the bracket cases above).
        ("etc.", "etc."),
        ("Done!", "Done!"),
        ("Yahoo!", "Yahoo!"),
        ("Jeopardy!", "Jeopardy!"),
        ("What Is Yahoo!?", "What Is Yahoo!?"),
        ("Yahoo!)", "Yahoo!"),
        ("What Is Yahoo!?)", "What Is Yahoo!?"),
        ("etc.)", "etc."),
        (".NET", ".NET"),
        ("Node.js", "Node.js"),
        ("C++", "C++"),
        # Surrounding quotes stripped, then the dangling closer.
        ('"Phase II)"', "Phase II"),
        # Internal unbalanced paren is left alone — we only trim edges.
        ("Foo (bar", "Foo (bar"),
        # Never returns empty: a name made only of strippable chars falls back to its stripped form.
        (")", ")"),
        ("  Quantum  ", "Quantum"),
    ],
)
def test_clean_display_name(raw, expected):
    assert clean_display_name(raw) == expected
