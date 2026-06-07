"""
Tag sanitization utilities.

Leaf module — no project imports, safe to import from models.py and vault.py.
"""

from __future__ import annotations

import re

# Valid Obsidian tag: [a-zA-Z0-9][a-zA-Z0-9_/-]*
# We enforce lowercase by convention (consistent with hardcoded tags: source, meta, index, query).
_INVALID_CHARS = re.compile(r"[^a-zA-Z0-9_/\-]")
_LEADING_NON_ALNUM = re.compile(r"^[^a-zA-Z0-9]+")


def sanitize_tag(raw: str) -> str:
    """Convert arbitrary string to a valid Obsidian tag (lowercase convention).

    Steps:
      1. Strip whitespace
      2. Replace spaces with hyphens
      3. Remove chars not in [a-zA-Z0-9_/-]
      4. Strip leading non-alphanumeric chars
      5. Lowercase
      6. Return "" if nothing remains
    """
    tag = raw.strip()
    tag = tag.replace(" ", "-")
    tag = _INVALID_CHARS.sub("", tag)
    tag = _LEADING_NON_ALNUM.sub("", tag)
    tag = tag.lower()
    return tag


# Quotes that wrap a name when an LLM echoes it back; stripped from both ends.
_SURROUNDING_QUOTES = re.compile(r"^[`'\"“”‘’«»]+|[`'\"“”‘’«»]+$")
# closer -> opener and opener -> closer for the bracket types we trim at the edges.
_BRACKET_CLOSERS = {")": "(", "]": "["}
_BRACKET_OPENERS = {"(": ")", "[": "]"}


def _trailing_closer_unmatched(s: str) -> bool:
    """Whether s's last char is a closing bracket with no opener of its type to pair with it.

    Scans s[:-1] left→right for that bracket type; the trailing closer is matched iff an opener is
    still open when we reach the end. A global opener/closer count is wrong here — in "(a))" the
    inner ")" is matched and only the final one dangles.
    """
    opener = _BRACKET_CLOSERS[s[-1]]
    closer = s[-1]
    balance = 0
    for ch in s[:-1]:
        if ch == opener:
            balance += 1
        elif ch == closer:
            balance = max(0, balance - 1)
    return balance == 0


def _leading_opener_unmatched(s: str) -> bool:
    """Whether s's first char is an opening bracket with no closer of its type to pair with it.

    Scans s[1:] right→left so an interior matched pair (e.g. the leading "(" in "(a) (b") is not
    mistaken for unmatched by a whole-string count.
    """
    closer = _BRACKET_OPENERS[s[0]]
    opener = s[0]
    balance = 0
    for ch in reversed(s[1:]):
        if ch == closer:
            balance += 1
        elif ch == opener:
            balance = max(0, balance - 1)
    return balance == 0


def clean_display_name(name: str) -> str:
    """Strip dangling/unbalanced edge brackets from a human-readable name.

    A name like ``Phase II)`` (trailing ``)`` with no matching ``(``) otherwise diverges from
    ``Phase II`` at the filename/wikilink boundary — ``sanitize_filename`` keeps parens, so the two
    become separate files. This removes only *unmatched edge brackets* so balanced titles survive:
    ``Extreme Programming (XP)``, ``f(x)``, ``(see note)``, ``(a) (b`` are unchanged, while
    ``Phase II)`` → ``Phase II`` and ``(draft`` → ``draft``. Ordinary trailing punctuation is
    preserved — ``Yahoo!``, ``What Is X?``, ``etc.``, ``C++``, and ``Yahoo!)`` → ``Yahoo!`` are
    kept verbatim apart from the unmatched bracket, since ``sanitize_filename`` keeps those
    characters and no divergence occurs.
    Never returns empty.
    """
    original = name.strip()
    cleaned = _SURROUNDING_QUOTES.sub("", original).strip()

    # Iterate: removing one unmatched edge bracket can expose another ("Phase II))" → "Phase II)").
    while cleaned:
        before = cleaned
        if cleaned[-1] in _BRACKET_CLOSERS and _trailing_closer_unmatched(cleaned):
            cleaned = cleaned[:-1].rstrip()
        if cleaned and cleaned[0] in _BRACKET_OPENERS and _leading_opener_unmatched(cleaned):
            cleaned = cleaned[1:].lstrip()
        if cleaned == before:
            break

    return cleaned or original


def sanitize_tags(raw_tags: list[str]) -> list[str]:
    """Map sanitize_tag over list, filter empties, deduplicate (preserving order)."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in raw_tags:
        tag = sanitize_tag(raw)
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result
