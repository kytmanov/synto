"""Concept label normalization helpers.

Neutral module imported by both state.py (resolve_label) and pipeline/ingest.py
(_normalize_concepts). Do not import from synto.pipeline here.
"""

from __future__ import annotations

import re
import unicodedata

from .sanitize import clean_display_name

_PAREN_ABBR_RE = re.compile(r"^(?P<base>.+?)\s*\((?P<abbr>[A-ZА-Я0-9][A-ZА-Я0-9.+-]{1,8})\)$")
_SURROUNDING_QUOTES_RE = re.compile(r"^[`'\"" "''«»]+|[`'\"" "''«»]+$")


def clean_concept_text(text: str) -> str:
    """NFKC-normalize, strip surrounding quotes, drop unbalanced punctuation."""
    text = unicodedata.normalize("NFKC", text).strip()
    text = _SURROUNDING_QUOTES_RE.sub("", text).strip()
    # Drop dangling/unbalanced punctuation so a stray char can't mint a concept name
    # that diverges into its own file. Runs before base_concept_name.
    text = clean_display_name(text)
    return re.sub(r"\s+", " ", text)


def concept_key(text: str) -> str:
    """Deterministic key for exact concept matching; not used as display text."""
    text = clean_concept_text(text).casefold()
    text = re.sub(r"[_\-/:]+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def base_concept_name(text: str) -> str:
    """Strip only safe parenthetical abbreviations, e.g. Extreme Programming (XP)."""
    cleaned = clean_concept_text(text)
    match = _PAREN_ABBR_RE.match(cleaned)
    if not match:
        return cleaned
    abbr = match.group("abbr")
    if not abbr.isupper():
        return cleaned
    return match.group("base").strip()


def match_key(text: str) -> str:
    """Folded key for plural/singular matching (decision 20).

    Folds on the *match level* only — uniqueness still uses concept_key/label_key.
    Per-token rules: only tokens len >= 4; skip ALL-CAPS/acronyms;
    ies->y, sses->ss, else strip trailing s unless token ends in ss/us/is.
    """
    base = concept_key(text)
    tokens = base.split()
    folded = [_fold_token(t) for t in tokens]
    return " ".join(folded)


def _fold_token(token: str) -> str:
    if len(token) < 4:
        return token
    # Skip all-caps / acronyms (e.g. "US", "NASA")
    if token.isupper():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]  # sses -> ss
    if token.endswith("s") and not (
        token.endswith("ss") or token.endswith("us") or token.endswith("is")
    ):
        return token[:-1]
    return token


def legacy_name_key(text: str) -> str:
    """Folding key for the remaining name-keyed legacy tables
    (rejections, stubs, blocked_concepts, knowledge_items, and some occurrence
    paths). This makes the 'still on display names' debt greppable and mechanical
    to migrate later. Reuses the same normalization as concept_key for consistency
    with how surfaces are matched elsewhere.
    """
    return concept_key(text)
