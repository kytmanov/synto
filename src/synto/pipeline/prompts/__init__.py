"""Source-type–specific system prompts for the ingest pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent

# Source types that have dedicated prompt files
_KNOWN_TYPES = frozenset(
    {"notes", "textbook", "paper", "api_docs", "web_article", "corp_docs"}
)


def load_prompt(source_type: str) -> str:
    """Return the system prompt string for the given source type.

    Falls back to the 'notes' prompt for unknown types and logs a warning.
    Trailing newline is stripped so callers get a clean string.
    """
    prompt_path = _PROMPTS_DIR / f"{source_type}.md"
    if not prompt_path.exists():
        if source_type not in ("notes", "unknown_text", "spec", "transcript"):
            log.warning("No prompt file for source_type=%r; falling back to notes", source_type)
        prompt_path = _PROMPTS_DIR / "notes.md"
    return prompt_path.read_text(encoding="utf-8").rstrip("\n")
