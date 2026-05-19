"""Tests for Feature 02: Source-type Prompts."""

from __future__ import annotations

from pathlib import Path

import pytest

from synto.pipeline.prompts import load_prompt

# ---------------------------------------------------------------------------
# Stage 1: notes prompt and backward-compat
# ---------------------------------------------------------------------------


def test_load_prompt_notes_non_empty() -> None:
    prompt = load_prompt("notes")
    assert prompt, "notes prompt must be non-empty"


def test_load_prompt_notes_matches_system_constant() -> None:
    """load_prompt('notes') must equal the _SYSTEM constant imported from ingest.py."""
    from synto.pipeline.ingest import _SYSTEM

    assert load_prompt("notes") == _SYSTEM


def test_notes_prompt_contains_language_field() -> None:
    prompt = load_prompt("notes")
    assert "ISO 639-1" in prompt
    assert "language" in prompt


def test_notes_md_file_exists() -> None:
    prompts_dir = Path(__file__).parent.parent / "src" / "synto" / "pipeline" / "prompts"
    assert (prompts_dir / "notes.md").exists()


# ---------------------------------------------------------------------------
# Stage 2: textbook prompt
# ---------------------------------------------------------------------------


def test_textbook_prompt_non_empty() -> None:
    prompt = load_prompt("textbook")
    assert prompt


def test_textbook_prompt_differs_from_notes() -> None:
    assert load_prompt("textbook") != load_prompt("notes")


def test_textbook_prompt_content() -> None:
    prompt = load_prompt("textbook").lower()
    assert "definitions" in prompt
    assert "examples" in prompt
    assert "exercise" in prompt


def test_ingest_uses_textbook_prompt(tmp_path: Path, config, db) -> None:
    """ingest_note must pass the textbook prompt when source_type='textbook'."""
    from unittest.mock import MagicMock, patch

    from synto.models import AnalysisResult, Concept
    from synto.pipeline.ingest import ingest_note

    note = tmp_path / "book.md"
    note.write_text("---\nsource_type: textbook\n---\nChapter 1: Definitions.\n")

    mock_result = AnalysisResult(
        summary="test",
        concepts=[Concept(name="Definitions", aliases=[], definition="", tags=[])],
        suggested_topics=[],
        named_references=[],
        quality="medium",
        language="en",
    )

    captured_system: list[str] = []

    def fake_request_structured(**kwargs):
        captured_system.append(kwargs.get("system", ""))
        return mock_result

    client = MagicMock()

    with patch("synto.pipeline.ingest.request_structured", side_effect=fake_request_structured):
        ingest_note(note, config, client, db)

    assert captured_system, "request_structured should have been called"
    assert captured_system[0] == load_prompt("textbook")


# ---------------------------------------------------------------------------
# Stage 3: paper prompt
# ---------------------------------------------------------------------------


def test_paper_prompt_non_empty() -> None:
    assert load_prompt("paper")


def test_paper_prompt_content() -> None:
    prompt = load_prompt("paper").lower()
    assert "claims" in prompt
    assert "methods" in prompt
    assert "findings" in prompt
    assert "limitations" in prompt


def test_ingest_uses_paper_prompt(tmp_path: Path, config, db) -> None:
    from unittest.mock import MagicMock, patch

    from synto.models import AnalysisResult, Concept
    from synto.pipeline.ingest import ingest_note

    note = tmp_path / "paper.md"
    note.write_text("---\nsource_type: paper\n---\nThis paper presents a new method.\n")

    mock_result = AnalysisResult(
        summary="test",
        concepts=[Concept(name="Method", aliases=[], definition="", tags=[])],
        suggested_topics=[],
        named_references=[],
        quality="medium",
        language="en",
    )
    captured: list[str] = []

    def fake_request(**kwargs):
        captured.append(kwargs.get("system", ""))
        return mock_result

    with patch("synto.pipeline.ingest.request_structured", side_effect=fake_request):
        ingest_note(note, config, MagicMock(), db)

    assert captured[0] == load_prompt("paper")


# ---------------------------------------------------------------------------
# Stage 4: remaining prompts and fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_type", ["api_docs", "web_article", "corp_docs"])
def test_remaining_prompts_non_empty(source_type: str) -> None:
    prompt = load_prompt(source_type)
    assert prompt, f"Prompt for {source_type!r} must be non-empty"


@pytest.mark.parametrize("source_type", ["api_docs", "web_article", "corp_docs"])
def test_remaining_prompts_differ_from_notes(source_type: str) -> None:
    assert load_prompt(source_type) != load_prompt("notes")


def test_unknown_type_falls_back_to_notes() -> None:
    assert load_prompt("totally_unknown_xyz") == load_prompt("notes")


def test_all_known_types_load() -> None:
    for st in ("notes", "textbook", "paper", "api_docs", "web_article", "corp_docs"):
        prompt = load_prompt(st)
        assert prompt, f"load_prompt({st!r}) returned empty string"
