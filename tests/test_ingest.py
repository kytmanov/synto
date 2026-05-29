"""Tests for pipeline/ingest.py — no Ollama required (mocked client)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from synto.cli import cli
from synto.config import Config
from synto.models import AnalysisResult, Concept
from synto.pipeline.ingest import (
    _SYSTEM,
    _analyze_body,
    _analyze_body_with_checkpoints,
    _base_concept_name,
    _build_analysis_prompt,
    _build_trusted_alias_rewrite_index,
    _canonical_prompt_contexts,
    _checkpoint_hash,
    _concept_key,
    _content_hash,
    _dedup_by_shared_alias,
    _display_aliases,
    _filter_concept_candidates,
    _ingest_prompt_version,
    _is_noise_concept,
    _meaningful_text_stats,
    _merge_chunk_results,
    _normalize_concepts,
    _preprocess_web_clip,
    _rewrite_candidates_to_canonicals,
    _safe_aliases_for_name,
    _suggested_topic_candidates,
    ingest_all,
    ingest_note,
    write_source_content_md,
)
from synto.state import StateDB

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / ".synto").mkdir()
    return tmp_path


@pytest.fixture
def config(vault):
    return Config(vault=vault)


@pytest.fixture
def db(config):
    return StateDB(config.state_db_path)


def _make_client(analysis_json: str) -> MagicMock:
    client = MagicMock()
    client.generate.return_value = analysis_json
    return client


def _write_raw(vault: Path, name: str, content: str) -> Path:
    p = vault / "raw" / name
    p.write_text(content, encoding="utf-8")
    return p


# ── _preprocess_web_clip ──────────────────────────────────────────────────────


def test_preprocess_strips_html_tags():
    content = (
        "<nav>Skip Navigation Menu</nav>\n\n"
        "# Real Content\n\n"
        "Full paragraph with enough words to pass the filter."
    )
    result = _preprocess_web_clip(content)
    assert "<nav>" not in result
    assert "Real Content" in result
    assert "Full paragraph" in result


def test_preprocess_strips_short_header_lines():
    # Short plain-text lines in first 30 lines (nav/banner) should be stripped
    # But markdown headings (starting with #) must be kept even if short
    lines = [
        "Home",
        "About",
        "Contact",
        "",
        "# Article Title",
        "",
        "This is a full substantive paragraph with many words that will not be stripped.",
    ]
    result = _preprocess_web_clip("\n".join(lines))
    assert "Home" not in result
    assert "Article Title" in result
    assert "substantive paragraph" in result


def test_preprocess_preserves_short_body_lines():
    """Short lines AFTER line 30 must NOT be stripped (bullets, code comments, etc.)."""
    header = ["Nav item"] * 31  # push past the 30-line scan window
    body = ["- Key insight", "- Another bullet", "Short sentence."]
    content = "\n".join(header + body)
    result = _preprocess_web_clip(content)
    assert "Key insight" in result
    assert "Another bullet" in result


def test_preprocess_preserves_body_html():
    """HTML after line 30 (body content) must be preserved."""
    header = ["Nav item"] * 31  # push past the 30-line scan window
    body = [
        "<details><summary>Collapse me</summary>",
        "Hidden content here.",
        "</details>",
        "Use <kbd>Ctrl+C</kbd> to copy.",
    ]
    content = "\n".join(header + body)
    result = _preprocess_web_clip(content)
    assert "<details>" in result
    assert "<kbd>Ctrl+C</kbd>" in result


def test_preprocess_strips_header_html():
    """HTML tags in first 30 lines must be stripped."""
    content = "<nav>Skip Navigation</nav>\n\n# Real Title\n\nBody content here."
    result = _preprocess_web_clip(content)
    assert "<nav>" not in result
    assert "Real Title" in result


def test_preprocess_preserves_blank_lines():
    content = "Home\n\n# Title\n\nContent."
    result = _preprocess_web_clip(content)
    assert "Title" in result


# ── _build_analysis_prompt ────────────────────────────────────────────────────


def test_build_prompt_includes_body():
    prompt = _build_analysis_prompt("Some content here.", [])
    assert "Some content here" in prompt


def test_build_prompt_includes_existing_concepts():
    prompt = _build_analysis_prompt("content", ["Quantum Computing", "Machine Learning"])
    assert "Quantum Computing" in prompt
    assert "Machine Learning" in prompt


def test_build_prompt_prioritizes_body_matched_existing_concepts():
    existing = [*(f"Noise {i}" for i in range(40)), "Template Catalog", "Каталог шаблонов"]

    prompt = _build_analysis_prompt(
        "Русский текст про каталог шаблонов и template catalog.",
        existing,
        "note.md",
    )

    hint_line = prompt.splitlines()[2]
    assert "Template Catalog" in hint_line
    assert "Каталог шаблонов" in hint_line
    assert "Noise 39" not in hint_line


def test_build_prompt_prioritizes_canonical_when_alias_matches_body():
    prompt = _build_analysis_prompt(
        "Русский текст про каталог шаблонов.",
        ["Template Catalog"],
        "note.md",
        prompt_concepts=_canonical_prompt_contexts(
            ["Template Catalog"], {"каталог шаблонов": "Template Catalog"}
        ),
    )

    hint_line = prompt.splitlines()[2]
    assert "Template Catalog" in hint_line
    assert "aliases: каталог шаблонов" in hint_line


def test_build_prompt_preserves_note_language_for_concepts():
    prompt = _build_analysis_prompt("Русский текст про каталог шаблонов.", [])
    # Surface-form based rule (not "same language" wording which was English-centric)
    assert "form they appear in the note text" in prompt
    assert "Do not translate concept names" in prompt
    # Acronym heuristic is present but scoped to pure acronyms; methodology terms are excluded
    assert "pure acronym or initialism" in prompt
    assert "API" in prompt
    assert "Methodology and domain terms" in prompt
    assert "not in this category" in prompt


def test_build_prompt_language_override_suppresses_note_language_rule():
    prompt = _build_analysis_prompt("Some content", [], language="de")
    assert "de (ISO 639-1)" in prompt
    assert "form they appear in the note text" not in prompt
    assert "pure acronym or initialism" not in prompt
    # Must include canonical-reuse exception so existing wiki concepts are not translated
    assert "exact canonical name shown" in prompt


def test_build_prompt_language_override_alias_instruction_requests_native_form():
    prompt = _build_analysis_prompt("Some content", [], language="ru")
    # Aliases are explicitly multi-lingual; native surface form must be requested
    assert "Aliases may be in any language" in prompt
    assert "surface form as the first alias" in prompt


def test_build_prompt_includes_full_body():
    # No truncation in _build_analysis_prompt — chunking happens in _analyze_body
    body = "x " * 5000  # ~10000 chars
    prompt = _build_analysis_prompt(body, [])
    assert body in prompt


def test_build_prompt_includes_chunk_label():
    prompt = _build_analysis_prompt("content", [], chunk_label="[part 2/4]")
    assert "[part 2/4]" in prompt


def test_build_prompt_no_chunk_label_by_default():
    prompt = _build_analysis_prompt("content", [])
    assert "[part" not in prompt


# ── _normalize_concepts ───────────────────────────────────────────────────────


def test_concept_key_normalizes_case_punctuation_and_unicode():
    assert _concept_key(" Extreme-Programming: XP ") == "extreme programming xp"
    assert _concept_key("Ｆｕｓｅ　Diagram") == "fuse diagram"


def test_base_concept_name_strips_only_safe_abbreviations():
    assert _base_concept_name("Extreme Programming (XP)") == "Extreme Programming"
    assert _base_concept_name("Scrum (framework)") == "Scrum (framework)"


def test_safe_aliases_for_name_extracts_parenthetical_abbreviation():
    aliases = _safe_aliases_for_name("Extreme Programming (XP)")
    assert "Extreme Programming" in aliases
    assert "XP" in aliases


def test_is_noise_concept_detects_generic_unknowns():
    assert _is_noise_concept("Image content unknown") is True
    assert _is_noise_concept("Untitled") is True
    assert _is_noise_concept("Extreme Programming") is False


def test_meaningful_text_stats_ignores_media_and_urls():
    chars, words = _meaningful_text_stats("![[image.png]] https://example.com")
    assert chars == 0
    assert words == 0


def test_suggested_topic_candidates_require_meaningful_text():
    result = AnalysisResult(
        summary="Only media.",
        concepts=[],
        suggested_topics=["Image content unknown"],
        quality="low",
    )
    assert _suggested_topic_candidates(result, "![[image.png]]") == []


def test_suggested_topic_candidates_allow_meaningful_text():
    result = AnalysisResult(
        summary="API note.",
        concepts=[],
        suggested_topics=["API Response Testing"],
        quality="high",
    )
    body = "API response testing validates response status, schema, headers, boundaries. " * 2
    candidates = _suggested_topic_candidates(result, body)
    assert [c.name for c in candidates] == ["API Response Testing"]


def test_filter_concept_candidates_keeps_medium_quality_concept_with_evidence():
    result = AnalysisResult(
        summary="Article about planning rituals.",
        concepts=[Concept(name="Project Coordination", aliases=[])],
        suggested_topics=[],
        quality="medium",
    )
    body = "This article discusses project coordination, team rituals, and meeting notes." * 2

    assert [c.name for c in _filter_concept_candidates(result.concepts, result, body)] == [
        "Project Coordination"
    ]


def test_filter_concept_candidates_drops_medium_quality_concept_without_evidence():
    result = AnalysisResult(
        summary="Article about planning rituals.",
        concepts=[Concept(name="Project Coordination", aliases=[])],
        suggested_topics=[],
        quality="medium",
    )
    body = "This article discusses team rituals, meeting notes, and project planning." * 2

    assert _filter_concept_candidates(result.concepts, result, body) == []


def test_filter_concept_candidates_keeps_low_quality_title_evidence():
    result = AnalysisResult(
        summary="Documentary notes.",
        concepts=[Concept(name="Example Artist documentary", aliases=[])],
        suggested_topics=[],
        quality="low",
    )

    kept = _filter_concept_candidates(
        result.concepts, result, "![[image.png]]", "Example Artist documentary.md"
    )

    assert [c.name for c in kept] == ["Example Artist documentary"]


def test_filter_concept_candidates_passes_translated_concept_via_validated_alias():
    """Translated concept name not in body survives when a validated alias provides evidence."""
    # Body uses the exact surface form that the alias captures; Russian nominative case
    body = "Каталог шаблонов применяется для управления проектными ресурсами." * 3
    result = AnalysisResult(
        summary="About template systems.",
        concepts=[Concept(name="Vorlagenkatalog", aliases=["Каталог шаблонов"])],
        suggested_topics=[],
        quality="medium",
    )
    kept = _filter_concept_candidates(result.concepts, result, body)
    assert [c.name for c in kept] == ["Vorlagenkatalog"]


def test_filter_concept_candidates_drops_noise_alias_false_positive():
    """Noise-concept alias ('document') must not rescue a concept with no real evidence."""
    body = "This document contains some information about the system."
    result = AnalysisResult(
        summary="A document.",
        concepts=[Concept(name="Vorlagenkatalog", aliases=["document"])],
        suggested_topics=[],
        quality="medium",
    )
    assert _filter_concept_candidates(result.concepts, result, body) == []


def _make_concepts(names):
    from synto.models import Concept

    return [Concept(name=n, aliases=[]) for n in names]


def test_normalize_reuses_canonical_case(vault, config, db):
    db.upsert_concepts("raw/a.md", ["Quantum Computing"])
    result = _normalize_concepts(_make_concepts(["quantum computing"]), db)
    assert [name for name, _ in result] == ["Quantum Computing"]


def test_normalize_reuses_canonical_for_punctuation_variant(vault, config, db):
    db.upsert_concepts("raw/a.md", ["Extreme Programming"])
    result = _normalize_concepts(_make_concepts(["extreme-programming"]), db)
    assert [name for name, _ in result] == ["Extreme Programming"]


def test_normalize_merges_parenthetical_abbreviation_variant(vault, config, db):
    db.upsert_concepts("raw/a.md", ["Extreme Programming"])
    result = _normalize_concepts(_make_concepts(["Extreme Programming (XP)"]), db)
    assert [name for name, _ in result] == ["Extreme Programming"]
    assert "XP" in result[0][1]


def test_normalize_uses_base_name_for_new_parenthetical_abbreviation(vault, config, db):
    result = _normalize_concepts(_make_concepts(["Extreme Programming (XP)"]), db)
    assert [name for name, _ in result] == ["Extreme Programming"]
    assert "XP" in result[0][1]


def test_normalize_deduplicates_same_note_parenthetical_variant(vault, config, db):
    result = _normalize_concepts(
        _make_concepts(["Extreme Programming", "Extreme Programming (XP)"]), db
    )
    assert [name for name, _ in result] == ["Extreme Programming"]


def test_normalize_merges_unambiguous_abbreviation(vault, config, db):
    db.upsert_concepts("raw/a.md", ["Extreme Programming (XP)"])
    result = _normalize_concepts(_make_concepts(["XP"]), db)
    assert [name for name, _ in result] == ["Extreme Programming (XP)"]


def test_normalize_does_not_merge_ambiguous_abbreviation(vault, config, db):
    db.upsert_concepts("raw/a.md", ["Extreme Programming (XP)"])
    db.upsert_concepts("raw/b.md", ["Experience Points (XP)"])
    result = _normalize_concepts(_make_concepts(["XP"]), db)
    assert [name for name, _ in result] == ["XP"]


def test_normalize_does_not_use_llm_aliases_for_merge(vault, config, db):
    db.upsert_concepts("raw/a.md", ["Scrum"])
    concept = Concept(name="Iterative development", aliases=["Scrum"])
    result = _normalize_concepts([concept], db)
    assert [name for name, _ in result] == ["Iterative development"]


def test_normalize_reuses_canonical_via_stored_alias(vault, config, db):
    db.upsert_concepts("raw/a.md", ["Template Catalog"])
    db.upsert_aliases("Template Catalog", ["Каталог шаблонов"])

    result = _normalize_concepts(_make_concepts(["Каталог шаблонов"]), db)

    assert result == [("Template Catalog", ["Каталог шаблонов"])]


def test_normalize_preserves_cross_language_surface_as_alias(vault, config, db):
    db.upsert_concepts("raw/a.md", ["Decision Heuristics"])
    db.upsert_aliases("Decision Heuristics", ["эвристики решений"])

    result = _normalize_concepts(_make_concepts(["эвристики решений"]), db)

    assert result == [("Decision Heuristics", ["эвристики решений"])]


def test_normalize_reuses_seeded_index_alias_when_db_empty(vault, config, db):
    result = _normalize_concepts(
        _make_concepts(["гибкая разработка"]),
        db,
        None,
        {"гибкая разработка": "Agile Development"},
    )

    assert result == [("Agile Development", ["гибкая разработка"])]


def test_normalize_reuses_seeded_canonical_case_when_db_empty(vault, config, db):
    result = _normalize_concepts(_make_concepts(["api testing"]), db, ["API Testing"])

    assert result == [("API Testing", [])]


def test_build_trusted_alias_rewrite_index_skips_alias_that_is_other_canonical():
    rewrite = _build_trusted_alias_rewrite_index(
        ["Template Catalog", "Каталог шаблонов"],
        {},
        {"Каталог шаблонов": "Template Catalog"},
    )

    assert _concept_key("Каталог шаблонов") not in rewrite


def test_rewrite_candidates_to_canonicals_rewrites_unambiguous_seed_alias(vault, config, db):
    rewritten = _rewrite_candidates_to_canonicals(
        [Concept(name="Разработка через тестирование", aliases=[])],
        ["Test Driven Development"],
        {},
        {"Разработка через тестирование": "Test Driven Development"},
    )

    assert [candidate.name for candidate in rewritten] == ["Test Driven Development"]
    assert _display_aliases(rewritten[0].aliases) == ["Разработка через тестирование"]


def test_rewrite_candidates_to_canonicals_skips_alias_that_is_also_other_canonical(
    vault, config, db
):
    rewritten = _rewrite_candidates_to_canonicals(
        [Concept(name="Каталог шаблонов", aliases=[])],
        ["Template Catalog", "Каталог шаблонов"],
        {},
        {"Каталог шаблонов": "Template Catalog"},
    )

    assert rewritten == [Concept(name="Каталог шаблонов", aliases=[])]


def test_rewrite_candidates_to_canonicals_uses_trusted_candidate_alias_match(vault, config, db):
    rewritten = _rewrite_candidates_to_canonicals(
        [Concept(name="Vorlagenkatalog", aliases=["Каталог шаблонов"])],
        ["Template Catalog"],
        {"каталог шаблонов": "Template Catalog"},
    )

    assert [candidate.name for candidate in rewritten] == ["Template Catalog"]
    assert _display_aliases(rewritten[0].aliases) == ["Vorlagenkatalog", "Каталог шаблонов"]


def test_normalize_deduplicates(vault, config, db):
    result = _normalize_concepts(_make_concepts(["ML", "ML", "Machine Learning"]), db)
    assert len(result) == 2
    assert any(name == "ML" for name, _ in result)


def test_normalize_strips_empty(vault, config, db):
    result = _normalize_concepts(_make_concepts(["", "  ", "Neural Networks"]), db)
    names = [name for name, _ in result]
    assert "" not in names
    assert "  " not in names
    assert "Neural Networks" in names


# ── ingest_note ───────────────────────────────────────────────────────────────


def _analysis_json(
    concepts=None,
    quality="high",
    summary="A summary.",
    suggested_topics=None,
    named_references=None,
):
    names = ["Quantum Computing", "Qubit"] if concepts is None else concepts
    topics = ["Quantum Computing"] if suggested_topics is None else suggested_topics
    return json.dumps(
        {
            "summary": summary,
            "concepts": [{"name": c, "aliases": []} for c in names],
            "suggested_topics": topics,
            "named_references": named_references or [],
            "quality": quality,
        }
    )


def test_ingest_note_returns_analysis_result(vault, config, db):
    path = _write_raw(vault, "quantum.md", "# Quantum Computing\n\nQubits are awesome.")
    client = _make_client(_analysis_json())
    result = ingest_note(path, config, client, db)
    assert result is not None
    assert result.quality == "high"
    assert len(result.concepts) >= 1


def test_ingest_note_stores_status_ingested(vault, config, db):
    path = _write_raw(vault, "note.md", "# Note\n\nSome content here.")
    client = _make_client(_analysis_json())
    ingest_note(path, config, client, db)
    rec = db.get_raw("raw/note.md")
    assert rec is not None
    assert rec.status == "ingested"


def test_ingest_note_skip_already_ingested(vault, config, db):
    path = _write_raw(vault, "dup.md", "# Dup\n\nContent.")
    client = _make_client(_analysis_json())
    ingest_note(path, config, client, db)
    # Second call without force — should skip
    result = ingest_note(path, config, client, db)
    assert result is None
    # Client called only once (for first ingest)
    assert client.generate.call_count == 1


def test_ingest_note_stores_prompt_version(vault, config, db):
    path = _write_raw(vault, "versioned.md", "# Note\n\nSome content here.")
    client = _make_client(_analysis_json())

    ingest_note(path, config, client, db)

    rec = db.get_raw("raw/versioned.md")
    assert rec is not None
    assert rec.prompt_version == _ingest_prompt_version(config)


def test_ingest_note_reingests_when_language_policy_changes(vault, config, db):
    path = _write_raw(vault, "policy.md", "# Note\n\nSome content here.")
    client = _make_client(_analysis_json())
    ingest_note(path, config, client, db)

    config.pipeline.language = "de"
    result = ingest_note(path, config, client, db)

    assert result is not None
    assert client.generate.call_count == 2
    rec = db.get_raw("raw/policy.md")
    assert rec is not None
    assert rec.prompt_version == _ingest_prompt_version(config)


def test_ingest_note_force_reingest(vault, config, db):
    path = _write_raw(vault, "forceme.md", "# Force\n\nContent.")
    client = _make_client(_analysis_json())
    ingest_note(path, config, client, db)
    result = ingest_note(path, config, client, db, force=True)
    assert result is not None
    assert client.generate.call_count == 2


def test_ingest_note_dedup_by_hash(vault, config, db):
    """Same content in two files → second skipped as duplicate."""
    content = "# Same\n\nIdentical body content here."
    p1 = _write_raw(vault, "first.md", content)
    p2 = _write_raw(vault, "second.md", content)
    client = _make_client(_analysis_json())
    ingest_note(p1, config, client, db)
    result = ingest_note(p2, config, client, db)
    assert result is None
    assert client.generate.call_count == 1


def test_ingest_note_stores_concepts(vault, config, db):
    path = _write_raw(vault, "ml.md", "# ML\n\nNeural networks and backprop.")
    client = _make_client(_analysis_json(concepts=["Neural Networks", "Backpropagation"]))
    ingest_note(path, config, client, db)
    names = db.list_all_concept_names()
    assert "Neural Networks" in names
    assert "Backpropagation" in names


def test_ingest_note_uses_suggested_topics_when_concepts_empty(vault, config, db):
    path = _write_raw(
        vault,
        "api.md",
        "# API testing\n\nChecklist for response validation with schemas, headers, status codes, "
        "boundary values, and data type checks.",
    )
    client = _make_client(_analysis_json(concepts=[], suggested_topics=["API Response Testing"]))

    ingest_note(path, config, client, db)

    assert db.list_all_concept_names() == ["API Response Testing"]


def test_ingest_note_does_not_use_suggested_topics_for_media_only_note(vault, config, db):
    path = _write_raw(vault, "image.md", "![[unknown_filename.png]]")
    client = _make_client(
        _analysis_json(concepts=[], quality="low", suggested_topics=["Image content unknown"])
    )

    ingest_note(path, config, client, db)

    assert db.list_all_concept_names() == []


def test_ingest_note_filters_low_quality_short_note_concept_without_evidence(vault, config, db):
    path = _write_raw(
        vault,
        "short.md",
        "A clipped note.",
    )
    client = _make_client(
        _analysis_json(concepts=["Planning Framework"], quality="low", suggested_topics=[])
    )

    ingest_note(path, config, client, db)

    assert db.list_all_concept_names() == []


def test_ingest_note_preserves_evidenced_named_reference_as_item(vault, config, db):
    path = _write_raw(
        vault,
        "reference.md",
        "This note mentions 設計の思想 as a named reference.",
    )
    client = _make_client(
        _analysis_json(
            concepts=[],
            quality="low",
            suggested_topics=[],
            named_references=["設計の思想"],
        )
    )

    ingest_note(path, config, client, db)

    assert db.list_all_concept_names() == []
    item = db.get_item("設計の思想")
    assert item is not None
    assert item.kind == "ambiguous"
    assert item.subtype == "named_reference"


def test_ingest_note_rejects_unevidenced_named_reference(vault, config, db):
    path = _write_raw(vault, "reference.md", "This note has no matching named reference.")
    client = _make_client(
        _analysis_json(
            concepts=[],
            quality="medium",
            suggested_topics=[],
            named_references=["Missing Reference"],
        )
    )

    ingest_note(path, config, client, db)

    assert db.get_item("Missing Reference") is None


def test_ingest_note_rejects_named_reference_matching_concept(vault, config, db):
    path = _write_raw(vault, "concept.md", "Example Concept is the main topic.")
    client = _make_client(
        _analysis_json(
            concepts=["Example Concept"],
            quality="high",
            suggested_topics=[],
            named_references=["Example Concept"],
        )
    )

    ingest_note(path, config, client, db)

    item = db.get_item("Example Concept")
    assert item is not None
    assert item.kind == "concept"


def test_ingest_note_failure_marks_db_status(vault, config, db):
    path = _write_raw(vault, "fail.md", "# Fail\n\nContent.")
    client = MagicMock()
    client.generate.side_effect = RuntimeError("Ollama timeout")
    result = ingest_note(path, config, client, db)
    assert result is None
    rec = db.get_raw("raw/fail.md")
    assert rec is not None
    assert rec.status == "failed"
    assert "timeout" in (rec.error or "").lower()


def test_ingest_note_creates_source_summary_page(vault, config, db):
    path = _write_raw(vault, "quantum.md", "# Quantum\n\nSuperposition and entanglement.")
    client = _make_client(_analysis_json(concepts=["Superposition", "Entanglement"]))
    ingest_note(path, config, client, db)
    sources = list((vault / "wiki" / "sources").glob("*.md"))
    assert sources, "Source summary page should be created"


def test_source_page_preserves_filename_casing(vault, config, db):
    path = _write_raw(vault, "Reference note.md", "# Reference\n\nContent.")
    client = _make_client(_analysis_json(concepts=["API Testing"]))
    ingest_note(path, config, client, db)

    from synto.vault import parse_note

    sources = list((vault / "wiki" / "sources").glob("*.md"))
    meta, _ = parse_note(sources[0])
    assert meta["title"] == "Reference note"


def test_source_page_uses_normalized_canonical_concepts(vault, config, db):
    db.upsert_concepts("raw/existing.md", ["Extreme Programming"])
    path = _write_raw(vault, "xp.md", "# XP\n\nExtreme Programming notes.")
    client = _make_client(_analysis_json(concepts=["Extreme Programming (XP)"]))
    ingest_note(path, config, client, db)

    sources = list((vault / "wiki" / "sources").glob("*.md"))
    source_text = sources[0].read_text()
    assert "[[Extreme Programming]]" in source_text
    assert "[[Extreme Programming (XP)]]" not in source_text


def test_source_page_yaml_with_colon_title(vault, config, db):
    """Source page title containing ':' must not break YAML parsing."""
    # Raw note uses quoted title (valid YAML) — the colon in title flows to source page
    path = _write_raw(vault, "guide.md", "---\ntitle: 'Python: A Guide'\n---\n\nContent here.")
    client = _make_client(_analysis_json(concepts=["Python"]))
    ingest_note(path, config, client, db)
    sources = list((vault / "wiki" / "sources").glob("*.md"))
    assert sources
    from synto.vault import parse_note

    meta, _ = parse_note(sources[0])
    assert meta["title"] == "Python: A Guide"


def test_source_page_aliases_are_list(vault, config, db):
    """Aliases must be a proper YAML list, not Python repr string."""
    path = _write_raw(vault, "ml.md", "# ML\n\nMachine Learning (ML) basics.")
    client = _make_client(_analysis_json(concepts=["Machine Learning"]))
    ingest_note(path, config, client, db)
    sources = list((vault / "wiki" / "sources").glob("*.md"))
    assert sources
    from synto.vault import parse_note

    meta, _ = parse_note(sources[0])
    assert isinstance(meta.get("aliases", []), list)


def test_source_page_roundtrip(vault, config, db):
    """Source page has all required fields with correct types."""
    path = _write_raw(vault, "q.md", "# Quantum\n\nContent.")
    client = _make_client(_analysis_json(concepts=["Qubits"]))
    ingest_note(path, config, client, db)
    sources = list((vault / "wiki" / "sources").glob("*.md"))
    assert sources
    from synto.vault import parse_note

    meta, body = parse_note(sources[0])
    assert "title" in meta
    assert meta["status"] == "published"
    assert meta["tags"] == ["source"]
    assert isinstance(meta["aliases"], list)
    assert "## Summary" in body
    assert "## Concepts" in body


def test_source_page_media_section(vault, config, db):
    """Raw note with images produces ## Media section in source page."""
    content = (
        "# Note\n\nSee ![[diagram.png]] for the architecture.\n"
        "Also ![Photo](http://example.com/photo.jpg) is relevant."
    )
    path = _write_raw(vault, "media-note.md", content)
    client = _make_client(_analysis_json(concepts=["Architecture"]))
    ingest_note(path, config, client, db)
    sources = list((vault / "wiki" / "sources").glob("*.md"))
    assert sources
    source_text = sources[0].read_text()
    assert "## Media" in source_text
    assert "diagram.png" in source_text
    assert "photo.jpg" in source_text


def test_source_page_no_media_section_when_none(vault, config, db):
    """Raw note without media produces no ## Media section."""
    path = _write_raw(vault, "text-only.md", "# Note\n\nJust text, no images.")
    client = _make_client(_analysis_json(concepts=["Text"]))
    ingest_note(path, config, client, db)
    sources = list((vault / "wiki" / "sources").glob("*.md"))
    assert sources
    source_text = sources[0].read_text()
    assert "## Media" not in source_text


def test_ingest_note_respects_max_concepts_per_source(vault, config, db):
    config2 = Config(vault=vault, pipeline={"max_concepts_per_source": 2})
    path = _write_raw(vault, "many.md", "# Many\n\nLots of concepts.")
    client = _make_client(_analysis_json(concepts=["A", "B", "C", "D", "E"]))
    ingest_note(path, config2, client, db)
    names = db.list_all_concept_names()
    # Only first 2 should be stored
    assert len(names) <= 2


def test_ingest_note_filters_before_medium_quality_cap(vault, config, db):
    config2 = Config(vault=vault, pipeline={"max_concepts_per_source": 8})
    path = _write_raw(
        vault,
        "agile.md",
        "# Agile\n\nThis note discusses Scrum, канбан, and Extreme Programming in one process.",
    )
    client = _make_client(
        _analysis_json(
            concepts=[
                "Noise One",
                "Noise Two",
                "Noise Three",
                "Noise Four",
                "Канбан",
                "Extreme Programming",
            ],
            quality="medium",
            suggested_topics=[],
        )
    )

    ingest_note(path, config2, client, db)

    names = db.list_all_concept_names()
    assert "Канбан" in names
    assert "Extreme Programming" in names


def test_ingest_all_updates_existing_topics_within_run(vault, config, db):
    _write_raw(vault, "a.md", "# A\n\nAlpha content.")
    _write_raw(vault, "b.md", "# B\n\nBeta content.")
    client = MagicMock()
    client.generate.side_effect = [
        _analysis_json(concepts=["Alpha Concept"]),
        _analysis_json(concepts=["Beta Concept"]),
    ]

    ingest_all(config, client, db)

    second_prompt = client.generate.call_args_list[1].kwargs["prompt"]
    assert "Alpha Concept" in second_prompt


def test_ingest_all_reports_progress(vault, config, db):
    _write_raw(vault, "a.md", "# A\n\nAlpha content.")
    _write_raw(vault, "b.md", "# B\n\nBeta content.")
    client = MagicMock()
    client.generate.side_effect = [
        _analysis_json(concepts=["Alpha Concept"]),
        _analysis_json(concepts=["Beta Concept"]),
    ]
    progress_events: list[tuple[int, int, str]] = []

    def on_progress(done: int, total: int, current_note_path: str) -> None:
        progress_events.append((done, total, current_note_path))

    ingest_all(config, client, db, on_progress=on_progress)

    assert progress_events == [
        (1, 2, "raw/a.md"),
        (2, 2, "raw/b.md"),
    ]


def test_ingest_all_seeds_existing_topics_from_index_when_db_empty(vault, config, db):
    _write_raw(vault, "a.md", "# A\n\nAlpha content.")
    (vault / ".synto" / "INDEX.json").write_text(
        json.dumps(
            {
                "articles": [
                    {
                        "name": "Template Catalog",
                        "aliases": ["Каталог шаблонов", "Справочник шаблонов"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    client = _make_client(_analysis_json(concepts=["Каталог шаблонов"]))

    ingest_all(config, client, db)

    prompt = client.generate.call_args.kwargs["prompt"]
    assert "Template Catalog" in prompt
    assert "aliases: каталог шаблонов" in prompt.lower()


def test_ingest_note_reuses_existing_canonical_via_trusted_alias_from_llm_alias(vault, config, db):
    db.upsert_concepts("raw/existing.md", ["Template Catalog"])
    db.upsert_aliases("Template Catalog", ["Каталог шаблонов"])
    path = _write_raw(
        vault,
        "catalog.md",
        "# Note\n\nКаталог шаблонов помогает организовать шаблоны проекта.",
    )
    analysis = json.dumps(
        {
            "summary": "A multilingual note.",
            "concepts": [{"name": "Vorlagenkatalog", "aliases": ["Каталог шаблонов"]}],
            "suggested_topics": [],
            "named_references": [],
            "quality": "medium",
        }
    )
    client = _make_client(analysis)

    ingest_note(path, config, client, db)

    assert db.get_concepts_for_sources(["raw/catalog.md"]) == ["Template Catalog"]
    assert "Vorlagenkatalog" not in db.list_all_concept_names()


def test_ingest_all_reuses_seeded_aliases_across_full_run(vault, config, db):
    _write_raw(vault, "a.md", "# A\n\nAlpha content.")
    _write_raw(vault, "b.md", "# B\n\nГибкая разработка и Scrum.")
    (vault / ".synto" / "INDEX.json").write_text(
        json.dumps(
            {
                "articles": [
                    {
                        "name": "Agile Development",
                        "aliases": ["гибкая разработка"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    client = MagicMock()
    client.generate.side_effect = [
        _analysis_json(concepts=["Alpha Concept"]),
        _analysis_json(concepts=["гибкая разработка"]),
    ]

    ingest_all(config, client, db)

    assert "Agile Development" in db.get_concepts_for_sources(["raw/b.md"])


def test_ingest_note_does_not_persist_rewrite_only_alias(vault, config, db):
    path = _write_raw(vault, "tdd.md", "# TDD\n\nРазработка через тестирование помогает команде.")
    client = _make_client(_analysis_json(concepts=["Разработка через тестирование"]))

    ingest_note(
        path,
        config,
        client,
        db,
        existing_topics=["Test Driven Development", "Разработка через тестирование"],
        seed_concepts=["Test Driven Development"],
        seed_alias_map={"Разработка через тестирование": "Test Driven Development"},
    )

    assert db.get_concepts_for_sources(["raw/tdd.md"]) == ["Test Driven Development"]
    assert db.get_aliases("Test Driven Development") == []


def test_ingest_all_does_not_rewrite_seed_alias_when_it_is_other_canonical(vault, config, db):
    _write_raw(vault, "a.md", "# A\n\nКаталог шаблонов и справочник шаблонов.")
    (vault / ".synto" / "INDEX.json").write_text(
        json.dumps(
            {
                "articles": [
                    {"name": "Template Catalog", "aliases": ["Каталог шаблонов"]},
                    {"name": "Каталог шаблонов", "aliases": []},
                ]
            }
        ),
        encoding="utf-8",
    )
    client = _make_client(_analysis_json(concepts=["Каталог шаблонов"]))

    ingest_all(config, client, db)

    assert db.get_concepts_for_sources(["raw/a.md"]) == ["Каталог шаблонов"]
    aliases = "".join(db.get_aliases("Каталог шаблонов"))
    assert "__olw_rewrite_alias__:" not in aliases
    assert "__synto_rewrite_alias__:" not in aliases


def test_ingest_all_reuses_matching_source_concept_seed(vault, config, db):
    path = _write_raw(
        vault,
        "api.md",
        "# API\n\nChecklist for status codes, schemas, headers, and boundary values.",
    )
    content_hash = _content_hash(path.read_text(encoding="utf-8"))
    (vault / ".synto" / "INDEX.json").write_text(
        json.dumps(
            {
                "articles": [
                    {"name": "Response Validation", "aliases": []},
                    {"name": "Testing Checklist", "aliases": []},
                ],
                "source_concepts": [
                    {
                        "source_path": "raw/api.md",
                        "content_hash": content_hash,
                        "concepts": ["Response Validation", "Testing Checklist"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    client = _make_client(_analysis_json(concepts=["Проверка статуса ответа API"]))

    ingest_all(config, client, db)

    assert db.get_concepts_for_sources(["raw/api.md"]) == [
        "Response Validation",
        "Testing Checklist",
    ]


def test_ingest_all_ignores_source_concept_seed_when_hash_changed(vault, config, db):
    _write_raw(
        vault,
        "api.md",
        "# API\n\nUpdated checklist for status codes, schemas, headers, and boundary values.",
    )
    (vault / ".synto" / "INDEX.json").write_text(
        json.dumps(
            {
                "articles": [
                    {"name": "Response Validation", "aliases": []},
                    {"name": "Testing Checklist", "aliases": []},
                ],
                "source_concepts": [
                    {
                        "source_path": "raw/api.md",
                        "content_hash": "old-hash",
                        "concepts": ["Response Validation", "Testing Checklist"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    client = _make_client(_analysis_json(concepts=["API Status Checks"]))

    ingest_all(config, client, db)

    assert db.get_concepts_for_sources(["raw/api.md"]) == ["API Status Checks"]


def test_cli_ingest_all_passes_progress_callback(vault, db, monkeypatch):
    _write_raw(vault, "a.md", "# A\n\nAlpha content.")
    captured: dict[str, object] = {}

    def fake_ingest_all(*, config, client, db, force, on_progress):
        captured["on_progress"] = on_progress
        on_progress(1, 1, "raw/a.md")
        return [(config.raw_dir / "a.md", None)]

    monkeypatch.setattr("synto.cli._load_deps", lambda cfg: (object(), db))
    monkeypatch.setattr("synto.pipeline.ingest.ingest_all", fake_ingest_all)
    monkeypatch.setattr(
        "synto.cli.generate_index",
        lambda config, db: None,
        raising=False,
    )
    monkeypatch.setattr(
        "synto.cli.append_log",
        lambda config, message: None,
        raising=False,
    )

    result = CliRunner().invoke(cli, ["ingest", "--vault", str(vault), "--all"])

    assert result.exit_code == 0
    assert callable(captured["on_progress"])


def test_ingest_note_reuses_matching_source_concept_seed_when_db_empty(vault, config, db):
    path = _write_raw(
        vault,
        "api.md",
        "# API\n\nChecklist for status codes, schemas, headers, and boundary values.",
    )
    content_hash = _content_hash(path.read_text(encoding="utf-8"))
    (vault / ".synto" / "INDEX.json").write_text(
        json.dumps(
            {
                "articles": [
                    {"name": "Response Validation", "aliases": []},
                    {"name": "Testing Checklist", "aliases": []},
                ],
                "source_concepts": [
                    {
                        "source_path": "raw/api.md",
                        "content_hash": content_hash,
                        "concepts": ["Response Validation", "Testing Checklist"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    client = _make_client(_analysis_json(concepts=["Проверка статуса ответа API"]))

    ingest_note(path, config, client, db)

    assert db.get_concepts_for_sources(["raw/api.md"]) == [
        "Response Validation",
        "Testing Checklist",
    ]


# ── _merge_chunk_results ──────────────────────────────────────────────────────


def _make_result(concepts, summary="Summary.", quality="high", topics=None, named_references=None):
    from synto.models import Concept

    return AnalysisResult(
        summary=summary,
        concepts=[Concept(name=c, aliases=[]) for c in concepts],
        suggested_topics=topics or ["Topic"],
        named_references=named_references or [],
        quality=quality,
    )


def test_merge_single_chunk_returns_unchanged():
    r = _make_result(["A", "B"])
    assert _merge_chunk_results([r]) is r


def test_merge_unions_concepts():
    r1 = _make_result(["A", "B"])
    r2 = _make_result(["B", "C"])
    merged = _merge_chunk_results([r1, r2])
    assert [c.name for c in merged.concepts] == ["A", "B", "C"]  # B deduped


def test_merge_concept_dedup_case_insensitive():
    r1 = _make_result(["Machine Learning"])
    r2 = _make_result(["machine learning", "Deep Learning"])
    merged = _merge_chunk_results([r1, r2])
    names_lower = [c.name.lower() for c in merged.concepts]
    assert names_lower.count("machine learning") == 1
    assert "deep learning" in names_lower


def test_merge_summary_from_first_chunk():
    r1 = _make_result(["A"], summary="First summary.")
    r2 = _make_result(["B"], summary="Second summary.")
    merged = _merge_chunk_results([r1, r2])
    assert merged.summary == "First summary."


def test_merge_quality_is_most_common_not_minimum():
    """Quality is the most common rating, not the conservative min.

    Why it matters: with segment-aligned chunking a single thin/peripheral section
    (title page, references) rates low and, under the old min, dragged a mostly-high
    paper down to low — which the quality cap then used to halve the concept count.
    A paper whose substantive majority is high should rate high.
    """
    # Substantive majority high, one peripheral low section → high (not low).
    highs = [_make_result([c], quality="high") for c in ["A", "B", "C"]]
    low = _make_result(["D"], quality="low")
    assert _merge_chunk_results([*highs, low]).quality == "high"

    # Genuine majority low → low.
    lows = [_make_result([c], quality="low") for c in ["A", "B"]]
    assert _merge_chunk_results([*lows, _make_result(["C"], quality="high")]).quality == "low"

    # Tie → broken toward the higher rank.
    tie = [_make_result(["A"], quality="high"), _make_result(["B"], quality="medium")]
    assert _merge_chunk_results(tie).quality == "high"


def test_merge_unions_topics():
    r1 = _make_result(["A"], topics=["Topic A"])
    r2 = _make_result(["B"], topics=["Topic B", "Topic A"])
    merged = _merge_chunk_results([r1, r2])
    assert len(merged.suggested_topics) == 2


def test_merge_unions_named_references():
    r1 = _make_result(["A"], named_references=["Ref A", "Ref B"])
    r2 = _make_result(["B"], named_references=["ref a", "Ref C"])
    merged = _merge_chunk_results([r1, r2])

    assert merged.named_references == ["Ref A", "Ref B", "Ref C"]


# ── _analyze_body ──────────────────────────────────────────────────────────────


def test_analyze_body_single_call_for_short_note(vault, config, db):
    """Body <= fast_ctx // 2 → exactly one generate call."""
    client = _make_client(_analysis_json())
    body = "Short note content."
    _analyze_body(body, [], "test.md", client, config)
    assert client.generate.call_count == 1


def test_analyze_body_multi_call_for_long_note(vault, config, db):
    """Body > fast_ctx // 2 → one call per chunk."""
    config2 = Config(vault=vault, ollama={"fast_ctx": 100})  # tiny ctx for test
    chunk_size = 100 // 2  # = 50 chars per chunk
    body = "x" * 200  # 200 chars → 4 chunks
    client = _make_client(_analysis_json())
    result = _analyze_body(body, [], "long.md", client, config2)
    expected_chunks = -(-200 // chunk_size)  # ceiling division
    assert client.generate.call_count == expected_chunks
    assert isinstance(result, AnalysisResult)


def test_analyze_body_chunk_labels_in_prompt(vault, config, db):
    """Multi-chunk prompts include [part N/M] labels."""
    config2 = Config(vault=vault, ollama={"fast_ctx": 100})
    body = "x" * 200
    client = _make_client(_analysis_json())
    _analyze_body(body, [], "long.md", client, config2)
    prompts = [call.kwargs["prompt"] for call in client.generate.call_args_list]
    assert any("[part 1/" in p for p in prompts)
    assert any("[part 2/" in p for p in prompts)


def test_analyze_body_passes_language_from_config(vault, config, db):
    """config.pipeline.language is forwarded to _build_analysis_prompt."""
    config.pipeline.language = "ru"
    client = _make_client(_analysis_json())
    _analyze_body("Short note.", [], "test.md", client, config)
    prompt = client.generate.call_args.kwargs["prompt"]
    assert "ru (ISO 639-1)" in prompt
    assert "same language used by the note" not in prompt


def test_analyze_body_passes_language_from_config_for_all_chunks(vault, config, db):
    config.pipeline.language = "ru"
    config.ollama.fast_ctx = 100
    body = "x" * 200
    client = _make_client(_analysis_json())

    _analyze_body(body, [], "long.md", client, config)

    prompts = [call.kwargs["prompt"] for call in client.generate.call_args_list]
    assert len(prompts) == 4
    assert all("ru (ISO 639-1)" in prompt for prompt in prompts)
    assert all("exact canonical name shown" in prompt for prompt in prompts)


def test_analyze_body_no_language_config_uses_note_language_rule(vault, config, db):
    """When config.pipeline.language is None, prompt uses the surface-form heuristic."""
    assert config.pipeline.language is None
    client = _make_client(_analysis_json())
    _analyze_body("Short note.", [], "test.md", client, config)
    prompt = client.generate.call_args.kwargs["prompt"]
    assert "form they appear in the note text" in prompt


def test_analyze_body_parallel_mode(vault):
    """ingest_parallel=True still produces one call per chunk, results merged."""
    config2 = Config(
        vault=vault,
        ollama={"fast_ctx": 100},
        pipeline={"ingest_parallel": True},
    )
    body = "x" * 200
    client = _make_client(_analysis_json(concepts=["A"]))
    result = _analyze_body(body, [], "long.md", client, config2)
    assert client.generate.call_count == -(-200 // 50)  # same chunk count
    assert isinstance(result, AnalysisResult)


def test_analyze_body_with_checkpoints_resumes_after_failure(vault, config, db):
    config2 = Config(vault=vault, ollama={"fast_ctx": 100})
    path = _write_raw(vault, "long.md", "x" * 200)
    body = path.read_text()
    content_hash = _content_hash(body)
    checkpoint_hash = _checkpoint_hash(content_hash, config2, [])

    client = MagicMock()
    calls = {"count": 0}

    def side_effect(**kwargs):
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("chunk timeout")
        return _analysis_json(concepts=[f"Chunk {calls['count']}"])

    client.generate.side_effect = side_effect

    with pytest.raises(RuntimeError, match="chunk timeout"):
        _analyze_body_with_checkpoints(body, [], path, content_hash, client, config2, db)

    rows = db.list_ingest_chunks("raw/long.md", checkpoint_hash, 4, 50)
    assert [row["chunk_index"] for row in rows] == [0, 1]

    client2 = _make_client(_analysis_json(concepts=["Recovered"]))
    result = _analyze_body_with_checkpoints(body, [], path, content_hash, client2, config2, db)
    assert isinstance(result, AnalysisResult)
    assert client2.generate.call_count == 2
    assert db.list_ingest_chunks("raw/long.md", checkpoint_hash, 4, 50) == []


def test_analyze_body_with_checkpoints_loads_existing_partial_resume(vault, config, db):
    config2 = Config(vault=vault, ollama={"fast_ctx": 100})
    path = _write_raw(vault, "resume.md", "x" * 200)
    body = path.read_text()
    content_hash = _content_hash(body)
    checkpoint_hash = _checkpoint_hash(content_hash, config2, [])

    db.upsert_ingest_chunk(
        "raw/resume.md",
        checkpoint_hash,
        0,
        4,
        50,
        _analysis_json(concepts=["Stored Alpha"]),
    )
    db.upsert_ingest_chunk(
        "raw/resume.md",
        checkpoint_hash,
        1,
        4,
        50,
        _analysis_json(concepts=["Stored Beta"]),
    )

    client = _make_client(_analysis_json(concepts=["Fresh Gamma"]))
    result = _analyze_body_with_checkpoints(body, [], path, content_hash, client, config2, db)

    assert client.generate.call_count == 2
    assert [concept.name for concept in result.concepts] == [
        "Stored Alpha",
        "Stored Beta",
        "Fresh Gamma",
    ]
    assert db.list_ingest_chunks("raw/resume.md", checkpoint_hash, 4, 50) == []


def test_analyze_body_with_checkpoints_purges_stale_chunks_for_short_note(vault, config, db):
    config2 = Config(vault=vault, ollama={"fast_ctx": 100})
    path = _write_raw(vault, "shortened.md", "short body")
    old_hash = _content_hash("x" * 200)
    new_hash = _content_hash(path.read_text())
    # Old row uses old_hash directly (pre-compound-hash, simulating stale data)
    db.upsert_ingest_chunk(
        "raw/shortened.md",
        old_hash,
        0,
        4,
        50,
        _analysis_json(concepts=["Old Chunk"]),
    )

    client = _make_client(_analysis_json(concepts=["Fresh Short"], quality="high"))
    result = _analyze_body_with_checkpoints(
        path.read_text(), [], path, new_hash, client, config2, db
    )

    assert [concept.name for concept in result.concepts] == ["Fresh Short"]
    assert db.list_ingest_chunks("raw/shortened.md", old_hash, 4, 50) == []


def test_analyze_body_with_checkpoints_language_change_invalidates_cache(vault, config, db):
    """Changing config.pipeline.language must not reuse checkpoints from prior language."""
    config_de = Config(vault=vault, ollama={"fast_ctx": 100}, pipeline={"language": "de"})

    path = _write_raw(vault, "multilang.md", "x" * 200)
    body = path.read_text()
    content_hash = _content_hash(body)

    # Partially ingest under language=None (2 of 4 chunks saved to checkpoints)
    checkpoint_hash_none = _checkpoint_hash(content_hash, config, [])
    db.upsert_ingest_chunk(
        "raw/multilang.md",
        checkpoint_hash_none,
        0,
        4,
        50,
        _analysis_json(concepts=["OldConcept"]),
    )
    db.upsert_ingest_chunk(
        "raw/multilang.md",
        checkpoint_hash_none,
        1,
        4,
        50,
        _analysis_json(concepts=["OldConcept2"]),
    )

    # Now re-run with language="de" — all 4 chunks must be re-analyzed, none reused
    client_de = _make_client(_analysis_json(concepts=["NeuKonzept"]))
    _analyze_body_with_checkpoints(body, [], path, content_hash, client_de, config_de, db)

    assert client_de.generate.call_count == 4  # all chunks re-analyzed, none reused
    # Stale language=None rows are purged by purge_ingest_chunks(keep_hash=checkpoint_hash_de)
    assert db.list_ingest_chunks("raw/multilang.md", checkpoint_hash_none, 4, 50) == []


def test_analyze_body_with_checkpoints_context_change_invalidates_cache(vault, config, db):
    config2 = Config(vault=vault, ollama={"fast_ctx": 100})
    path = _write_raw(vault, "context.md", "x" * 200)
    body = path.read_text()
    content_hash = _content_hash(body)
    old_contexts = _canonical_prompt_contexts(["Alpha"], {"alpha alias": "Alpha"})
    old_checkpoint_hash = _checkpoint_hash(content_hash, config2, old_contexts)
    db.upsert_ingest_chunk(
        "raw/context.md",
        old_checkpoint_hash,
        0,
        4,
        50,
        _analysis_json(concepts=["Old Chunk"]),
    )

    client = _make_client(_analysis_json(concepts=["Fresh Chunk"]))
    new_contexts = _canonical_prompt_contexts(["Beta"], {"beta alias": "Beta"})
    _analyze_body_with_checkpoints(
        body,
        ["Beta"],
        path,
        content_hash,
        client,
        config2,
        db,
        prompt_contexts=new_contexts,
    )

    assert client.generate.call_count == 4
    assert db.list_ingest_chunks("raw/context.md", old_checkpoint_hash, 4, 50) == []


def test_analyze_body_with_checkpoints_source_type_change_invalidates_cache(vault, config, db):
    config2 = Config(vault=vault, ollama={"fast_ctx": 100})
    path = _write_raw(vault, "source-type.md", "x" * 200)
    body = path.read_text()
    content_hash = _content_hash(body)
    paper_hash = _checkpoint_hash(content_hash, config2, [], source_type="paper")
    db.upsert_ingest_chunk(
        "raw/source-type.md",
        paper_hash,
        0,
        4,
        50,
        _analysis_json(concepts=["Paper Chunk"]),
    )

    client = _make_client(_analysis_json(concepts=["Notes Chunk"]))
    _analyze_body_with_checkpoints(
        body,
        [],
        path,
        content_hash,
        client,
        config2,
        db,
        source_type="notes",
    )

    assert client.generate.call_count == 4
    assert db.list_ingest_chunks("raw/source-type.md", paper_hash, 4, 50) == []


def test_analyze_body_with_checkpoints_ignores_previous_schema_rows(vault, config, db):
    config2 = Config(vault=vault, ollama={"fast_ctx": 100})
    path = _write_raw(vault, "schema.md", "x" * 200)
    body = path.read_text()
    content_hash = _content_hash(body)
    current_checkpoint_hash = _checkpoint_hash(content_hash, config2, [])
    db.upsert_ingest_chunk(
        "raw/schema.md",
        current_checkpoint_hash,
        0,
        4,
        50,
        _analysis_json(concepts=["Old Schema"]),
        checkpoint_schema=1,
    )

    client = _make_client(_analysis_json(concepts=["Fresh Schema"]))
    _analyze_body_with_checkpoints(body, [], path, content_hash, client, config2, db)

    assert client.generate.call_count == 4
    assert (
        db.list_ingest_chunks("raw/schema.md", current_checkpoint_hash, 4, 50, checkpoint_schema=1)
        != []
    )


def test_ingest_note_replaces_stale_source_concepts(vault, config, db):
    path = _write_raw(vault, "note.md", "Initial body")
    first = _make_client(_analysis_json(concepts=["Alpha", "Beta"]))
    ingest_note(path, config, first, db)
    assert set(db.get_concepts_for_sources(["raw/note.md"])) == {"Alpha", "Beta"}

    path.write_text("Updated body", encoding="utf-8")
    second = _make_client(_analysis_json(concepts=["Beta", "Gamma"]))
    ingest_note(path, config, second, db, force=True)

    assert set(db.get_concepts_for_sources(["raw/note.md"])) == {"Beta", "Gamma"}
    assert db.get_compile_state("Alpha", "raw/note.md") is None


def test_ingest_note_reanalyzes_changed_compiled_note_without_force(vault, config, db):
    path = _write_raw(vault, "compiled.md", "Initial body")
    first = _make_client(_analysis_json(concepts=["Alpha"]))
    ingest_note(path, config, first, db)
    db.mark_concept_compile_state("Alpha", ["raw/compiled.md"], "compiled")
    assert db.get_raw("raw/compiled.md").status == "compiled"

    path.write_text("Updated body", encoding="utf-8")
    second = _make_client(_analysis_json(concepts=["Beta"]))
    ingest_note(path, config, second, db)

    assert second.generate.call_count == 1
    assert set(db.get_concepts_for_sources(["raw/compiled.md"])) == {"Beta"}
    assert db.get_compile_state("Alpha", "raw/compiled.md") is None
    assert db.get_raw("raw/compiled.md").content_hash == _content_hash("Updated body")


# ── Language tests ─────────────────────────────────────────────────────────────


def test_system_prompt_contains_language_detection_instruction():
    assert "ISO 639-1" in _SYSTEM
    assert "language" in _SYSTEM


def test_analysis_result_stores_language_in_db(vault, config, db):
    path = _write_raw(vault, "french_note.md", "# Bonjour\n\nCeci est une note en français.")
    analysis = json.dumps(
        {
            "summary": "A French note.",
            "concepts": [{"name": "Bonjour", "aliases": []}],
            "suggested_topics": ["Salutations"],
            "quality": "high",
            "language": "fr",
        }
    )
    client = _make_client(analysis)
    ingest_note(path, config, client, db)
    assert db.get_note_language("raw/french_note.md") == "fr"


def test_analysis_result_language_none_stored(vault, config, db):
    path = _write_raw(vault, "unknown.md", "# Mixed content\n\nSome text.")
    analysis = json.dumps(
        {
            "summary": "Unknown language note.",
            "concepts": [{"name": "Mixed", "aliases": []}],
            "suggested_topics": [],
            "quality": "medium",
            "language": None,
        }
    )
    client = _make_client(analysis)
    ingest_note(path, config, client, db)
    assert db.get_note_language("raw/unknown.md") is None


def test_merge_chunk_results_picks_first_detected_language():
    make = lambda lang: AnalysisResult(  # noqa: E731
        summary="s", concepts=[], suggested_topics=[], quality="high", language=lang
    )
    merged = _merge_chunk_results([make(None), make("de"), make("fr")])
    assert merged.language == "de"


# ── AnalysisResult.coerce_concepts ────────────────────────────────────────────


def test_analysis_result_coerces_string_concepts():
    r = AnalysisResult(
        summary="s",
        concepts=["Foo", "Bar"],
        suggested_topics=[],
        quality="high",
        language=None,
    )
    assert r.concepts == [Concept(name="Foo", aliases=[]), Concept(name="Bar", aliases=[])]


def test_analysis_result_accepts_null_summary():
    r = AnalysisResult(
        summary=None,
        concepts=["Example Concept"],
        named_references=["Example Reference"],
        suggested_topics=[],
        quality="low",
        language=None,
    )

    assert r.summary == "Source references: Example Concept, Example Reference."


def test_analysis_result_accepts_mixed_concepts():
    r = AnalysisResult(
        summary="s",
        concepts=[{"name": "A", "aliases": ["a"]}, "B"],
        suggested_topics=[],
        quality="high",
        language=None,
    )
    assert r.concepts[0].aliases == ["a"]
    assert r.concepts[1].name == "B"
    assert r.concepts[1].aliases == []


# ── _dedup_by_shared_alias ────────────────────────────────────────────────────


def test_dedup_by_shared_alias_drops_when_name_matches_earlier_alias():
    """Candidate whose name appears as an alias of an earlier candidate is dropped."""
    a = Concept(name="Extreme Programming", aliases=["XP", "Экстремальное программирование"])
    b = Concept(name="Экстремальное программирование", aliases=["Extreme Programming"])
    result = _dedup_by_shared_alias([a, b])
    assert [c.name for c in result] == ["Extreme Programming"]


def test_dedup_by_shared_alias_keeps_distinct_concepts():
    a = Concept(name="Scrum", aliases=["Scrum framework"])
    b = Concept(name="Kanban", aliases=["Kanban board"])
    result = _dedup_by_shared_alias([a, b])
    assert [c.name for c in result] == ["Scrum", "Kanban"]


def test_dedup_by_shared_alias_empty_list():
    assert _dedup_by_shared_alias([]) == []


# ── suggested_topics cap ──────────────────────────────────────────────────────


def test_suggested_topic_candidates_medium_quality_requires_evidence():
    """Medium-quality suggested topics without body evidence are dropped."""
    result = AnalysisResult(
        summary="API testing guide.",
        concepts=[],
        suggested_topics=["Проверка статуса ответа API", "API Testing"],
        quality="medium",
    )
    body = "This guide covers API testing and how to verify response status codes." * 3
    candidates = _suggested_topic_candidates(result, body, "Api testing example.md")
    names = [c.name for c in candidates]
    assert "API Testing" in names
    assert "Проверка статуса ответа API" not in names


# ---------------------------------------------------------------------------
# write_source_content_md
# ---------------------------------------------------------------------------


def _seg(text: str, locator: str | None = None):
    from types import SimpleNamespace

    return SimpleNamespace(text=text, structural_locator=locator)


def test_write_source_content_md_with_locators(tmp_path):
    segs = [_seg("Intro text.", "Introduction"), _seg("Background text.", "Background")]
    path = write_source_content_md("src-001", "paper", "My Paper", segs, tmp_path)
    assert path == tmp_path / "raw" / "src-001.md"
    content = path.read_text()
    assert "source_type: paper" in content
    assert "## Introduction" in content
    assert "Intro text." in content
    assert "## Background" in content


def test_write_source_content_md_no_locator(tmp_path):
    segs = [_seg("Plain text content.")]
    path = write_source_content_md("src-002", "notes", None, segs, tmp_path)
    content = path.read_text()
    assert "Plain text content." in content
    assert "##" not in content
    assert "source_type: notes" in content


def test_write_source_content_md_preserves_image_refs(tmp_path):
    from types import SimpleNamespace

    segs = [
        SimpleNamespace(
            text="Segment text.",
            structural_locator="Images",
            image_refs=["assets/src-003/img-0-0.png"],
        )
    ]
    path = write_source_content_md("src-003", "paper", "With Images", segs, tmp_path)
    content = path.read_text()
    assert "### Media" in content
    assert "![[assets/src-003/img-0-0.png]]" in content


def test_media_section_stripped_before_llm_analysis(vault, config, db):
    """### Media + ![[...]] embeds are removed from body before LLM sees it.

    The raw/ file retains the embeds for human readers in Obsidian;
    only the in-memory body passed to the LLM is cleaned.
    """
    body = (
        "## section:intro\n\nSome meaningful text.\n\n"
        "### Media\n- ![[assets/src-001/img-0-0.png]]\n- ![[assets/src-001/img-0-1.png]]\n\n"
        "## section:results\n\nMore meaningful text.\n\n"
        "### Media\n- ![[assets/src-001/img-1-0.png]]\n"
    )
    path = _write_raw(vault, "pdf-source.md", body)
    client = _make_client(_analysis_json())
    ingest_note(path, config, client, db)

    prompt_sent = client.generate.call_args.kwargs["prompt"]
    assert "### Media" not in prompt_sent
    assert "![[assets" not in prompt_sent
    assert "Some meaningful text." in prompt_sent
    assert "More meaningful text." in prompt_sent
    # Raw file on disk is unchanged — embeds are still there for Obsidian
    assert "![[assets/src-001/img-0-0.png]]" in path.read_text()
