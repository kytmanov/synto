"""Tests for compile pipeline — mocked LLM, no Ollama required."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from conftest import as_router

from synto.config import Config
from synto.models import RawNoteRecord
from synto.openai_compat_client import LLMBadRequestError
from synto.pipeline.compile import (
    _resolve_language,
    _write_concept_prompt,
    _write_prompt_legacy,
    approve_drafts,
    compile_concepts,
    compile_notes,
    reject_draft,
)
from synto.state import StateDB


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


def _make_client(plan_json: str, article_json: str, config=None):
    """Mock client: first call returns plan, subsequent return article."""
    client = MagicMock()
    call_count = [0]

    def generate_side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return plan_json
        return article_json

    client.generate.side_effect = generate_side_effect
    return as_router(client, config)


def test_compile_creates_draft(vault, config, db, fixtures_dir):
    # Setup: ingested raw note
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("---\ntitle: Note\n---\n\nQuantum entanglement content.")
    db.upsert_raw(
        RawNoteRecord(
            path="raw/note.md",
            content_hash="abc",
            status="ingested",
        )
    )

    plan_json = (fixtures_dir / "compile_plan_valid.json").read_text()
    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_client(plan_json, article_json)

    drafts, failed = compile_notes(config=config, router=client, db=db)

    assert len(drafts) == 1
    assert len(failed) == 0
    assert drafts[0].exists()
    assert drafts[0].parent == config.drafts_dir


def test_draft_has_correct_frontmatter(vault, config, db, fixtures_dir):
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("# Note\n\nContent here.")
    db.upsert_raw(RawNoteRecord(path="raw/note.md", content_hash="h", status="ingested"))

    plan_json = (fixtures_dir / "compile_plan_valid.json").read_text()
    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_client(plan_json, article_json)

    drafts, _ = compile_notes(config=config, router=client, db=db)
    assert drafts

    from synto.vault import parse_note

    meta, body = parse_note(drafts[0])
    assert meta["status"] == "draft"
    assert "title" in meta
    assert "tags" in meta
    assert 0.0 <= meta["confidence"] <= 1.0


def test_dry_run_writes_nothing(vault, config, db, fixtures_dir):
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(RawNoteRecord(path="raw/note.md", content_hash="h", status="ingested"))

    plan_json = (fixtures_dir / "compile_plan_valid.json").read_text()
    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_client(plan_json, article_json)

    drafts, _ = compile_notes(config=config, router=client, db=db, dry_run=True)
    assert drafts == []
    assert list(config.drafts_dir.glob("*.md")) == []


def test_legacy_compile_uses_article_max_tokens_from_config(vault, config, db, fixtures_dir):
    """Regression for issue #48: compile_notes (legacy --legacy path) was hardcoded
    to 4096 and ignored config.pipeline.article_max_tokens. After the fix, num_predict
    must reflect the user's configured cap."""
    config.pipeline.article_max_tokens = 12000
    config.provider = config.effective_provider.model_copy(update={"heavy_ctx": 32768})

    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("# Note\n\nShort content.")
    db.upsert_raw(RawNoteRecord(path="raw/note.md", content_hash="h", status="ingested"))

    plan_json = (fixtures_dir / "compile_plan_valid.json").read_text()
    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_client(plan_json, article_json, config)

    compile_notes(config=config, router=client, db=db)

    # Find the article-write call (second generate call) and inspect num_predict
    write_calls = [c for c in client.generate.call_args_list if c.kwargs.get("num_predict")]
    assert write_calls, "expected at least one generate call with num_predict"
    article_call = write_calls[-1]
    assert article_call.kwargs["num_predict"] == 12000


def test_legacy_compile_ignores_concept_draft_soft_cap(vault, config, db, fixtures_dir):
    """concept_draft_soft_cap applies only to concept-driven compile, not --legacy."""
    config.pipeline.article_max_tokens = 5000
    config.pipeline.concept_draft_soft_cap = 1200
    config.provider = config.effective_provider.model_copy(update={"heavy_ctx": 32768})

    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("# Note\n\nShort content.")
    db.upsert_raw(RawNoteRecord(path="raw/note.md", content_hash="h", status="ingested"))

    plan_json = (fixtures_dir / "compile_plan_valid.json").read_text()
    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_client(plan_json, article_json, config)

    compile_notes(config=config, router=client, db=db)

    write_calls = [c for c in client.generate.call_args_list if c.kwargs.get("num_predict")]
    assert write_calls, "expected at least one generate call with num_predict"
    article_call = write_calls[-1]
    assert article_call.kwargs["num_predict"] == 5000


def test_approve_moves_draft_to_wiki(vault, config, db):
    from synto.models import WikiArticleRecord
    from synto.vault import write_note

    draft_path = config.drafts_dir / "article.md"
    write_note(draft_path, {"title": "Article", "status": "draft", "tags": []}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Article",
            sources=[],
            content_hash="h",
            status="draft",
        )
    )

    published = approve_drafts(config, db, [draft_path])
    assert len(published) == 1
    assert published[0].exists()
    assert published[0].parent == config.wiki_dir
    assert not draft_path.exists()

    # State updated
    record = db.get_article(str(published[0].relative_to(vault)))
    assert record is not None
    assert record.is_draft is False


def test_reject_deletes_draft(vault, config, db):
    from synto.models import WikiArticleRecord
    from synto.vault import write_note

    draft_path = config.drafts_dir / "bad.md"
    write_note(draft_path, {"title": "Bad", "status": "draft"}, "Wrong content.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Bad",
            sources=[],
            content_hash="h",
            status="draft",
        )
    )

    reject_draft(draft_path, config, db, feedback="Hallucinated content")
    assert not draft_path.exists()


# ── Concept-driven compile tests ───────────────────────────────────────────────


def _make_concept_client(article_json: str):
    """Mock client that returns a single article for any generate() call."""
    client = MagicMock()
    client.generate.return_value = article_json
    return as_router(client)


def test_compile_concepts_creates_draft(vault, config, db, fixtures_dir):
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("---\ntitle: Note\n---\n\nQuantum entanglement content.")
    db.upsert_raw(
        __import__("synto.models", fromlist=["RawNoteRecord"]).RawNoteRecord(
            path="raw/note.md", content_hash="abc", status="ingested"
        )
    )
    db.upsert_concepts("raw/note.md", ["Quantum Entanglement"])

    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_concept_client(article_json)

    drafts, failed, _ = compile_concepts(config=config, router=client, db=db)

    assert len(drafts) == 1
    assert len(failed) == 0
    assert drafts[0].exists()
    assert drafts[0].parent == config.drafts_dir


def test_compile_concepts_skips_when_no_concepts_needing_compile(vault, config, db):
    db.upsert_raw(
        __import__("synto.models", fromlist=["RawNoteRecord"]).RawNoteRecord(
            path="raw/note.md", content_hash="abc", status="compiled"
        )
    )
    db.upsert_concepts("raw/note.md", ["Some Concept"])
    db.mark_concept_compile_state("Some Concept", ["raw/note.md"], "compiled")

    client = as_router(MagicMock())
    drafts, failed, _ = compile_concepts(config=config, router=client, db=db)

    assert drafts == []
    assert failed == []
    client.generate.assert_not_called()


def test_compile_concepts_dry_run(vault, config, db, fixtures_dir, capsys):
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(
        __import__("synto.models", fromlist=["RawNoteRecord"]).RawNoteRecord(
            path="raw/note.md", content_hash="abc", status="ingested"
        )
    )
    db.upsert_concepts("raw/note.md", ["Concept A"])

    client = as_router(MagicMock())
    drafts, _, _ = compile_concepts(config=config, router=client, db=db, dry_run=True)

    assert drafts == []
    assert list(config.drafts_dir.glob("*.md")) == []
    captured = capsys.readouterr()
    assert "Concept A" in captured.out


def test_compile_concepts_manual_edit_protection(vault, config, db, fixtures_dir):
    """Article with content_hash mismatch (manually edited) should be skipped."""
    from synto.models import WikiArticleRecord

    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(
        __import__("synto.models", fromlist=["RawNoteRecord"]).RawNoteRecord(
            path="raw/note.md", content_hash="abc", status="ingested"
        )
    )
    db.upsert_concepts("raw/note.md", ["Quantum Entanglement"])

    # Simulate published article with a DIFFERENT content_hash than what's on disk
    wiki_path = config.wiki_dir / "Quantum Entanglement.md"
    wiki_path.write_text("---\ntitle: Quantum Entanglement\n---\n\nManually edited content.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(wiki_path.relative_to(vault)),
            title="Quantum Entanglement",
            sources=["raw/note.md"],
            content_hash="original_hash_before_edit",  # differs from file on disk
            status="published",
        )
    )

    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_concept_client(article_json)

    drafts, failed, _ = compile_concepts(config=config, router=client, db=db)

    # Should skip the manually-edited article
    assert drafts == []
    client.generate.assert_not_called()


def test_compile_concepts_force_overrides_edit_protection(vault, config, db, fixtures_dir):
    """--force should recompile even manually-edited articles."""
    from synto.models import WikiArticleRecord

    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(
        __import__("synto.models", fromlist=["RawNoteRecord"]).RawNoteRecord(
            path="raw/note.md", content_hash="abc", status="ingested"
        )
    )
    db.upsert_concepts("raw/note.md", ["Quantum Entanglement"])

    wiki_path = config.wiki_dir / "Quantum Entanglement.md"
    wiki_path.write_text("---\ntitle: Quantum Entanglement\n---\n\nManually edited.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(wiki_path.relative_to(vault)),
            title="Quantum Entanglement",
            sources=["raw/note.md"],
            content_hash="old_hash",
            status="published",
        )
    )

    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_concept_client(article_json)

    drafts, failed, _ = compile_concepts(config=config, router=client, db=db, force=True)

    assert len(drafts) == 1


def test_write_concept_prompt_has_tag_instructions():
    prompt = _write_concept_prompt("Quantum Computing", "source text", [])
    assert "hyphen-separated" in prompt
    assert "machine-learning" in prompt


def test_write_prompt_legacy_has_tag_instructions():
    from synto.models import ArticlePlan

    plan = ArticlePlan(
        title="Test",
        action="create",
        path="test.md",
        reasoning="needed",
        source_paths=[],
    )
    prompt = _write_prompt_legacy(plan, "source text", [])
    assert "hyphen-separated" in prompt


def test_compile_concepts_marks_sources_compiled(vault, config, db, fixtures_dir):
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(
        __import__("synto.models", fromlist=["RawNoteRecord"]).RawNoteRecord(
            path="raw/note.md", content_hash="abc", status="ingested"
        )
    )
    db.upsert_concepts("raw/note.md", ["Concept A"])

    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_concept_client(article_json)

    compile_concepts(config=config, router=client, db=db)

    record = db.get_raw("raw/note.md")
    assert record.status == "compiled"


def test_compile_concepts_failed_same_source_stays_queued(vault, config, db):
    import json

    db.upsert_raw(RawNoteRecord(path="raw/note.md", content_hash="abc", status="ingested"))
    db.upsert_concepts("raw/note.md", ["Alpha", "Beta"])
    (vault / "raw" / "note.md").write_text("Body.")

    client = as_router(MagicMock())
    client.generate.side_effect = [
        json.dumps({"title": "Alpha", "content": "Alpha content.", "tags": []}),
        "not valid json",
        "not valid json",
        "not valid json",
    ]

    drafts, failed, _ = compile_concepts(config=config, router=client, db=db)

    assert len(drafts) == 1
    assert failed == ["Beta"]
    assert db.get_raw("raw/note.md").status == "ingested"
    assert "Beta" in db.concepts_needing_compile()
    assert "Alpha" not in db.concepts_needing_compile()


def test_compile_concepts_isolates_provider_error(vault, config, db):
    """A provider error (issue #25: e.g. OpenRouter 2xx rate-limit envelope, which
    generate() now raises as LLMBadRequestError) must fail only that concept and
    leave it queued for retry — never crash the whole compile run."""
    db.upsert_raw(RawNoteRecord(path="raw/note.md", content_hash="abc", status="ingested"))
    db.upsert_concepts("raw/note.md", ["Alpha"])
    (vault / "raw" / "note.md").write_text("Body.")

    client = as_router(MagicMock())
    client.generate.side_effect = LLMBadRequestError("openrouter: Rate limit exceeded (code=429)")

    drafts, failed, _ = compile_concepts(config=config, router=client, db=db)

    assert drafts == []
    assert failed == ["Alpha"]
    assert db.get_raw("raw/note.md").status == "ingested"
    assert "Alpha" in db.concepts_needing_compile()


# ── Language tests ─────────────────────────────────────────────────────────────


def test_write_concept_prompt_no_language():
    prompt = _write_concept_prompt("Topic", "source", [])
    assert "same language as the source notes" in prompt


def test_write_concept_prompt_with_language():
    prompt = _write_concept_prompt("Topic", "source", [], language="fr")
    assert "Output language: fr" in prompt
    assert "same language as the source notes" not in prompt


def test_write_prompt_legacy_no_language():
    from synto.models import ArticlePlan

    plan = ArticlePlan(title="T", action="create", path="t.md", reasoning="r", source_paths=[])
    prompt = _write_prompt_legacy(plan, "source", [])
    assert "same language as the source notes" in prompt


def test_write_prompt_legacy_with_language():
    from synto.models import ArticlePlan

    plan = ArticlePlan(title="T", action="create", path="t.md", reasoning="r", source_paths=[])
    prompt = _write_prompt_legacy(plan, "source", [], language="de")
    assert "Output language: de" in prompt


def test_resolve_language_uses_config_over_detected(config, db):
    r = RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested", language="fr")
    db.upsert_raw(r)
    config.pipeline.language = "en"
    assert _resolve_language(["raw/a.md"], db, config) == "en"


def test_resolve_language_uses_detected_when_unambiguous(config, db):
    for path, lang in [("raw/a.md", "fr"), ("raw/b.md", "fr")]:
        db.upsert_raw(RawNoteRecord(path=path, content_hash=path, status="ingested", language=lang))
    assert _resolve_language(["raw/a.md", "raw/b.md"], db, config) == "fr"


def test_resolve_language_none_when_mixed(config, db):
    for path, lang in [("raw/a.md", "fr"), ("raw/b.md", "de")]:
        db.upsert_raw(RawNoteRecord(path=path, content_hash=path, status="ingested", language=lang))
    assert _resolve_language(["raw/a.md", "raw/b.md"], db, config) is None


def test_resolve_language_none_when_no_detected(config, db):
    r = RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested", language=None)
    db.upsert_raw(r)
    assert _resolve_language(["raw/a.md"], db, config) is None


# ── unknown_filename filter ───────────────────────────────────────────────────


def test_apply_draft_media_mode_drops_unknown_filename():
    from synto.pipeline.compile import _apply_draft_media_mode

    content = "Text before\n![[_resources/clip/unknown_filename.jpeg]]\nText after."
    result = _apply_draft_media_mode(content, "reference")
    assert "unknown_filename" not in result
    assert "Text before" in result
    assert "Text after" in result


def test_apply_draft_media_mode_drops_unknown_filename_in_omit_mode():
    from synto.pipeline.compile import _apply_draft_media_mode

    content = "![[unknown_filename.png]] and normal ![[diagram.png]]"
    result = _apply_draft_media_mode(content, "omit")
    assert "unknown_filename" not in result
    assert "diagram" not in result


def test_gather_sources_strips_ocr_picture_text(vault, config):
    """Source material handed to the heavy model must not include OCR picture-text
    gibberish — otherwise the writer can copy it verbatim into the article."""
    from synto.pipeline.compile import _gather_sources

    raw = config.vault / "raw" / "paper.md"
    raw.write_text(
        "# Paper\n\nReal source sentence.\n\n"
        "**==> picture [499 x 200] intentionally omitted <==**\n\n"
        "**----- Start of picture text -----**<br>\n"
        "1.0<br>Ne VIII 775 HI Ly<br>F125LP<br>**----- End of picture text -----**<br>\n",
        encoding="utf-8",
    )
    combined, resolved = _gather_sources(["raw/paper.md"], config.vault)
    assert "Real source sentence." in combined
    assert "Start of picture text" not in combined
    assert "Ne VIII 775" not in combined
    assert "intentionally omitted" not in combined


# ── confidence gate ───────────────────────────────────────────────────────────


def test_approve_drafts_holds_back_below_min_confidence(vault, config, db):
    from synto.models import WikiArticleRecord
    from synto.vault import write_note

    low_draft = config.drafts_dir / "low.md"
    high_draft = config.drafts_dir / "high.md"
    write_note(low_draft, {"title": "Low", "status": "draft", "confidence": 0.3, "tags": []}, "A")
    write_note(high_draft, {"title": "High", "status": "draft", "confidence": 0.7, "tags": []}, "B")
    for path, title in [(low_draft, "Low"), (high_draft, "High")]:
        db.upsert_article(
            WikiArticleRecord(
                path=str(path.relative_to(vault)),
                title=title,
                sources=[],
                content_hash="h",
                status="draft",
            )
        )

    published = approve_drafts(config, db, [low_draft, high_draft], min_confidence=0.5)
    assert len(published) == 1
    assert published[0].name == "high.md"
    assert low_draft.exists()  # still in drafts


def test_approve_drafts_zero_min_confidence_publishes_all(vault, config, db):
    from synto.models import WikiArticleRecord
    from synto.vault import write_note

    draft = config.drafts_dir / "any.md"
    write_note(draft, {"title": "Any", "status": "draft", "confidence": 0.1, "tags": []}, "C")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft.relative_to(vault)),
            title="Any",
            sources=[],
            content_hash="h",
            status="draft",
        )
    )

    published = approve_drafts(config, db, [draft], min_confidence=0.0)
    assert len(published) == 1


def test_approve_db_updated_before_draft_removed(vault, config, db, monkeypatch):
    """DB must reach published state even if draft unlink fails mid-operation."""
    from pathlib import Path

    from synto.models import WikiArticleRecord
    from synto.vault import write_note

    draft_path = config.drafts_dir / "article.md"
    write_note(draft_path, {"title": "Article", "status": "draft", "tags": []}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Article",
            sources=[],
            content_hash="h",
            status="draft",
        )
    )
    target = config.wiki_dir / "article.md"

    # Simulate crash: unlink raises after DB updates have already run
    original_unlink = Path.unlink

    def failing_unlink(self, missing_ok=False):
        if self == draft_path:
            raise OSError("simulated disk full")
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with pytest.raises(OSError, match="simulated disk full"):
        approve_drafts(config, db, [draft_path])

    # DB must be consistent — article is published
    record = db.get_article(str(target.relative_to(vault)))
    assert record is not None
    assert record.is_draft is False

    # Published file must exist on disk
    assert target.exists()
