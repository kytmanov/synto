"""Tests for per-source-type ingest config.

Covers SourceTypeOverride, effective_max_concepts, and integration with ingest_note.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from test_ingest import _analysis_json, _make_client, _write_raw

from synto.config import Config, SourceTypeOverride
from synto.pipeline.ingest import ingest_note
from synto.state import StateDB

# ── Stage 1: SourceTypeOverride model + source_overrides field ──────────────


def test_source_overrides_parse_direct():
    config = Config(
        vault="/tmp/v",
        pipeline={
            "source_overrides": {
                "textbook": {"max_concepts_per_source": 25},
                "paper": {"max_concepts_per_source": 15},
            }
        },
    )
    assert config.pipeline.source_overrides["textbook"].max_concepts_per_source == 25
    assert config.pipeline.source_overrides["paper"].max_concepts_per_source == 15


def test_source_overrides_parse_from_toml(tmp_path):
    (tmp_path / "synto.toml").write_text(
        "[pipeline.source_overrides.textbook]\nmax_concepts_per_source = 25\n"
    )
    config = Config.from_vault(tmp_path)
    assert config.pipeline.source_overrides["textbook"].max_concepts_per_source == 25


def test_source_overrides_empty_by_default():
    config = Config(vault="/tmp/v")
    assert config.pipeline.source_overrides == {}


def test_source_override_none_field():
    assert SourceTypeOverride().max_concepts_per_source is None


def test_source_override_rejects_non_positive():
    with pytest.raises(ValidationError):
        SourceTypeOverride(max_concepts_per_source=0)


def test_unknown_source_type_warns_not_raises(caplog):
    with caplog.at_level("WARNING"):
        config = Config(
            vault="/tmp/v",
            pipeline={"source_overrides": {"textbok": {"max_concepts_per_source": 25}}},
        )
    assert "textbok" in config.pipeline.source_overrides
    assert any("unknown source type" in r.message for r in caplog.records)


# ── Stage 2: effective_max_concepts() ──────────────────────────────────────


def test_effective_max_concepts_override():
    config = Config(
        vault="/tmp/v",
        pipeline={"source_overrides": {"textbook": {"max_concepts_per_source": 25}}},
    )
    assert config.pipeline.effective_max_concepts("textbook") == 25


def test_effective_max_concepts_global_fallback():
    # Types without a built-in default fall through to the global default.
    config = Config(vault="/tmp/v")
    assert config.pipeline.effective_max_concepts("notes") == 8
    assert config.pipeline.effective_max_concepts("spec") == 8


def test_effective_max_concepts_unknown_type():
    config = Config(
        vault="/tmp/v",
        pipeline={"source_overrides": {"textbook": {"max_concepts_per_source": 25}}},
    )
    assert config.pipeline.effective_max_concepts("notes") == 8


def test_effective_max_concepts_none_field():
    # An override with no value is not a value; resolution falls through to the built-in.
    config = Config(
        vault="/tmp/v",
        pipeline={"source_overrides": {"textbook": {}}},
    )
    assert config.pipeline.effective_max_concepts("textbook") == 25


def test_builtin_book_defaults_apply_without_config():
    """Long-form source types are book-appropriate out of the box.

    Why it matters: a book ingested as `source_type: textbook` with no override must not
    be capped at the short-note default of 8 — the whole point of source types.
    """
    config = Config(vault="/tmp/v")
    assert config.pipeline.effective_max_concepts("textbook") == 25
    assert config.pipeline.effective_max_concepts("paper") == 15
    assert config.pipeline.effective_max_concepts("notes") == 8


def test_explicit_override_beats_builtin_default():
    # Explicit per-type override wins, even when it lowers the built-in.
    config = Config(
        vault="/tmp/v",
        pipeline={"source_overrides": {"textbook": {"max_concepts_per_source": 10}}},
    )
    assert config.pipeline.effective_max_concepts("textbook") == 10


def test_raised_global_beats_lower_builtin():
    # A built-in default never silently lowers a user's raised global.
    config = Config(vault="/tmp/v", pipeline={"max_concepts_per_source": 40})
    assert config.pipeline.effective_max_concepts("paper") == 40
    assert config.pipeline.effective_max_concepts("notes") == 40


# ── Stage 3: Integration with ingest_note() ────────────────────────────────


def _make_vault(base: Path) -> Path:
    (base / "raw").mkdir(parents=True)
    (base / "wiki").mkdir(parents=True)
    (base / "wiki" / ".drafts").mkdir(parents=True)
    (base / "wiki" / "sources").mkdir(parents=True)
    (base / ".synto").mkdir(parents=True)
    return base


def _ingest_count(vault_dir, source_type, pipeline_cfg, concepts, quality):
    config = Config(vault=vault_dir, pipeline=pipeline_cfg)
    db = StateDB(config.state_db_path)
    # Concept names must appear in the body: at medium/low quality
    # _filter_concept_candidates drops evidence-free concepts before the cap applies.
    body = "\n".join(concepts)
    path = _write_raw(
        vault_dir,
        f"{source_type}.md",
        f"---\nsource_type: {source_type}\n---\n# {source_type}\n\n{body}",
    )
    client = _make_client(_analysis_json(concepts=concepts, quality=quality))
    ingest_note(path, config, client, db)
    return len(db.list_all_concept_names())


_TWENTY = [f"Topic Alpha {i}" for i in range(20)]


def test_ingest_textbook_override_lifts_high_quality_cap(tmp_path):
    vault = _make_vault(tmp_path / "tb")
    count = _ingest_count(
        vault,
        "textbook",
        {"source_overrides": {"textbook": {"max_concepts_per_source": 25}}},
        _TWENTY,
        "high",
    )
    assert count == 20


def test_ingest_notes_unaffected_by_override(tmp_path):
    vault = _make_vault(tmp_path / "nt")
    count = _ingest_count(
        vault,
        "notes",
        {"source_overrides": {"textbook": {"max_concepts_per_source": 25}}},
        _TWENTY,
        "high",
    )
    assert count == 8


def test_ingest_medium_quality_still_clamped_within_override(tmp_path):
    vault = _make_vault(tmp_path / "md")
    count = _ingest_count(
        vault,
        "textbook",
        {"source_overrides": {"textbook": {"max_concepts_per_source": 25}}},
        _TWENTY,
        "medium",
    )
    assert count == 4


# ── Stage 4: Multi-chunk path (the #52 bug) ────────────────────────────────
# A tiny fast_ctx (chunk_size = ctx // 2 = 50 chars) splits the body into several chunks,
# exercising _merge_chunk_results — the path where per-call output limits were wrongly
# applied as whole-document caps. The single-chunk tests above never hit it.


def _multichunk_body(names: list[str]) -> str:
    # Spans several chunks at fast_ctx=100, with every name present so concepts and
    # references have in-body evidence regardless of where chunk boundaries fall.
    return "\n".join(names) + "\n" + "filler text " * 20


def test_concepts_honor_config_on_multichunk_source(tmp_path):
    """A long, multi-chunk textbook yields its configured concept count, not 8.

    Why it matters: the per-call merge limit previously truncated the candidate pool to 8
    before ingest_note's configured cap was consulted, so the config was silently ignored
    for exactly the book-length sources it was meant for (#52).
    """
    vault = _make_vault(tmp_path / "mc_concepts")
    config = Config(
        vault=vault,
        ollama={"fast_ctx": 100},
        pipeline={"source_overrides": {"textbook": {"max_concepts_per_source": 25}}},
    )
    db = StateDB(config.state_db_path)
    path = _write_raw(
        vault, "textbook.md", f"---\nsource_type: textbook\n---\n{_multichunk_body(_TWENTY)}"
    )
    client = _make_client(_analysis_json(concepts=_TWENTY, quality="high"))
    ingest_note(path, config, client, db)

    assert client.generate.call_count > 1  # multi-chunk path actually exercised
    assert len(db.list_all_concept_names()) == 20


def test_named_references_not_capped_at_eight_on_multichunk_source(tmp_path):
    """Named references from a long source aren't bounded by the per-call limit of 8.

    Why it matters: named references have no downstream cap, so the merge slice was their
    only (chunk-order-biased) ceiling — a multi-chunk book lost every reference past 8.
    """
    vault = _make_vault(tmp_path / "mc_refs")
    config = Config(vault=vault, ollama={"fast_ctx": 100})
    db = StateDB(config.state_db_path)
    refs = [f"Reference Person {i}" for i in range(12)]
    concepts = ["Topic Alpha", "Topic Beta"]
    body = _multichunk_body([*concepts, *refs])
    path = _write_raw(vault, "textbook.md", f"---\nsource_type: textbook\n---\n{body}")
    client = _make_client(_analysis_json(concepts=concepts, named_references=refs, quality="high"))
    ingest_note(path, config, client, db)

    assert client.generate.call_count > 1
    stored_refs = [item for item in db.list_items() if item.subtype == "named_reference"]
    assert len(stored_refs) > 8
