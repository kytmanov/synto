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
    config = Config(vault="/tmp/v")
    assert config.pipeline.effective_max_concepts("textbook") == 8


def test_effective_max_concepts_unknown_type():
    config = Config(
        vault="/tmp/v",
        pipeline={"source_overrides": {"textbook": {"max_concepts_per_source": 25}}},
    )
    assert config.pipeline.effective_max_concepts("notes") == 8


def test_effective_max_concepts_none_field():
    config = Config(
        vault="/tmp/v",
        pipeline={"source_overrides": {"textbook": {}}},
    )
    assert config.pipeline.effective_max_concepts("textbook") == 8


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
    path = _write_raw(
        vault_dir,
        f"{source_type}.md",
        f"---\nsource_type: {source_type}\n---\n# {source_type}\n\nBody text.",
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
    assert count > 8


def test_ingest_notes_unaffected_by_override(tmp_path):
    vault = _make_vault(tmp_path / "nt")
    count = _ingest_count(
        vault,
        "notes",
        {"source_overrides": {"textbook": {"max_concepts_per_source": 25}}},
        _TWENTY,
        "high",
    )
    assert count <= 8


def test_ingest_medium_quality_still_clamped_within_override(tmp_path):
    vault = _make_vault(tmp_path / "md")
    count = _ingest_count(
        vault,
        "textbook",
        {"source_overrides": {"textbook": {"max_concepts_per_source": 25}}},
        _TWENTY,
        "medium",
    )
    assert count <= 4
