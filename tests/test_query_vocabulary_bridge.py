"""Tests for the query vocabulary bridge (feature #35).

Covers:
  Stage 1 — StateDB.load_concept_alias_map()
  Stage 2 — _expand_query() pure function
  Stage 3 — expansion wired into _query_core() routing prompt only
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synto.config import Config
from synto.pipeline.query import _expand_query, _query_core
from synto.state import StateDB
from synto.vault import write_note

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


def _seed_aliases(db: StateDB, concept_name: str, aliases: list[str]) -> None:
    """Insert concept + aliases directly into the DB."""
    with db._tx():
        for alias in aliases:
            db._conn.execute(
                "INSERT OR IGNORE INTO concept_aliases (concept_name, alias) VALUES (?, ?)",
                (concept_name, alias),
            )


def _write_index(config: Config, content: str) -> None:
    (config.wiki_dir / "index.md").write_text(content, encoding="utf-8")


def _write_concept_page(config: Config, title: str, body: str = "") -> Path:
    path = config.wiki_dir / f"{title}.md"
    write_note(
        path,
        {"title": title, "tags": [], "status": "published"},
        body or f"Content about {title}.",
    )
    return path


def _fast_client(pages: list[str]) -> MagicMock:
    c = MagicMock()
    c.generate.return_value = json.dumps({"pages": pages})
    return c


def _heavy_client(answer: str = "Answer.", title: str = "Topic") -> MagicMock:
    c = MagicMock()
    c.generate.return_value = json.dumps({"answer": answer, "title": title})
    return c


def _routing_prompt(fast_mock: MagicMock) -> str:
    return fast_mock.generate.call_args.kwargs["prompt"]


def _answer_prompt(heavy_mock: MagicMock) -> str:
    return heavy_mock.generate.call_args.kwargs["prompt"]


# ── Stage 1: StateDB.load_concept_alias_map() ─────────────────────────────────


def test_load_alias_map_empty_table(db):
    result = db.load_concept_alias_map()
    assert result == {}


def test_load_alias_map_groups_multiple_aliases(db):
    _seed_aliases(db, "Idempotency", ["idempotent", "ack", "exactly-once"])
    result = db.load_concept_alias_map()
    assert set(result["Idempotency"]) == {"idempotent", "ack", "exactly-once"}
    assert len(result["Idempotency"]) == 3


def test_load_alias_map_skips_concepts_without_aliases(db):
    _seed_aliases(db, "Idempotency", ["ack"])
    # No aliases seeded for "Caching"
    result = db.load_concept_alias_map()
    assert "Idempotency" in result
    assert "Caching" not in result


def test_load_alias_map_no_table(db):
    db._conn.execute("DROP TABLE IF EXISTS concept_aliases")
    result = db.load_concept_alias_map()
    assert result == {}


# ── Stage 2: _expand_query() ──────────────────────────────────────────────────


def test_expand_query_alias_hit():
    alias_map = {"Idempotency": ["ack", "exactly-once"]}
    result = _expand_query("how does ack work in distributed systems?", alias_map)
    assert "Idempotency" in result


def test_expand_query_concept_name_hit():
    alias_map = {"Idempotency": ["ack"]}
    result = _expand_query("explain idempotency please", alias_map)
    assert "Idempotency" in result


def test_expand_query_no_match():
    alias_map = {"Idempotency": ["ack", "exactly-once"]}
    q = "how do I bake bread?"
    assert _expand_query(q, alias_map) == q


def test_expand_query_empty_alias_map():
    q = "any question"
    assert _expand_query(q, {}) == q


def test_expand_query_empty_question():
    alias_map = {"Idempotency": ["ack"]}
    assert _expand_query("", alias_map) == ""


def test_expand_query_word_boundary_no_false_match():
    alias_map = {"REST": ["rest"]}
    q = "my restaurant business"
    assert _expand_query(q, alias_map) == q


def test_expand_query_word_boundary_with_punctuation():
    alias_map = {"ERP": ["erp"]}
    result = _expand_query("sync to ERP, then continue", alias_map)
    assert "ERP" in result


def test_expand_query_min_length_excludes_short_aliases():
    alias_map = {"ML": ["ml", "ai"]}  # both under _MIN_ALIAS_LEN=3
    q = "what is ml ai about"
    assert _expand_query(q, alias_map) == q


def test_expand_query_caps_at_max_matches():
    alias_map = {f"Concept{i}": [f"term{i}"] for i in range(15)}
    q = " ".join(f"term{i}" for i in range(15))
    result = _expand_query(q, alias_map)
    assert "(Routing hint — related wiki concepts:" in result
    # At most 10 concepts in the hint
    hint_line = [line for line in result.splitlines() if "Routing hint" in line][0]
    concept_count = hint_line.count(",") + 1
    assert concept_count <= 10


def test_expand_query_dedupes_concept_via_multiple_aliases():
    alias_map = {"Idempotency": ["ack", "idempotent", "exactly-once"]}
    result = _expand_query("ack idempotent exactly-once message", alias_map)
    assert result.count("Idempotency") == 1


def test_expand_query_case_insensitive():
    alias_map = {"Idempotency": ["idempotent"]}
    result = _expand_query("IDEMPOTENT operations", alias_map)
    assert "Idempotency" in result


def test_expand_query_filters_unknown_titles():
    alias_map = {"Idempotency": ["ack"]}
    known = {"SomeOtherConcept"}
    q = "how does ack work?"
    assert _expand_query(q, alias_map, known_titles=known) == q


def test_expand_query_keeps_known_titles():
    alias_map = {"Idempotency": ["ack"]}
    known = {"Idempotency"}
    result = _expand_query("how does ack work?", alias_map, known_titles=known)
    assert "Idempotency" in result


def test_expand_query_known_titles_none_keeps_all():
    alias_map = {"Idempotency": ["ack"], "Caching": ["cache"]}
    result = _expand_query("ack and cache matter", alias_map, known_titles=None)
    assert "Idempotency" in result
    assert "Caching" in result


def test_expand_query_hint_on_separate_line():
    alias_map = {"Idempotency": ["ack"]}
    result = _expand_query("how does ack work?", alias_map)
    assert "\n\n(Routing hint — related wiki concepts:" in result


def test_expand_query_cjk_silent_noop():
    # \b word-boundary doesn't work for CJK — documented v1 limitation.
    # This test pins the no-op behaviour so any future fix is an intentional change.
    alias_map = {"Sync": ["同步"]}
    q = "同步订单"
    assert _expand_query(q, alias_map) == q


# ── Stage 3: expansion wired into _query_core() ───────────────────────────────


def test_expansion_injected_into_routing_prompt(vault, config, db):
    _write_index(config, "# Index\n\n- [[Idempotency]]\n")
    _write_concept_page(config, "Idempotency", "Idempotent operations.")
    _seed_aliases(db, "Idempotency", ["ack"])

    fast = _fast_client(["Idempotency"])
    heavy = _heavy_client()

    _query_core(config, fast, heavy, db, "how does ack work?")

    prompt = _routing_prompt(fast)
    assert "(Routing hint — related wiki concepts:" in prompt
    assert "Idempotency" in prompt


def test_expansion_absent_from_answer_prompt(vault, config, db):
    _write_index(config, "# Index\n\n- [[Idempotency]]\n")
    _write_concept_page(config, "Idempotency", "Idempotent operations.")
    _seed_aliases(db, "Idempotency", ["ack"])

    fast = _fast_client(["Idempotency"])
    heavy = _heavy_client()

    _query_core(config, fast, heavy, db, "how does ack work?")

    prompt = _answer_prompt(heavy)
    assert "Routing hint" not in prompt


def test_query_with_db_none_runs_without_error(vault, config):
    _write_index(config, "# Index\n\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    fast = _fast_client(["Topic"])
    heavy = _heavy_client()

    result = _query_core(config, fast, heavy, None, "what is topic?")
    assert result.answer == "Answer."


def test_query_with_empty_alias_map_passes_through(vault, config, db):
    _write_index(config, "# Index\n\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")
    # No aliases seeded

    fast = _fast_client(["Topic"])
    heavy = _heavy_client()

    _query_core(config, fast, heavy, db, "what is topic?")

    prompt = _routing_prompt(fast)
    assert "Routing hint" not in prompt


def test_query_with_no_matching_aliases_passes_through(vault, config, db):
    _write_index(config, "# Index\n\n- [[Idempotency]]\n")
    _write_concept_page(config, "Idempotency")
    _seed_aliases(db, "Idempotency", ["ack"])

    fast = _fast_client(["Idempotency"])
    heavy = _heavy_client()

    _query_core(config, fast, heavy, db, "what is bread?")

    prompt = _routing_prompt(fast)
    assert "Routing hint" not in prompt


def test_query_filters_concepts_without_articles(vault, config, db):
    _write_index(config, "# Index\n\n- [[OtherConcept]]\n")
    _write_concept_page(config, "OtherConcept")
    # "Idempotency" has a matching alias but NO published article in the vault
    _seed_aliases(db, "Idempotency", ["ack"])

    fast = _fast_client(["OtherConcept"])
    heavy = _heavy_client()

    _query_core(config, fast, heavy, db, "how does ack work?")

    prompt = _routing_prompt(fast)
    assert "Idempotency" not in prompt


def test_query_passes_question_to_answer_prompt_verbatim(vault, config, db):
    _write_index(config, "# Index\n\n- [[Idempotency]]\n")
    _write_concept_page(config, "Idempotency", "Idempotent operations.")
    _seed_aliases(db, "Idempotency", ["ack"])

    fast = _fast_client(["Idempotency"])
    heavy = _heavy_client()

    original = "how does ack work?"
    _query_core(config, fast, heavy, db, original)

    prompt = _answer_prompt(heavy)
    assert original in prompt
    assert "Routing hint" not in prompt
