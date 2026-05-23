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


def test_load_alias_map_skips_ambiguous_aliases(db):
    # "cache" is claimed by both concepts — must be filtered from the map.
    _seed_aliases(db, "Caching", ["cache"])
    _seed_aliases(db, "CDN", ["cache"])
    result = db.load_concept_alias_map()
    assert "cache" not in result.get("Caching", [])
    assert "cache" not in result.get("CDN", [])


def test_load_alias_map_keeps_unambiguous_aliases(db):
    _seed_aliases(db, "Caching", ["caching-layer"])
    _seed_aliases(db, "CDN", ["edge-cache"])
    result = db.load_concept_alias_map()
    assert result["Caching"] == ["caching-layer"]
    assert result["CDN"] == ["edge-cache"]


def test_load_alias_map_partial_filter(db):
    # "cache" is ambiguous (both concepts), "caching-layer" is unique to Caching.
    # Caching keeps the unique alias; the ambiguous one is dropped.
    _seed_aliases(db, "Caching", ["cache", "caching-layer"])
    _seed_aliases(db, "CDN", ["cache"])
    result = db.load_concept_alias_map()
    assert result.get("Caching") == ["caching-layer"]
    assert "CDN" not in result  # CDN had only the ambiguous alias


# ── Stage 2: _expand_query() ──────────────────────────────────────────────────


def test_expand_query_alias_hit():
    alias_map = {"Idempotency": ["ack", "exactly-once"]}
    result = _expand_query("how does ack work in distributed systems?", alias_map)
    assert "Idempotency" in result


def test_expand_query_concept_name_alone_does_not_hint():
    # Concept names are not matchable surfaces — the LLM sees them in the index
    # already, so re-hinting them is redundant and eats cap slots.
    alias_map = {"Idempotency": ["ack"]}
    q = "explain idempotency please"
    assert _expand_query(q, alias_map) == q


def test_expand_query_concept_name_plus_alias_hints_once():
    # When the question contains BOTH the concept name and an alias, the
    # alias triggers the hint and the concept is listed exactly once.
    alias_map = {"Idempotency": ["ack"]}
    result = _expand_query("explain idempotency and ack semantics", alias_map)
    assert result.count("Idempotency") == 1
    assert "(Routing hint" in result


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


def test_expand_query_occurrence_order_preserved():
    # Z-concept matched first; alphabetical sort would push it last. The hint
    # must list Zeitgeist before Caching.
    alias_map = {
        "Zeitgeist": ["zeitgeist-term"],
        "Caching": ["caching-term"],
    }
    result = _expand_query("zeitgeist-term then caching-term", alias_map)
    z_idx = result.index("Zeitgeist")
    c_idx = result.index("Caching")
    assert z_idx < c_idx


def test_expand_query_caps_drops_later_occurrences():
    # Build 15 concepts and a question that mentions them in known order
    # (term0 ... term14). After cap=10, the first 10 (by occurrence) survive;
    # term10..term14 are dropped.
    alias_map = {f"Concept{i:02d}": [f"term{i:02d}"] for i in range(15)}
    q = " ".join(f"term{i:02d}" for i in range(15))
    result = _expand_query(q, alias_map)
    hint_line = next(line for line in result.splitlines() if "Routing hint" in line)
    for i in range(10):
        assert f"Concept{i:02d}" in hint_line
    for i in range(10, 15):
        assert f"Concept{i:02d}" not in hint_line


def test_expand_query_multi_word_alias():
    alias_map = {"Eventual Consistency": ["exactly once", "at least once"]}
    result = _expand_query("we need exactly once delivery", alias_map)
    assert "Eventual Consistency" in result


def test_expand_query_multi_word_alias_word_boundary():
    # "exactly oncestop" should NOT match "exactly once" — \b anchor on the
    # trailing edge of the alias keeps this honest.
    alias_map = {"Eventual Consistency": ["exactly once"]}
    q = "exactly oncestop"
    assert _expand_query(q, alias_map) == q


def test_expand_query_uppercase_two_letter_acronym_hits():
    # "AI" (all-caps, len 2) bypasses the length-3 floor.
    alias_map = {"Artificial Intelligence": ["AI"]}
    result = _expand_query("use AI for routing", alias_map)
    assert "Artificial Intelligence" in result


def test_expand_query_lowercase_two_letter_still_filtered():
    # Lowercase "ai" is NOT the all-caps acronym path — length-3 floor applies.
    # Pins that the exception is uppercase-only.
    alias_map = {"Artificial Intelligence": ["ai"]}
    q = "use ai for routing"
    assert _expand_query(q, alias_map) == q


def test_expand_query_uppercase_one_letter_filtered():
    # The acronym exception requires len >= 2. A single uppercase letter is
    # still filtered (would match too aggressively otherwise).
    alias_map = {"Existential": ["X"]}
    q = "explain X to me"
    assert _expand_query(q, alias_map) == q


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


def test_query_hints_concepts_past_80_article_cap(vault, config, db):
    # Pin Fix A: vault with 90 articles, bridge alias on the 85th — the answer-
    # prompt wikilink whitelist is still capped at 80, but expansion sees all
    # titles. Without the fix, this concept would be silently invisible.
    titles = [f"Concept{i:03d}" for i in range(90)]
    bridge_concept = titles[85]
    index_lines = ["# Index", ""] + [f"- [[{t}]]" for t in titles]
    _write_index(config, "\n".join(index_lines) + "\n")
    for t in titles:
        _write_concept_page(config, t)
    _seed_aliases(db, bridge_concept, ["wibbletron"])

    fast = _fast_client([bridge_concept])
    heavy = _heavy_client()

    _query_core(config, fast, heavy, db, "what is a wibbletron?")

    prompt = _routing_prompt(fast)
    assert bridge_concept in prompt
    assert "(Routing hint" in prompt


def test_query_logs_routing_hint_when_fired(vault, config, db, caplog):
    _write_index(config, "# Index\n\n- [[Idempotency]]\n")
    _write_concept_page(config, "Idempotency")
    _seed_aliases(db, "Idempotency", ["ack"])

    fast = _fast_client(["Idempotency"])
    heavy = _heavy_client()

    with caplog.at_level("INFO", logger="synto.pipeline.query"):
        _query_core(config, fast, heavy, db, "how does ack work?")

    hint_records = [r for r in caplog.records if "query.routing_hint" in r.getMessage()]
    assert len(hint_records) == 1
    assert "Idempotency" in hint_records[0].getMessage()


def test_query_does_not_log_routing_hint_when_no_match(vault, config, db, caplog):
    _write_index(config, "# Index\n\n- [[Idempotency]]\n")
    _write_concept_page(config, "Idempotency")
    _seed_aliases(db, "Idempotency", ["ack"])

    fast = _fast_client(["Idempotency"])
    heavy = _heavy_client()

    with caplog.at_level("INFO", logger="synto.pipeline.query"):
        _query_core(config, fast, heavy, db, "how do I bake bread?")

    hint_records = [r for r in caplog.records if "query.routing_hint" in r.getMessage()]
    assert hint_records == []
