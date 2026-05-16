from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synto.config import Config
from synto.engines import QueryEngine
from synto.pipeline.query import run_query
from synto.readers import VaultReader
from synto.state import StateDB
from synto.vault import write_note


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / ".synto").mkdir()
    return tmp_path


@pytest.fixture
def config(vault: Path) -> Config:
    return Config(vault=vault)


@pytest.fixture
def db(config: Config) -> StateDB:
    return StateDB(config.state_db_path)


def _make_client(selection_json: str, answer_json: str) -> MagicMock:
    client = MagicMock()
    call_count = [0]

    def side_effect(**kwargs):
        call_count[0] += 1
        return selection_json if call_count[0] == 1 else answer_json

    client.generate.side_effect = side_effect
    return client


def _write_index(config: Config, content: str) -> None:
    (config.wiki_dir / "index.md").write_text(content, encoding="utf-8")


def _write_concept_page(config: Config, title: str, body: str = "") -> None:
    write_note(
        config.wiki_dir / f"{title}.md",
        {"title": title, "tags": [], "status": "published"},
        body or f"Content about {title}.",
    )


def test_query_engine_returns_answer_shape_and_selected_pages(config: Config, db: StateDB) -> None:
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Quantum Computing]]\n")
    _write_concept_page(config, "Quantum Computing", "Qubits exploit superposition.")
    fast_client = MagicMock()
    heavy_client = MagicMock()
    fast_client.generate.return_value = json.dumps({"pages": ["Quantum Computing"]})
    heavy_client.generate.return_value = json.dumps(
        {"answer": "[[Quantum Computing]] uses qubits.", "title": "Quantum Computing"}
    )

    engine = QueryEngine(VaultReader(config.vault), fast_client, heavy_client, config, db=db)
    answer = engine.query("What is quantum computing?")

    assert answer.text == "[[Quantum Computing]] uses qubits."
    assert answer.title == "Quantum Computing"
    assert answer.citations == ()
    assert engine.last_selected_pages == ("Quantum Computing",)
    assert fast_client.generate.call_count == 1
    assert heavy_client.generate.call_count == 1


def test_query_engine_uses_fast_and_heavy_clients_for_distinct_stages(
    config: Config, db: StateDB
) -> None:
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic", "Topic body.")
    fast_client = MagicMock()
    heavy_client = MagicMock()
    fast_client.generate.return_value = json.dumps({"pages": ["Topic"]})
    heavy_client.generate.return_value = json.dumps(
        {"answer": "Answer about [[Topic]].", "title": "Topic"}
    )

    engine = QueryEngine(VaultReader(config.vault), fast_client, heavy_client, config, db=db)
    answer = engine.query("Tell me about Topic")

    assert answer.text == "Answer about [[Topic]]."
    assert fast_client.generate.call_count == 1
    assert heavy_client.generate.call_count == 1


def test_query_engine_matches_run_query_output(config: Config, db: StateDB) -> None:
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic", "Topic body.")
    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Answer about [[Topic]].", "title": "Topic"})

    engine = QueryEngine(
        VaultReader(config.vault),
        _make_client(selection_json, answer_json),
        _make_client(selection_json, answer_json),
        config,
        db=db,
    )
    engine_answer = engine.query("Tell me about Topic")

    result = run_query(config, _make_client(selection_json, answer_json), db, "Tell me about Topic")

    assert result.answer == engine_answer.text
    assert result.selected_pages == list(engine.last_selected_pages)


def test_query_engine_tracks_missing_index_without_saving(config: Config, db: StateDB) -> None:
    engine = QueryEngine(VaultReader(config.vault), MagicMock(), MagicMock(), config, db=db)

    answer = engine.query("Any question")

    assert "index" in answer.text.lower()
    assert engine.last_index_found is False
    assert engine.last_selected_pages == ()
