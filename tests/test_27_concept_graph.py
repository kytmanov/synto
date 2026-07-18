"""Feature 27 Stage 1: graph/graph.json in pack export + "graph" capability.

Mirrors the seeding pattern of tests/test_phase1a_pack_export.py: a seeded state db +
published wiki articles on disk, exported via export_pack, then asserted against the
files export_pack writes to out_dir.
"""

from __future__ import annotations

import json
from pathlib import Path

from synto.concept_text import concept_key
from synto.config import Config
from synto.models import WikiArticleRecord
from synto.pack_export import export_pack
from synto.readers import PackReader, VaultReader
from synto.state import StateDB
from synto.vault import write_note


def _write_wiki_toml(vault: Path) -> None:
    (vault / "synto.toml").write_text(
        """
[models]
fast = "test-fast"
heavy = "test-heavy"

[provider]
name = "ollama"
url = "http://localhost:11434"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _seed_two_concepts_with_articles(vault: Path, config: Config, db: StateDB) -> None:
    _write_wiki_toml(vault)
    write_note(config.wiki_dir / "Vector-Clocks.md", {"title": "Vector Clocks"}, "Body")
    write_note(config.wiki_dir / "Causal-Consistency.md", {"title": "Causal Consistency"}, "Body")
    db.upsert_concepts("raw/a.md", ["Vector Clocks", "Causal Consistency"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Vector-Clocks.md",
            title="Vector Clocks",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Causal-Consistency.md",
            title="Causal Consistency",
            sources=["raw/a.md"],
            content_hash="h2",
            status="published",
        )
    )


def test_pack_export_writes_graph_json_with_relations(vault: Path, config: Config, db: StateDB):
    _seed_two_concepts_with_articles(vault, config, db)
    db.upsert_relation(
        subject="Vector Clocks",
        predicate="implemented_by",
        object_="Causal Consistency",
        confidence=0.9,
        source_segment_id="note:a:0",
        evidence_text="Vector clocks implement causal consistency.",
    )
    db.upsert_relation(
        subject="Causal Consistency",
        predicate="depends_on",
        object_="Network Partitions",
        confidence=0.5,
        source_segment_id="note:a:1",
        evidence_text="Causal consistency depends on handling network partitions.",
    )

    out_dir = vault / ".knowledge" / "agents"
    export_pack(config, target="agents", out=out_dir)

    manifest = json.loads((out_dir / "agent" / "manifest.json").read_text(encoding="utf-8"))
    assert "graph" in manifest["pack"]["capabilities"]

    graph_path = out_dir / "graph" / "graph.json"
    assert graph_path.exists()
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1

    node_ids = {node["id"] for node in payload["nodes"]}
    concepts = json.loads((out_dir / "agent" / "concepts.json").read_text(encoding="utf-8"))[
        "concepts"
    ]
    assert node_ids == {concept_key(c["name"]) for c in concepts}

    by_name = {node["label"]: node for node in payload["nodes"]}
    assert by_name["Vector Clocks"]["article_id"] is not None
    assert by_name["Causal Consistency"]["article_id"] is not None

    edges = {(e["from_id"], e["to_id"], e["predicate"], e["confidence"]) for e in payload["edges"]}
    assert (
        concept_key("Vector Clocks"),
        concept_key("Causal Consistency"),
        "implemented_by",
        0.9,
    ) in edges
    # "Network Partitions" is not a known concept — edge exports anyway, no filtering.
    assert (
        concept_key("Causal Consistency"),
        concept_key("Network Partitions"),
        "depends_on",
        0.5,
    ) in edges
    assert len(payload["edges"]) == 2


def test_pack_export_omits_graph_json_without_relations(vault: Path, config: Config, db: StateDB):
    _seed_two_concepts_with_articles(vault, config, db)

    out_dir = vault / ".knowledge" / "agents"
    export_pack(config, target="agents", out=out_dir)

    manifest = json.loads((out_dir / "agent" / "manifest.json").read_text(encoding="utf-8"))
    assert "graph" not in manifest["pack"]["capabilities"]
    assert not (out_dir / "graph" / "graph.json").exists()


def test_vault_reader_capabilities_gate_graph_on_relations(
    vault: Path, config: Config, db: StateDB
):
    reader = VaultReader(vault)
    assert "graph" not in reader.capabilities

    db.upsert_relation(
        subject="A",
        predicate="related_to",
        object_="B",
        confidence=0.5,
        source_segment_id="s1",
        evidence_text="A relates to B.",
    )

    reader_after = VaultReader(vault)
    assert "graph" in reader_after.capabilities


def test_pack_reader_graph_returns_parsed_payload_or_none(vault: Path, config: Config, db: StateDB):
    _seed_two_concepts_with_articles(vault, config, db)
    db.upsert_relation(
        subject="Vector Clocks",
        predicate="implemented_by",
        object_="Causal Consistency",
        confidence=0.9,
        source_segment_id="note:a:0",
        evidence_text="Vector clocks implement causal consistency.",
    )

    with_graph = vault / ".knowledge" / "with-graph"
    export_pack(config, target="agents", out=with_graph)
    graph = PackReader(with_graph).graph()
    assert graph is not None
    assert graph["schema_version"] == 1
    assert len(graph["edges"]) == 1

    without_graph = vault / ".knowledge" / "without-graph"
    db2_vault = vault.parent / "no-relations-vault"
    db2_vault.mkdir()
    (db2_vault / "raw").mkdir()
    (db2_vault / "wiki").mkdir()
    (db2_vault / "wiki" / ".drafts").mkdir()
    (db2_vault / ".synto").mkdir()
    _write_wiki_toml(db2_vault)
    config2 = Config(vault=db2_vault)
    db2 = StateDB(config2.state_db_path)
    write_note(config2.wiki_dir / "Solo.md", {"title": "Solo"}, "Body")
    db2.upsert_concepts("raw/a.md", ["Solo"])
    db2.upsert_article(
        WikiArticleRecord(
            path="wiki/Solo.md",
            title="Solo",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )
    export_pack(config2, target="agents", out=without_graph)
    assert PackReader(without_graph).graph() is None
