"""Feature 27 Stage 1: graph/graph.json in pack export + "graph" capability.

Mirrors the seeding pattern of tests/test_phase1a_pack_export.py: a seeded state db +
published wiki articles on disk, exported via export_pack, then asserted against the
files export_pack writes to out_dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from conftest import as_endpoint

from synto.concept_text import concept_key
from synto.config import Config
from synto.engines import QueryEngine
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

    # Seed order above is (Vector Clocks, ...) then (Causal Consistency, ...) — the reverse
    # of sorted-by-subject order — so this only passes if the export sorts edges rather than
    # relying on insertion/rowid order.
    edge_tuples = [
        (e["from_id"], e["to_id"], e["predicate"], e["confidence"]) for e in payload["edges"]
    ]
    assert edge_tuples == [
        (
            concept_key("Causal Consistency"),
            concept_key("Network Partitions"),
            "depends_on",
            0.5,
        ),
        # "Network Partitions" is not a known concept — edge exports anyway, no filtering.
        (
            concept_key("Vector Clocks"),
            concept_key("Causal Consistency"),
            "implemented_by",
            0.9,
        ),
    ]


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


# ---------------------------------------------------------------------------
# Stage 2: synto find reverse-lookup CLI command
# ---------------------------------------------------------------------------


def _seed_find_fixtures(config: Config, db: StateDB) -> None:
    """Seed three published articles exercising the concept/title/body tiers.

    Only the first is registered as a concept — the title and body matches must
    come purely from article metadata/content, not from concept resolution, so
    the test can't accidentally pass via the wrong tier.
    """
    _write_wiki_toml(config.vault)
    write_note(
        config.wiki_dir / "Raft.md",
        {"title": "Raft"},
        "Raft is a consensus algorithm for managing replicated logs.",
    )
    db.upsert_concepts("raw/a.md", ["Raft"])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Raft.md",
            title="Raft",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )

    write_note(
        config.wiki_dir / "Raft-Variants.md",
        {"title": "Raft Variants Overview"},
        "This page surveys leader-election strategies in replicated systems.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Raft-Variants.md",
            title="Raft Variants Overview",
            sources=["raw/b.md"],
            content_hash="h2",
            status="published",
        )
    )

    write_note(
        config.wiki_dir / "Distributed-Systems-Notes.md",
        {"title": "Distributed Systems Notes"},
        "This note explains how Raft achieves consensus in distributed logs.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Distributed-Systems-Notes.md",
            title="Distributed Systems Notes",
            sources=["raw/c.md"],
            content_hash="h3",
            status="published",
        )
    )


def test_find_ranks_concept_title_body_and_dedupes(config: Config, db: StateDB) -> None:
    from click.testing import CliRunner

    from synto.cli import cli

    _seed_find_fixtures(config, db)

    runner = CliRunner()
    result = runner.invoke(cli, ["find", "--vault", str(config.vault), "raft"])

    assert result.exit_code == 0, result.output
    assert "wiki/Raft.md" in result.output
    assert "wiki/Raft-Variants.md" in result.output
    assert "wiki/Distributed-Systems-Notes.md" in result.output

    # Tier order preserved: concept match, then title match, then body match.
    concept_pos = result.output.index("wiki/Raft.md")
    title_pos = result.output.index("wiki/Raft-Variants.md")
    body_pos = result.output.index("wiki/Distributed-Systems-Notes.md")
    assert concept_pos < title_pos < body_pos

    # The concept's own article must not also appear as a title-tier duplicate.
    assert result.output.count("wiki/Raft.md") == 1


def test_find_no_matches_exits_cleanly(config: Config, db: StateDB) -> None:
    from click.testing import CliRunner

    from synto.cli import cli

    _seed_find_fixtures(config, db)

    runner = CliRunner()
    result = runner.invoke(cli, ["find", "--vault", str(config.vault), "zzz-nonexistent"])

    assert result.exit_code == 0, result.output
    assert "No articles found" in result.output


# ---------------------------------------------------------------------------
# Stage 3: 1-hop graph expansion in QueryEngine
# ---------------------------------------------------------------------------


def _write_query_article(config: Config, title: str, body: str) -> None:
    write_note(
        config.wiki_dir / f"{title.replace(' ', '-')}.md",
        {"title": title, "tags": [], "status": "published"},
        body,
    )


def _seed_query_graph_fixtures(vault: Path, config: Config, db: StateDB) -> None:
    """Two published articles, no relations yet — callers add relations as needed."""
    _write_wiki_toml(vault)
    (config.wiki_dir / "index.md").write_text(
        "# Wiki Index\n\n## Concepts\n- [[Raft]]\n- [[Consensus]]\n", encoding="utf-8"
    )
    _write_query_article(config, "Raft", "Raft is a consensus algorithm for replicated logs.")
    _write_query_article(config, "Consensus", "Consensus content: agreement across replicas.")


def _make_query_clients(pages: list[str]) -> tuple[MagicMock, MagicMock]:
    fast_client = MagicMock()
    heavy_client = MagicMock()
    fast_client.generate.return_value = json.dumps({"pages": pages})
    heavy_client.generate.return_value = json.dumps({"answer": "Raft answer.", "title": "Raft"})
    return fast_client, heavy_client


def test_graph_expansion_adds_high_confidence_neighbor(
    vault: Path, config: Config, db: StateDB
) -> None:
    """A high-confidence relation should widen context to the neighbor's page

    so the heavy model can use it, even though the fast model only selected the
    subject page.
    """
    _seed_query_graph_fixtures(vault, config, db)
    db.upsert_relation(
        subject="Raft",
        predicate="depends_on",
        object_="Consensus",
        confidence=0.9,
        source_segment_id="note:a:0",
        evidence_text="Raft depends on consensus.",
    )
    fast_client, heavy_client = _make_query_clients(["Raft"])

    engine = QueryEngine(
        VaultReader(config.vault),
        as_endpoint(fast_client),
        as_endpoint(heavy_client),
        config,
        db=db,
    )
    engine.query("How does Raft achieve agreement?")

    heavy_prompt = heavy_client.generate.call_args.kwargs["prompt"]
    assert "Consensus content" in heavy_prompt
    assert "Consensus" in engine.last_selected_pages


def test_graph_expansion_caps_at_two_extras(vault: Path, config: Config, db: StateDB) -> None:
    """The cap is 2 extras total, not 2 per selected page — three eligible neighbors
    of a single selected page must not all be pulled in."""
    _write_wiki_toml(vault)
    (config.wiki_dir / "index.md").write_text(
        "# Wiki Index\n\n## Concepts\n- [[Raft]]\n", encoding="utf-8"
    )
    _write_query_article(config, "Raft", "Raft is a consensus algorithm.")
    for name in ["Consensus", "Replication", "Leader Election"]:
        _write_query_article(config, name, f"Content about {name}.")
        db.upsert_relation(
            subject="Raft",
            predicate="depends_on",
            object_=name,
            confidence=0.9,
            source_segment_id=f"note:a:{name}",
            evidence_text=f"Raft depends on {name}.",
        )
    fast_client, heavy_client = _make_query_clients(["Raft"])

    engine = QueryEngine(
        VaultReader(config.vault),
        as_endpoint(fast_client),
        as_endpoint(heavy_client),
        config,
        db=db,
    )
    engine.query("Tell me about Raft")

    assert len(engine.last_selected_pages) == 3


def test_graph_expansion_skips_below_confidence_threshold(
    vault: Path, config: Config, db: StateDB
) -> None:
    """A relation below the noise threshold must not widen context — otherwise
    low-confidence extraction errors would leak unrelated pages into every answer."""
    _seed_query_graph_fixtures(vault, config, db)
    db.upsert_relation(
        subject="Raft",
        predicate="depends_on",
        object_="Consensus",
        confidence=0.5,
        source_segment_id="note:a:0",
        evidence_text="weak link",
    )
    fast_client, heavy_client = _make_query_clients(["Raft"])

    engine = QueryEngine(
        VaultReader(config.vault),
        as_endpoint(fast_client),
        as_endpoint(heavy_client),
        config,
        db=db,
    )
    engine.query("Tell me about Raft")

    assert engine.last_selected_pages == ("Raft",)


def test_graph_expansion_noop_without_relations(vault: Path, config: Config, db: StateDB) -> None:
    """No relations recorded yet (e.g. relation extraction never ran) must behave
    exactly like today: no crash, no extras, despite graph_expand_hops defaulting to 1."""
    _seed_query_graph_fixtures(vault, config, db)
    fast_client, heavy_client = _make_query_clients(["Raft"])

    engine = QueryEngine(
        VaultReader(config.vault),
        as_endpoint(fast_client),
        as_endpoint(heavy_client),
        config,
        db=db,
    )
    answer = engine.query("Tell me about Raft")

    assert answer.text == "Raft answer."
    assert engine.last_selected_pages == ("Raft",)
