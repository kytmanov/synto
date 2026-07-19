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

    # Closed-graph invariant: "Network Partitions" is not a known concept, so its edge
    # must NOT export — a consumer may assume every edge endpoint resolves to a node.
    edge_tuples = [
        (e["from_id"], e["to_id"], e["predicate"], e["confidence"]) for e in payload["edges"]
    ]
    assert edge_tuples == [
        (
            concept_key("Vector Clocks"),
            concept_key("Causal Consistency"),
            "implemented_by",
            0.9,
        ),
    ]
    assert concept_key("Network Partitions") not in {t[1] for t in edge_tuples}


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


def test_find_article_title_verbatim(config: Config, db: StateDB) -> None:
    """A published article title is LLM-synthesized and can contain a colon-enclosed
    token; the find table must render it verbatim, not mangled by rich's
    emoji-shortcode parsing (":a:" -> "🅰") — same invariant the trace tables carry."""
    from click.testing import CliRunner

    from synto.cli import cli

    concept_name = "Raft :a: Consensus"
    _write_wiki_toml(config.vault)
    write_note(
        config.wiki_dir / "Raft-Consensus.md",
        {"title": concept_name},
        "Raft is a consensus algorithm.",
    )
    db.upsert_concepts("raw/a.md", [concept_name])
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Raft-Consensus.md",
            title=concept_name,
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["find", "--vault", str(config.vault), "raft"])

    assert result.exit_code == 0, result.output
    assert ":a:" in result.output
    assert "🅰" not in result.output


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
    of a single selected page must not all be pulled in. Confidences are distinct, and
    deliberately out of alphabetical order, so the cap must keep the two STRONGEST
    neighbors (Zab Protocol, Consensus) rather than falling back to SQLite's implicit
    UNION-dedup ordering (alphabetical), which would keep Availability instead of the
    strongest link, Zab Protocol."""
    _write_wiki_toml(vault)
    (config.wiki_dir / "index.md").write_text(
        "# Wiki Index\n\n## Concepts\n- [[Raft]]\n", encoding="utf-8"
    )
    _write_query_article(config, "Raft", "Raft is a consensus algorithm.")
    confidences = {"Availability": 0.71, "Consensus": 0.72, "Zab Protocol": 0.99}
    for name, confidence in confidences.items():
        _write_query_article(config, name, f"Content about {name}.")
        db.upsert_relation(
            subject="Raft",
            predicate="depends_on",
            object_=name,
            confidence=confidence,
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
    assert set(engine.last_selected_pages) == {"Raft", "Zab Protocol", "Consensus"}


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
    exactly like today: no crash, no extras, despite graph_expand defaulting to True."""
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


# ---------------------------------------------------------------------------
# Stage 4: synto trace term/relation/citation CLI commands
# ---------------------------------------------------------------------------


def test_trace_term_command(config: Config, db: StateDB) -> None:
    """Occurrences (with confidence) and the concept's published articles must be
    discoverable by canonical name, so a user can audit where a term came from."""
    from click.testing import CliRunner

    from synto.cli import cli
    from synto.models import TermRecord

    _write_wiki_toml(config.vault)
    db.upsert_concepts("raw/a.md", ["Vector Clocks"])
    db.upsert_concept_occurrences(
        [
            TermRecord(
                name="Vector Clocks",
                definition="A logical clock scheme.",
                source_segment_id="note:a:0",
                provenance="extracted",
                confidence=0.87,
            )
        ],
        source_segment_id="note:a:0",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Vector-Clocks.md",
            title="Vector Clocks",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["trace", "term", "--vault", str(config.vault), "Vector Clocks"])

    assert result.exit_code == 0, result.output
    assert "note:a:0" in result.output
    assert "0.87" in result.output
    assert "Vector Clocks" in result.output


def test_trace_term_covering_article_title_verbatim(config: Config, db: StateDB) -> None:
    """A published article title is LLM-synthesized and can contain a colon-enclosed
    token (e.g. from a quoted phrase); it must render verbatim in the covering-articles
    table, not get mangled by rich's emoji-shortcode parsing (":a:" -> "🅰"). The
    filename is sanitized separately from the title (vault.sanitize_filename), so a
    real published article can carry this token in its title while its path stays clean.
    """
    from click.testing import CliRunner

    from synto.cli import cli
    from synto.models import TermRecord

    concept_name = "Raft :a: Consensus"
    _write_wiki_toml(config.vault)
    db.upsert_concepts("raw/a.md", [concept_name])
    db.upsert_concept_occurrences(
        [
            TermRecord(
                name=concept_name,
                definition="A consensus algorithm.",
                source_segment_id="note:b:0",
                provenance="extracted",
                confidence=0.9,
            )
        ],
        source_segment_id="note:b:0",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Raft-Consensus.md",
            title=concept_name,
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["trace", "term", "--vault", str(config.vault), concept_name])

    assert result.exit_code == 0, result.output
    assert concept_name in result.output
    # The table title also happens to embed concept_name (already Text-safe), so a bare
    # substring check alone can't tell the two tables apart — assert the emoji-shortcode
    # corruption doesn't appear anywhere, which only the (also unwrapped) article-row cell
    # in the covering-articles table can introduce.
    assert "🅰" not in result.output


def test_trace_term_unknown(config: Config) -> None:
    """An unknown term must not crash trace — it should degrade to a friendly message."""
    from click.testing import CliRunner

    from synto.cli import cli

    _write_wiki_toml(config.vault)
    runner = CliRunner()
    result = runner.invoke(cli, ["trace", "term", "--vault", str(config.vault), "Nonexistent Term"])

    assert result.exit_code == 0
    assert "Nonexistent Term" in result.output


def test_trace_relation_command(config: Config, db: StateDB) -> None:
    """Looking up a relation id must surface subject/predicate/object, confidence,
    and the evidence text it was extracted from — the audit trail for feature 26."""
    from click.testing import CliRunner

    from synto.cli import cli

    _write_wiki_toml(config.vault)
    relation_id = db.upsert_relation(
        subject="Raft",
        predicate="depends_on",
        object_="Consensus",
        confidence=0.92,
        source_segment_id="note:a:0",
        evidence_text="Raft depends on consensus.",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["trace", "relation", "--vault", str(config.vault), relation_id])

    assert result.exit_code == 0, result.output
    assert "Raft" in result.output
    assert "depends_on" in result.output
    assert "Consensus" in result.output
    assert "0.92" in result.output
    assert "Raft depends on consensus." in result.output
    assert "note:a:0" in result.output


def test_trace_relation_unknown(config: Config) -> None:
    """A bogus relation id must not crash trace — it should degrade to a friendly message."""
    from click.testing import CliRunner

    from synto.cli import cli

    _write_wiki_toml(config.vault)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["trace", "relation", "--vault", str(config.vault), "deadbeefdeadbeef"]
    )

    assert result.exit_code == 0
    assert "deadbeefdeadbeef" in result.output


def test_trace_citation_command(config: Config, db: StateDB) -> None:
    """A segment id must resolve to the concepts occurring in it and the published
    articles that cite them, with the compile run that produced each article."""
    from click.testing import CliRunner

    from synto.cli import cli
    from synto.models import TermRecord

    _write_wiki_toml(config.vault)
    db.upsert_concepts("raw/a.md", ["Vector Clocks"])
    db.upsert_concept_occurrences(
        [
            TermRecord(
                name="Vector Clocks",
                definition="A logical clock scheme.",
                source_segment_id="note:a:0",
                provenance="extracted",
                confidence=0.9,
            )
        ],
        source_segment_id="note:a:0",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Vector-Clocks.md",
            title="Vector Clocks",
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )
    db.start_compile_run("run123", "{}", "test-fast", "test-heavy")
    db.update_article_compile_run("wiki/Vector-Clocks.md", "run123")

    runner = CliRunner()
    result = runner.invoke(cli, ["trace", "citation", "--vault", str(config.vault), "note:a:0"])

    assert result.exit_code == 0, result.output
    assert "Vector Clocks" in result.output
    assert "run123"[:16] in result.output
    # Segment id must render verbatim, not get mangled by rich's emoji-shortcode
    # parsing (":a:" between colons is a real shortcode, "🅰").
    assert "note:a:0" in result.output


def test_trace_citation_article_title_verbatim(config: Config, db: StateDB) -> None:
    """A published article title is LLM-synthesized and can contain a colon-enclosed
    token; it must render verbatim in the citation table, not get mangled by rich's
    emoji-shortcode parsing (":a:" -> "🅰"). The filename is sanitized separately from
    the title (vault.sanitize_filename), so a real published article can carry this
    token in its title while its path stays clean.
    """
    from click.testing import CliRunner

    from synto.cli import cli
    from synto.models import TermRecord

    concept_name = "Raft :a: Consensus"
    _write_wiki_toml(config.vault)
    db.upsert_concepts("raw/a.md", [concept_name])
    db.upsert_concept_occurrences(
        [
            TermRecord(
                name=concept_name,
                definition="A consensus algorithm.",
                source_segment_id="note:b:0",
                provenance="extracted",
                confidence=0.9,
            )
        ],
        source_segment_id="note:b:0",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Raft-Consensus.md",
            title=concept_name,
            sources=["raw/a.md"],
            content_hash="h1",
            status="published",
        )
    )
    db.start_compile_run("run123", "{}", "test-fast", "test-heavy")
    db.update_article_compile_run("wiki/Raft-Consensus.md", "run123")

    runner = CliRunner()
    result = runner.invoke(cli, ["trace", "citation", "--vault", str(config.vault), "note:b:0"])

    assert result.exit_code == 0, result.output
    assert concept_name in result.output


def test_trace_citation_unknown(config: Config) -> None:
    """An unknown 'note:' segment id must not crash trace — it should degrade to a friendly
    message that also explains WHY: plain notes are never chunked into source_segments, so
    they never get concept_occurrences rows. Without that sentence, a user has no way to
    tell "wrong id" apart from "this kind of id never has data"."""
    from click.testing import CliRunner

    from synto.cli import cli

    _write_wiki_toml(config.vault)
    runner = CliRunner()
    result = runner.invoke(cli, ["trace", "citation", "--vault", str(config.vault), "note:none:0"])

    assert result.exit_code == 0
    assert "note:none:0" in result.output
    normalized = " ".join(result.output.casefold().split())
    assert "plain-note segments have no occurrence records" in normalized
